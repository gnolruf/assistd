//! A bash lexer for policy checks: it finds every simple command a script
//! can run. Where it cannot follow bash exactly it errs toward seeing
//! commands that are not there, never toward hiding ones that are.

use std::mem;
use std::ops::Range;

/// How deeply constructs may nest before [`parse`] gives up.
const MAX_DEPTH: usize = 64;

/// Redirection operators other than here-documents, longest first.
const REDIRECT_OPERATORS: &[&str] = &["&>>", "&>", ">>", ">|", ">&", "<&", "<>", ">", "<"];

/// A shell word after quote removal.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct Word {
    /// Quotes and escapes removed; expansions kept as written.
    pub text: String,
    /// Some part is only known at run time (expansion, substitution, glob).
    pub dynamic: bool,
    /// It may become several words at run time.
    pub splits: bool,
    /// It [`splits`](Self::splits) or is dynamic after its last `/`.
    pub dynamic_name: bool,
    /// Some part was quoted or escaped, so it is never a reserved word.
    pub quoted: bool,
}

impl Word {
    /// A word whose text is known exactly, as in an argv.
    pub(super) fn literal(text: &str) -> Self {
        Self {
            text: text.to_string(),
            ..Self::default()
        }
    }
}

/// A redirection other than a here-document or here-string.
#[derive(Debug)]
pub(super) struct Redirect {
    /// One of [`REDIRECT_OPERATORS`], without any file-descriptor prefix.
    pub operator: &'static str,
    pub target: Word,
}

impl Redirect {
    /// Whether it opens its target for writing, rather than reading it or
    /// duplicating or closing a file descriptor (`2>&1`, `>&-`).
    pub(super) fn writes(&self) -> bool {
        match self.operator {
            "<" | "<&" => false,
            ">&" => !names_fd(&self.target),
            _ => true,
        }
    }
}

/// One simple command and the input the script spells out for it.
#[derive(Debug, Default)]
pub(super) struct SimpleCommand {
    pub words: Vec<Word>,
    /// Indexes into [`Script::inputs`] of what is fed to its stdin.
    pub inputs: Vec<usize>,
    pub redirects: Vec<Redirect>,
}

/// Every simple command in a script, in source order, with those inside
/// substitutions and subshells flattened in.
#[derive(Debug, Default)]
pub(super) struct Script {
    pub commands: Vec<SimpleCommand>,
    /// Here-document bodies and here-string words, as written.
    pub inputs: Vec<String>,
    /// Each function defined, with the index into [`Script::commands`] of
    /// the first command after its definition began.
    pub functions: Vec<(String, usize)>,
    /// The script ended inside a quote, substitution or subshell.
    pub incomplete: bool,
}

impl Script {
    /// The inputs the script feeds `command`'s stdin.
    pub(super) fn inputs_of<'s>(
        &'s self,
        command: &'s SimpleCommand,
    ) -> impl Iterator<Item = &'s str> {
        command.inputs.iter().map(|&i| self.inputs[i].as_str())
    }
}

/// Constructs nest more than [`MAX_DEPTH`] levels deep.
#[derive(Debug)]
pub(super) struct TooDeep;

struct Heredoc {
    input: usize,
    delimiter: String,
    strip_tabs: bool,
    expands: bool,
}

struct Lexer<'s, 'o> {
    src: &'s str,
    pos: usize,
    out: &'o mut Script,
    depth: usize,
    heredocs: Vec<Heredoc>,
    incomplete: bool,
    arithmetic: bool,
}

impl<'s, 'o> Lexer<'s, 'o> {
    fn new(src: &'s str, out: &'o mut Script, depth: usize) -> Self {
        Self {
            src,
            pos: 0,
            out,
            depth,
            heredocs: Vec::new(),
            incomplete: false,
            arithmetic: false,
        }
    }

    fn peek(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn peek_at(&self, n: usize) -> Option<char> {
        self.src[self.pos..].chars().nth(n)
    }

    fn next(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn skip(&mut self, chars: usize) {
        for _ in 0..chars {
            self.next();
        }
    }

    fn eat(&mut self, s: &str) -> bool {
        let found = self.src[self.pos..].starts_with(s);
        if found {
            self.pos += s.len();
        }
        found
    }

    fn skip_blanks(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t')) {
            self.next();
        }
    }

    fn descend(&mut self) -> Result<(), TooDeep> {
        if self.depth >= MAX_DEPTH {
            return Err(TooDeep);
        }
        self.depth += 1;
        Ok(())
    }

    /// Lex commands up to the end of input or, `in_parens`, the `)` that
    /// closes them.
    fn commands(&mut self, in_parens: bool) -> Result<(), TooDeep> {
        self.descend()?;
        let mut cmd = SimpleCommand::default();
        loop {
            let Some(c) = self.peek() else {
                self.incomplete |= in_parens;
                break;
            };
            match c {
                ' ' | '\t' => self.skip(1),
                '\n' => {
                    self.skip(1);
                    self.end_command(&mut cmd);
                    self.heredoc_bodies()?;
                }
                '#' => self.skip_comment(),
                ';' | '|' => {
                    self.skip(1);
                    self.end_command(&mut cmd);
                }
                '&' if self.peek_at(1) != Some('>') => {
                    self.skip(1);
                    self.end_command(&mut cmd);
                }
                '(' if self.peek_at(1) == Some('(') => {
                    self.end_command(&mut cmd);
                    self.skip(2);
                    self.double_parens()?;
                }
                '(' => {
                    self.skip(1);
                    if self.function_definition(&mut cmd) {
                        continue;
                    }
                    self.end_command(&mut cmd);
                    self.commands(true)?;
                }
                ')' => {
                    self.skip(1);
                    if in_parens {
                        break;
                    }
                    self.end_command(&mut cmd);
                }
                '&' | '<' | '>' if c == '&' || self.peek_at(1) != Some('(') => {
                    self.redirect(&mut cmd)?;
                }
                _ => self.command_word(&mut cmd)?,
            }
        }
        self.end_command(&mut cmd);
        self.depth -= 1;
        Ok(())
    }

    fn skip_comment(&mut self) {
        while self.peek().is_some_and(|c| c != '\n') {
            self.next();
        }
    }

    /// Lex one word into `cmd`, dropping a file-descriptor prefix on a
    /// redirection.
    fn command_word(&mut self, cmd: &mut SimpleCommand) -> Result<(), TooDeep> {
        if let Some(word) = self.word()? {
            let fd = matches!(self.peek(), Some('<' | '>')) && is_fd(&word);
            if !fd {
                cmd.words.push(word);
            }
        }
        Ok(())
    }

    /// After a `(`: whether it opens a `name ()` function definition. If
    /// so, consumes the `)` and records the name in place of a command.
    fn function_definition(&mut self, cmd: &mut SimpleCommand) -> bool {
        let [name] = cmd.words.as_slice() else {
            return false;
        };
        if name.quoted || name.dynamic {
            return false;
        }
        self.skip_blanks();
        if self.peek() != Some(')') {
            return false;
        }
        self.skip(1);
        let name = cmd.words.remove(0).text;
        self.out.functions.push((name, self.out.commands.len()));
        true
    }

    fn end_command(&mut self, cmd: &mut SimpleCommand) {
        let cmd = mem::take(cmd);
        if let [keyword, name, ..] = cmd.words.as_slice()
            && keyword.text == "function"
            && !keyword.quoted
        {
            self.out
                .functions
                .push((name.text.clone(), self.out.commands.len()));
        }
        if !cmd.words.is_empty() || !cmd.inputs.is_empty() || !cmd.redirects.is_empty() {
            self.out.commands.push(cmd);
        }
    }

    fn push_input(&mut self, text: String) -> usize {
        self.out.inputs.push(text);
        self.out.inputs.len() - 1
    }

    fn redirect(&mut self, cmd: &mut SimpleCommand) -> Result<(), TooDeep> {
        if self.eat("<<<") {
            self.skip_blanks();
            if let Some(word) = self.word()? {
                let input = self.push_input(word.text);
                cmd.inputs.push(input);
            }
            return Ok(());
        }
        if self.eat("<<") {
            let strip_tabs = self.eat("-");
            self.skip_blanks();
            let delimiter = self.word()?.unwrap_or_default();
            let input = self.push_input(String::new());
            cmd.inputs.push(input);
            self.heredocs.push(Heredoc {
                input,
                delimiter: delimiter.text,
                strip_tabs,
                expands: !delimiter.quoted,
            });
            return Ok(());
        }
        let Some(operator) = REDIRECT_OPERATORS.iter().copied().find(|op| self.eat(op)) else {
            return Ok(());
        };
        self.skip_blanks();
        if let Some(target) = self.word()?
            && !self.arithmetic
        {
            cmd.redirects.push(Redirect { operator, target });
        }
        Ok(())
    }

    fn heredoc_bodies(&mut self) -> Result<(), TooDeep> {
        for doc in mem::take(&mut self.heredocs) {
            let (body, closed) = self.heredoc_body(&doc);
            self.incomplete |= !closed;
            if doc.expands {
                let mut lexer = Lexer::new(&body, &mut *self.out, self.depth);
                lexer.double_quoted(&mut WordBuf::default(), false)?;
                self.incomplete |= lexer.incomplete;
            }
            self.out.inputs[doc.input] = body;
        }
        Ok(())
    }

    /// Consume lines up to `doc`'s delimiter, returning the body and
    /// whether the delimiter was found.
    fn heredoc_body(&mut self, doc: &Heredoc) -> (String, bool) {
        let src = self.src;
        let mut body = String::new();
        while self.pos < src.len() {
            let rest = &src[self.pos..];
            let line = &rest[..rest.find('\n').map_or(rest.len(), |i| i + 1)];
            self.pos += line.len();
            let bare = line.strip_suffix('\n').unwrap_or(line);
            let bare = if doc.strip_tabs {
                bare.trim_start_matches('\t')
            } else {
                bare
            };
            if bare == doc.delimiter {
                return (body, true);
            }
            body.push_str(line);
        }
        (body, false)
    }

    /// Lex `src` as a script of its own, or as an `arithmetic` expression,
    /// adding its commands to ours.
    fn nested(&mut self, src: &str, arithmetic: bool) -> Result<(), TooDeep> {
        let mut lexer = Lexer::new(src, &mut *self.out, self.depth);
        lexer.arithmetic = arithmetic;
        lexer.commands(false)?;
        self.incomplete |= lexer.incomplete;
        Ok(())
    }

    /// Consume up to the `close` that balances `depth` consumed `open`s and
    /// lex what lies between as commands. Used for subscripts, extglobs and
    /// `${ …; }`, so a `<<` shift inside is never read as a here-document.
    fn balanced(&mut self, open: char, close: char, depth: usize) -> Result<(), TooDeep> {
        let src = self.src;
        let span = self.balanced_span(open, close, depth);
        self.nested(&src[span.text], false)
    }

    /// After `((` or `$((`: an arithmetic expression or, as bash reads it
    /// when the second `(` does not close right before the first, a subshell.
    fn double_parens(&mut self) -> Result<(), TooDeep> {
        let src = self.src;
        let span = self.balanced_span('(', ')', 2);
        self.nested(&src[span.text], span.closes_together)
    }

    /// Consume up to the `close` that balances `depth` consumed `open`s.
    fn balanced_span(&mut self, open: char, close: char, mut depth: usize) -> Span {
        let start = self.pos;
        let mut second_closed_at = None;
        while let Some(c) = self.next() {
            match c {
                c if c == open => depth += 1,
                c if c == close => {
                    depth -= 1;
                    let close_start = self.pos - close.len_utf8();
                    if depth == 0 {
                        return Span {
                            text: start..close_start,
                            closes_together: second_closed_at == Some(close_start),
                        };
                    }
                    if depth == 1 && second_closed_at.is_none() {
                        second_closed_at = Some(self.pos);
                    }
                }
                '\\' => {
                    self.next();
                }
                '\'' | '"' => self.skip_past(c),
                _ => {}
            }
        }
        self.incomplete = true;
        Span {
            text: start..self.src.len(),
            closes_together: false,
        }
    }

    /// Consume the `( … )` of an array assignment, whose elements are
    /// values but whose substitutions still run.
    fn array(&mut self) -> Result<(), TooDeep> {
        self.descend()?;
        loop {
            match self.peek() {
                None => {
                    self.incomplete = true;
                    break;
                }
                Some(')') => {
                    self.skip(1);
                    break;
                }
                Some(' ' | '\t' | '\n') => self.skip(1),
                Some('#') => self.skip_comment(),
                Some(_) => {
                    let before = self.pos;
                    self.word()?;
                    if self.pos == before {
                        self.skip(1);
                    }
                }
            }
        }
        self.depth -= 1;
        Ok(())
    }

    fn skip_past(&mut self, quote: char) {
        loop {
            match self.next() {
                None => {
                    self.incomplete = true;
                    return;
                }
                Some(c) if c == quote => return,
                Some('\\') if quote == '"' => {
                    self.next();
                }
                Some(_) => {}
            }
        }
    }

    fn word(&mut self) -> Result<Option<Word>, TooDeep> {
        let src = self.src;
        let mut buf = WordBuf::default();
        if matches!(self.peek(), Some('<' | '>')) && self.peek_at(1) == Some('(') {
            self.process_substitution(&mut buf)?;
        }
        while let Some(c) = self.peek() {
            match c {
                ' ' | '\t' | '\n' | ';' | '&' | '|' | ')' | '<' | '>' => break,
                '(' if buf.before_extglob() => {
                    let start = self.pos;
                    self.skip(1);
                    self.balanced('(', ')', 1)?;
                    buf.expansion(&src[start..self.pos], true);
                }
                '(' if buf.is_array_assignment() => {
                    let start = self.pos;
                    self.skip(1);
                    self.array()?;
                    buf.expansion(&src[start..self.pos], false);
                }
                '[' if buf.is_bare_name() => {
                    let start = self.pos;
                    self.skip(1);
                    self.balanced('[', ']', 1)?;
                    buf.expansion(&src[start..self.pos], true);
                }
                '(' => break,
                '\\' => {
                    self.skip(1);
                    match self.next() {
                        Some('\n') | None => {}
                        Some(escaped) => buf.quoted_char(escaped),
                    }
                }
                '\'' => self.single_quoted(&mut buf),
                '"' => {
                    self.skip(1);
                    buf.open_quote();
                    self.double_quoted(&mut buf, true)?;
                }
                '$' => self.dollar(&mut buf, false)?,
                '`' => self.backtick(&mut buf, false)?,
                _ => {
                    self.skip(1);
                    buf.unquoted_char(c);
                }
            }
        }
        Ok(buf.finish())
    }

    fn process_substitution(&mut self, buf: &mut WordBuf) -> Result<(), TooDeep> {
        let src = self.src;
        let start = self.pos;
        self.skip(2);
        self.substitution()?;
        buf.expansion(&src[start..self.pos], true);
        Ok(())
    }

    fn substitution(&mut self) -> Result<(), TooDeep> {
        let arithmetic = mem::take(&mut self.arithmetic);
        let lexed = self.commands(true);
        self.arithmetic = arithmetic;
        lexed
    }

    fn single_quoted(&mut self, buf: &mut WordBuf) {
        self.skip(1);
        buf.open_quote();
        loop {
            match self.next() {
                Some('\'') => return,
                Some(c) => buf.quoted_char(c),
                None => {
                    self.incomplete = true;
                    return;
                }
            }
        }
    }

    /// Lex double-quoted text up to its closing `"` or, when not `closed`,
    /// to the end of input.
    fn double_quoted(&mut self, buf: &mut WordBuf, closed: bool) -> Result<(), TooDeep> {
        loop {
            let Some(c) = self.peek() else {
                self.incomplete |= closed;
                return Ok(());
            };
            match c {
                '"' if closed => {
                    self.skip(1);
                    return Ok(());
                }
                '\\' => {
                    self.skip(1);
                    match self.peek() {
                        Some('\n') => self.skip(1),
                        Some(escaped @ ('"' | '\\' | '$' | '`')) => {
                            self.skip(1);
                            buf.quoted_char(escaped);
                        }
                        _ => buf.quoted_char('\\'),
                    }
                }
                '$' => self.dollar(buf, true)?,
                '`' => self.backtick(buf, true)?,
                _ => {
                    self.skip(1);
                    buf.quoted_char(c);
                }
            }
        }
    }

    fn dollar(&mut self, buf: &mut WordBuf, quoted: bool) -> Result<(), TooDeep> {
        let src = self.src;
        let start = self.pos;
        self.skip(1);
        if !quoted && self.peek() == Some('"') {
            self.skip(1);
            buf.open_quote();
            return self.double_quoted(buf, true);
        }
        match self.after_dollar(quoted)? {
            Some(splits) => {
                let raw = &src[start..self.pos];
                buf.expansion(raw, splits || raw.contains('@'));
            }
            None if quoted => buf.quoted_char('$'),
            None => buf.unquoted_char('$'),
        }
        Ok(())
    }

    /// Consume the expansion after a `$`, returning whether it splits, or
    /// `None` when the `$` is literal.
    fn after_dollar(&mut self, quoted: bool) -> Result<Option<bool>, TooDeep> {
        match self.peek() {
            Some('(') if self.peek_at(1) == Some('(') => {
                self.skip(2);
                self.double_parens()?;
            }
            Some('[') => {
                self.skip(1);
                self.balanced('[', ']', 1)?;
            }
            Some('{') if matches!(self.peek_at(1), Some(' ' | '\t' | '\n' | '|')) => {
                self.skip(1);
                self.balanced('{', '}', 1)?;
            }
            Some('(') => {
                self.skip(1);
                self.substitution()?;
            }
            Some('{') => {
                self.skip(1);
                self.braced(quoted)?;
            }
            Some('\'') if !quoted => {
                self.skip(1);
                self.ansi_c();
                return Ok(Some(false));
            }
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {
                while self
                    .peek()
                    .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
                {
                    self.next();
                }
            }
            Some(c) if c.is_ascii_digit() || "@*#?-$!".contains(c) => self.skip(1),
            _ => return Ok(None),
        }
        Ok(Some(!quoted))
    }

    /// Consume a `${…}` body, lexing the substitutions inside it.
    fn braced(&mut self, quoted: bool) -> Result<(), TooDeep> {
        self.descend()?;
        let mut scratch = WordBuf::default();
        loop {
            let Some(c) = self.peek() else {
                self.incomplete = true;
                break;
            };
            match c {
                '}' => {
                    self.skip(1);
                    break;
                }
                '\\' => self.skip(2),
                '\'' if !quoted => {
                    self.skip(1);
                    self.skip_past('\'');
                }
                '"' => {
                    self.skip(1);
                    self.double_quoted(&mut scratch, true)?;
                }
                '$' => self.dollar(&mut scratch, true)?,
                '`' => self.backtick(&mut scratch, true)?,
                _ => self.skip(1),
            }
        }
        self.depth -= 1;
        Ok(())
    }

    fn ansi_c(&mut self) {
        loop {
            match self.next() {
                None => {
                    self.incomplete = true;
                    return;
                }
                Some('\'') => return,
                Some('\\') => {
                    self.next();
                }
                Some(_) => {}
            }
        }
    }

    /// A backquoted substitution, lexed after bash's backslash unescaping.
    fn backtick(&mut self, buf: &mut WordBuf, quoted: bool) -> Result<(), TooDeep> {
        let src = self.src;
        let start = self.pos;
        self.skip(1);
        let mut body = String::new();
        loop {
            match self.next() {
                None => {
                    self.incomplete = true;
                    break;
                }
                Some('`') => break,
                Some('\\') => match self.next() {
                    Some(escaped @ ('`' | '\\' | '$')) => body.push(escaped),
                    Some(other) => {
                        body.push('\\');
                        body.push(other);
                    }
                    None => body.push('\\'),
                },
                Some(c) => body.push(c),
            }
        }
        self.nested(&body, false)?;
        buf.expansion(&src[start..self.pos], !quoted);
        Ok(())
    }
}

/// Where [`Lexer::balanced_span`] found text between brackets.
struct Span {
    text: Range<usize>,
    /// The bracket closing the second opening one came right before the
    /// last.
    closes_together: bool,
}

#[derive(Default)]
struct WordBuf {
    text: String,
    started: bool,
    quoted: bool,
    dynamic: bool,
    splits: bool,
    tail_dynamic: bool,
    bracket: bool,
    /// An open `{`; `Some(true)` once it has seen `,` or `..`.
    brace: Option<bool>,
    last_unquoted: Option<char>,
}

impl WordBuf {
    fn literal(&mut self, c: char) {
        self.started = true;
        self.text.push(c);
        if c == '/' {
            self.tail_dynamic = false;
        }
    }

    fn open_quote(&mut self) {
        self.started = true;
        self.quoted = true;
        self.last_unquoted = None;
    }

    fn quoted_char(&mut self, c: char) {
        self.quoted = true;
        self.literal(c);
        self.last_unquoted = None;
    }

    fn unquoted_char(&mut self, c: char) {
        let closes_brace_expansion = c == '}' && self.brace.take() == Some(true);
        match c {
            '*' | '?' => self.glob(),
            '[' => self.bracket = true,
            ']' if self.bracket => self.glob(),
            '}' if closes_brace_expansion => self.glob(),
            '{' => self.brace = Some(false),
            ',' if self.brace.is_some() => self.brace = Some(true),
            '.' if self.last_unquoted == Some('.') && self.brace.is_some() => {
                self.brace = Some(true);
            }
            _ => {}
        }
        self.literal(c);
        self.last_unquoted = Some(c);
    }

    fn glob(&mut self) {
        self.dynamic = true;
        self.splits = true;
        self.tail_dynamic = true;
    }

    fn expansion(&mut self, raw: &str, splits: bool) {
        self.started = true;
        self.text.push_str(raw);
        self.dynamic = true;
        self.splits |= splits;
        self.tail_dynamic = true;
        self.last_unquoted = None;
    }

    /// The word so far is a bare name, so a `[` opens a subscript.
    fn is_bare_name(&self) -> bool {
        !self.quoted && !self.dynamic && is_name(&self.text)
    }

    /// The word so far is `name=` or `name+=`, so a `(` opens an array.
    fn is_array_assignment(&self) -> bool {
        !self.quoted
            && !self.dynamic
            && self
                .text
                .strip_suffix('=')
                .map(|name| name.strip_suffix('+').unwrap_or(name))
                .is_some_and(is_name)
    }

    fn before_extglob(&self) -> bool {
        matches!(self.last_unquoted, Some('*' | '?' | '+' | '@' | '!'))
    }

    fn finish(self) -> Option<Word> {
        self.started.then_some(Word {
            dynamic_name: self.splits || self.tail_dynamic,
            text: self.text,
            dynamic: self.dynamic,
            splits: self.splits,
            quoted: self.quoted,
        })
    }
}

/// Lex `src` into its simple commands.
///
/// # Errors
/// [`TooDeep`] when constructs nest more than [`MAX_DEPTH`] levels deep.
pub(super) fn parse(src: &str) -> Result<Script, TooDeep> {
    let mut script = Script::default();
    let mut lexer = Lexer::new(src, &mut script, 0);
    lexer.commands(false)?;
    script.incomplete = lexer.incomplete;
    Ok(script)
}

/// Whether `text` is a shell variable name.
pub(super) fn is_name(text: &str) -> bool {
    text.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        && text.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// A `>&` or `<&` target that names a file descriptor to duplicate or
/// close (`1`, `3-`, `-`) rather than a file.
fn names_fd(target: &Word) -> bool {
    let text = target.text.as_str();
    !target.dynamic
        && !text.is_empty()
        && text
            .strip_suffix('-')
            .unwrap_or(text)
            .bytes()
            .all(|b| b.is_ascii_digit())
}

/// A redirection's file-descriptor prefix: `2` of `2>&1`, `{fd}` of
/// `{fd}>file`.
fn is_fd(word: &Word) -> bool {
    let text = word.text.as_str();
    !word.quoted
        && !word.dynamic
        && !text.is_empty()
        && (text.bytes().all(|b| b.is_ascii_digit())
            || (text.starts_with('{') && text.ends_with('}')))
}
