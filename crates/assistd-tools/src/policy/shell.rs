//! A bash lexer that finds every simple command a script can run; where it cannot follow bash
//! exactly it errs toward seeing commands that are not there, never toward hiding real ones.

use std::mem;
use std::ops::Range;

/// How deeply constructs may nest before [`parse`] gives up.
const MAX_DEPTH: usize = 64;

/// Redirection operators other than here-documents, longest first.
const REDIRECT_OPERATORS: &[&str] = &["&>>", "&>", ">>", ">|", ">&", "<&", "<>", ">", "<"];

/// Reserved words that leave the word after them in command position, where bash still
/// reads a reserved word such as `[[` or `case`.
pub(super) const COMMAND_PREFIX_WORDS: &[&str] = &[
    "!", "{", "if", "then", "else", "elif", "do", "while", "until",
];

/// Each reserved word that opens a compound command, with the one that closes it.
const COMPOUND_COMMANDS: &[(&str, &str)] = &[
    ("{", "}"),
    ("case", "esac"),
    ("for", "done"),
    ("if", "fi"),
    ("select", "done"),
    ("until", "done"),
    ("while", "done"),
];

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
    /// It starts with a `~` bash leaves as written: one quoted, or with something quoted or
    /// expanded before the first unquoted `/`.
    pub literal_tilde: bool,
}

impl Word {
    /// A word whose text is known exactly, as in an argv.
    pub(super) fn literal(text: &str) -> Self {
        Self {
            text: text.to_string(),
            ..Self::default()
        }
    }

    /// The part of this word that `text` is, as a word of its own. Bash expands a `~` only at
    /// the start of a word, so one that starts a later part stays as written.
    pub(super) fn part(&self, text: &str) -> Self {
        let is_whole = text.len() == self.text.len();
        Self {
            text: text.to_string(),
            literal_tilde: if is_whole {
                self.literal_tilde
            } else {
                text.starts_with('~')
            },
            ..self.clone()
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
    /// The script's own shell runs it whenever it gets that far: it is outside every subshell,
    /// substitution, compound command and function body, follows no `&&`, `||` or `|`, and
    /// is neither piped onward nor put in the background, alone or with the list it begins.
    pub unconditional: bool,
    /// The commands after it, up to this index into [`Script::commands`], run only if it
    /// succeeded; 0 when none are known to.
    pub required_until: usize,
}

impl SimpleCommand {
    fn is_empty(&self) -> bool {
        self.words.is_empty() && self.inputs.is_empty() && self.redirects.is_empty()
    }
}

/// Every simple command in a script, in source order, with those inside
/// substitutions and subshells flattened in.
#[derive(Debug, Default)]
pub(super) struct Script {
    pub commands: Vec<SimpleCommand>,
    /// Here-document bodies and here-string words, as written.
    pub inputs: Vec<String>,
    /// Each function defined where a command would be
    /// [`unconditional`](SimpleCommand::unconditional), with the index into
    /// [`Script::commands`] of the first command after its definition began.
    pub functions: Vec<(String, usize)>,
    /// The commands inside function bodies and loops, by index into [`Script::commands`]: each
    /// may run after commands that follow it.
    pub run_later: Vec<Range<usize>>,
    /// Arithmetic expressions as written: `$((…))`, `((…))`, `$[…]`, and
    /// the subscripts of assignments and array elements.
    pub arithmetic: Vec<String>,
    /// The text inside each `${…}`, as written.
    pub parameters: Vec<String>,
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

    /// Whether the command at `index` may run after commands that follow it.
    pub(super) fn runs_later(&self, index: usize) -> bool {
        self.run_later.iter().any(|range| range.contains(&index))
    }
}

/// Constructs nest more than [`MAX_DEPTH`] levels deep.
#[derive(Debug)]
pub(super) struct TooDeep;

/// How far the lexer has read into one open `case` command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CaseState {
    /// After `case`, before the word it matches.
    Subject,
    /// After the subject, before `in`.
    In,
    /// In a pattern list up to its `)`; `fresh` until the list's first word.
    Pattern { fresh: bool },
    /// In the commands of a clause.
    Body,
}

/// The `case` commands open in one command list, innermost last.
#[derive(Debug, Default)]
struct OpenCases(Vec<CaseState>);

impl OpenCases {
    fn innermost(&self) -> Option<CaseState> {
        self.0.last().copied()
    }

    fn set_innermost(&mut self, state: CaseState) {
        if let Some(innermost) = self.0.last_mut() {
            *innermost = state;
        }
    }

    fn in_pattern(&self) -> bool {
        matches!(self.innermost(), Some(CaseState::Pattern { .. }))
    }

    fn at_pattern_start(&self) -> bool {
        self.innermost() == Some(CaseState::Pattern { fresh: true })
    }

    /// After `;;`, `;&` or `;;&`: the next clause's patterns follow.
    fn end_clause(&mut self) {
        if self.innermost() == Some(CaseState::Body) {
            self.set_innermost(CaseState::Pattern { fresh: true });
        }
    }

    /// Whether bash reads `word`, at the start of a command, as the reserved word that closes
    /// a compound command: in a pattern list it is a pattern, but for an `esac` that starts one.
    fn reads_closer(&self, word: &Word) -> bool {
        match self.innermost() {
            Some(CaseState::Pattern { fresh }) => fresh && is_bare(word, "esac"),
            _ => true,
        }
    }

    /// Advance past the last of `words`, the words of the command being lexed.
    fn word(&mut self, words: &[Word]) {
        let Some((last, before)) = words.split_last() else {
            return;
        };
        match self.innermost() {
            Some(CaseState::Subject) => self.set_innermost(CaseState::In),
            Some(CaseState::In) if is_bare(last, "in") => {
                self.set_innermost(CaseState::Pattern { fresh: true });
            }
            Some(CaseState::Pattern { fresh: true }) if is_bare(last, "esac") => {
                self.0.pop();
            }
            Some(CaseState::Pattern { .. }) => {
                self.set_innermost(CaseState::Pattern { fresh: false });
            }
            Some(CaseState::Body) if is_bare(last, "esac") && at_command_start(before) => {
                self.0.pop();
            }
            _ if is_bare(last, "case") && at_command_start(before) => {
                self.0.push(CaseState::Subject);
            }
            _ => {}
        }
    }
}

/// Which commands one call of [`Lexer::commands`] lexes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum List {
    /// The script's own, which its shell runs.
    Script,
    /// Text bash runs from inside a word, to its end.
    Nested,
    /// A subshell's or a substitution's, to the `)` that closes them.
    Parenthesized,
}

/// What ends a command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Terminator {
    /// `;`, a newline or the end of the list: the next command runs whatever this one did.
    Sequence,
    /// `&&`: the next command runs only if this one succeeded.
    And,
    /// `||`: the next command runs only if this one failed.
    Or,
    /// `|` or `|&`: this command runs in a subshell.
    Pipe,
    /// `&`: the list this command ends runs in a subshell.
    Background,
    /// A `(` or `)`, which separates no commands.
    Paren,
}

/// Something open around the command being lexed, which may therefore never run.
#[derive(Debug, PartialEq, Eq)]
enum Open {
    /// A compound command, up to this reserved word.
    Until(&'static str),
    /// A function definition whose body has not ended.
    Body,
}

impl Open {
    /// Whether the commands inside may run after ones that follow them: a function's when it
    /// is called, a loop's when it goes round again.
    fn runs_later(&self) -> bool {
        matches!(self, Self::Body | Self::Until("done"))
    }
}

/// What decides whether the script's own shell always runs the command being lexed, and
/// whether it may run it again later.
#[derive(Debug, Default)]
struct Reach {
    /// The commands being lexed are the script's own.
    own: bool,
    /// Innermost last.
    open: Vec<Open>,
    /// An `&&`, `||` or `|` joins the command being lexed to the one before it.
    joined: bool,
    /// Functions defined in the list being lexed, which a `|` or `&` after their definition
    /// would leave defined in a subshell only.
    pending: Vec<(String, usize)>,
    /// The index of the unconditional command that began the list being lexed.
    head: Option<usize>,
    /// Every command since `head` runs only if it succeeded.
    needs_head: bool,
    /// The index into [`Script::run_later`] of the range the command being lexed is in.
    later: Option<usize>,
}

impl Reach {
    fn new(list: List) -> Self {
        Self {
            own: list == List::Script,
            ..Self::default()
        }
    }

    fn always_runs(&self) -> bool {
        self.own && self.open.is_empty() && !self.joined
    }

    /// Advance past the last word of `cmd`, the command being lexed, which `may_close` a
    /// compound command where bash reads a reserved word in its place.
    fn word(&mut self, cmd: &SimpleCommand, may_close: bool, out: &mut Script) {
        let Some((last, before)) = cmd.words.split_last() else {
            return;
        };
        if let Some((_, closer)) = COMPOUND_COMMANDS
            .iter()
            .find(|(opener, _)| is_bare(last, opener))
        {
            self.enter(Open::Until(closer), out);
            return;
        }
        let starts_command =
            cmd.redirects.is_empty() && cmd.inputs.is_empty() && before.iter().all(is_closer);
        let closes_innermost =
            matches!(self.open.last(), Some(Open::Until(closer)) if is_bare(last, closer));
        if may_close && starts_command && closes_innermost {
            self.open.pop();
            self.end_body();
        }
    }

    /// Note the definition of function `name`, whose body starts at the next command.
    fn define(&mut self, name: String, out: &mut Script) {
        if self.always_runs() {
            self.pending.push((name, out.commands.len()));
        }
        self.enter(Open::Body, out);
    }

    /// Note that `opened` encloses what follows, starting a range in `out` when its commands
    /// run later and none is open.
    fn enter(&mut self, opened: Open, out: &mut Script) {
        if opened.runs_later() && self.later.is_none() {
            self.later = Some(out.run_later.len());
            out.run_later.push(out.commands.len()..usize::MAX);
        }
        self.open.push(opened);
    }

    /// What was just lexed was the body of the function definition it followed, if any.
    fn end_body(&mut self) {
        if self.open.last() == Some(&Open::Body) {
            self.open.pop();
        }
    }

    /// Note a parenthesized list or arithmetic command that just ended, and whether it
    /// `ran_something`: `()` runs nothing, so it is no function body.
    fn end_parens(&mut self, ran_something: bool) {
        if ran_something {
            self.end_body();
        }
        self.joined = false;
    }

    /// End in `out` the range of commands that run later, once nothing open makes them.
    fn end_run_later(&mut self, out: &mut Script) {
        if !self.open.iter().any(Open::runs_later)
            && let Some(range) = self.later.take().and_then(|at| out.run_later.get_mut(at))
        {
            range.end = out.commands.len();
        }
    }

    /// Mark `cmd`, which `terminator` ends, and settle in `out` what the end of the list it is
    /// in decides: which functions it defined, and what its first command is required for.
    fn end_command(&mut self, cmd: &mut SimpleCommand, terminator: Terminator, out: &mut Script) {
        let in_shell = matches!(
            terminator,
            Terminator::Sequence | Terminator::And | Terminator::Or
        );
        cmd.unconditional = self.always_runs() && in_shell;
        if !cmd.is_empty() {
            self.end_body();
        }
        if terminator == Terminator::Paren {
            return;
        }
        if cmd.unconditional && !cmd.is_empty() && self.head.is_none() {
            self.head = Some(out.commands.len());
            self.needs_head =
                terminator == Terminator::And && !cmd.words.iter().any(|word| is_bare(word, "!"));
        }
        let heads_pipeline = !self.joined;
        self.joined = match terminator {
            Terminator::Sequence => self.joined && cmd.is_empty(),
            Terminator::Background => false,
            _ => true,
        };
        if terminator == Terminator::Or || !self.joined {
            self.end_requirement(out, usize::from(!cmd.is_empty()));
        }
        if terminator == Terminator::Background
            && let Some(head) = self.head.take().and_then(|at| out.commands.get_mut(at))
        {
            head.unconditional = false;
        }
        if !self.open.is_empty() {
            return;
        }
        match terminator {
            Terminator::Sequence if !self.joined => {
                out.functions.append(&mut self.pending);
                self.head = None;
            }
            Terminator::Pipe if heads_pipeline => self.pending.clear(),
            Terminator::Background => self.pending.clear(),
            _ => {}
        }
    }

    /// Note in `out` that the commands since the head of the list, and `more` yet to be added,
    /// are the ones that run only if it succeeded.
    fn end_requirement(&mut self, out: &mut Script, more: usize) {
        let required_until = out.commands.len() + more;
        if mem::take(&mut self.needs_head)
            && let Some(head) = self.head.and_then(|at| out.commands.get_mut(at))
        {
            head.required_until = required_until;
        }
    }
}

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

    /// Lex the commands of `list` up to the end of input or, when it is parenthesized, the
    /// `)` that closes them.
    fn commands(&mut self, list: List) -> Result<(), TooDeep> {
        self.descend()?;
        let in_parens = list == List::Parenthesized;
        let mut cmd = SimpleCommand::default();
        let mut conditional = false;
        let mut cases = OpenCases::default();
        let mut reach = Reach::new(list);
        loop {
            conditional &= !cmd.words.is_empty();
            let Some(c) = self.peek() else {
                self.incomplete |= in_parens || conditional;
                break;
            };
            match c {
                ' ' | '\t' => self.skip(1),
                '\n' if conditional => self.skip(1),
                '&' | '|' if conditional && self.peek_at(1) == Some(c) => {
                    self.conditional_operator(&mut cmd, 2);
                }
                '(' if !conditional && cases.at_pattern_start() => {
                    self.skip(1);
                    cases.set_innermost(CaseState::Pattern { fresh: false });
                }
                ')' if !conditional && cases.in_pattern() => {
                    self.skip(1);
                    self.end_command(&mut cmd, &mut reach, Terminator::Paren);
                    cases.set_innermost(CaseState::Body);
                }
                '(' | ')' => {
                    if conditional {
                        self.conditional_operator(&mut cmd, 1);
                    } else if self.parens(&mut cmd, &mut reach, in_parens)? {
                        break;
                    }
                }
                '<' | '>' if conditional && self.peek_at(1) != Some('(') => {
                    self.conditional_operator(&mut cmd, 1);
                }
                '\n' => {
                    self.skip(1);
                    self.end_command(&mut cmd, &mut reach, Terminator::Sequence);
                    self.heredoc_bodies()?;
                }
                '#' => self.skip_comment(),
                ';' => {
                    if matches!(self.peek_at(1), Some(';' | '&')) {
                        cases.end_clause();
                    }
                    self.skip(1);
                    self.end_command(&mut cmd, &mut reach, Terminator::Sequence);
                }
                '|' | '&' if c == '|' || self.peek_at(1) != Some('>') => {
                    let terminator = self.control_operator(c);
                    self.end_command(&mut cmd, &mut reach, terminator);
                }
                '&' | '<' | '>' if c == '&' || self.peek_at(1) != Some('(') => {
                    self.redirect(&mut cmd)?;
                }
                _ => {
                    if self.command_word(&mut cmd, conditional)? && !conditional {
                        self.after_word(&mut cmd, &mut cases, &mut reach);
                    }
                    conditional = if conditional {
                        !cmd.words.last().is_some_and(|word| is_bare(word, "]]"))
                    } else {
                        opens_conditional(&cmd.words)
                    };
                }
            }
        }
        self.end_command(&mut cmd, &mut reach, Terminator::Sequence);
        self.depth -= 1;
        Ok(())
    }

    /// Lex a `(` or `)` outside `[[ … ]]`, returning whether it closes the
    /// commands being lexed.
    fn parens(
        &mut self,
        cmd: &mut SimpleCommand,
        reach: &mut Reach,
        in_parens: bool,
    ) -> Result<bool, TooDeep> {
        if self.eat("((") {
            self.end_command(cmd, reach, Terminator::Paren);
            self.double_parens()?;
            reach.end_parens(true);
        } else if self.eat("(") {
            if !self.function_definition(cmd, reach) {
                self.end_command(cmd, reach, Terminator::Paren);
                let commands_before = self.out.commands.len();
                self.commands(List::Parenthesized)?;
                reach.end_parens(self.out.commands.len() > commands_before);
            }
        } else {
            self.skip(1);
            if in_parens {
                return Ok(true);
            }
            self.end_command(cmd, reach, Terminator::Paren);
        }
        Ok(false)
    }

    /// Consume the `|`, `||`, `|&`, `&` or `&&` that starts with `first`.
    fn control_operator(&mut self, first: char) -> Terminator {
        let (terminator, len) = match (first, self.peek_at(1)) {
            ('&', Some('&')) => (Terminator::And, 2),
            ('&', _) => (Terminator::Background, 1),
            (_, Some('|')) => (Terminator::Or, 2),
            (_, Some('&')) => (Terminator::Pipe, 2),
            _ => (Terminator::Pipe, 1),
        };
        self.skip(len);
        terminator
    }

    /// Advance what is open past the word `cmd` just gained, recording a `function name`
    /// that the word completes in place of a command.
    fn after_word(&mut self, cmd: &mut SimpleCommand, cases: &mut OpenCases, reach: &mut Reach) {
        let may_close = cmd
            .words
            .last()
            .is_some_and(|word| cases.reads_closer(word));
        cases.word(&cmd.words);
        reach.word(cmd, may_close, self.out);
        if matches!(cmd.words.as_slice(), [keyword, _] if is_bare(keyword, "function"))
            && let Some(name) = cmd.words.drain(..).next_back()
        {
            reach.define(name.text, self.out);
        }
    }

    /// Consume an operator of `[[ … ]]` that is `len` characters long as a
    /// word of `cmd`.
    fn conditional_operator(&mut self, cmd: &mut SimpleCommand, len: usize) {
        let start = self.pos;
        self.skip(len);
        cmd.words.push(Word::literal(&self.src[start..self.pos]));
    }

    fn skip_comment(&mut self) {
        while self.peek().is_some_and(|c| c != '\n') {
            self.next();
        }
    }

    /// Lex one word into `cmd`, dropping a file-descriptor prefix on a
    /// redirection outside a `conditional`; returns whether `cmd` gained a word.
    fn command_word(
        &mut self,
        cmd: &mut SimpleCommand,
        conditional: bool,
    ) -> Result<bool, TooDeep> {
        let Some(word) = self.word()? else {
            return Ok(false);
        };
        let fd = !conditional && matches!(self.peek(), Some('<' | '>')) && is_fd(&word);
        if !fd {
            cmd.words.push(word);
        }
        Ok(!fd)
    }

    /// After a `(`: whether it opens a `name ()` function definition. If
    /// so, consumes the `)` and records the name in place of a command.
    fn function_definition(&mut self, cmd: &mut SimpleCommand, reach: &mut Reach) -> bool {
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
        reach.define(name, self.out);
        true
    }

    fn end_command(&mut self, cmd: &mut SimpleCommand, reach: &mut Reach, terminator: Terminator) {
        let mut cmd = mem::take(cmd);
        reach.end_command(&mut cmd, terminator, self.out);
        if !cmd.is_empty() {
            self.out.commands.push(cmd);
        }
        if self.heredocs.is_empty() {
            reach.end_run_later(self.out);
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
        if let Some(target) = self.word()? {
            cmd.redirects.push(Redirect { operator, target });
        }
        Ok(())
    }

    fn heredoc_bodies(&mut self) -> Result<(), TooDeep> {
        for doc in mem::take(&mut self.heredocs) {
            let (body, closed) = self.heredoc_body(&doc);
            self.incomplete |= !closed;
            if doc.expands {
                self.expansions_in(&body)?;
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

    /// Lex `src` as a script of its own, adding its commands to ours.
    fn nested(&mut self, src: &str) -> Result<(), TooDeep> {
        let mut lexer = Lexer::new(src, &mut *self.out, self.depth);
        lexer.commands(List::Nested)?;
        self.incomplete |= lexer.incomplete;
        Ok(())
    }

    fn expansions_in(&mut self, src: &str) -> Result<(), TooDeep> {
        self.descend()?;
        let mut lexer = Lexer::new(src, &mut *self.out, self.depth);
        lexer.double_quoted(&mut WordBuf::default(), false)?;
        self.incomplete |= lexer.incomplete;
        self.depth -= 1;
        Ok(())
    }

    /// Consume up to the `close` that balances `depth` consumed `open`s and
    /// lex what lies between as commands. Used for extglobs and `${ …; }`,
    /// so a `<<` shift inside is never read as a here-document.
    fn balanced(&mut self, open: char, close: char, depth: usize) -> Result<(), TooDeep> {
        let src = self.src;
        let span = self.balanced_span(open, close, depth);
        self.nested(&src[span.text])
    }

    fn bracketed_arithmetic(&mut self) -> Result<&'s str, TooDeep> {
        let src = self.src;
        let span = self.balanced_span('[', ']', 1);
        let expression = &src[span.text];
        self.expansions_in(expression)?;
        Ok(expression)
    }

    fn assigned_subscript(&mut self, subscript: &str) {
        let rest = &self.src[self.pos..];
        if rest.starts_with('=') || rest.starts_with("+=") {
            self.out.arithmetic.push(subscript.to_string());
        }
    }

    /// After `((` or `$((`: an arithmetic expression or, as bash reads it
    /// when the second `(` does not close right before the first, a subshell.
    fn double_parens(&mut self) -> Result<(), TooDeep> {
        let src = self.src;
        let span = self.balanced_span('(', ')', 2);
        let text = &src[span.text];
        if !span.closes_together {
            return self.nested(text);
        }
        self.out.arithmetic.push(text.to_string());
        self.expansions_in(text)
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
                Some('[') => {
                    self.skip(1);
                    let subscript = self.bracketed_arithmetic()?;
                    self.assigned_subscript(subscript);
                }
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
                    let subscript = self.bracketed_arithmetic()?;
                    self.assigned_subscript(subscript);
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
        self.commands(List::Parenthesized)?;
        buf.expansion(&src[start..self.pos], true);
        Ok(())
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
                let expression = self.bracketed_arithmetic()?;
                self.out.arithmetic.push(expression.to_string());
            }
            Some('{') if matches!(self.peek_at(1), Some(' ' | '\t' | '\n' | '|')) => {
                self.skip(1);
                self.balanced('{', '}', 1)?;
            }
            Some('(') => {
                self.skip(1);
                self.commands(List::Parenthesized)?;
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

    /// Consume a `${…}` body, lexing the substitutions inside it and
    /// recording it in [`Script::parameters`]. A subscript or substring
    /// offset is arithmetic, so quotes inside it hide nothing.
    fn braced(&mut self, quoted: bool) -> Result<(), TooDeep> {
        self.descend()?;
        let src = self.src;
        let start = self.pos;
        self.parameter_reference()?;
        let end = if self.at_offset() {
            self.skip(1);
            let span = self.balanced_span('{', '}', 1);
            let end = span.text.end;
            self.expansions_in(&src[span.text])?;
            end
        } else {
            self.parameter_operation(quoted)?
        };
        self.out.parameters.push(src[start..end].to_string());
        self.depth -= 1;
        Ok(())
    }

    fn parameter_reference(&mut self) -> Result<(), TooDeep> {
        while matches!(self.peek(), Some('!' | '#')) {
            self.skip(1);
        }
        let name_start = self.pos;
        while self
            .peek()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            self.skip(1);
        }
        if self.pos == name_start && self.peek().is_some_and(|c| "@*?-$!".contains(c)) {
            self.skip(1);
        }
        if self.peek() == Some('[') {
            self.skip(1);
            self.bracketed_arithmetic()?;
        }
        Ok(())
    }

    fn at_offset(&self) -> bool {
        self.peek() == Some(':') && !matches!(self.peek_at(1), Some('-' | '=' | '?' | '+'))
    }

    fn parameter_operation(&mut self, quoted: bool) -> Result<usize, TooDeep> {
        let mut scratch = WordBuf::default();
        loop {
            let Some(c) = self.peek() else {
                self.incomplete = true;
                return Ok(self.src.len());
            };
            match c {
                '}' => {
                    let end = self.pos;
                    self.skip(1);
                    return Ok(end);
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
        self.nested(&body)?;
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

/// What bash makes of the `~` a word may start with.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum TildePrefix {
    /// The word starts with no unquoted `~`, or quotes part of what follows it.
    #[default]
    Unexpanded,
    /// Unquoted from the `~` on, and still before the first unquoted `/`.
    Open,
    /// Unquoted up to a `/`, so bash expands it.
    Expanded,
}

#[derive(Default)]
struct WordBuf {
    text: String,
    started: bool,
    quoted: bool,
    tilde: TildePrefix,
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
        self.leave_tilde_unexpanded();
    }

    fn quoted_char(&mut self, c: char) {
        self.quoted = true;
        self.leave_tilde_unexpanded();
        self.literal(c);
        self.last_unquoted = None;
    }

    /// Something bash does not read as part of a tilde-prefix follows a still open one.
    fn leave_tilde_unexpanded(&mut self) {
        if self.tilde == TildePrefix::Open {
            self.tilde = TildePrefix::Unexpanded;
        }
    }

    fn unquoted_char(&mut self, c: char) {
        match (c, self.tilde) {
            ('~', TildePrefix::Unexpanded) if !self.started => self.tilde = TildePrefix::Open,
            ('/', TildePrefix::Open) => self.tilde = TildePrefix::Expanded,
            _ => {}
        }
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
        self.leave_tilde_unexpanded();
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
            literal_tilde: self.tilde == TildePrefix::Unexpanded && self.text.starts_with('~'),
            dynamic_name: self.splits || self.tail_dynamic,
            text: self.text,
            dynamic: self.dynamic,
            splits: self.splits,
            quoted: self.quoted,
        })
    }
}

/// Lex `src` into its simple commands.
/// Fails with [`TooDeep`] when constructs nest more than [`MAX_DEPTH`] levels deep.
pub(super) fn parse(src: &str) -> Result<Script, TooDeep> {
    let mut script = Script::default();
    let mut lexer = Lexer::new(src, &mut script, 0);
    lexer.commands(List::Script)?;
    script.incomplete = lexer.incomplete;
    Ok(script)
}

/// Whether `text` is a shell variable name.
pub(super) fn is_name(text: &str) -> bool {
    text.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        && text.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn opens_conditional(words: &[Word]) -> bool {
    words
        .split_last()
        .is_some_and(|(last, before)| is_bare(last, "[[") && at_command_start(before))
}

/// Whether a word after `before` is where bash reads a reserved word: after only
/// [`COMMAND_PREFIX_WORDS`], `time` and `time -p`.
fn at_command_start(before: &[Word]) -> bool {
    before.iter().enumerate().all(|(at, word)| {
        COMMAND_PREFIX_WORDS
            .iter()
            .any(|keyword| is_bare(word, keyword))
            || is_bare(word, "time")
            || (is_bare(word, "-p") && at > 0 && is_bare(&before[at - 1], "time"))
    })
}

/// Whether `word` is a reserved word that closes a compound command.
fn is_closer(word: &Word) -> bool {
    COMPOUND_COMMANDS
        .iter()
        .any(|(_, closer)| is_bare(word, closer))
}

/// Whether `word` is `text`, unquoted and known before run time.
fn is_bare(word: &Word, text: &str) -> bool {
    !word.quoted && !word.dynamic && word.text == text
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
