//! Text bash evaluates at run time as arithmetic, a variable name or a
//! prompt string, where a subscript in a value runs command substitutions.

use std::slice;

use super::changed_variables;
use crate::policy::shell::{self, Script, Word};

/// Characters of an arithmetic expression other than numbers.
const ARITHMETIC_OPERATORS: &str = "+-*/%<>=!&|^~?:,;() \t\n";

/// `[[` operators that evaluate both operands as arithmetic.
const ARITHMETIC_COMPARISONS: &[&str] = &["-eq", "-ne", "-lt", "-le", "-gt", "-ge"];

/// Test operators whose operand is a variable name.
const NAME_TESTS: &[&str] = &["-v", "-R"];

/// Special parameters that always expand to a single number.
const NUMERIC_PARAMETERS: &[&str] = &["$#", "$?", "$$", "$!"];

/// Builtins that declare variables and set their attributes.
const DECLARATIONS: &[&str] = &["declare", "export", "local", "readonly", "typeset"];

/// Of [`DECLARATIONS`], those whose `-i` and `-n` make later values code.
const ATTRIBUTE_DECLARATIONS: &[&str] = &["declare", "local", "typeset"];

const PRINTF: NameSyntax = NameSyntax {
    name_options: "v",
    value_options: "",
    operands_are_names: false,
};

const READ: NameSyntax = NameSyntax {
    name_options: "a",
    value_options: "dinNptu",
    operands_are_names: true,
};

const UNSET: NameSyntax = NameSyntax {
    name_options: "",
    value_options: "",
    operands_are_names: true,
};

const WAIT: NameSyntax = NameSyntax {
    name_options: "p",
    value_options: "",
    operands_are_names: false,
};

/// How a builtin that assigns or reads variables by name takes its
/// arguments.
struct NameSyntax {
    /// Options whose value is a variable name.
    name_options: &'static str,
    /// Options whose value is anything else.
    value_options: &'static str,
    /// Its operands are variable names; otherwise its options end at the
    /// first operand.
    operands_are_names: bool,
}

/// Why `script` needs confirmation for text bash evaluates as code: an
/// arithmetic expression or `${…}` that reads a variable, a `[[` operand,
/// or a change to the tracing prompt.
pub(super) fn in_script(script: &Script) -> Option<String> {
    script
        .arithmetic
        .iter()
        .find(|expression| !is_constant(expression))
        .map(|expression| reads_variables(expression))
        .or_else(|| {
            script
                .parameters
                .iter()
                .find(|body| evaluates_value(body))
                .map(|body| {
                    format!("`${{{body}}}` evaluates a variable's value, which can run commands")
                })
        })
        .or_else(|| conditional_operand(script))
        .or_else(|| changes_trace_prompt(script))
}

/// Why running builtin `program` with `args` needs confirmation, when it
/// evaluates arithmetic or a variable name the review cannot see.
pub(super) fn by_builtin(program: &str, args: &[Word]) -> Option<String> {
    if program == "let" {
        return args
            .iter()
            .find(|arg| !is_constant(&arg.text))
            .map(|arg| reads_variables(&arg.text));
    }
    if DECLARATIONS.contains(&program) {
        return declaration(program, args);
    }
    let hidden = match program {
        "getopts" => args.iter().take(2).find(|arg| may_split(arg)).or_else(|| {
            args.get(1)
                .filter(|name| may_evaluate_subscript(&name.text))
        }),
        "printf" => hidden_name(args, &PRINTF),
        "read" => hidden_name(args, &READ),
        "unset" => hidden_name(args, &UNSET),
        "wait" => hidden_name(args, &WAIT),
        "test" | "[" => args
            .iter()
            .find(|arg| may_split(arg))
            .or_else(|| tested_name(args)),
        _ => None,
    };
    hidden.map(|arg| subscript_reason(program, &arg.text))
}

fn reads_variables(expression: &str) -> String {
    format!(
        "arithmetic `{}` evaluates variable values, which can run commands",
        expression.trim()
    )
}

fn subscript_reason(program: &str, arg: &str) -> String {
    format!("`{program} {arg}` may evaluate an array subscript, which can run commands")
}

/// Whether `expression` holds only numbers and operators, so evaluating it
/// reads no variable.
fn is_constant(expression: &str) -> bool {
    expression
        .split(|c| ARITHMETIC_OPERATORS.contains(c))
        .all(|token| {
            token.is_empty()
                || (token.starts_with(|c: char| c.is_ascii_digit())
                    && token
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '#' | '@' | '_')))
        })
}

/// Whether `${body}` evaluates text the review cannot see: a prompt
/// expansion, an indirection, or a subscript or offset that reads a
/// variable.
fn evaluates_value(body: &str) -> bool {
    if body.ends_with("@P") || is_indirection(body) {
        return true;
    }
    let target = body.trim_start_matches(['!', '#']);
    let after_name = match split_name(target) {
        ("", _) => target.get(1..).unwrap_or_default(),
        (_, after_name) => after_name,
    };
    skip_subscript(after_name).is_none_or(|after| {
        after
            .strip_prefix(':')
            .is_some_and(|range| !range.starts_with(['-', '=', '?', '+']) && !is_constant(range))
    })
}

/// `${!…}` other than the keys of an array or the names with a prefix.
fn is_indirection(body: &str) -> bool {
    let Some(target) = body.strip_prefix('!').filter(|target| !target.is_empty()) else {
        return false;
    };
    let (name, after) = split_name(target);
    !(shell::is_name(name) && matches!(after, "@" | "*" | "[@]" | "[*]"))
}

/// Whether bash, taking `text` as a variable name, maybe followed by
/// `=value`, may evaluate a subscript the review cannot see.
fn may_evaluate_subscript(text: &str) -> bool {
    let (_, after_name) = split_name(text);
    skip_subscript(after_name).is_none_or(|after| {
        !(after.is_empty() || after.starts_with('=') || after.starts_with("+="))
    })
}

/// `text` split after its leading variable name.
fn split_name(text: &str) -> (&str, &str) {
    let end = text
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(text.len());
    text.split_at(end)
}

/// What follows a leading `[subscript]` in `text`, or `text` when it has
/// none; `None` when the subscript may read a variable.
fn skip_subscript(text: &str) -> Option<&str> {
    let Some(inner) = text.strip_prefix('[') else {
        return Some(text);
    };
    let (subscript, after) = inner.split_once(']')?;
    (subscript == "@" || is_constant(subscript)).then_some(after)
}

/// Whether `word` may expand to several words, and so to options and
/// names no check can see.
fn may_split(word: &Word) -> bool {
    word.splits && !NUMERIC_PARAMETERS.contains(&word.text.as_str())
}

fn declaration(program: &str, args: &[Word]) -> Option<String> {
    let attributes_evaluate = ATTRIBUTE_DECLARATIONS.contains(&program);
    args.iter().find_map(|arg| {
        let text = arg.text.as_str();
        match text.strip_prefix(['-', '+']) {
            Some(_) if arg.dynamic => Some(subscript_reason(program, text)),
            Some(flags)
                if attributes_evaluate && text.starts_with('-') && flags.contains(['i', 'n']) =>
            {
                Some(format!(
                    "`{program} {text}` makes bash evaluate values assigned later, which can run commands"
                ))
            }
            Some(_) => None,
            None => may_evaluate_subscript(text).then(|| subscript_reason(program, text)),
        }
    })
}

/// The first of `args` bash may take as a variable name with a subscript
/// the review cannot see, or that may split into such a name.
fn hidden_name<'w>(args: &'w [Word], syntax: &NameSyntax) -> Option<&'w Word> {
    let mut args = args.iter();
    let mut options = true;
    while let Some(arg) = args.next() {
        if may_split(arg) {
            return Some(arg);
        }
        if options && !arg.dynamic && arg.text == "--" {
            options = false;
            continue;
        }
        let cluster = arg
            .text
            .strip_prefix('-')
            .filter(|cluster| options && !cluster.is_empty());
        match cluster {
            Some(_) if arg.dynamic => return Some(arg),
            Some(cluster) => {
                if let Some(hidden) = option_value(cluster, arg, &mut args, syntax) {
                    return Some(hidden);
                }
            }
            None if syntax.operands_are_names => {
                if may_evaluate_subscript(&arg.text) {
                    return Some(arg);
                }
            }
            None => {
                let may_be_option = arg.dynamic && arg.text.starts_with(['$', '`']);
                return args
                    .next()
                    .filter(|next| may_be_option && may_evaluate_subscript(&next.text));
            }
        }
    }
    None
}

/// Consume the value of the option in `cluster` that takes one, returning
/// it when it may hide a variable name.
fn option_value<'w>(
    cluster: &str,
    arg: &'w Word,
    rest: &mut slice::Iter<'w, Word>,
    syntax: &NameSyntax,
) -> Option<&'w Word> {
    let (at, option) = cluster
        .char_indices()
        .find(|&(_, c)| syntax.name_options.contains(c) || syntax.value_options.contains(c))?;
    let attached = &cluster[at + option.len_utf8()..];
    let (value, name) = if attached.is_empty() {
        let value = rest.next()?;
        (value, value.text.as_str())
    } else {
        (arg, attached)
    };
    let hidden =
        may_split(value) || (syntax.name_options.contains(option) && may_evaluate_subscript(name));
    hidden.then_some(value)
}

/// The operand of a `-v` or `-R` test that may hide a subscript.
fn tested_name(words: &[Word]) -> Option<&Word> {
    words
        .windows(2)
        .find(|pair| {
            NAME_TESTS.contains(&pair[0].text.as_str()) && may_evaluate_subscript(&pair[1].text)
        })
        .map(|pair| &pair[1])
}

/// An operand of `[[` evaluated as arithmetic or as a variable name.
fn conditional_operand(script: &Script) -> Option<String> {
    script
        .commands
        .iter()
        .filter(|cmd| {
            cmd.words
                .iter()
                .any(|word| !word.quoted && word.text == "[[")
        })
        .find_map(|cmd| hidden_operand(&cmd.words))
}

fn hidden_operand(words: &[Word]) -> Option<String> {
    let constant = |operand: Option<&Word>| {
        operand.is_some_and(|word| {
            word.text.contains(|c: char| c.is_ascii_digit()) && is_constant(&word.text)
        })
    };
    words
        .iter()
        .enumerate()
        .filter(|(_, word)| !word.quoted)
        .find_map(|(at, word)| {
            let operator = word.text.as_str();
            let before = at.checked_sub(1).and_then(|before| words.get(before));
            let after = words.get(at + 1);
            if ARITHMETIC_COMPARISONS.contains(&operator) {
                return (!constant(before) || !constant(after)).then(|| {
                    format!(
                        "`[[` evaluates the operands of `{operator}` as arithmetic, which can run commands"
                    )
                });
            }
            after
                .filter(|name| {
                    NAME_TESTS.contains(&operator) && may_evaluate_subscript(&name.text)
                })
                .map(|name| subscript_reason(&format!("[[ {operator}"), &name.text))
        })
}

fn changes_trace_prompt(script: &Script) -> Option<String> {
    script
        .commands
        .iter()
        .flat_map(|cmd| &cmd.words)
        .flat_map(|word| changed_variables(&word.text))
        .any(|name| name == "PS4")
        .then(|| "the script changes PS4, which bash expands as a prompt string".to_string())
}
