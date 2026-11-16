//! What the classifier knows of each program it vets: the wrappers that run a command, the
//! programs that only read, and those that edit the paths they name, each with its options.

use super::options::Flag::{Long, Primary, Short};
use super::options::Takes::{Command, Nothing, Value};
use super::options::{Flag, OptionSpec, OptionsEnd, Takes};

/// Builtins and programs that run a command given as their arguments.
const WRAPPERS: &[Wrapper] = &[
    Wrapper {
        name: "builtin",
        options: OptionSpec::LEADING,
        operands: 0,
    },
    Wrapper {
        name: "command",
        options: OptionSpec {
            flags: "pVv",
            ..OptionSpec::LEADING
        },
        operands: 0,
    },
    Wrapper {
        name: "coproc",
        options: OptionSpec::LEADING,
        operands: 0,
    },
    Wrapper {
        name: "env",
        options: OptionSpec {
            flags: "0iv",
            values: "CSu",
            long_flags: &[
                "debug",
                "help",
                "ignore-environment",
                "list-signal-handling",
                "null",
                "version",
            ],
            long_values: &["chdir", "split-string", "unset"],
            long_optional: &["block-signal", "default-signal", "ignore-signal"],
            ..OptionSpec::LEADING
        },
        operands: 0,
    },
    Wrapper {
        name: "exec",
        options: OptionSpec {
            flags: "cl",
            ..OptionSpec::LEADING
        },
        operands: 0,
    },
    Wrapper {
        name: "nice",
        options: OptionSpec {
            flags: "0123456789",
            values: "n",
            long_flags: &["help", "version"],
            long_values: &["adjustment"],
            ..OptionSpec::LEADING
        },
        operands: 0,
    },
    Wrapper {
        name: "nohup",
        options: OptionSpec {
            long_flags: &["help", "version"],
            ..OptionSpec::LEADING
        },
        operands: 0,
    },
    Wrapper {
        name: "stdbuf",
        options: OptionSpec {
            values: "eio",
            long_flags: &["help", "version"],
            long_values: &["error", "input", "output"],
            ..OptionSpec::LEADING
        },
        operands: 0,
    },
    Wrapper {
        name: "time",
        options: OptionSpec {
            flags: "p",
            ..OptionSpec::LEADING
        },
        operands: 0,
    },
    Wrapper {
        name: "timeout",
        options: OptionSpec {
            flags: "fpv",
            values: "ks",
            long_flags: &[
                "foreground",
                "help",
                "preserve-status",
                "verbose",
                "version",
            ],
            long_values: &["kill-after", "signal"],
            ..OptionSpec::LEADING
        },
        operands: 1,
    },
    Wrapper {
        name: "xargs",
        options: OptionSpec {
            flags: "0oprtx",
            values: "adEILnPs",
            optional: "eil",
            long_flags: &[
                "exit",
                "help",
                "interactive",
                "no-run-if-empty",
                "null",
                "open-tty",
                "show-limits",
                "verbose",
                "version",
            ],
            long_values: &[
                "arg-file",
                "delimiter",
                "max-args",
                "max-chars",
                "max-procs",
                "process-slot-var",
            ],
            long_optional: &["eof", "max-lines", "replace"],
            ..OptionSpec::LEADING
        },
        operands: 0,
    },
];

/// The options `md5sum` and the `sha*sum` programs share.
const CHECKSUM_OPTIONS: OptionSpec = OptionSpec {
    flags: "bctwz",
    long_flags: &[
        "binary",
        "check",
        "help",
        "ignore-missing",
        "quiet",
        "status",
        "strict",
        "tag",
        "text",
        "version",
        "warn",
        "zero",
    ],
    ..OptionSpec::NONE
};

/// The options `grep`, `egrep` and `fgrep` share.
const GREP_OPTIONS: OptionSpec = OptionSpec {
    flags: "abcEFGhHiIlLnoPqrRsTuUvVwxyzZ0123456789",
    values: "ABCdDefmX",
    long_flags: &[
        "basic-regexp",
        "binary",
        "byte-offset",
        "count",
        "dereference-recursive",
        "extended-regexp",
        "files-with-matches",
        "files-without-match",
        "fixed-strings",
        "help",
        "ignore-case",
        "initial-tab",
        "invert-match",
        "line-buffered",
        "line-number",
        "line-regexp",
        "no-filename",
        "no-group-separator",
        "no-ignore-case",
        "no-messages",
        "null",
        "null-data",
        "only-matching",
        "perl-regexp",
        "quiet",
        "recursive",
        "silent",
        "text",
        "version",
        "with-filename",
        "word-regexp",
    ],
    long_values: &[
        "after-context",
        "before-context",
        "binary-files",
        "context",
        "devices",
        "directories",
        "exclude",
        "exclude-dir",
        "exclude-from",
        "file",
        "group-separator",
        "include",
        "label",
        "max-count",
        "regexp",
    ],
    long_optional: &["color", "colour"],
    ..OptionSpec::NONE
};

/// `grep`'s options that make it read every file below a directory.
const GREP_RECURSIVE: &[Flag] = &[
    Short('d'),
    Short('r'),
    Short('R'),
    Long("dereference-recursive"),
    Long("directories"),
    Long("recursive"),
];

/// `grep`'s options naming a file of patterns or of globs.
const GREP_FILE_OPTIONS: &[Flag] = &[Short('f'), Long("exclude-from"), Long("file")];

/// `grep`'s options that give its pattern, leaving every operand a file to search.
const GREP_PATTERN_OPTIONS: &[Flag] = &[Short('e'), Short('f'), Long("file"), Long("regexp")];

/// `find`'s operators, options, tests and actions.
const FIND_PRIMARIES: &[(&str, Takes)] = &[
    ("(", Nothing),
    (")", Nothing),
    ("!", Nothing),
    (",", Nothing),
    ("-a", Nothing),
    ("-amin", Value),
    ("-and", Nothing),
    ("-anewer", Value),
    ("-atime", Value),
    ("-cmin", Value),
    ("-cnewer", Value),
    ("-context", Value),
    ("-ctime", Value),
    ("-d", Nothing),
    ("-daystart", Nothing),
    ("-delete", Nothing),
    ("-depth", Nothing),
    ("-empty", Nothing),
    ("-exec", Command),
    ("-execdir", Command),
    ("-executable", Nothing),
    ("-false", Nothing),
    ("-files0-from", Value),
    ("-fls", Value),
    ("-follow", Nothing),
    ("-fprint", Value),
    ("-fprint0", Value),
    ("-fprintf", Value),
    ("-fstype", Value),
    ("-gid", Value),
    ("-group", Value),
    ("-help", Nothing),
    ("-ignore_readdir_race", Nothing),
    ("-ilname", Value),
    ("-iname", Value),
    ("-inum", Value),
    ("-ipath", Value),
    ("-iregex", Value),
    ("-iwholename", Value),
    ("-links", Value),
    ("-lname", Value),
    ("-ls", Nothing),
    ("-maxdepth", Value),
    ("-mindepth", Value),
    ("-mmin", Value),
    ("-mount", Nothing),
    ("-mtime", Value),
    ("-name", Value),
    ("-newer", Value),
    ("-newerat", Value),
    ("-newerct", Value),
    ("-newermt", Value),
    ("-nogroup", Nothing),
    ("-noignore_readdir_race", Nothing),
    ("-noleaf", Nothing),
    ("-not", Nothing),
    ("-nouser", Nothing),
    ("-nowarn", Nothing),
    ("-o", Nothing),
    ("-ok", Command),
    ("-okdir", Command),
    ("-or", Nothing),
    ("-path", Value),
    ("-perm", Value),
    ("-print", Nothing),
    ("-print0", Nothing),
    ("-printf", Value),
    ("-prune", Nothing),
    ("-quit", Nothing),
    ("-readable", Nothing),
    ("-regex", Value),
    ("-regextype", Value),
    ("-samefile", Value),
    ("-size", Value),
    ("-true", Nothing),
    ("-type", Value),
    ("-uid", Value),
    ("-used", Value),
    ("-user", Value),
    ("-version", Nothing),
    ("-warn", Nothing),
    ("-wholename", Value),
    ("-writable", Nothing),
    ("-xdev", Nothing),
    ("-xtype", Value),
];

/// Programs that only read, each with every option it takes.
const READ_ONLY_PROGRAMS: &[ReadOnlyProgram] = &[
    ReadOnlyProgram::reading(
        "b2sum",
        OptionSpec {
            flags: "bctwz",
            values: "l",
            long_flags: CHECKSUM_OPTIONS.long_flags,
            long_values: &["length"],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram::reading(
        "base64",
        OptionSpec {
            flags: "di",
            values: "w",
            long_flags: &["decode", "help", "ignore-garbage", "version"],
            long_values: &["wrap"],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram::naming(
        "basename",
        OptionSpec {
            flags: "az",
            values: "s",
            long_flags: &["help", "multiple", "version", "zero"],
            long_values: &["suffix"],
            ..OptionSpec::LEADING
        },
    ),
    ReadOnlyProgram::reading(
        "cat",
        OptionSpec {
            flags: "AbeEnstTuv",
            long_flags: &[
                "help",
                "number",
                "number-nonblank",
                "show-all",
                "show-ends",
                "show-nonprinting",
                "show-tabs",
                "squeeze-blank",
                "version",
            ],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram::reading(
        "cksum",
        OptionSpec {
            flags: "bctwz",
            values: "al",
            long_flags: &[
                "base64",
                "check",
                "debug",
                "help",
                "ignore-missing",
                "quiet",
                "raw",
                "status",
                "strict",
                "tag",
                "untagged",
                "version",
                "warn",
                "zero",
            ],
            long_values: &["algorithm", "length"],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram::reading(
        "cmp",
        OptionSpec {
            flags: "bclsv",
            values: "in",
            long_flags: &[
                "help",
                "print-bytes",
                "quiet",
                "silent",
                "verbose",
                "version",
            ],
            long_values: &["bytes", "ignore-initial"],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram::reading(
        "column",
        OptionSpec {
            flags: "dehJKLmtVx",
            values: "cCEHilnNoOprRsSTW",
            long_flags: &[
                "fillrows",
                "help",
                "json",
                "keep-empty-lines",
                "table",
                "table-header-as-columns",
                "table-header-repeat",
                "table-maxout",
                "table-noheadings",
                "version",
            ],
            long_values: &[
                "input-separator",
                "output-separator",
                "output-width",
                "separator",
                "table-colorscheme",
                "table-column",
                "table-columns",
                "table-columns-limit",
                "table-hide",
                "table-name",
                "table-noextreme",
                "table-order",
                "table-right",
                "table-truncate",
                "table-wrap",
                "tree",
                "tree-id",
                "tree-parent",
                "use-spaces",
                "wrap-separator",
            ],
            long_optional: &["color"],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram::reading(
        "comm",
        OptionSpec {
            flags: "123z",
            long_flags: &[
                "check-order",
                "help",
                "nocheck-order",
                "total",
                "version",
                "zero-terminated",
            ],
            long_values: &["output-delimiter"],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram::reading(
        "cut",
        OptionSpec {
            flags: "nswz",
            values: "bcdfFO",
            long_flags: &[
                "complement",
                "help",
                "no-partial",
                "only-delimited",
                "version",
                "zero-terminated",
            ],
            long_values: &[
                "bytes",
                "characters",
                "delimiter",
                "fields",
                "output-delimiter",
            ],
            long_optional: &["whitespace-delimited"],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram {
        disqualifiers: &[Short('s'), Long("set")],
        file_options: &[Short('f'), Long("file")],
        ..ReadOnlyProgram::naming(
            "date",
            OptionSpec {
                flags: "Ru",
                values: "dfrs",
                optional: "I",
                long_flags: &[
                    "debug",
                    "help",
                    "resolution",
                    "rfc-2822",
                    "rfc-822",
                    "rfc-email",
                    "universal",
                    "utc",
                    "version",
                ],
                long_values: &["date", "file", "reference", "rfc-3339", "set"],
                long_optional: &["iso-8601"],
                ..OptionSpec::NONE
            },
        )
    },
    ReadOnlyProgram::naming(
        "df",
        OptionSpec {
            flags: "ahHiklmPTv",
            values: "BFtx",
            long_flags: &[
                "all",
                "help",
                "human-readable",
                "inodes",
                "local",
                "no-sync",
                "portability",
                "print-type",
                "si",
                "sync",
                "total",
                "version",
            ],
            long_values: &["block-size", "exclude-type", "type"],
            long_optional: &["output"],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram {
        file_options: &[
            Short('X'),
            Long("exclude-from"),
            Long("from-file"),
            Long("to-file"),
        ],
        recursive: &[Short('r'), Long("recursive")],
        ..ReadOnlyProgram::reading(
            "diff",
            OptionSpec {
                flags: "abBcdeEfhHilnNpPqrstTuvwyZ0123456789",
                values: "CDFILSUWxX",
                long_flags: &[
                    "brief",
                    "ed",
                    "expand-tabs",
                    "help",
                    "ignore-all-space",
                    "ignore-blank-lines",
                    "ignore-case",
                    "ignore-file-name-case",
                    "ignore-space-change",
                    "ignore-tab-expansion",
                    "ignore-trailing-space",
                    "initial-tab",
                    "left-column",
                    "minimal",
                    "new-file",
                    "no-dereference",
                    "no-ignore-file-name-case",
                    "normal",
                    "paginate",
                    "rcs",
                    "recursive",
                    "report-identical-files",
                    "show-c-function",
                    "side-by-side",
                    "speed-large-files",
                    "strip-trailing-cr",
                    "suppress-blank-empty",
                    "suppress-common-lines",
                    "text",
                    "unidirectional-new-file",
                    "version",
                ],
                long_values: &[
                    "changed-group-format",
                    "exclude",
                    "exclude-from",
                    "from-file",
                    "horizon-lines",
                    "ifdef",
                    "ignore-matching-lines",
                    "label",
                    "line-format",
                    "new-group-format",
                    "new-line-format",
                    "old-group-format",
                    "old-line-format",
                    "palette",
                    "show-function-line",
                    "starting-file",
                    "tabsize",
                    "to-file",
                    "unchanged-group-format",
                    "unchanged-line-format",
                    "width",
                ],
                long_optional: &["color", "context", "unified"],
                ..OptionSpec::NONE
            },
        )
    },
    ReadOnlyProgram::naming(
        "dirname",
        OptionSpec {
            flags: "z",
            long_flags: &["help", "version", "zero"],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram {
        file_options: &[Short('X'), Long("exclude-from"), Long("files0-from")],
        ..ReadOnlyProgram::naming(
            "du",
            OptionSpec {
                flags: "0aAbcDhHklLmPsSx",
                values: "BdtX",
                long_flags: &[
                    "all",
                    "apparent-size",
                    "bytes",
                    "count-links",
                    "dereference",
                    "dereference-args",
                    "help",
                    "human-readable",
                    "inodes",
                    "no-dereference",
                    "null",
                    "one-file-system",
                    "separate-dirs",
                    "si",
                    "summarize",
                    "total",
                    "version",
                ],
                long_values: &[
                    "block-size",
                    "exclude",
                    "exclude-from",
                    "files0-from",
                    "max-depth",
                    "threshold",
                    "time-style",
                ],
                long_optional: &["time"],
                ..OptionSpec::NONE
            },
        )
    },
    ReadOnlyProgram::searching("egrep"),
    ReadOnlyProgram::reading(
        "expand",
        OptionSpec {
            flags: "i0123456789,",
            values: "t",
            long_flags: &["help", "initial", "version"],
            long_values: &["tabs"],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram::searching("fgrep"),
    ReadOnlyProgram {
        disqualifiers: &[
            Primary("-delete"),
            Primary("-fls"),
            Primary("-fprint"),
            Primary("-fprint0"),
            Primary("-fprintf"),
        ],
        file_options: &[Primary("-files0-from")],
        ..ReadOnlyProgram::naming(
            "find",
            OptionSpec {
                flags: "HLP",
                values: "D",
                optional: "O",
                long_flags: &["help", "version"],
                primaries: FIND_PRIMARIES,
                end: OptionsEnd::Never,
                ..OptionSpec::NONE
            },
        )
    },
    ReadOnlyProgram::reading(
        "fold",
        OptionSpec {
            flags: "bcs0123456789",
            values: "w",
            long_flags: &["bytes", "characters", "help", "spaces", "version"],
            long_values: &["width"],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram::searching("grep"),
    ReadOnlyProgram::reading(
        "head",
        OptionSpec {
            flags: "qvz0123456789",
            values: "cn",
            long_flags: &[
                "help",
                "quiet",
                "silent",
                "verbose",
                "version",
                "zero-terminated",
            ],
            long_values: &["bytes", "lines"],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram {
        file_options: &[Short('f'), Long("format-file")],
        ..ReadOnlyProgram::reading(
            "hexdump",
            OptionSpec {
                flags: "bcCdhovVxX",
                values: "efns",
                optional: "L",
                long_flags: &[
                    "canonical",
                    "help",
                    "no-squeezing",
                    "one-byte-char",
                    "one-byte-hex",
                    "one-byte-octal",
                    "two-bytes-decimal",
                    "two-bytes-hex",
                    "two-bytes-octal",
                    "version",
                ],
                long_values: &["format", "format-file", "length", "skip"],
                long_optional: &["color"],
                ..OptionSpec::NONE
            },
        )
    },
    ReadOnlyProgram::naming(
        "id",
        OptionSpec {
            flags: "agGnruzZ",
            long_flags: &[
                "context", "group", "groups", "help", "name", "real", "user", "version", "zero",
            ],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram::reading(
        "join",
        OptionSpec {
            flags: "iz",
            values: "12aejotv",
            long_flags: &[
                "check-order",
                "header",
                "help",
                "ignore-case",
                "nocheck-order",
                "version",
                "zero-terminated",
            ],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram::naming(
        "ls",
        OptionSpec {
            flags: "1aAbBcCdDfFgGhHiklLmnNopqQrRsStuUvxXZ",
            values: "ITw",
            long_flags: &[
                "all",
                "almost-all",
                "author",
                "context",
                "dereference",
                "dereference-command-line",
                "dereference-command-line-symlink-to-dir",
                "directory",
                "dired",
                "escape",
                "file-type",
                "full-time",
                "group-directories-first",
                "help",
                "hide-control-chars",
                "human-readable",
                "ignore-backups",
                "inode",
                "kibibytes",
                "literal",
                "no-group",
                "numeric-uid-gid",
                "quote-name",
                "recursive",
                "reverse",
                "show-control-chars",
                "si",
                "size",
                "version",
                "zero",
            ],
            long_values: &[
                "block-size",
                "format",
                "hide",
                "ignore",
                "indicator-style",
                "quoting-style",
                "sort",
                "tabsize",
                "time",
                "time-style",
                "width",
            ],
            long_optional: &["classify", "color", "hyperlink"],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram::reading("md5sum", CHECKSUM_OPTIONS),
    ReadOnlyProgram::reading(
        "nl",
        OptionSpec {
            flags: "p",
            values: "bdfhilnsvw",
            long_flags: &["help", "no-renumber", "version"],
            long_values: &[
                "body-numbering",
                "footer-numbering",
                "header-numbering",
                "join-blank-lines",
                "line-increment",
                "number-format",
                "number-separator",
                "number-width",
                "section-delimiter",
                "starting-line-number",
            ],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram::reading(
        "od",
        OptionSpec {
            flags: "abBcdDefFhHiIlLoOsvxX",
            values: "AjNSt",
            optional: "w",
            long_flags: &["help", "output-duplicates", "traditional", "version"],
            long_values: &[
                "address-radix",
                "endian",
                "format",
                "read-bytes",
                "skip-bytes",
            ],
            long_optional: &["strings", "width"],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram::reading(
        "paste",
        OptionSpec {
            flags: "sz",
            values: "d",
            long_flags: &["help", "serial", "version", "zero-terminated"],
            long_values: &["delimiters"],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram::naming(
        "printenv",
        OptionSpec {
            flags: "0",
            long_flags: &["help", "null", "version"],
            ..OptionSpec::LEADING
        },
    ),
    ReadOnlyProgram::naming(
        "readlink",
        OptionSpec {
            flags: "efmnqsvz",
            long_flags: &[
                "canonicalize",
                "canonicalize-existing",
                "canonicalize-missing",
                "help",
                "no-newline",
                "quiet",
                "silent",
                "verbose",
                "version",
                "zero",
            ],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram::naming(
        "realpath",
        OptionSpec {
            flags: "eELmPqsz",
            long_flags: &[
                "canonicalize",
                "canonicalize-existing",
                "canonicalize-missing",
                "help",
                "logical",
                "no-symlinks",
                "physical",
                "quiet",
                "strip",
                "version",
                "zero",
            ],
            long_values: &["relative-base", "relative-to"],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram::reading(
        "rev",
        OptionSpec {
            flags: "0hV",
            long_flags: &["help", "version", "zero"],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram::naming(
        "seq",
        OptionSpec {
            flags: "w0123456789.",
            values: "fs",
            long_flags: &["equal-width", "help", "version"],
            long_values: &["format", "separator"],
            ..OptionSpec::LEADING
        },
    ),
    ReadOnlyProgram::reading("sha1sum", CHECKSUM_OPTIONS),
    ReadOnlyProgram::reading("sha256sum", CHECKSUM_OPTIONS),
    ReadOnlyProgram::reading("sha512sum", CHECKSUM_OPTIONS),
    ReadOnlyProgram::naming(
        "sleep",
        OptionSpec {
            long_flags: &["help", "version"],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram {
        disqualifiers: &[Short('o'), Long("compress-program"), Long("output")],
        file_options: &[Long("files0-from"), Long("random-source")],
        ..ReadOnlyProgram::reading(
            "sort",
            OptionSpec {
                flags: "bcCdfghimMnrRsuVz",
                values: "koStTy",
                long_flags: &[
                    "debug",
                    "dictionary-order",
                    "general-numeric-sort",
                    "help",
                    "human-numeric-sort",
                    "ignore-case",
                    "ignore-leading-blanks",
                    "ignore-nonprinting",
                    "merge",
                    "month-sort",
                    "numeric-sort",
                    "random-sort",
                    "reverse",
                    "stable",
                    "unique",
                    "version",
                    "version-sort",
                    "zero-terminated",
                ],
                long_values: &[
                    "batch-size",
                    "buffer-size",
                    "compress-program",
                    "field-separator",
                    "files0-from",
                    "key",
                    "output",
                    "parallel",
                    "random-source",
                    "sort",
                    "temporary-directory",
                ],
                long_optional: &["check"],
                ..OptionSpec::NONE
            },
        )
    },
    ReadOnlyProgram::naming(
        "stat",
        OptionSpec {
            flags: "fLt",
            values: "c",
            long_flags: &["dereference", "file-system", "help", "terse", "version"],
            long_values: &["cached", "format", "printf"],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram {
        argument_files: true,
        ..ReadOnlyProgram::reading(
            "strings",
            OptionSpec {
                flags: "adfhHovVw0123456789",
                values: "enstTU",
                long_flags: &[
                    "all",
                    "data",
                    "help",
                    "include-all-whitespace",
                    "print-file-name",
                    "version",
                ],
                long_values: &[
                    "bytes",
                    "encoding",
                    "output-separator",
                    "radix",
                    "target",
                    "unicode",
                ],
                ..OptionSpec::NONE
            },
        )
    },
    ReadOnlyProgram::reading(
        "tac",
        OptionSpec {
            flags: "br",
            values: "s",
            long_flags: &["before", "help", "regex", "version"],
            long_values: &["separator"],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram::reading(
        "tail",
        OptionSpec {
            flags: "bfFlqvz0123456789",
            values: "cns",
            long_flags: &[
                "debug",
                "help",
                "quiet",
                "retry",
                "silent",
                "verbose",
                "version",
                "zero-terminated",
            ],
            long_values: &[
                "bytes",
                "lines",
                "max-unchanged-stats",
                "pid",
                "sleep-interval",
            ],
            long_optional: &["follow"],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram::reading(
        "tr",
        OptionSpec {
            flags: "AcCdst",
            long_flags: &[
                "complement",
                "delete",
                "help",
                "squeeze-repeats",
                "truncate-set1",
                "version",
            ],
            ..OptionSpec::LEADING
        },
    ),
    ReadOnlyProgram::naming(
        "uname",
        OptionSpec {
            flags: "aAimnoprsv",
            long_flags: &[
                "all",
                "all-labeled",
                "hardware-platform",
                "help",
                "kernel-name",
                "kernel-release",
                "kernel-version",
                "machine",
                "nodename",
                "operating-system",
                "processor",
                "version",
            ],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram::reading(
        "unexpand",
        OptionSpec {
            flags: "a0123456789,",
            values: "t",
            long_flags: &["all", "first-only", "help", "version"],
            long_values: &["tabs"],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram {
        file_options: &[Long("files0-from")],
        ..ReadOnlyProgram::reading(
            "wc",
            OptionSpec {
                flags: "clLmw",
                long_flags: &[
                    "bytes",
                    "chars",
                    "debug",
                    "help",
                    "lines",
                    "max-line-length",
                    "version",
                    "words",
                ],
                long_values: &["files0-from", "total"],
                ..OptionSpec::NONE
            },
        )
    },
    ReadOnlyProgram::naming(
        "which",
        OptionSpec {
            flags: "aivV",
            long_flags: &[
                "all",
                "help",
                "read-alias",
                "read-functions",
                "show-dot",
                "show-tilde",
                "skip-alias",
                "skip-dot",
                "skip-functions",
                "skip-tilde",
                "tty-only",
                "version",
            ],
            ..OptionSpec::NONE
        },
    ),
    ReadOnlyProgram::naming(
        "whoami",
        OptionSpec {
            long_flags: &["help", "version"],
            ..OptionSpec::NONE
        },
    ),
];

/// Programs that write or delete the literal paths they name, with the options vetted for each.
const EDIT_PROGRAMS: &[EditProgram] = &[
    EditProgram {
        name: "chmod",
        options: OptionSpec {
            flags: "cfvR",
            long_flags: &[
                "changes",
                "no-preserve-root",
                "preserve-root",
                "quiet",
                "recursive",
                "silent",
                "verbose",
            ],
            ..OptionSpec::NONE
        },
        targets: Targets::ChangesAllButFirst,
    },
    EditProgram {
        name: "cp",
        options: OptionSpec {
            flags: "adfHiLnPprRTuvx",
            long_flags: &[
                "archive",
                "dereference",
                "force",
                "interactive",
                "no-clobber",
                "no-dereference",
                "no-target-directory",
                "one-file-system",
                "recursive",
                "update",
                "verbose",
            ],
            ..OptionSpec::NONE
        },
        targets: Targets::WritesLast,
    },
    EditProgram {
        name: "mkdir",
        options: OptionSpec {
            flags: "pv",
            values: "m",
            long_flags: &["parents", "verbose"],
            ..OptionSpec::NONE
        },
        targets: Targets::WritesAll,
    },
    EditProgram {
        name: "mv",
        options: OptionSpec {
            flags: "finTuv",
            long_flags: &[
                "force",
                "interactive",
                "no-clobber",
                "no-target-directory",
                "update",
                "verbose",
            ],
            ..OptionSpec::NONE
        },
        targets: Targets::Moves,
    },
    EditProgram {
        name: "rm",
        options: OptionSpec {
            flags: "dfIirRv",
            long_flags: &[
                "dir",
                "force",
                "interactive",
                "no-preserve-root",
                "one-file-system",
                "preserve-root",
                "recursive",
                "verbose",
            ],
            ..OptionSpec::NONE
        },
        targets: Targets::Removes,
    },
    EditProgram {
        name: "tee",
        options: OptionSpec {
            flags: "aip",
            long_flags: &["append", "ignore-interrupts"],
            ..OptionSpec::NONE
        },
        targets: Targets::WritesAll,
    },
    EditProgram {
        name: "touch",
        options: OptionSpec {
            flags: "achm",
            values: "dt",
            long_flags: &["no-create", "no-dereference"],
            ..OptionSpec::NONE
        },
        targets: Targets::WritesAll,
    },
];

/// A builtin or program that runs a command given as its arguments.
#[derive(Debug)]
pub(super) struct Wrapper {
    pub name: &'static str,
    /// Every option it takes that is vetted, read up to its first operand.
    pub options: OptionSpec,
    /// How many operands come before the command it runs.
    pub operands: usize,
}

/// A program that only reads, unless given one of its `disqualifiers`.
#[derive(Debug)]
pub(super) struct ReadOnlyProgram {
    pub name: &'static str,
    /// Every option it takes.
    pub options: OptionSpec,
    /// The options that make it write a file or run a program.
    pub disqualifiers: &'static [Flag],
    /// The options naming a file it reads.
    pub file_options: &'static [Flag],
    /// The options that make it read every file below a directory operand.
    pub recursive: &'static [Flag],
    pub operands: Operands,
    /// It replaces an `@file` argument with the arguments that file holds.
    pub argument_files: bool,
}

impl ReadOnlyProgram {
    /// One that reads the files its operands name, and no other.
    const fn reading(name: &'static str, options: OptionSpec) -> Self {
        Self {
            name,
            options,
            disqualifiers: &[],
            file_options: &[],
            recursive: &[],
            operands: Operands::Files,
            argument_files: false,
        }
    }

    /// One that reads the contents of no file its operands name.
    const fn naming(name: &'static str, options: OptionSpec) -> Self {
        Self {
            operands: Operands::Unread,
            ..Self::reading(name, options)
        }
    }

    /// One that takes `grep`'s arguments.
    const fn searching(name: &'static str) -> Self {
        Self {
            file_options: GREP_FILE_OPTIONS,
            recursive: GREP_RECURSIVE,
            operands: Operands::PatternThenFiles(GREP_PATTERN_OPTIONS),
            ..Self::reading(name, GREP_OPTIONS)
        }
    }
}

/// What a [`ReadOnlyProgram`]'s operands are to it.
#[derive(Debug, Clone, Copy)]
pub(super) enum Operands {
    /// Not files whose contents it reads.
    Unread,
    /// Files it reads.
    Files,
    /// A pattern, unless one of these options gives it, and then files it searches.
    PatternThenFiles(&'static [Flag]),
}

/// A program that writes or deletes the paths it names.
#[derive(Debug)]
pub(super) struct EditProgram {
    pub name: &'static str,
    /// The options vetted for it.
    pub options: OptionSpec,
    pub targets: Targets,
}

/// Which operands an [`EditProgram`] writes or deletes.
#[derive(Debug, Clone, Copy)]
pub(super) enum Targets {
    /// Deletes every operand.
    Removes,
    /// Writes every operand.
    WritesAll,
    /// Reads all but its last operand, which it writes.
    WritesLast,
    /// Deletes all but its last operand, which it writes.
    Moves,
    /// Changes the mode of every operand after the first, which names the change.
    ChangesAllButFirst,
}

/// The wrapper called `name`.
pub(super) fn wrapper(name: &str) -> Option<&'static Wrapper> {
    WRAPPERS.iter().find(|wrapper| wrapper.name == name)
}

/// The read-only program called `name`.
pub(super) fn read_only(name: &str) -> Option<&'static ReadOnlyProgram> {
    READ_ONLY_PROGRAMS
        .iter()
        .find(|program| program.name == name)
}

/// The edit program called `name`.
pub(super) fn edit(name: &str) -> Option<&'static EditProgram> {
    EDIT_PROGRAMS.iter().find(|program| program.name == name)
}
