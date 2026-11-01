use std::path::Path;

use super::*;

/// Scripts that once ran without confirmation and each needed a fix in the reviewer.
const SHIPPED_BYPASSES: &[&str] = &[
    "rg --pre rm x .",
    "rg --pre=sh x .",
    "rg --hostname-bin=sh x",
    "RIPGREP_CONFIG_PATH=/tmp/rc rg x",
    "export RIPGREP_CONFIG_PATH=/tmp/rc; cat x",
    "sort --compress-program=sh a",
    "sort --comp=sh a",
    "sort -o a b",
    "sort -ro a b",
    "sort --output=a b",
    "ls | xargs sort",
    "sort *",
    "tree -o out",
    "tree -R -H .",
    "file -C -m magic",
    "file --compile -m magic",
    "find . -delete",
    "find . -fprint out",
    "find . -fprint0 out",
    "find . -fprintf out %p",
    "find . -fls out",
    "find . -name '*.o' -exec rm {} +",
    "find . -exec {} \\;",
    "printf 'rm -rf ~' | xargs env",
    "xargs env FOO=1",
    "xargs nice -n 5",
    "xargs timeout 5",
    "xargs -0 xargs",
    "history -s x; history -w ~/.bashrc",
    "env -S 'rm a'",
    "env -Srm a",
    "env --split-string=rm a",
    "hash -p /usr/bin/rm ls; ls",
    "PATH=/tmp:$PATH ls",
    "LD_PRELOAD=/tmp/x.so ls",
    "BASH_ENV=/tmp/x ls",
    "GIT_DIR=/tmp/x ls",
    "PS4='$(rm a)'; set -x; ls",
    "echo $((a[$(rm b)]))",
    "read 'a[$(rm b)]'",
    "printf -v 'a[$(rm b)]' %s y",
    "declare -n ref='a[$(rm b)]'",
    "[[ x -eq y ]]",
    "cmd=rm; $cmd a",
    r#""${EDITOR:-rm}" a"#,
    "'time' -o out ls",
    "/usr/bin/rm a",
    "./ls",
    "source /dev/stdin <<< 'rm a'",
    "echo 'rm a' | bash",
    r#"eval "rm a""#,
    "trap 'rm a' EXIT",
    "alias ll='rm -i'",
    "echo hi > out.txt",
];

fn writes(paths: &[&str]) -> Vec<String> {
    paths.iter().map(ToString::to_string).collect()
}

fn delete(path: &str, recursive: bool) -> Delete {
    Delete {
        path: path.to_string(),
        recursive,
    }
}

#[test]
fn no_shipped_bypass_is_read_only() {
    for script in SHIPPED_BYPASSES {
        assert_ne!(classify(script, None), Class::ReadOnly, "{script:?}");
    }
}

#[test]
fn edits_record_their_targets_as_written() {
    let scratch = Path::new("/run/scratch");
    let cases: &[(&str, Class)] = &[
        (
            "ls -la | sort -k2 > /dev/null 2>&1; grep -rn x . > /run/scratch/out",
            Class::ReadOnly,
        ),
        (
            "rm -rf build && rm -- -f && mkdir -p out/x && echo hi >> ~/notes",
            Class::Edit {
                writes: writes(&["out/x", "~/notes"]),
                deletes: vec![delete("build", true), delete("-f", false)],
            },
        ),
        (
            "mv a b dest/ && cp -r src /tmp/copy && ln -s /etc/hosts",
            Class::Edit {
                writes: writes(&["dest/", "/tmp/copy", "hosts"]),
                deletes: vec![delete("a", true), delete("b", true)],
            },
        ),
        (
            "echo hi | tee",
            Class::Edit {
                writes: Vec::new(),
                deletes: Vec::new(),
            },
        ),
    ];
    for (script, expected) in cases {
        assert_eq!(&classify(script, Some(scratch)), expected, "{script:?}");
    }
    for script in [
        "rm \"$f\"",
        "rm *.o",
        "cp -t dir a",
        "cd sub && rm a",
        "env -C sub rm a",
        "echo x > \"$out\"",
        "ls | xargs rm",
    ] {
        assert!(
            matches!(classify(script, Some(scratch)), Class::Unvetted { .. }),
            "{script:?}"
        );
    }
}
