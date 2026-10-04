Closes #

## What changed

## How it was checked

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`
- [ ] `cargo test --workspace`
- [ ] `config/config.sample.toml` updated (if a config key changed)
