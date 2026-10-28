Closes #

## What changed

## How it was checked

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets --features assistd-llm/test-support,assistd-mcp/test-support,assistd-voice/test-support -- -D warnings`
- [ ] `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --features assistd-llm/test-support,assistd-mcp/test-support,assistd-voice/test-support`
- [ ] `cargo test --workspace --features assistd-llm/test-support,assistd-mcp/test-support,assistd-voice/test-support`
- [ ] `config/config.sample.toml` updated (if a config key changed)
