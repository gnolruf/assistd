//! Lets `assistd` find the sherpa-onnx shared libraries beside it in the
//! build tree, or in `../lib/assistd` once installed.

fn main() {
    println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN:$ORIGIN/../lib/assistd");
}
