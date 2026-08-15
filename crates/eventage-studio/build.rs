//! Make sure `ui/dist` exists before `rust_embed` looks for it.
//!
//! The front-end is built by `ui/build.sh` and deliberately not committed, so
//! a fresh clone has no `ui/dist` and would otherwise fail to compile with an
//! error about a missing folder rather than a missing build step. Writing a
//! placeholder keeps `cargo build` working and puts the instruction in front
//! of whoever opens the app.
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=ui/dist");
    let dist = Path::new("ui/dist");
    let index = dist.join("index.html");
    if index.exists() {
        return;
    }
    let _ = std::fs::create_dir_all(dist);
    let _ = std::fs::write(
        &index,
        "<!doctype html><meta charset=\"utf-8\"><title>Eventage Studio</title>\
         <body style=\"font:14px system-ui;padding:2rem\">\
         <h1>Front-end not built</h1>\
         <p>Run <code>crates/eventage-studio/ui/build.sh</code>, then rebuild.</p>\
         </body>",
    );
}
