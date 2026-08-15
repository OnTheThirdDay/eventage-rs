//! Proof that `lsp_rename` works against a real language server.
//!
//! Ignored by default: it spawns rust-analyzer and waits for it to index, so
//! it is far too slow for the normal suite and useless where the server is
//! not installed. Run it with `--ignored` when touching the rename path.

use eventage::agent::Tool;
use eventage_code::lsp::LspPool;
use eventage_code::tools::intel::{LspReferences, LspRename};
use eventage_code::workspace::Workspace;
use serde_json::json;
use std::sync::Arc;

#[tokio::test]
#[ignore = "spawns rust-analyzer"]
async fn rust_analyzer_renames_across_files_and_leaves_lookalikes_alone() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::write(
        root.join("Cargo.toml"),
        "[package]\nname = \"sample\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::create_dir(root.join("src")).unwrap();

    // `width` is defined in one file, used in another, and — crucially —
    // also appears as an unrelated local and inside a comment and a string.
    std::fs::write(
        root.join("src/shape.rs"),
        "pub struct Rect {\n    pub width: u32,\n}\n\nimpl Rect {\n    pub fn area(&self) -> u32 {\n        // width times one\n        self.width * 1\n    }\n}\n",
    )
    .unwrap();
    std::fs::write(
        root.join("src/main.rs"),
        "mod shape;\nuse shape::Rect;\n\nfn main() {\n    let width = 7;\n    let r = Rect { width };\n    println!(\"width {}\", r.width + width);\n}\n",
    )
    .unwrap();

    let ws = Arc::new(Workspace::open(root).unwrap());
    let lsp = Arc::new(LspPool::new(root));
    let (ws2, lsp2) = (ws.clone(), lsp.clone());
    let tool = LspRename {
        ws,
        lsp,
        client: None,
    };

    // rust-analyzer answers position queries with "no references" until it
    // has loaded the crate graph, so poll rather than guess a sleep.
    let references = LspReferences { ws: ws2, lsp: lsp2 };
    let mut resolved = false;
    for _ in 0..30 {
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        if let Ok(out) = references
            .execute(json!({ "path": "src/shape.rs", "line": 2, "character": 9 }))
            .await
        {
            eprintln!("references: {out}");
            if out["count"].as_u64().unwrap_or(0) >= 2 {
                resolved = true;
                break;
            }
        }
    }
    assert!(resolved, "rust-analyzer never indexed the sample crate");

    // `pub width: u32` — line 2, column 9 is inside the field name.
    let out = tool
        .execute(json!({ "path": "src/shape.rs", "line": 2, "character": 9, "new_name": "span" }))
        .await
        .expect("rename failed");

    eprintln!("{}", serde_json::to_string_pretty(&out).unwrap());
    assert_eq!(out["files"], 2, "the rename should reach both files");

    let shape = std::fs::read_to_string(root.join("src/shape.rs")).unwrap();
    let main = std::fs::read_to_string(root.join("src/main.rs")).unwrap();

    assert!(shape.contains("pub span: u32"), "{shape}");
    assert!(shape.contains("self.span * 1"), "{shape}");
    // The comment says "width" and must still say it.
    assert!(
        shape.contains("// width times one"),
        "comment was rewritten:\n{shape}"
    );

    assert!(
        main.contains("Rect { span: width }") || main.contains("span: width"),
        "{main}"
    );
    // The unrelated local binding keeps its name...
    assert!(
        main.contains("let width = 7;"),
        "local was renamed:\n{main}"
    );
    // ...and so does the string literal.
    assert!(
        main.contains("\"width {}\""),
        "string was rewritten:\n{main}"
    );
    assert!(main.contains("r.span"), "{main}");
}
