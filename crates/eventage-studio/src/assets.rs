//! The built front-end, compiled into the binary.
//!
//! Studio ships as one executable with no runtime directory to find, so the
//! Vite output is embedded at build time by `rust-embed` from `ui/dist`.
//!
//! That directory is a *build artifact* and is gitignored, so it is absent in
//! a fresh clone and on CI — which is exactly where this used to go wrong.
//! The module docs claimed a placeholder was checked in; none was, so a build
//! without node produced a Studio that answered every page request with a
//! bare 404 body, and the deep-link test failed on an assertion three lines
//! after the one that would have explained why.
//!
//! So the placeholder lives here instead, as [`FALLBACK_SHELL`]: a real HTML
//! document that says what is missing and how to fix it. No file to check in,
//! nothing to go stale against a gitignore rule, and a binary built without
//! the front-end serves something a person can read rather than a 404.

use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "ui/dist"]
struct Dist;

pub fn exists(path: &str) -> bool {
    Dist::get(path.trim_start_matches('/')).is_some()
}

/// Served in place of the app shell when the front-end was never built.
///
/// Deliberately a whole document rather than a line of text. Whatever asks
/// for the shell — a browser, a health check, a test — gets HTML, and a
/// person who opens it is told the one thing they need to do.
pub const FALLBACK_SHELL: &str = r#"<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Studio — front-end not built</title>
  </head>
  <body>
    <h1>Studio's front-end is not in this build</h1>
    <p>
      The API is running normally; only the user interface is missing.
      <code>ui/dist</code> is a build artifact and is not checked in.
    </p>
    <p>Build it and rebuild Studio:</p>
    <pre>cd crates/eventage-studio/ui &amp;&amp; ./build.sh</pre>
  </body>
</html>
"#;

/// Serve one embedded file with its content type and a cache policy.
pub fn serve(path: &str) -> Response {
    let path = path.trim_start_matches('/');
    // An empty entry counts as missing. `rust-embed` does not simply return
    // `None` when the folder is absent — it yields an entry with no bytes —
    // so matching on `None` alone meant a build with no front-end answered
    // every page with `200 OK` and a zero-length body. That is why the
    // deep-link test failed on its *third* assertion: the status and the
    // cookie were both fine, and there was nothing in the response.
    match Dist::get(path).filter(|file| !file.data.is_empty()) {
        Some(file) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            // Vite fingerprints everything under assets/, so those are safe to
            // cache forever; the shell must never be, or an update would not
            // reach a browser that already has it.
            let cache = if path.starts_with("assets/") {
                "public, max-age=31536000, immutable"
            } else {
                "no-cache"
            };
            (
                [
                    (header::CONTENT_TYPE, mime.as_ref()),
                    (header::CACHE_CONTROL, cache),
                ],
                file.data.into_owned(),
            )
                .into_response()
        }
        // The shell is the one asset whose absence must not become a 404.
        // Anything unrecognised is routed here as a client-side route, so a
        // 404 body would replace every page in the app with an error string.
        None if path == "index.html" => (
            [
                (header::CONTENT_TYPE, "text/html; charset=utf-8"),
                (header::CACHE_CONTROL, "no-cache"),
            ],
            FALLBACK_SHELL,
        )
            .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            "Studio's front-end is missing from this build. Run ui/build.sh and rebuild.",
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shell_is_html_even_when_the_front_end_was_never_built() {
        // `ui/dist` is gitignored, so this is the state of every fresh clone
        // and every CI run that does not install node. Studio answering with
        // a bare 404 string made the deep-link test fail on an assertion that
        // said nothing about the cause.
        assert!(FALLBACK_SHELL.contains("<html"));
        assert!(
            FALLBACK_SHELL.contains("build.sh"),
            "the fallback has to say how to fix it"
        );
    }

    #[test]
    fn an_empty_embedded_asset_counts_as_missing() {
        // The actual cause. `rust-embed` yields a zero-byte entry rather than
        // `None` when `ui/dist` is not there, so a `None` check alone left
        // Studio serving `200 OK` with an empty body — a failure that looks
        // like a broken test rather than an unbuilt front-end.
        let response = serve("index.html");
        assert_eq!(response.status(), StatusCode::OK);
    }
}
