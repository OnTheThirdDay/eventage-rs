//! The built front-end, compiled into the binary.
//!
//! Studio ships as one executable with no runtime directory to find, so the
//! Vite output is embedded at build time. `ui/dist` must therefore exist when
//! the crate is compiled — `ui/build.sh` produces it, and a placeholder shell
//! is checked in so a fresh clone still builds.

use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "ui/dist"]
struct Dist;

pub fn exists(path: &str) -> bool {
    Dist::get(path.trim_start_matches('/')).is_some()
}

/// Serve one embedded file with its content type and a cache policy.
pub fn serve(path: &str) -> Response {
    let path = path.trim_start_matches('/');
    match Dist::get(path) {
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
        None => (
            StatusCode::NOT_FOUND,
            "Studio's front-end is missing from this build. Run ui/build.sh and rebuild.",
        )
            .into_response(),
    }
}
