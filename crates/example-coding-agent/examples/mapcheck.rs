//! Print the repository map a session would be given.
//!
//! ```sh
//! cargo run -p example-coding-agent --example mapcheck -- [path] [token-budget]
//! ```
//!
//! Useful when the agent looks in the wrong place: the map is the first thing
//! it reads, so if a directory is missing here it was missing for the agent.

fn main() {
    let mut args = std::env::args().skip(1);
    let root = args.next().unwrap_or_else(|| ".".into());
    let budget: usize = args.next().and_then(|b| b.parse().ok()).unwrap_or(1_800);

    let started = std::time::Instant::now();
    let map = eventage_code::repomap::build(std::path::Path::new(&root), budget);
    let elapsed = started.elapsed();

    if map.is_empty() {
        eprintln!("no source files found under {root}");
        return;
    }
    print!("{map}");
    eprintln!(
        "\n[{} chars, ~{} tokens, built in {:?}]",
        map.len(),
        map.len() / 4,
        elapsed
    );
}
