//! The README's quickstart, as code that has to compile.
//!
//! It lived only in the README, where it named two crates that do not exist
//! (`eventage_provided_impl`, `eventage_llm`) and so had never compiled. A
//! quickstart is the first thing a reader runs; being wrong there costs more
//! trust than being wrong almost anywhere else. Keeping it here means CI
//! notices before a reader does.
//!
//! Run it against a local Ollama:
//!
//! ```text
//! ollama serve && ollama pull qwen3:4b
//! cargo run --example quickstart
//! ```

use eventage::agent::Session;
use eventage::llm::OpenAiProvider;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Build a session with an Ollama provider.
    let mut session = Session::builder()
        .llm(OpenAiProvider::ollama("qwen3:4b"))
        .system_prompt("You are a concise, helpful assistant.")
        // .tool(MyCustomTool)
        .build();

    // 2. Chat with the agent.
    // Under the hood this publishes a `user.message` to the EventBus, runs
    // the `ReactStrategy` loop, and awaits the final `assistant.message`.
    let reply = session.chat("What is the capital of France?").await?;
    println!("Agent: {reply}");

    Ok(())
}
