fn main() {
    let library = eventage::agent::prompts::PromptLibrary::with_defaults();
    for (name, tokens) in library.inventory() {
        println!("{tokens:>5} tok  {name}");
    }
}
