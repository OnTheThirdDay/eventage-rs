//! Prompts as addressable assets.
//!
//! A harness sends far more prompts to a model than the one the user writes.
//! It summarizes conversations, titles sessions, classifies state, briefs
//! subagents, and asks for judgement in a dozen other places. When each of
//! those lives as a string literal at its call site, three things follow, all
//! bad: nobody can see the full inventory, nobody can change one without
//! forking the crate, and nothing can express the same job done two different
//! ways — a full prompt and a degraded one, a coordinator and a worker.
//!
//! A [`PromptLibrary`] makes each of them a named entry with a default. The
//! framework asks for prompts by name, so an application can replace any of
//! them — including the ones the harness uses on its own behalf — without
//! touching this code.
//!
//! ```
//! use eventage::agent::prompts::{PromptLibrary, names};
//!
//! let library = PromptLibrary::with_defaults();
//!
//! // Every prompt the framework will send is listed and countable.
//! assert!(library.names().contains(&names::SUMMARIZE_FRESH.to_string()));
//!
//! // Any of them can be replaced without forking.
//! library.set(names::SUMMARIZE_FRESH, "Summarize tersely:\n\n{conversation}");
//! let rendered = library.render(names::SUMMARIZE_FRESH, &[("conversation", "hi")]);
//! assert_eq!(rendered, "Summarize tersely:\n\nhi");
//! ```

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Names of the prompts the framework itself sends.
///
/// Applications may register any other name they like; these are the ones
/// with built-in defaults, and replacing one changes framework behaviour.
pub mod names {
    /// First compression of a conversation. Variables: `conversation`.
    pub const SUMMARIZE_FRESH: &str = "context.summarize.fresh";
    /// Extending an existing summary. Variables: `summary`, `conversation`.
    pub const SUMMARIZE_EXTEND: &str = "context.summarize.extend";
    /// Hint shown to a model that appears to be looping. No variables.
    pub const STUCK_HINT: &str = "loop.stuck_hint";
    /// Wrap-up instruction when the step budget is spent. No variables.
    pub const WRAP_UP: &str = "loop.wrap_up";
}

/// A named collection of prompt templates.
///
/// Cheap to clone: clones share one set of prompts, so a library handed to
/// several components stays a single source of truth.
#[derive(Clone)]
pub struct PromptLibrary {
    prompts: Arc<RwLock<HashMap<String, String>>>,
}

impl Default for PromptLibrary {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl PromptLibrary {
    /// An empty library. Rendering an unregistered name yields an empty
    /// string, so a missing prompt degrades rather than panicking mid-session.
    pub fn empty() -> Self {
        Self {
            prompts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// A library carrying the framework's built-in prompts.
    pub fn with_defaults() -> Self {
        let library = Self::empty();
        for (name, body) in DEFAULTS {
            library.set(*name, *body);
        }
        library
    }

    /// Register or replace a prompt.
    pub fn set(&self, name: impl Into<String>, template: impl Into<String>) {
        self.prompts
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(name.into(), template.into());
    }

    /// The raw template, if registered.
    pub fn get(&self, name: &str) -> Option<String> {
        self.prompts
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(name)
            .cloned()
    }

    /// Render a prompt, substituting `{variable}` placeholders.
    ///
    /// Unknown names render empty and unknown placeholders are left alone: a
    /// prompt that is slightly wrong beats a session that dies, and the
    /// leftover `{placeholder}` is visible in the trace.
    pub fn render(&self, name: &str, vars: &[(&str, &str)]) -> String {
        let Some(template) = self.get(name) else {
            tracing::warn!(prompt = name, "no such prompt; rendering empty");
            return String::new();
        };
        let mut out = template;
        for (key, value) in vars {
            out = out.replace(&format!("{{{key}}}"), value);
        }
        out
    }

    /// Every registered name, sorted.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .prompts
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .cloned()
            .collect();
        names.sort();
        names
    }

    /// Name and rough token cost of every prompt.
    ///
    /// Prompts are a standing charge on every request that carries them, and
    /// they grow quietly. Being able to print the bill is most of what keeps
    /// them honest.
    pub fn inventory(&self) -> Vec<(String, usize)> {
        let prompts = self.prompts.read().unwrap_or_else(|e| e.into_inner());
        let mut rows: Vec<(String, usize)> = prompts
            .iter()
            .map(|(name, body)| (name.clone(), body.len().div_ceil(4)))
            .collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        rows
    }
}

/// The framework's built-in prompts.
const DEFAULTS: &[(&str, &str)] = &[
    (
        names::SUMMARIZE_FRESH,
        "Summarize the following conversation history. \
         Your summary MUST start with a \"User instructions\" bullet list \
         that quotes, verbatim, every short instruction or correction the \
         user gave. \
         After the bullet list, write a concise narrative preserving all \
         important context, decisions, and results.\n\n{conversation}",
    ),
    (
        names::SUMMARIZE_EXTEND,
        "You have a summary of earlier conversation history:\n\n\
         {summary}\n\n\
         Extend this summary to also cover the following new messages. \
         Your updated summary MUST start with a \"User instructions\" bullet \
         list that quotes, verbatim, every short instruction or correction \
         the user gave (including any from the previous summary). \
         After the bullet list, write a concise narrative preserving all \
         important context, decisions, and results.\n\n{conversation}",
    ),
    (
        names::STUCK_HINT,
        "You appear to be repeating the same action or error. \
         Try a different approach, use different arguments, \
         or ask the user for clarification.",
    ),
    (
        names::WRAP_UP,
        "You have reached the step limit for this turn. \
         Summarize what you did, what remains, and stop calling tools.",
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_cover_every_prompt_the_framework_sends() {
        let library = PromptLibrary::with_defaults();
        for name in [
            names::SUMMARIZE_FRESH,
            names::SUMMARIZE_EXTEND,
            names::STUCK_HINT,
            names::WRAP_UP,
        ] {
            assert!(library.get(name).is_some(), "missing default for {name}");
        }
    }

    #[test]
    fn an_application_can_replace_a_framework_prompt() {
        // The point of the registry: changing how the harness asks for a
        // summary must not require forking the assembler that asks.
        let library = PromptLibrary::with_defaults();
        library.set(names::SUMMARIZE_FRESH, "Just the facts: {conversation}");
        assert_eq!(
            library.render(names::SUMMARIZE_FRESH, &[("conversation", "abc")]),
            "Just the facts: abc"
        );
    }

    #[test]
    fn clones_share_one_set_of_prompts() {
        let a = PromptLibrary::with_defaults();
        let b = a.clone();
        b.set("custom", "hello");
        assert_eq!(a.get("custom").as_deref(), Some("hello"));
    }

    #[test]
    fn substitutes_every_occurrence_of_a_variable() {
        let library = PromptLibrary::empty();
        library.set("p", "{x} and {x} again, plus {y}");
        assert_eq!(
            library.render("p", &[("x", "1"), ("y", "2")]),
            "1 and 1 again, plus 2"
        );
    }

    #[test]
    fn an_unknown_prompt_renders_empty_rather_than_panicking() {
        // Mid-session is the worst possible time to discover a typo.
        assert_eq!(PromptLibrary::empty().render("nope", &[]), "");
    }

    #[test]
    fn an_unfilled_placeholder_stays_visible() {
        let library = PromptLibrary::empty();
        library.set("p", "value: {missing}");
        assert_eq!(library.render("p", &[]), "value: {missing}");
    }

    #[test]
    fn the_inventory_lists_the_biggest_prompts_first() {
        let library = PromptLibrary::empty();
        library.set("small", "hi");
        library.set("big", "x".repeat(400));
        let inventory = library.inventory();
        assert_eq!(inventory[0].0, "big");
        assert_eq!(inventory[0].1, 100);
        assert_eq!(inventory[1].0, "small");
    }
}
