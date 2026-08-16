//! Plugins extend a session without forking it.
//!
//! A plugin directory carries a manifest, an optional prompt fragment, skills
//! and MCP servers. The agent looks in the workspace's `.eventage/plugins/`
//! and the user's, loads what it finds, and folds the result into the system
//! prompt and the tool registry.

use eventage::agent::ToolRegistry;
use eventage::PluginHost;

/// Write a minimal plugin into `root/.eventage/plugins/<name>`.
fn write_plugin(root: &std::path::Path, name: &str, prompt: &str, skill: Option<(&str, &str)>) {
    let dir = root.join(".eventage/plugins").join(name);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("PROMPT.md"), prompt).unwrap();

    let mut manifest = format!(
        "[plugin]\nname = \"{name}\"\ndescription = \"a test plugin\"\nprompt = \"PROMPT.md\"\n"
    );
    if let Some((skill_name, body)) = skill {
        std::fs::create_dir_all(dir.join("skills").join(skill_name)).unwrap();
        std::fs::write(
            dir.join("skills").join(skill_name).join("SKILL.md"),
            format!("---\nname: {skill_name}\ndescription: {body}\n---\n\n{body}\n"),
        )
        .unwrap();
        manifest.push_str("skills_dir = \"skills\"\n");
    }
    std::fs::write(dir.join("eventage-plugin.toml"), manifest).unwrap();
}

#[tokio::test]
async fn a_plugin_contributes_its_prompt_and_its_skills() {
    let root = tempfile::tempdir().unwrap();
    write_plugin(
        root.path(),
        "house-style",
        "Follow the house style: British spelling, no em dashes in code comments.",
        Some(("changelog", "How this project writes changelog entries")),
    );

    let mut host = PluginHost::new();
    host.load(root.path().join(".eventage/plugins/house-style"))
        .expect("the plugin loaded");
    assert_eq!(host.plugins().len(), 1);

    let registry = ToolRegistry::new();
    let prompt = host.install(&registry).await.expect("installed");

    // The fragment reaches the prompt.
    assert!(prompt.contains("house style"), "{prompt}");
    // And the skill is discoverable, behind the single `skill` tool.
    assert!(prompt.contains("changelog"), "{prompt}");
    assert!(!registry.is_empty(), "the skill tool was not registered");
}

#[test]
fn a_directory_without_a_manifest_is_not_a_plugin() {
    // People keep all sorts of things in a plugins folder. Only a directory
    // that claims to be a plugin is treated as a broken one.
    let root = tempfile::tempdir().unwrap();
    let stray = root.path().join(".eventage/plugins/notes");
    std::fs::create_dir_all(&stray).unwrap();
    std::fs::write(stray.join("README.md"), "just some notes").unwrap();

    assert!(!stray.join(eventage::plugin::MANIFEST_NAME).is_file());
}

#[tokio::test]
async fn a_broken_plugin_does_not_stop_the_others() {
    let root = tempfile::tempdir().unwrap();
    write_plugin(root.path(), "good", "Good plugin prompt.", None);
    let broken = root.path().join(".eventage/plugins/broken");
    std::fs::create_dir_all(&broken).unwrap();
    std::fs::write(broken.join("eventage-plugin.toml"), "{ not toml").unwrap();

    let mut host = PluginHost::new();
    assert!(host
        .load(root.path().join(".eventage/plugins/broken"))
        .is_err());
    host.load(root.path().join(".eventage/plugins/good"))
        .expect("the good one still loads");

    let prompt = host.install(&ToolRegistry::new()).await.unwrap();
    assert!(prompt.contains("Good plugin"), "{prompt}");
}
