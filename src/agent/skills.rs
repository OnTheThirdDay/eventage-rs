//! Agent Skills — `SKILL.md` bundles with progressive disclosure.
//!
//! A *skill* is a directory containing a `SKILL.md` file with YAML
//! frontmatter (`name`, `description`) followed by markdown instructions,
//! optionally alongside supporting files (scripts, templates, references).
//! This is the same format used by Claude Code / Claude Agent Skills, so
//! existing skill directories work as-is.
//!
//! Skills load in three levels of detail (progressive disclosure), keeping
//! the context lean:
//!
//! 1. **Discovery** — [`SkillsLibrary::system_prompt_section`] renders one
//!    line per skill (name + description) for the system prompt.
//! 2. **On demand** — the model calls the [`SkillTool`] with a skill name and
//!    receives the full `SKILL.md` body plus a listing of bundled files.
//! 3. **Deep dive** — the model opens bundled files with its regular file
//!    tools as the skill instructs.
//!
//! ```no_run
//! use eventage::agent::skills::{SkillsLibrary, SkillTool};
//!
//! let library = SkillsLibrary::discover("./skills").unwrap();
//! let system_prompt = format!("You are helpful.\n\n{}", library.system_prompt_section());
//! let skill_tool = SkillTool::new(library);
//! // AgentBuilder::new().system_prompt(system_prompt).tool(skill_tool)...
//! ```

use super::error::AgentError;
use super::tool::Tool;
use crate::llm::types::ToolDefinition;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::warn;

/// One discovered skill.
#[derive(Debug, Clone)]
pub struct Skill {
    /// Frontmatter `name` (falls back to the directory name).
    pub name: String,
    /// Frontmatter `description` — the model decides from this line whether
    /// the skill is relevant, so it should say what the skill does *and when
    /// to use it*.
    pub description: String,
    /// Directory containing `SKILL.md` and any bundled files.
    pub dir: PathBuf,
    /// Markdown body of `SKILL.md` (frontmatter stripped).
    pub instructions: String,
}

/// A collection of skills discovered from one or more directories.
#[derive(Debug, Clone, Default)]
pub struct SkillsLibrary {
    skills: BTreeMap<String, Skill>,
}

impl SkillsLibrary {
    pub fn new() -> Self {
        Self::default()
    }

    /// Scan `dir` for `<skill-name>/SKILL.md` entries.
    ///
    /// Malformed skills are skipped with a warning; an empty or missing
    /// directory yields an empty library rather than an error.
    pub fn discover(dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let mut library = Self::new();
        library.add_dir(dir)?;
        Ok(library)
    }

    /// Add every skill under `dir` to this library (later duplicates win).
    pub fn add_dir(&mut self, dir: impl AsRef<Path>) -> std::io::Result<()> {
        let dir = dir.as_ref();
        if !dir.is_dir() {
            return Ok(());
        }
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let skill_dir = entry.path();
            if !skill_dir.is_dir() {
                continue;
            }
            let skill_md = skill_dir.join("SKILL.md");
            if !skill_md.is_file() {
                continue;
            }
            match Self::parse_skill(&skill_dir, &skill_md) {
                Ok(skill) => {
                    self.skills.insert(skill.name.clone(), skill);
                }
                Err(e) => warn!(path = %skill_md.display(), "skipping malformed skill: {e}"),
            }
        }
        Ok(())
    }

    /// Register a skill directly (e.g. built in code rather than on disk).
    pub fn add_skill(&mut self, skill: Skill) {
        self.skills.insert(skill.name.clone(), skill);
    }

    fn parse_skill(dir: &Path, skill_md: &Path) -> std::io::Result<Skill> {
        let raw = std::fs::read_to_string(skill_md)?;
        let (frontmatter, body) = split_frontmatter(&raw);
        let name = frontmatter
            .get("name")
            .cloned()
            .or_else(|| dir.file_name().map(|n| n.to_string_lossy().into_owned()))
            .unwrap_or_default();
        let description = frontmatter.get("description").cloned().unwrap_or_default();
        Ok(Skill {
            name,
            description,
            dir: dir.to_path_buf(),
            instructions: body.trim().to_string(),
        })
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.skills.get(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Skill> {
        self.skills.values()
    }

    /// Render the discovery-level listing for the system prompt.
    ///
    /// Returns an empty string when no skills are loaded.
    pub fn system_prompt_section(&self) -> String {
        if self.skills.is_empty() {
            return String::new();
        }
        let mut out = String::from(
            "## Skills\n\
             The following skills are available. When a task matches a skill's \
             description, call the `skill` tool with its name to load the full \
             instructions before proceeding.\n",
        );
        for skill in self.skills.values() {
            out.push_str(&format!("- {}: {}\n", skill.name, skill.description));
        }
        out
    }
}

/// Split `---` YAML frontmatter from a markdown document.
///
/// Parses only simple top-level `key: value` scalars — enough for skill
/// metadata without a YAML dependency.
fn split_frontmatter(raw: &str) -> (BTreeMap<String, String>, &str) {
    let mut meta = BTreeMap::new();
    let trimmed = raw.trim_start_matches('\u{feff}');
    let Some(rest) = trimmed.strip_prefix("---") else {
        return (meta, raw);
    };
    let Some(end) = rest.find("\n---") else {
        return (meta, raw);
    };
    let frontmatter = &rest[..end];
    let body_start = &rest[end + 4..];
    let body = body_start.strip_prefix('\n').unwrap_or(body_start);

    for line in frontmatter.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once(':') {
            // Nested/multiline values are ignored (value would be empty or a
            // continuation, which simple skills don't use).
            let value = value.trim().trim_matches('"').trim_matches('\'');
            if !value.is_empty() {
                meta.insert(key.trim().to_string(), value.to_string());
            }
        }
    }
    (meta, body)
}

// ── SkillTool ─────────────────────────────────────────────────────────────────

/// The `skill` tool: loads a skill's full instructions on demand.
pub struct SkillTool {
    library: Arc<SkillsLibrary>,
}

impl SkillTool {
    pub fn new(library: SkillsLibrary) -> Self {
        Self {
            library: Arc::new(library),
        }
    }

    pub fn from_arc(library: Arc<SkillsLibrary>) -> Self {
        Self { library }
    }

    /// List the files bundled with a skill (relative paths, excluding SKILL.md).
    fn bundled_files(dir: &Path) -> Vec<String> {
        let mut files = Vec::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(current) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&current) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.file_name().and_then(|n| n.to_str()) != Some("SKILL.md") {
                    if let Ok(rel) = path.strip_prefix(dir) {
                        files.push(rel.to_string_lossy().into_owned());
                    }
                }
            }
        }
        files.sort();
        files
    }
}

#[async_trait]
impl Tool for SkillTool {
    fn definition(&self) -> ToolDefinition {
        let names: Vec<&str> = self.library.iter().map(|s| s.name.as_str()).collect();
        ToolDefinition::function(
            "skill",
            "Load the full instructions for a named skill. Call this before \
             attempting a task that matches a skill's description.",
            json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill to load",
                        "enum": names,
                    }
                },
                "required": ["name"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::Tool("missing 'name' argument".into()))?;
        let skill = self.library.get(name).ok_or_else(|| {
            AgentError::Tool(format!(
                "unknown skill '{name}'; available: {}",
                self.library
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;

        let files = Self::bundled_files(&skill.dir);
        Ok(json!({
            "name": skill.name,
            "instructions": skill.instructions,
            "directory": skill.dir.to_string_lossy(),
            "bundled_files": files,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_skill(root: &Path, dir_name: &str, frontmatter_name: &str, extra_file: Option<&str>) {
        let dir = root.join(dir_name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("SKILL.md"),
            format!(
                "---\nname: {frontmatter_name}\ndescription: does {frontmatter_name} things. Use when asked.\n---\n\n# Steps\n1. Do the thing.\n"
            ),
        )
        .unwrap();
        if let Some(f) = extra_file {
            std::fs::write(dir.join(f), "helper").unwrap();
        }
    }

    #[test]
    fn discovers_and_renders_prompt_section() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(tmp.path(), "pdf", "pdf-processing", Some("reference.md"));
        write_skill(tmp.path(), "csv", "csv-analysis", None);

        let library = SkillsLibrary::discover(tmp.path()).unwrap();
        assert_eq!(library.len(), 2);

        let section = library.system_prompt_section();
        assert!(section.contains("pdf-processing: does pdf-processing things"));
        assert!(section.contains("csv-analysis"));
        assert!(section.contains("`skill` tool"));
    }

    #[tokio::test]
    async fn skill_tool_loads_instructions_and_files() {
        let tmp = tempfile::tempdir().unwrap();
        write_skill(tmp.path(), "pdf", "pdf-processing", Some("reference.md"));

        let library = SkillsLibrary::discover(tmp.path()).unwrap();
        let tool = SkillTool::new(library);

        // Enum constraint exposes the available names.
        let def = tool.definition();
        assert_eq!(
            def.function.parameters["properties"]["name"]["enum"][0],
            "pdf-processing"
        );

        let result = tool
            .execute(json!({ "name": "pdf-processing" }))
            .await
            .unwrap();
        assert!(result["instructions"]
            .as_str()
            .unwrap()
            .contains("Do the thing"));
        assert_eq!(result["bundled_files"][0], "reference.md");

        let err = tool.execute(json!({ "name": "nope" })).await.unwrap_err();
        assert!(err.to_string().contains("available: pdf-processing"));
    }

    #[test]
    fn frontmatter_edge_cases() {
        let (meta, body) = split_frontmatter("no frontmatter here");
        assert!(meta.is_empty());
        assert_eq!(body, "no frontmatter here");

        let (meta, body) = split_frontmatter("---\nname: x\ndescription: \"quoted\"\n---\nbody");
        assert_eq!(meta["name"], "x");
        assert_eq!(meta["description"], "quoted");
        assert_eq!(body, "body");
    }

    #[test]
    fn missing_dir_is_empty_library() {
        let library = SkillsLibrary::discover("/nonexistent/path/xyz").unwrap();
        assert!(library.is_empty());
        assert_eq!(library.system_prompt_section(), "");
    }
}
