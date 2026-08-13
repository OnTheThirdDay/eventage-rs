//! Lightweight JSON-Schema validation for tool-call arguments.
//!
//! Covers the subset of JSON Schema that tool definitions actually use —
//! `type`, `required`, `properties`, `items`, and `enum` — with no external
//! dependency. Violations are returned as human-readable strings intended to
//! be fed back to the model so it can correct the call.

use serde_json::Value;

/// Validate `args` against a (subset of) JSON Schema.
///
/// Checks `type`, `required`, `properties` (recursively), `items`, and
/// `enum`. Unknown keywords are ignored, and extra properties are allowed
/// (matching the permissive behavior of LLM providers). Returns the first
/// violation found, phrased for the model.
pub fn validate_args(schema: &Value, args: &Value) -> Result<(), String> {
    validate_at(schema, args, "arguments")
}

fn validate_at(schema: &Value, value: &Value, path: &str) -> Result<(), String> {
    let Some(schema_obj) = schema.as_object() else {
        return Ok(()); // malformed / boolean schema — don't block the call
    };

    if let Some(expected) = schema_obj.get("type").and_then(|t| t.as_str()) {
        if !type_matches(expected, value) {
            return Err(format!(
                "{path} must be of type '{expected}' but got {}",
                type_name(value)
            ));
        }
    }

    if let Some(allowed) = schema_obj.get("enum").and_then(|e| e.as_array()) {
        if !allowed.contains(value) {
            let opts: Vec<String> = allowed.iter().map(|v| v.to_string()).collect();
            return Err(format!(
                "{path} must be one of [{}] but got {value}",
                opts.join(", ")
            ));
        }
    }

    if let Some(obj) = value.as_object() {
        if let Some(required) = schema_obj.get("required").and_then(|r| r.as_array()) {
            for req in required.iter().filter_map(|r| r.as_str()) {
                if !obj.contains_key(req) {
                    return Err(format!("{path} is missing required property '{req}'"));
                }
            }
        }
        if let Some(props) = schema_obj.get("properties").and_then(|p| p.as_object()) {
            for (key, prop_schema) in props {
                if let Some(prop_value) = obj.get(key) {
                    validate_at(prop_schema, prop_value, &format!("{path}.{key}"))?;
                }
            }
        }
    }

    if let (Some(items_schema), Some(arr)) = (schema_obj.get("items"), value.as_array()) {
        for (i, item) in arr.iter().enumerate() {
            validate_at(items_schema, item, &format!("{path}[{i}]"))?;
        }
    }

    Ok(())
}

fn type_matches(expected: &str, value: &Value) -> bool {
    match expected {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        _ => true, // unknown type keyword — permissive
    }
}

fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "count": { "type": "integer" },
                "mode": { "type": "string", "enum": ["read", "write"] },
                "tags": { "type": "array", "items": { "type": "string" } }
            },
            "required": ["path"]
        })
    }

    #[test]
    fn accepts_valid_args() {
        let args = json!({ "path": "a.txt", "count": 3, "mode": "read", "tags": ["x"] });
        assert!(validate_args(&schema(), &args).is_ok());
    }

    #[test]
    fn rejects_missing_required() {
        let err = validate_args(&schema(), &json!({ "count": 3 })).unwrap_err();
        assert!(err.contains("required property 'path'"), "{err}");
    }

    #[test]
    fn rejects_wrong_type() {
        let err = validate_args(&schema(), &json!({ "path": 42 })).unwrap_err();
        assert!(err.contains("arguments.path"), "{err}");
        assert!(err.contains("'string'"), "{err}");
    }

    #[test]
    fn rejects_bad_enum() {
        let err = validate_args(&schema(), &json!({ "path": "a", "mode": "append" })).unwrap_err();
        assert!(err.contains("must be one of"), "{err}");
    }

    #[test]
    fn rejects_bad_array_item() {
        let err = validate_args(&schema(), &json!({ "path": "a", "tags": ["ok", 7] })).unwrap_err();
        assert!(err.contains("tags[1]"), "{err}");
    }

    #[test]
    fn extra_properties_are_allowed() {
        let args = json!({ "path": "a", "unknown": true });
        assert!(validate_args(&schema(), &args).is_ok());
    }
}
