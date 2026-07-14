//! JSON-schema → anchored-regex compiler for constrained decoding.
//!
//! The [`LogitFSM`](crate::engine::logit_fsm) grammar engine already enforces
//! an anchored byte-DFA over the *entire* generated text (`regex:` mode). A
//! JSON schema with a fixed structure — the shape OpenAI's "structured outputs"
//! strict mode produces — is expressible as such a regex, so we compile the
//! schema to a pattern and hand it to that engine unchanged. No separate JSON
//! state machine, no counting stack: the finite, non-recursive schema unrolls
//! into a finite regex.
//!
//! Two entry points:
//! - [`schema_to_regex`] — a concrete schema (`json_schema:` mode / OpenAI
//!   `response_format: json_schema`).
//! - [`any_json_regex`] — a generic, bounded-depth JSON *value* (the shapeless
//!   `json:` mode / `response_format: json_object`).
//!
//! ## Supported schema subset
//!
//! `type` of `object` (with `properties`), `array` (with `items`), `string`,
//! `integer`, `number`, `boolean`, `null`; plus `enum` (any JSON literals).
//! Following OpenAI strict semantics, **every declared property is emitted, in
//! declaration order** — optional-property subsets would blow the regex up
//! combinatorially, so a schema that wants a field omitted should not declare
//! it. `minItems`/`maxItems` bound arrays; `pattern` embeds a string regex.
//!
//! Unsupported (rejected with an error the caller surfaces as 400): `$ref`,
//! `oneOf`/`anyOf`/`allOf`, `patternProperties`, numeric `minimum`/`maximum`
//! (not expressible as a small regex). Depth is capped to guard adversarial
//! schemas.

use serde_json::Value;

/// Max schema/JSON nesting the compiler will unroll before erroring. Guards
/// against pathological or cyclic (`$ref`) schemas producing an unbounded
/// regex or blowing the stack.
const MAX_DEPTH: usize = 32;

/// Default nesting depth for the shapeless `json_object` value grammar. Bounds
/// the regex (a DFA cannot count arbitrary nesting) while covering essentially
/// all real model output.
const ANY_JSON_DEFAULT_DEPTH: usize = 5;

// JSON primitive value patterns (RFC 8259 shapes, compact).
const STRING: &str = r#""([^"\\]|\\(["\\/bfnrt]|u[0-9a-fA-F]{4}))*""#;
const NUMBER: &str = r#"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?"#;
const BOOL: &str = r#"(true|false)"#;
const NULL: &str = r#"null"#;
// Zero-or-one space after structural `:` / `,` — matches the `"k": v, ...`
// style most models are trained on, while staying bounded so whitespace can't
// loop forever (the mask must always drive generation toward termination).
const SP: &str = " ?";

/// Backslash-escape regex metacharacters so a literal (object key, enum value)
/// matches itself. Deliberately conservative — escapes every ASCII punctuation
/// char that could be special in `regex-automata`'s syntax.
fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for c in s.chars() {
        if "\\.+*?()|[]{}^$".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// A JSON literal, as it appears in output, regex-escaped. Used for `enum`
/// members and `const`. Strings are JSON-encoded (quoted, escaped) first.
fn literal_regex(v: &Value) -> Result<String, String> {
    let json = serde_json::to_string(v).map_err(|e| format!("enum literal: {e}"))?;
    Ok(esc(&json))
}

/// Compile a concrete JSON schema to an anchored regex matching exactly the
/// JSON documents that satisfy it. The returned pattern is *not* wrapped in
/// `regex:` — pass it to `LogitFSM::compile("regex:...")` or use the
/// `json_schema:` grammar mode which does that for you.
pub fn schema_to_regex(schema: &Value) -> Result<String, String> {
    compile(schema, 0)
}

/// A generic JSON *value* of bounded nesting depth — the `json_object` mode
/// (any well-formed JSON, no fixed shape). `depth` caps nesting.
pub fn any_json_regex(depth: usize) -> String {
    any_value(depth.min(MAX_DEPTH))
}

/// The default-depth shapeless JSON value grammar.
pub fn any_json_regex_default() -> String {
    any_json_regex(ANY_JSON_DEFAULT_DEPTH)
}

fn any_value(depth: usize) -> String {
    let mut alts = vec![STRING.to_string(), NUMBER.to_string(), BOOL.to_string(), NULL.to_string()];
    if depth > 0 {
        // array of value_{depth-1}
        let inner = any_value(depth - 1);
        alts.push(format!(r"\[{SP}({inner}({SP},{SP}{inner})*)?{SP}\]"));
        // object of string : value_{depth-1}
        alts.push(format!(
            r"\{{{SP}({STRING}{SP}:{SP}{inner}({SP},{SP}{STRING}{SP}:{SP}{inner})*)?{SP}\}}"
        ));
    }
    format!("({})", alts.join("|"))
}

fn compile(schema: &Value, depth: usize) -> Result<String, String> {
    if depth > MAX_DEPTH {
        return Err(format!("schema nesting exceeds {MAX_DEPTH} — refusing to unroll"));
    }
    let obj = schema
        .as_object()
        .ok_or_else(|| "schema must be a JSON object".to_string())?;

    // Reject constructs we can't faithfully compile, rather than silently
    // producing a laxer grammar.
    for k in ["$ref", "oneOf", "anyOf", "allOf", "not", "patternProperties", "if"] {
        if obj.contains_key(k) {
            return Err(format!("schema keyword {k:?} is not supported by the regex compiler"));
        }
    }

    // `const` — a single fixed literal.
    if let Some(c) = obj.get("const") {
        return literal_regex(c);
    }
    // `enum` — alternation of fixed literals.
    if let Some(Value::Array(vals)) = obj.get("enum") {
        if vals.is_empty() {
            return Err("enum must be non-empty".into());
        }
        let alts: Result<Vec<_>, _> = vals.iter().map(literal_regex).collect();
        return Ok(format!("({})", alts?.join("|")));
    }

    let ty = obj.get("type").and_then(|t| t.as_str()).ok_or_else(|| {
        "schema needs a \"type\" (or enum/const); unions via type-arrays are not supported".to_string()
    })?;

    match ty {
        "string" => {
            if let Some(p) = obj.get("pattern").and_then(|p| p.as_str()) {
                // A `pattern` constrains the *contents* of the JSON string.
                // Embed it between the quotes. The author owns its correctness.
                Ok(format!("\"{p}\""))
            } else {
                Ok(STRING.to_string())
            }
        }
        "integer" => Ok(r"-?(0|[1-9][0-9]*)".to_string()),
        "number" => Ok(NUMBER.to_string()),
        "boolean" => Ok(BOOL.to_string()),
        "null" => Ok(NULL.to_string()),
        "array" => compile_array(obj, depth),
        "object" => compile_object(obj, depth),
        other => Err(format!("unsupported schema type {other:?}")),
    }
}

fn compile_array(
    obj: &serde_json::Map<String, Value>,
    depth: usize,
) -> Result<String, String> {
    let items = obj
        .get("items")
        .ok_or_else(|| "array schema needs \"items\"".to_string())?;
    let item = compile(items, depth + 1)?;
    let min = obj.get("minItems").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let max = obj.get("maxItems").and_then(|v| v.as_u64()).map(|v| v as usize);

    if let Some(max) = max {
        if max < min {
            return Err("maxItems < minItems".into());
        }
        if max == 0 {
            return Ok(format!(r"\[{SP}\]"));
        }
    }

    // Build the element list honoring min/max. `sep` joins elements.
    let sep = format!("{SP},{SP}");
    let body = match (min, max) {
        (0, None) => format!("({item}({sep}{item})*)?"), // zero or more
        (n, None) => {
            // at least n
            let required = std::iter::repeat(item.clone())
                .take(n)
                .collect::<Vec<_>>()
                .join(&sep);
            format!("{required}({sep}{item})*")
        }
        (n, Some(m)) => {
            // between n and m
            let mut parts = Vec::new();
            for count in n.max(1)..=m {
                parts.push(
                    std::iter::repeat(item.clone())
                        .take(count)
                        .collect::<Vec<_>>()
                        .join(&sep),
                );
            }
            let mut alt = format!("({})", parts.join("|"));
            if n == 0 {
                alt = format!("({alt})?");
            }
            alt
        }
    };
    Ok(format!(r"\[{SP}{body}{SP}\]"))
}

fn compile_object(
    obj: &serde_json::Map<String, Value>,
    depth: usize,
) -> Result<String, String> {
    let props = match obj.get("properties") {
        Some(Value::Object(p)) => p,
        // An object with no declared properties → any bounded JSON object.
        _ => {
            return Ok(format!(
                r"\{{{SP}({STRING}{SP}:{SP}{v}({SP},{SP}{STRING}{SP}:{SP}{v})*)?{SP}\}}",
                v = any_value(ANY_JSON_DEFAULT_DEPTH.saturating_sub(depth.min(ANY_JSON_DEFAULT_DEPTH)))
            ))
        }
    };
    if props.is_empty() {
        return Ok(format!(r"\{{{SP}\}}"));
    }
    // OpenAI strict semantics: emit every property, in declaration order.
    // serde_json preserves object key order only with the `preserve_order`
    // feature; without it, iterate the schema's declared order is lost, so we
    // fall back to the map's iteration order (sorted) — deterministic either
    // way. `required` is advisory here; all declared props are produced.
    let mut pairs = Vec::with_capacity(props.len());
    for (key, subschema) in props {
        let val = compile(subschema, depth + 1)?;
        pairs.push(format!("\"{}\"{SP}:{SP}{val}", esc(key)));
    }
    let sep = format!("{SP},{SP}");
    Ok(format!(r"\{{{SP}{}{SP}\}}", pairs.join(&sep)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use regex_automata::dfa::{dense, Automaton};
    use regex_automata::{Anchored, Input};
    use serde_json::json;

    /// Compile `pattern` to an anchored DFA and test whether `text` is a
    /// complete match — mirrors how `LogitFSM` accepts a finished generation.
    fn full_match(pattern: &str, text: &str) -> bool {
        let dfa = dense::DFA::new(pattern).expect("pattern compiles");
        let mut state = dfa
            .start_state_forward(&Input::new("").anchored(Anchored::Yes))
            .unwrap();
        for &b in text.as_bytes() {
            state = dfa.next_state(state, b);
            if dfa.is_dead_state(state) {
                return false;
            }
        }
        dfa.is_match_state(dfa.next_eoi_state(state))
    }

    fn accepts(schema: &Value, text: &str) -> bool {
        let re = schema_to_regex(schema).expect("schema compiles");
        full_match(&re, text)
    }
    fn rejects(schema: &Value, text: &str) -> bool {
        !accepts(schema, text)
    }

    #[test]
    fn primitives() {
        assert!(accepts(&json!({"type":"boolean"}), "true"));
        assert!(accepts(&json!({"type":"boolean"}), "false"));
        assert!(rejects(&json!({"type":"boolean"}), "True"));
        assert!(accepts(&json!({"type":"integer"}), "-42"));
        assert!(rejects(&json!({"type":"integer"}), "4.5"));
        assert!(accepts(&json!({"type":"number"}), "3.14"));
        assert!(accepts(&json!({"type":"number"}), "-1.2e10"));
        assert!(accepts(&json!({"type":"null"}), "null"));
        assert!(accepts(&json!({"type":"string"}), "\"hi\""));
        assert!(rejects(&json!({"type":"string"}), "hi")); // needs quotes
    }

    #[test]
    fn string_with_escapes() {
        assert!(accepts(&json!({"type":"string"}), r#""a\"b""#));
        assert!(accepts(&json!({"type":"string"}), r#""tab\tend""#));
        assert!(accepts(&json!({"type":"string"}), r#""é""#));
    }

    #[test]
    fn enums() {
        let s = json!({"enum": ["red", "green", "blue"]});
        assert!(accepts(&s, "\"green\""));
        assert!(rejects(&s, "\"yellow\""));
        let n = json!({"enum": [1, 2, 3]});
        assert!(accepts(&n, "2"));
        assert!(rejects(&n, "4"));
    }

    #[test]
    fn const_literal() {
        let s = json!({"const": "yes"});
        assert!(accepts(&s, "\"yes\""));
        assert!(rejects(&s, "\"no\""));
    }

    #[test]
    fn simple_object() {
        let s = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        });
        assert!(accepts(&s, r#"{"name":"Ada","age":36}"#));
        assert!(accepts(&s, r#"{"name": "Ada", "age": 36}"#)); // optional spaces
        assert!(rejects(&s, r#"{"age":36,"name":"Ada"}"#)); // wrong order (strict)
        assert!(rejects(&s, r#"{"name":"Ada"}"#)); // missing prop (strict)
        assert!(rejects(&s, r#"{"name":"Ada","age":"36"}"#)); // wrong type
    }

    #[test]
    fn nested_object_and_array() {
        let s = json!({
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
                "meta": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}}
                }
            }
        });
        assert!(accepts(&s, r#"{"tags":["a","b"],"meta":{"ok":true}}"#));
        assert!(accepts(&s, r#"{"tags":[],"meta":{"ok":false}}"#));
        assert!(rejects(&s, r#"{"tags":[1],"meta":{"ok":true}}"#)); // wrong item type
    }

    #[test]
    fn array_bounds() {
        let s = json!({"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":3});
        assert!(rejects(&s, "[1]"));
        assert!(accepts(&s, "[1,2]"));
        assert!(accepts(&s, "[1,2,3]"));
        assert!(rejects(&s, "[1,2,3,4]"));
    }

    #[test]
    fn enum_with_regex_meta_in_value() {
        // A value containing regex metacharacters must be matched literally.
        let s = json!({"enum": ["a.b", "c+d"]});
        assert!(accepts(&s, "\"a.b\""));
        assert!(rejects(&s, "\"axb\"")); // '.' must be literal, not wildcard
        assert!(accepts(&s, "\"c+d\""));
    }

    #[test]
    fn key_with_special_chars() {
        let s = json!({"type":"object","properties":{"a.b":{"type":"integer"}}});
        assert!(accepts(&s, r#"{"a.b":1}"#));
        assert!(rejects(&s, r#"{"axb":1}"#));
    }

    #[test]
    fn unsupported_constructs_error() {
        assert!(schema_to_regex(&json!({"$ref": "#/x"})).is_err());
        assert!(schema_to_regex(&json!({"anyOf": [{"type":"string"}]})).is_err());
        assert!(schema_to_regex(&json!({"type": "object", "properties": {"x": {}}})).is_err()); // subschema has no type
        assert!(schema_to_regex(&json!(42)).is_err()); // not an object
    }

    #[test]
    fn any_json_value() {
        let re = any_json_regex_default();
        assert!(full_match(&re, "true"));
        assert!(full_match(&re, "-3.5"));
        assert!(full_match(&re, r#""hi""#));
        assert!(full_match(&re, r#"{"a":1,"b":[1,2,{"c":true}]}"#));
        assert!(full_match(&re, "[]"));
        assert!(full_match(&re, "{}"));
        assert!(!full_match(&re, "{not json")); // malformed
    }

    #[test]
    fn tool_call_shape_compiles() {
        // The shape a function-call is constrained to — proves the schema
        // compiler covers the tool-calling case end to end.
        let s = json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"enum": ["celsius", "fahrenheit"]}
            }
        });
        assert!(accepts(&s, r#"{"location":"Paris","unit":"celsius"}"#));
        assert!(rejects(&s, r#"{"location":"Paris","unit":"kelvin"}"#));
    }
}
