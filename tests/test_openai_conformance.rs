//! OpenAI-surface conformance (V1_PLAN M1 exit criterion): spawn a real
//! zllm server on the local 1B model and verify the parameter contract —
//! stop strings never appear in output, seeds reproduce, logit_bias
//! changes the distribution, unsupported params 400, and the utility
//! endpoints (embeddings / tokenize / detokenize) hold their shapes.
//!
//! Model-gated like test_real_inference: skips without `models/`.
//! Local-only; CI runs the unit/smoke suites.

use std::io::Write;
use std::path::Path;
use std::process::{Child, Command, Stdio};

const MODEL_PATH: &str = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
const TOKENIZER_PATH: &str = "models/tokenizer.json";
const PORT: u16 = 8199;

fn model_available() -> bool {
    Path::new(MODEL_PATH).exists() && Path::new(TOKENIZER_PATH).exists()
}

/// Kills the spawned server even when an assertion panics.
struct ServerGuard(Child);
impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn base() -> String {
    format!("http://127.0.0.1:{PORT}")
}

fn spawn_server() -> ServerGuard {
    let dir = std::env::temp_dir().join(format!("zllm-conformance-{PORT}"));
    std::fs::create_dir_all(&dir).unwrap();
    let cfg_path = dir.join("conformance.toml");
    let model_abs = std::fs::canonicalize(MODEL_PATH).unwrap();
    let mut f = std::fs::File::create(&cfg_path).unwrap();
    write!(
        f,
        r#"[server]
rest_port = {PORT}
max_concurrent = 8

[model]
path = "{model}"
quantization = "q4"
max_seq_len = 8192
tokenizer_path = ""

[engine]
encoder_layers = 8
reasoning_layers = 8
max_loops = 16
confidence_threshold = 0.9
default_temperature = 1.0
default_top_k = 50
default_top_p = 0.9
memory_inject_alpha = 0.0
backend_pool_size = 1

[memory]
block_size = 16
max_blocks = 65536
"#,
        model = model_abs.display().to_string().replace('\\', "/"),
    )
    .unwrap();

    let child = Command::new(env!("CARGO_BIN_EXE_zllm"))
        .args(["serve", "--config", cfg_path.to_str().unwrap()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn zllm serve");
    let guard = ServerGuard(child);

    // Wait for /health (model load takes a few seconds).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    loop {
        if ureq::get(&format!("{}/health", base())).timeout(std::time::Duration::from_secs(2)).call().is_ok() {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "server did not come up");
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    guard
}

fn chat(body: serde_json::Value) -> Result<serde_json::Value, u16> {
    match ureq::post(&format!("{}/v1/chat/completions", base()))
        .timeout(std::time::Duration::from_secs(300))
        .send_json(body)
    {
        Ok(resp) => Ok(resp.into_json().unwrap()),
        Err(ureq::Error::Status(code, _)) => Err(code),
        Err(e) => panic!("transport error: {e}"),
    }
}

fn chat_text(body: serde_json::Value) -> String {
    let v = chat(body).expect("expected 200");
    v["choices"][0]["message"]["content"].as_str().unwrap().to_string()
}

#[test]
fn openai_conformance() {
    if !model_available() {
        println!("SKIP: model not found at {MODEL_PATH}");
        return;
    }
    let _server = spawn_server();
    let ask = |extra: serde_json::Value| {
        let mut body = serde_json::json!({
            "messages": [{"role": "user", "content": "Count from one to ten as words, separated by spaces."}],
            "max_tokens": 24,
            "temperature": 0.0,
        });
        for (k, v) in extra.as_object().unwrap() {
            body[k] = v.clone();
        }
        body
    };

    // 1. Stop strings: the match must never appear in the returned text,
    //    whatever the model generated.
    let text = chat_text(ask(serde_json::json!({"stop": [" four"]})));
    assert!(!text.contains(" four"), "stop string leaked into output: {text:?}");
    println!("stop-strings OK: {text:?}");

    // 2. Seed determinism at temperature 0.9: same seed ⇒ same text.
    let a = chat_text(ask(serde_json::json!({"temperature": 0.9, "seed": 42})));
    let b = chat_text(ask(serde_json::json!({"temperature": 0.9, "seed": 42})));
    assert_eq!(a, b, "same seed must reproduce identical output");
    println!("seed determinism OK: {a:?}");

    // 3. Penalties are accepted and change greedy output vs baseline is
    //    not guaranteed — but the request must succeed and repetition
    //    of one token should be discouraged under a strong penalty.
    let p = chat(ask(serde_json::json!({
        "presence_penalty": 1.5, "frequency_penalty": 0.5, "repeat_penalty": 1.3
    })));
    assert!(p.is_ok(), "penalties must be accepted");
    println!("penalties OK");

    // 4. logit_bias: banning the first token of the greedy answer must
    //    change the output.
    let baseline = chat_text(ask(serde_json::json!({})));
    let first_word: String = baseline.chars().take_while(|c| !c.is_whitespace()).collect();
    let tok_resp: serde_json::Value = ureq::post(&format!("{}/tokenize", base()))
        .send_json(serde_json::json!({"content": first_word}))
        .unwrap()
        .into_json()
        .unwrap();
    // encode() prepends BOS — ban the first CONTENT token, not BOS.
    let toks = tok_resp["tokens"].as_array().unwrap();
    let first_id = toks.get(1).unwrap_or(&toks[0]).as_u64().unwrap();
    let banned = chat_text(ask(serde_json::json!({
        "logit_bias": { first_id.to_string(): -100.0 }
    })));
    assert_ne!(baseline, banned, "banning the first token must change greedy output");
    println!("logit_bias OK: {baseline:?} -> {banned:?}");

    // 5. Unsupported params fail loudly with 400.
    for (name, extra) in [
        ("tools", serde_json::json!({"tools": [{"type": "function"}]})),
        ("n>1", serde_json::json!({"n": 2})),
        ("response_format", serde_json::json!({"response_format": {"type": "json_object"}})),
    ] {
        match chat(ask(extra)) {
            Err(400) => println!("400 OK for {name}"),
            other => panic!("{name} should 400, got {other:?}"),
        }
    }

    // 6. Embeddings: shape, count, unit norm.
    let emb: serde_json::Value = ureq::post(&format!("{}/v1/embeddings", base()))
        .send_json(serde_json::json!({"input": ["hello world", "goodbye world"]}))
        .unwrap()
        .into_json()
        .unwrap();
    let data = emb["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    let v0: Vec<f64> = data[0]["embedding"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    let v1: Vec<f64> = data[1]["embedding"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    assert_eq!(v0.len(), v1.len());
    let norm: f64 = v0.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "embedding should be L2-normalized, norm={norm}");
    println!("embeddings OK: dim={} norm={norm:.4}", v0.len());

    // 7. tokenize/detokenize round-trip.
    let toks: serde_json::Value = ureq::post(&format!("{}/tokenize", base()))
        .send_json(serde_json::json!({"content": "The capital of France is Paris."}))
        .unwrap()
        .into_json()
        .unwrap();
    let ids: Vec<u32> = toks["tokens"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    assert!(!ids.is_empty());
    let detok: serde_json::Value = ureq::post(&format!("{}/detokenize", base()))
        .send_json(serde_json::json!({"tokens": ids}))
        .unwrap()
        .into_json()
        .unwrap();
    assert!(detok["content"].as_str().unwrap().contains("capital of France"));
    println!("tokenize/detokenize OK");

    // 8. min_p accepted.
    assert!(chat(ask(serde_json::json!({"temperature": 0.8, "min_p": 0.1, "seed": 1}))).is_ok());
    println!("min_p OK");
}
