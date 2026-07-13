//! V1_PLAN M4 robustness suite: one live server (auth on, 2K window via
//! ZLLM_MAX_SEQ) driven through the failure modes a product must survive:
//! missing/wrong API key, corrupt-GGUF swap, context overflow, degenerate
//! params, and concurrent load. Model-gated like the other live suites.

use std::io::Write;
use std::path::Path;
use std::process::{Child, Command, Stdio};

const MODEL_PATH: &str = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
const TOKENIZER_PATH: &str = "models/tokenizer.json";
const PORT: u16 = 8198;
const KEY: &str = "test-key-1337";

fn model_available() -> bool {
    Path::new(MODEL_PATH).exists() && Path::new(TOKENIZER_PATH).exists()
}

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

fn spawn_server(models_dir: &Path) -> ServerGuard {
    let dir = std::env::temp_dir().join(format!("zllm-hardening-{PORT}"));
    std::fs::create_dir_all(&dir).unwrap();
    let cfg_path = dir.join("hardening.toml");
    let model_abs = std::fs::canonicalize(MODEL_PATH).unwrap();
    let mut f = std::fs::File::create(&cfg_path).unwrap();
    write!(
        f,
        r#"[server]
rest_port = {PORT}
max_concurrent = 8

[model]
path = "{model}"
dir = "{dir}"
quantization = "q4"
max_seq_len = 32768
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
        dir = models_dir.display().to_string().replace('\\', "/"),
    )
    .unwrap();

    let child = Command::new(env!("CARGO_BIN_EXE_zllm"))
        .args(["serve", "--config", cfg_path.to_str().unwrap()])
        .env("ZLLM_API_KEY", KEY)
        .env("ZLLM_MAX_SEQ", "2048") // small window: fast KV + testable overflow
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn zllm serve");
    let guard = ServerGuard(child);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    loop {
        if ureq::get(&format!("{}/health", base()))
            .timeout(std::time::Duration::from_secs(2))
            .call()
            .is_ok()
        {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "server did not come up");
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    guard
}

fn chat_raw(body: serde_json::Value, key: Option<&str>) -> Result<serde_json::Value, u16> {
    let mut req = ureq::post(&format!("{}/v1/chat/completions", base()))
        .timeout(std::time::Duration::from_secs(300));
    if let Some(k) = key {
        req = req.set("Authorization", &format!("Bearer {k}"));
    }
    match req.send_json(body) {
        Ok(resp) => Ok(resp.into_json().unwrap()),
        Err(ureq::Error::Status(code, _)) => Err(code),
        Err(e) => panic!("transport: {e}"),
    }
}

fn simple(content: &str, max_tokens: u32) -> serde_json::Value {
    serde_json::json!({
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    })
}

#[test]
fn hardening_suite() {
    if !model_available() {
        println!("SKIP: model not found");
        return;
    }
    // A "GGUF" of garbage in the scanned models dir for the swap test.
    let junk_dir = std::env::temp_dir().join("zllm-hardening-models");
    std::fs::create_dir_all(&junk_dir).unwrap();
    std::fs::write(junk_dir.join("corrupt.gguf"), b"this is not a gguf at all").unwrap();
    let _server = spawn_server(&junk_dir);

    // 1. Auth: no key → 401 (health stays open); right key → 200.
    match chat_raw(simple("Say OK.", 4), None) {
        Err(401) => println!("auth: 401 without key OK"),
        other => panic!("expected 401 without key, got {other:?}"),
    }
    match chat_raw(simple("Say OK.", 4), Some("wrong-key")) {
        Err(401) => println!("auth: 401 with wrong key OK"),
        other => panic!("expected 401 with wrong key, got {other:?}"),
    }
    let ok = chat_raw(simple("Say OK.", 8), Some(KEY)).expect("valid key must pass");
    println!("auth: 200 with key OK ({:?})", ok["choices"][0]["message"]["content"]);

    // 2. Corrupt-GGUF swap → 4xx, pool untouched, chat still works.
    let swap = ureq::post(&format!("{}/v1/models/select", base()))
        .set("Authorization", &format!("Bearer {KEY}"))
        .send_json(serde_json::json!({"id": "corrupt"}));
    match swap {
        Err(ureq::Error::Status(code, _)) if (400..500).contains(&code) => {
            println!("corrupt swap: {code} OK")
        }
        other => panic!("corrupt swap should 4xx, got {other:?}"),
    }
    chat_raw(simple("Still alive?", 8), Some(KEY)).expect("server must survive corrupt swap");
    println!("corrupt swap: server still serves OK");

    // 3. Context overflow (window capped at 2048) → 400 with code.
    let big = "word ".repeat(2500);
    match chat_raw(simple(&big, 8), Some(KEY)) {
        Err(400) => println!("context overflow: 400 OK"),
        other => panic!("overflow should 400, got {other:?}"),
    }

    // 4. Degenerate params: max_tokens 0 and empty messages must not
    //    kill the server (any orderly status is fine).
    let _ = chat_raw(simple("hi", 0), Some(KEY));
    let _ = chat_raw(
        serde_json::json!({"messages": [], "max_tokens": 4, "temperature": 0.0}),
        Some(KEY),
    );
    ureq::get(&format!("{}/health", base())).call().expect("alive after degenerate params");
    println!("degenerate params: server alive OK");

    // 5. Absurd max_tokens is clamped to the window; a stop string keeps
    //    the run short.
    let mut body = simple("Write one short sentence about rain.", 1_000_000);
    body["stop"] = serde_json::json!(["."]);
    let r = chat_raw(body, Some(KEY)).expect("clamped absurd max_tokens must succeed");
    println!("absurd max_tokens: OK ({:?})", r["choices"][0]["finish_reason"]);

    // 6. Concurrency: 4 parallel chats on a 1-slot pool — all succeed
    //    (serialized), none 503 under the limit of 8.
    let handles: Vec<_> = (0..4)
        .map(|i| {
            std::thread::spawn(move || {
                chat_raw(simple(&format!("Reply with the number {i}."), 8), Some(KEY))
                    .map(|_| ())
            })
        })
        .collect();
    for (i, h) in handles.into_iter().enumerate() {
        h.join().unwrap().unwrap_or_else(|c| panic!("concurrent req {i} failed: {c}"));
    }
    println!("concurrency: 4/4 parallel chats OK");
}
