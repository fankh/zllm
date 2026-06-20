//! Offline GLSL→SPIR-V build tool for the raw-Vulkan kernels, so `.comp` files
//! can be edited and recompiled WITHOUT a full Vulkan SDK install.
//!
//! It shells out to `glslangValidator` — the only compiler whose output is
//! verified bit-exact here (naga's GLSL frontend mis-compiles these kernels:
//! rmsnorm via naga gave 0/24 token agreement vs candle, glslang gives 24/24).
//! Get a prebuilt one (no SDK) from the Khronos `glslang` releases, e.g.:
//!   Invoke-WebRequest -Uri https://github.com/KhronosGroup/glslang/releases/download/main-tot/glslang-master-windows-Release.zip -OutFile g.zip
//!   Expand-Archive g.zip target/glslang
//! The tool finds it via $GLSLANG, then `target/glslang/bin/glslangValidator.exe`, then PATH.
//!
//! Compile one kernel (writes `<name>.spv` next to it):
//!   GLSL_COMPILE=<name>.comp cargo test --test glsl_compile emit -- --nocapture
//! Recompile every `.comp` and report (verify the toolchain before editing):
//!   cargo test --test glsl_compile all -- --nocapture

use std::path::{Path, PathBuf};
use std::process::Command;

fn shaders_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/backend/vulkan/shaders")
}

fn glslang() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("GLSLANG") {
        let p = PathBuf::from(p);
        if p.exists() { return Some(p); }
    }
    let local = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/glslang/bin/glslangValidator.exe");
    if local.exists() { return Some(local); }
    // Fall back to PATH.
    if Command::new("glslangValidator").arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
        return Some(PathBuf::from("glslangValidator"));
    }
    None
}

/// Compile `name.comp` → `name.spv` in the shaders dir. Returns Ok(spv_bytes_len).
fn compile_one(glslang: &Path, name: &str) -> Result<u64, String> {
    let dir = shaders_dir();
    let comp = dir.join(name);
    let spv = dir.join(name.replace(".comp", ".spv"));
    // target-env vulkan1.3: RDNA3.5 kernels use subgroup arithmetic (flash SDPA)
    // and KHR cooperative matrix.
    let out = Command::new(glslang)
        .arg("-V").arg("--target-env").arg("vulkan1.3").arg(&comp).arg("-o").arg(&spv)
        .output().map_err(|e| format!("spawn: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stdout).trim().to_string());
    }
    Ok(std::fs::metadata(&spv).map(|m| m.len()).unwrap_or(0))
}

#[test]
fn emit() {
    let Some(g) = glslang() else { eprintln!("glslangValidator not found — see this file's docs to fetch it"); return; };
    let Ok(name) = std::env::var("GLSL_COMPILE") else { eprintln!("set GLSL_COMPILE=<name>.comp"); return; };
    match compile_one(&g, &name) {
        Ok(n) => eprintln!("compiled {name} -> {} bytes of SPIR-V", n),
        Err(e) => panic!("glslang failed:\n{e}"),
    }
}

#[test]
fn all() {
    let Some(g) = glslang() else { eprintln!("glslangValidator not found — see this file's docs to fetch it; skipping"); return; };
    eprintln!("using {}", g.display());
    let dir = shaders_dir();
    let mut comps: Vec<String> = std::fs::read_dir(&dir).expect("read shaders dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "comp").unwrap_or(false))
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    comps.sort();
    let (mut ok, mut fail) = (0, 0);
    for name in &comps {
        match compile_one(&g, name) {
            Ok(n) => { ok += 1; eprintln!("OK    {name}  ({n} bytes)"); }
            Err(e) => { fail += 1; eprintln!("FAIL  {name}  -> {}", e.lines().next().unwrap_or("")); }
        }
    }
    eprintln!("\nglslang: {ok} OK, {fail} FAIL of {} kernels", comps.len());
    assert_eq!(fail, 0, "some kernels failed to compile");
}
