# Reproduce the BENCHMARKS.md numbers on one command.
#
#   .\scripts\bench.ps1 -Model C:\models\llama-3.2-1b\Llama-3.2-1B-Instruct-Q4_K_M.gguf
#   .\scripts\bench.ps1 -Model <gguf> -Vk        # raw-Vulkan decode lane (ZLLM_VK=1)
#
# Decode throughput is measured by the CLI's own steady-state counter —
# the same numbers quoted against llama.cpp in BENCHMARKS.md. For a fair
# llama.cpp comparison run llama-bench on the SAME gguf on the SAME
# thermal state (run zllm second on a hot GPU per the recorded A/B
# discipline), and prefer the all-Q4 file: Q4_K_M's Q6 tensors are the
# slow path (see memory: quote tok/s only from Q4pure).
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [int]$MaxTokens = 128,
    [int]$Reps = 3,
    [switch]$Vk
)

$exe = Join-Path $PSScriptRoot "..\target\release\zllm.exe"
if (-not (Test-Path $exe)) {
    Write-Error "build first: cargo build --release"
    exit 1
}
if ($Vk) { $env:ZLLM_VK = "1" } else { Remove-Item Env:ZLLM_VK -ErrorAction SilentlyContinue }

$prompt = "Write a detailed paragraph about the history of computing."
$results = @()
for ($i = 1; $i -le $Reps; $i++) {
    $out = & $exe generate --model $Model --prompt $prompt --max-tokens $MaxTokens --temperature 0.0 2>&1 |
        Select-String -Pattern "tok/s"
    Write-Host "run ${i}: $out"
    if ("$out" -match '([\d.]+) tok/s') { $results += [double]$Matches[1] }
}
if ($results.Count -gt 0) {
    $avg = [math]::Round(($results | Measure-Object -Average).Average, 1)
    Write-Host "---"
    Write-Host "decode avg over $($results.Count) runs: $avg tok/s ($(if ($Vk) {'ZLLM_VK'} else {'CPU'}))"
}
