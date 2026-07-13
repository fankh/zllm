# Install the latest zllm release binary (Windows x86_64, CPU build).
#   irm https://raw.githubusercontent.com/fankh/zllm/main/scripts/install.ps1 | iex
# or from a checkout:  .\scripts\install.ps1

$ErrorActionPreference = "Stop"
$repo = "fankh/zllm"
$dest = Join-Path $env:LOCALAPPDATA "zllm"

$rel = Invoke-RestMethod "https://api.github.com/repos/$repo/releases/latest"
$asset = $rel.assets | Where-Object name -like "*windows-x86_64-cpu.zip" | Select-Object -First 1
if (-not $asset) {
    Write-Error "no windows-x86_64-cpu asset on release $($rel.tag_name) — see https://github.com/$repo/releases"
}
Write-Host "installing zllm $($rel.tag_name) -> $dest"
New-Item -ItemType Directory -Force $dest | Out-Null
$zip = Join-Path $env:TEMP $asset.name
Invoke-WebRequest $asset.browser_download_url -OutFile $zip
Expand-Archive $zip -DestinationPath $dest -Force
Remove-Item $zip

$exe = Get-ChildItem $dest -Recurse -Filter zllm.exe | Select-Object -First 1
Write-Host "installed: $($exe.FullName)"
Write-Host "add to PATH (current user):"
Write-Host "  [Environment]::SetEnvironmentVariable('Path', `$env:Path + ';$($exe.DirectoryName)', 'User')"
Write-Host "then: zllm serve --config $($exe.DirectoryName)\configs\default.toml"
