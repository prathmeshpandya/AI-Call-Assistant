# Downloads a prebuilt llama.cpp (CPU, x64) and a GGUF model into ./llm.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\setup-llm.ps1 [-Model qwen3-4b|qwen2.5-1.5b|llama3.2-3b] [-Build cpu|vulkan]
param(
    [ValidateSet("qwen3-4b", "qwen2.5-1.5b", "llama3.2-3b")]
    [string]$Model = "qwen3-4b",
    [ValidateSet("cpu", "vulkan")]
    [string]$Build = "cpu",
    [string]$LlmDir = (Join-Path $PSScriptRoot "..\llm")
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
New-Item -ItemType Directory -Force $LlmDir | Out-Null
$LlmDir = (Resolve-Path $LlmDir).Path

$models = @{
    "qwen3-4b"     = "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    "qwen2.5-1.5b" = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    "llama3.2-3b"  = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
}

# 1. llama.cpp binaries (latest release from ggml-org)
$binDir = Join-Path $LlmDir "llama.cpp"
if (-not (Test-Path (Join-Path $binDir "llama-server.exe"))) {
    # "releases/latest" can point at a tag without Windows binaries, so take the newest release that has them.
    $releases = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
    $asset = $releases | ForEach-Object { $_.assets } |
        Where-Object { $_.name -like "llama-*-bin-win-$Build-x64.zip" } | Select-Object -First 1
    if (-not $asset) { throw "No win-$Build-x64 asset in the 10 most recent llama.cpp releases" }
    $zip = Join-Path $LlmDir $asset.name
    Write-Host "[download] $($asset.browser_download_url)"
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $binDir -Force
    Remove-Item $zip
    # Some releases nest binaries in a sub-folder; flatten so llama-server.exe sits in $binDir.
    $exe = Get-ChildItem $binDir -Recurse -Filter "llama-server.exe" | Select-Object -First 1
    if ($exe.DirectoryName -ne $binDir) { Get-ChildItem $exe.DirectoryName | Move-Item -Destination $binDir -Force }
} else { Write-Host "[skip] llama.cpp already present" }

# 2. Model weights
$url = $models[$Model]
$gguf = Join-Path $LlmDir ([IO.Path]::GetFileName($url))
if (-not (Test-Path $gguf)) {
    Write-Host "[download] $url"
    Invoke-WebRequest -Uri $url -OutFile $gguf
} else { Write-Host "[skip] $gguf already present" }

Write-Host "Done. Start the server with: powershell -ExecutionPolicy Bypass -File scripts\start-llm.ps1 -ModelFile `"$gguf`""
