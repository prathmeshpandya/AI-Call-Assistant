# Starts llama-server (OpenAI-compatible API) on http://127.0.0.1:8080.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\start-llm.ps1 [-ModelFile path\to\model.gguf] [-Threads 6] [-Ctx 4096]
param(
    [string]$ModelFile = "",
    [int]$Port = 8080,
    [int]$Threads = 6,         # physical cores usually beat logical threads for token generation
    [int]$Ctx = 4096,
    [int]$GpuLayers = 0        # >0 only with a GPU (Vulkan/CUDA) build
)

$ErrorActionPreference = "Stop"
$llmDir = Join-Path $PSScriptRoot "..\llm"
$exe = Join-Path $llmDir "llama.cpp\llama-server.exe"
if (-not (Test-Path $exe)) { throw "llama-server.exe not found. Run scripts\setup-llm.ps1 first." }
if (-not $ModelFile) {
    $ModelFile = (Get-ChildItem $llmDir -Filter *.gguf | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
}
if (-not $ModelFile -or -not (Test-Path $ModelFile)) { throw "No .gguf model found. Run scripts\setup-llm.ps1 or pass -ModelFile." }

Write-Host "Starting llama-server with $ModelFile on port $Port"
# --alias gives the model a stable name the backend can request (LLM_MODEL=local-slm).
& $exe -m $ModelFile --host 127.0.0.1 --port $Port -c $Ctx -t $Threads -ngl $GpuLayers --alias local-slm --jinja
