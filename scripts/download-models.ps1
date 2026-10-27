# Downloads the speech models (streaming ASR, Silero VAD, Kokoro TTS) into ./models.
# Usage (from the project root):  powershell -ExecutionPolicy Bypass -File scripts\download-models.ps1
param(
    [string]$ModelsDir = (Join-Path $PSScriptRoot "..\models"),
    [string]$AsrModel = "sherpa-onnx-streaming-zipformer-en-2023-06-26",
    [string]$PiperVoice = "vits-piper-en_US-ljspeech-medium",  # default TTS (public-domain LJSpeech data)
    [string]$KokoroModel = "kokoro-en-v0_19",                   # optional quality TTS; fp32 (int8 is slower on x64)
    [switch]$SkipKokoro
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is ~10x faster without the progress bar
New-Item -ItemType Directory -Force $ModelsDir | Out-Null
$ModelsDir = (Resolve-Path $ModelsDir).Path

$base = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

function Get-Archive([string]$url, [string]$name) {
    $target = Join-Path $ModelsDir $name
    if (Test-Path $target) { Write-Host "[skip] $name already present"; return }
    $archive = Join-Path $ModelsDir "$name.tar.bz2"
    Write-Host "[download] $url"
    Invoke-WebRequest -Uri $url -OutFile $archive
    Write-Host "[extract] $archive"
    tar -xjf $archive -C $ModelsDir      # bsdtar ships with Windows 10+ and handles .tar.bz2
    Remove-Item $archive
}

Get-Archive "$base/asr-models/$AsrModel.tar.bz2" $AsrModel
Get-Archive "$base/tts-models/$PiperVoice.tar.bz2" $PiperVoice
if (-not $SkipKokoro) { Get-Archive "$base/tts-models/$KokoroModel.tar.bz2" $KokoroModel }

$vad = Join-Path $ModelsDir "silero_vad.onnx"
if (-not (Test-Path $vad)) {
    Write-Host "[download] silero_vad.onnx"
    Invoke-WebRequest -Uri "$base/asr-models/silero_vad.onnx" -OutFile $vad
} else { Write-Host "[skip] silero_vad.onnx already present" }

Write-Host "Done. Models in $ModelsDir"
Get-ChildItem $ModelsDir | Format-Table Name, Length
