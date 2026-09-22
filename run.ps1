param(
    [switch]$Demo,
    [string]$User,
    [string]$Query
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$model = "qwen2.5:3b"
$ollamaUrl = "http://127.0.0.1:11434/api/tags"

Set-Location $projectRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.10 or newer is required and must be available on PATH."
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama is required. Install it from https://ollama.com/download and ensure ollama is on PATH."
}

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    python -m venv .venv
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $projectRoot "requirements.txt")

try {
    Invoke-RestMethod -Uri $ollamaUrl -TimeoutSec 2 | Out-Null
}
catch {
    Write-Host "Starting Ollama..."
    Start-Process ollama -ArgumentList "serve" -WindowStyle Minimized
    $ready = $false
    for ($attempt = 1; $attempt -le 20; $attempt++) {
        Start-Sleep -Seconds 1
        try {
            Invoke-RestMethod -Uri $ollamaUrl -TimeoutSec 2 | Out-Null
            $ready = $true
            break
        }
        catch {
            # Keep waiting for the local service to become available.
        }
    }
    if (-not $ready) {
        throw "Ollama did not become available on http://127.0.0.1:11434."
    }
}

& ollama pull $model

$arguments = @("-m", "src.enterprise_rag.cli")
if ($Demo) {
    $arguments += "--demo"
}
if ($User) {
    $arguments += @("--user", $User)
}
if ($Query) {
    $arguments += @("--query", $Query)
}

& $venvPython @arguments
exit $LASTEXITCODE
