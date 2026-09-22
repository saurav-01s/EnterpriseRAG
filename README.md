# Enterprise RAG Intelligence System

A local, role-aware Retrieval-Augmented Generation (RAG) system for querying enterprise documents across HR, Finance, IT, Compliance, and General information silos.

The project enforces access control using demo users and roles, searches both semantic embeddings and keyword signals, and returns grounded answers with source citations.

## What this project does

- Uses RBAC-aware routing so each user only sees allowed categories
- Indexes enterprise documents in ChromaDB
- Combines vector similarity with keyword scoring
- Queries a local Ollama model for grounded answers
- Runs from a small interactive CLI or one-shot demo commands

## Prerequisites

- Windows 10/11 with PowerShell
- Python 3.10 or newer on PATH
- [Ollama](https://ollama.com/download) installed and available on PATH
- Internet access for the first setup so Python packages and models can be downloaded

## Installation

From a fresh clone:

```powershell
git clone https://github.com/<your-username>/EnterpriseRAG.git
cd EnterpriseRAG
```

Create and activate a virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

Start Ollama and pull the model used by the app:

```powershell
ollama serve
ollama pull qwen2.5:3b
```

> If `ollama` is not recognized, install Ollama and reopen PowerShell so the PATH is refreshed.

## Running the app

Run the project from the repository root so the app resolves the bundled dataset under `./enterprise_data`:

```powershell
python -m src.enterprise_rag.cli
```

This opens the interactive menu. Supported commands inside the CLI include:

- `whoami` - show the active user, role, and allowed categories
- `switch` - change the active demo user
- `stats` - display vector store and retrieval settings
- `help` - show available commands
- `exit` or `quit` - leave the application

## Quick commands

Run the built-in demo sequence:

```powershell
.\run.ps1 -Demo
```

Run one non-interactive query as a specific user:

```powershell
.\run.ps1 -User alice -Query "What was Q2 2025 revenue?"
```

Direct Python invocation without the launcher:

```powershell
python -m src.enterprise_rag.cli --demo
python -m src.enterprise_rag.cli --user alice --query "What was Q2 2025 revenue?"
```

## Bootstrap script

The repository includes a convenience launcher at `run.ps1` that handles the setup steps automatically:

- creates `.venv` if missing
- installs `requirements.txt`
- starts Ollama if needed
- downloads the Ollama model
- runs the app from the correct module entry point

Example:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Demo
```

## Demo access model

The built-in users are defined in `src/enterprise_rag/cli.py`:

| User | Role | Access |
| --- | --- | --- |
| alice | admin | HR, Finance, IT, Compliance, General |
| bob | hr_manager | HR, General |
| carol | finance_analyst | Finance, General |
| dave | it_engineer | IT, General |
| eve | compliance_officer | Compliance, HR, Finance, IT, General |

## Repository structure

```text
EnterpriseRAG/
|-- README.md
|-- requirements.txt
|-- run.ps1
|-- enterprise_data/                  # Source documents used by the app
|-- data/                             # Local extracted/ingested workspace data
|-- src/
|   `-- enterprise_rag/
|       `-- cli.py                   # Main application and CLI entry point
|-- chroma_db/                       # Chroma vector database
|-- docs/
`-- tests/
```

## Troubleshooting

### First run is slow

The first execution downloads Python packages, the sentence-transformer model, and the local Ollama model. Subsequent runs are much faster because they reuse the local caches.

### The app cannot find the data folder

Run commands from the repository root, not from a nested folder. The app is designed to use the top-level `enterprise_data` folder.

### Startup fails with a missing module error

Recreate the environment and reinstall dependencies:

```powershell
Remove-Item -Recurse -Force .\.venv
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

## Notes

This is a local demonstration project intended for enterprise-style RAG experiments. It is not a production-grade identity or authorization system.
