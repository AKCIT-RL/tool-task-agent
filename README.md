# MCP Benchmark

Run agent benchmarks in an environment accessed via **MCP** (Model Context Protocol) tools. The agent perceives the world through MCP tools, reasons about a task, and selects actions until the task succeeds or a step limit is reached.

## Prerequisites

- **Python 3.11+**
- **MCP server** exposing the benchmark tools (see [MCP server](#mcp-server))
- **OpenAI API key** (for the LLM used by the agent)

---

## Environment setup

### 1. Clone and enter the repo

```bash
git clone <repository-url>
cd tool-task-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Install dependencies

```bash
pip install -e .
```

Or without editable install:

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root (never commit it):

```env
OPENAI_API_KEY=sk-your-openai-api-key
```

Optional overrides:

```env
MCP_URL=http://localhost:8000/mcp
BENCHMARK_MODEL=gpt-4o-mini
LOG_LEVEL=INFO
USE_IMAGE=true
GOAL_SUFFIX=
```

---

## MCP server

Start your MCP server **before** running the benchmark (e.g. on `http://localhost:8000/mcp`).

---

### RUN (evaluation)

Runs several episodes, saves per-episode JSON and LLM history.

```bash
python run_evaluation.py --num-episodes 3 --output-dir eval_results --max-steps 100 --no-image
```

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--num-episodes` | 1 | Number of episodes |
| `--output-dir` | eval_results | Directory for `episode_N.json` files |
| `--max-steps` | 100 | Max steps per episode |
| `--no-image` | — | Disable `get_image` in perception |

Example:

```bash
python run_evaluation.py --num-episodes 5 --output-dir eval_results --max-steps 100 --no-image
```

Output:

- **eval_results/episode_N.json** — goal, task_success, steps, llm_history
- **logs/llm_history_epN_YYYYMMDD_HHMMSS.txt** — human-readable LLM log per episode

### Test MCP connection

Quick check that the MCP server is reachable and returns a task:

```bash
python scripts/test_mcp.py
```
