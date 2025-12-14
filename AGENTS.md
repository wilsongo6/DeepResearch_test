# Repository Guidelines

## Project Structure & Module Organization
- `inference/`: ReAct runtime (`react_agent.py`, `run_multi_react.py`), prompts, and core tools.
- `evaluation/`: Official benchmark scripts; name new runners after the dataset they target.
- `WebAgent/`: Historical agent families (WebWalker, WebSailor, etc.) kept mostly read-only for reference.
- `eval_data/`: Local JSON/JSONL inputs plus evidence under `file_corpus/`; mirror benchmark aliases in filenames.
- Root docs, assets, `requirements.txt`, and `.env.example` live at the top level for quick onboarding.

## Build, Test, and Development Commands
Use Python 3.10 in an isolated environment:
```bash
conda create -n deepresearch python=3.10 && conda activate deepresearch
pip install -r requirements.txt
cp .env.example .env  # then edit secrets
```
Smoke-test any change with a constrained rollout before scaling:
```bash
python inference/run_multi_react.py \
  --model /models/tongyi \
  --dataset eval_data/sample.jsonl \
  --output outputs/sample \
  --max_workers 2 --roll_out_count 1 --total_splits 1 --worker_split 0
```
`bash inference/run_react_infer.sh` wraps the same parameters once MODEL_PATH/DATASET/OUTPUT_PATH are set. Evaluate via `python evaluation/evaluate_deepsearch_official.py --input_fp outputs/sample --dataset browsecomp` or `python evaluation/evaluate_hle_official.py --input_fp outputs/hle --model_path /models/qwen`.

## Coding Style & Naming Conventions
Write Python with 4-space indents, type hints for public functions, and docstrings describing tool side effects. Keep methods `snake_case`, classes `CamelCase`, and new utilities named `tool_<verb>.py` so they register through `TOOL_MAP`. Group imports (stdlib/third-party/local) and avoid new wildcard imports. Preserve XML tags (`<tool_call>`, `<tool_response>`, `<answer>`) because downstream parsing depends on them.

## Testing Guidelines
Treat every dataset change as a test: place fixture JSON/JSONL in `eval_data/`, keep referenced documents in `file_corpus/`, and confirm rollouts emit `<answer>` plus tool traces under `outputs/<run>/`. Before submitting leaderboard results, run the matching `evaluation/*.py` script and attach the metrics in the PR. When debugging quotas, tune `MAX_LLM_CALL_PER_RUN` via env vars and capture logs that show retry/backoff behavior.

## Commit & Pull Request Guidelines
Recent commits (e.g., `Fix typo in visit tool`, `删除 scholar、file工具`) show the expected short, imperative summary; keep messages under ~50 characters and mention the touched module when helpful. Pull requests must cover scope, config/env changes, logs from `run_multi_react.py` or `evaluation/*.py`, and screenshots only when assets move. Link issues, document API limits, and tag maintainers of the directories you touched.

## Configuration & Secrets
Never hard-code credentials. Copy `.env.example`, populate `SERPER_KEY_ID`, `JINA_API_KEYS`, `API_KEY/API_BASE`, `DASHSCOPE_API_KEY`, and `SANDBOX_FUSION_ENDPOINT`, and source the file locally. Document new env vars inline plus here, and gate optional tools behind flags so other contributors can develop without provisioning extra keys.
