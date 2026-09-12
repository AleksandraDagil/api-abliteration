# API exposure evaluation

This branch adds a lightweight, API-only evaluation track to the FAR.AI Safety
Gap Toolkit. It preserves the original GPU/vLLM pipeline and begins from
upstream commit `9482423f16f0372c937aab4d030bdc76bdaedb5a`.

The phase-one experiment evaluates hosted abliterated models on Bio. The
approved full EU run uses `abliterated-model` because that account's regional
endpoint does not expose `abliterated-model-large`:

- WMDP-Bio multiple-choice capability using generated answer letters;
- FAR.AI's `propensity_bio.csv` prompts;
- self-judged StrongREJECT refusal, convincingness, and specificity;
- binary and continuous effective-dangerous-capability proxies;
- latency, token, cache, credit, retry, reliability, and cost measurements.

Because there is no aligned comparison model in phase one, these outputs are an
API exposure study, not a measured safety gap. Web search is enabled with the
largest documented search context, so the result measures the hosted API system
(model plus search), not only the underlying weights. The same target model is
also the StrongREJECT judge, so those scores must be labelled self-judged.

## Installation

Create and activate a Python 3.10+ virtual environment, then install only the
API dependencies:

```powershell
python -m pip install -r requirements-api.txt
```

## Configuration and dry run

The complete named experiment is in
`safety_gap/hydra_config/api_experiment/abliterated_model_large_bio.yaml`.
Running without overrides is a dry run and makes no API calls:

```powershell
python api_main.py
```

When `sample_limit` is set, the runner uses the first eligible rows in dataset
order. This makes small review pilots deterministic and easy to audit.

You can inspect the resolved Hydra configuration with:

```powershell
python api_main.py --cfg job
```

## Live run

Do not put the API key in YAML or on the command line. Set it only in the
process environment:

```powershell
$env:ABLIT_KEY = 'replace-with-token'
python api_main.py execution.live=true execution.max_total_estimated_cost_usd=10
```

The dollar value is an example, not a recommended full-run budget. Agree and
set the intended ceiling before running.

The approved full run is defined in
`safety_gap/hydra_config/api_experiment/abliterated_model_bio_full.yaml`. It
uses four workers, a $70 ceiling, a fixed resumable run ID, and indefinite
capped-backoff reconnect handling. Start it in the background with:

```powershell
.\scripts\start_full_bio_background.ps1
```

Check progress without opening the transcript with:

```powershell
.\scripts\status_full_bio.ps1
```

Every completed sample is appended and synced to disk immediately. Every API
result, including transient failures, is also synced to
`request_events.jsonl`. Re-running the launcher skips completed transcript
rows. An abrupt process or power loss can repeat up to four in-flight calls,
but cannot erase already durable rows. Parallel in-flight calls can also cause
a small overshoot after the recorded-cost ceiling is reached.

## Local outputs

Every run creates `outputs/api_runs/<run-id>/` containing:

- `transcript.csv`: every question, full target answer and reasoning trace,
  target request/response metadata, the complete judge conversation and answer,
  parsed judgment, latency, retry, token, credit, and cost fields;
- `summary.csv` and `summary.json`: aggregate capability, compliance, EDC,
  affordability, reliability, cache, and latency measurements;
- `manifest.json`: resolved configuration, timestamps, sample counts, and run
  completion status.
- `request_events.jsonl`: append-only, disk-synced API attempts for recovery and
  diagnostics.

The complete `outputs/` directory is ignored by Git because it contains raw
potentially dangerous prompts and responses. Target-call costs are reported
separately from judge costs: only target cost represents actor-facing access
cost; judge cost is research overhead.

Provider-reported `estimated_cost_usd` is the primary cost field. The runner
also recalculates cost from documented token rates for cross-checking. Full raw
response JSON is retained in the transcript so newly added usage fields are not
lost.
