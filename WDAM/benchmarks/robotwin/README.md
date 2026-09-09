# RoboTwin Benchmark Evaluation

These scripts assume the LD4WAM policy server is **already running** (see
`scripts/deploy.sh`). They only cover the RoboTwin side of the evaluation loop
and never load model weights themselves.

## Files

| File | Description |
|---|---|
| `openwam2robotwin_interface.py` | RoboTwin policy client — talks to the WebSocket server. |
| `eval_policy_wrapper.py` | Runs RoboTwin's `eval_policy.py` without its fragile render self-test. |
| `prompt_template.py` | Wraps a raw task instruction in the prompt format used at training time. |
| `policy_config.yml` | Client config template; `host` / `port` are injected at runtime. |
| `single_eval.sh` | Evaluate one task. |
| `multi_eval.sh` | Evaluate several tasks sequentially. |
| `parallel_eval.sh` | Shared task queue across N already-running servers (pairs with `scripts/deploy_multi.sh`). |
| `export_results_csv.py` | Merge `summary.tsv` + per-task `Success rate` lines into one CSV. |
| `step_limits.yml` | Optional per-task `step_lim` overrides (ships empty = stock RoboTwin). |

## Environment setup

1. Install RoboTwin following the [official guide](https://github.com/RoboTwin-Platform/RoboTwin)
   (repo checkout + Conda env, default name `robotwin`).
2. Export the environment variables the scripts read:

   ```bash
   export ROBOTWIN_PATH=/path/to/RoboTwin   # RoboTwin repo root (required)
   export ROBOTWIN_ENV=robotwin             # Conda env name (default: robotwin)
   ```

3. Match `policy_config.yml` to the checkpoint's saved `config.yaml`:
   - `action_mode: eef` / `state_dim: 20` (default) → `action_type: ee`, `state_dim: 20`.
   - `action_mode: joint` / `state_dim: 14` → `action_type: qpos`, `state_dim: 14`.
   - Keep `send_state: true` for checkpoints with `use_proprioception: true`;
     the client fails fast when the extracted state dimension does not match.

4. Start the policy server (can be on another machine):

   ```bash
   bash scripts/deploy.sh --ckpt-dir /path/to/ckpt_dir --port 8848
   ```

## Usage

### Single task

```bash
bash single_eval.sh <task_name> <task_config> <ckpt_setting> <gpu_id> [port] [host]
# e.g.
bash single_eval.sh adjust_bottle demo_clean ld4wam 0 8848 127.0.0.1
```

| Argument | Description |
|---|---|
| `task_name` | RoboTwin task (e.g. `adjust_bottle`). |
| `task_config` | `demo_clean` or `demo_randomized`. |
| `ckpt_setting` | Label written into result file names. |
| `gpu_id` | CUDA device for the simulator. |
| `port` / `host` | Policy server address (default `8848` / `127.0.0.1`). |

### Multiple tasks

```bash
bash multi_eval.sh -m <demo_clean|demo_randomized> -n <name> -d <ckpt_dir> [--host H] [--port P] [-g GPU] <tasks...|all|tasks.txt>
```

`-d` only names the log directory (`<ckpt_dir>/robotwin_eval_logs/...`); the
evaluator still talks to the running server. `all` expands to the 50 RoboTwin 2.0
tasks; a file argument is read one task per line (`#` comments allowed).

### Parallel evaluation

Start N servers (`bash scripts/deploy_multi.sh /path/to/ckpt_dir`, GPU *i* → port `8848+i`),
then let N workers pull tasks from a shared queue:

```bash
ROBOTWIN_PATH=/path/to/RoboTwin \
bash parallel_eval.sh -m all -n run1 -w 8 --port 8848 --gpu-start 0 all
```

Run `bash parallel_eval.sh -h` for all options. Progress can be watched in a
browser with the dashboard:

```bash
python benchmarks/web_control.py <log_dir> --benchmark robotwin --host 0.0.0.0 --port 8765
```

Open `http://<node-ip>:8765/` (`--host 127.0.0.1` for local-only access);
`python benchmarks/web_control.py -h` lists all options.

### Export results

```bash
python benchmarks/robotwin/export_results_csv.py /path/to/log_dir [-o results.csv] [--strict]
```

Combines `<log_dir>/summary.tsv` with each task log's last `Success rate` line;
`--strict` exits non-zero on missing or unparseable logs.

## Client protocol (for other benchmarks / robots)

`openwam2robotwin_interface.py` is a thin client over `benchmarks/utils/client.py`;
any other environment can talk to the server the same way. One message per
control step:

```json
{"type": "obs",
 "images": {"head_camera": "<base64 JPEG>",
            "left_wrist_camera": "<base64 JPEG>|null",
            "right_wrist_camera": "<base64 JPEG>|null"},
 "prompt": "pick up the red bottle",
 "state": [0.0, 0.1]}
```

`head_camera` is required; the wrist cameras are optional (missing ones become
black frames on multi-view checkpoints); `prompt` is forwarded verbatim (see
`prompt_template.py`); `state` is required when the checkpoint has
`use_proprioception: true`. The reply is
`{"type": "action", "action": [...], "step": n, "latency_ms": t}` with the
action already in physical units. The server resizes images and composes the
multi-view canvas itself, and caches an action chunk internally (first call runs
the model, later calls pop the buffer), so **send `{"type": "reset"}` between
episodes**. `{"type": "ping"}` is a liveness check.

## FAQ

**Render Error on headless servers** — SAPIEN needs an X display:

```bash
xvfb-run -a bash single_eval.sh adjust_bottle demo_clean ld4wam 0 8848 127.0.0.1
```

**`policy_config.yml` options** — `send_state` / `state_dim` (proprio forwarding
and fail-fast check), `action_type` (`ee`: 20-D EEF from the server converted to
RoboTwin's 16-D xyz+quat+grip; `qpos`: 14-D joints passed straight through),
`request_timeout`, and `debug` / `debug_dir` (dump the per-camera JPEGs sent to
the server). Cameras and resolution need no client config: the server reads the
checkpoint's `config.yaml` and applies the training-time preprocessing.
