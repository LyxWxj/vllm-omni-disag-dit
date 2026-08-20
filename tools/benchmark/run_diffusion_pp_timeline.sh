#!/usr/bin/env bash
# Run one diffusion PP timeline experiment end to end.
#
# Example:
#   tools/benchmark/run_diffusion_pp_timeline.sh \
#     --name a100 --port 8000 \
#     --server-command 'CUDA_VISIBLE_DEVICES=0,1 vllm serve MODEL --omni --pipeline-parallel-size 2' \
#     --request-command 'curl -fsS http://127.0.0.1:8000/v1/images/generations -H "Content-Type: application/json" -d @request.json'

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMELINE_TOOL="${SCRIPT_DIR}/trace_diffusion_pp_timeline.py"
name=pp; server_command=""; request_command=""; output_root="${PWD}/pp-timeline-results"
port=""; startup_timeout_s=180; request_timeout_s=1800; shutdown_timeout_s=30; bin_us=100
sync_trace=1; keep_server_log=1

usage() {
    sed -n '1,10p' "$0"
    cat <<'EOF'

Options:
  --name NAME, --server-command COMMAND, --request-command COMMAND
  --output-root DIR, --port PORT, --startup-timeout-s SEC
  --request-timeout-s SEC, --shutdown-timeout-s SEC, --bin-us US
  --no-sync-trace, --remove-server-log, -h/--help
EOF
}

while (($#)); do
    case "$1" in
        --name) name="$2"; shift 2;;
        --server-command) server_command="$2"; shift 2;;
        --request-command) request_command="$2"; shift 2;;
        --output-root) output_root="$2"; shift 2;;
        --port) port="$2"; shift 2;;
        --startup-timeout-s) startup_timeout_s="$2"; shift 2;;
        --request-timeout-s) request_timeout_s="$2"; shift 2;;
        --shutdown-timeout-s) shutdown_timeout_s="$2"; shift 2;;
        --bin-us) bin_us="$2"; shift 2;;
        --no-sync-trace) sync_trace=0; shift;;
        --remove-server-log) keep_server_log=0; shift;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;;
    esac
done

if [[ -z "$server_command" || -z "$request_command" ]]; then
    echo "Both --server-command and --request-command are required." >&2; exit 2
fi
if ! [[ "$startup_timeout_s" =~ ^[0-9]+$ && "$request_timeout_s" =~ ^[0-9]+$ && "$shutdown_timeout_s" =~ ^[0-9]+$ && "$bin_us" =~ ^[0-9]+$ ]]; then
    echo "Timeouts and --bin-us must be non-negative integers." >&2; exit 2
fi

result_dir="${output_root}/${name}"; trace_dir="${result_dir}/trace"; server_log="${result_dir}/server.log"
mkdir -p "$trace_dir"
find "$trace_dir" -maxdepth 1 -type f -name 'pp_rank_*.jsonl' -delete
export VLLM_OMNI_PP_TRACE_DIR="$trace_dir"
export VLLM_OMNI_PP_TRACE_SYNC="$sync_trace"

server_pid=""
cleanup() {
    local status=$?
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        for ((i=0; i<shutdown_timeout_s*10; i++)); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 0.1
        done
        kill -KILL "$server_pid" 2>/dev/null || true
    fi
    [[ "$status" -eq 0 ]] || echo "Experiment failed; see ${server_log}" >&2
    exit "$status"
}
trap cleanup EXIT INT TERM

echo "[pp-timeline] result_dir=${result_dir} trace_sync=${sync_trace}"
bash -lc "$server_command" >"$server_log" 2>&1 &
server_pid=$!
ready=0
for ((i=0; i<startup_timeout_s*10; i++)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        tail -80 "$server_log" >&2 || true; exit 1
    fi
    if [[ -n "$port" ]]; then
        if (echo >/dev/tcp/127.0.0.1/"$port") 2>/dev/null; then ready=1; break; fi
    elif grep -Eiq 'server.*(running|started)|application startup complete|uvicorn running' "$server_log"; then
        ready=1; break
    fi
    sleep 0.1
done
if [[ "$ready" -ne 1 ]]; then
    echo "Server startup probe timed out; inspect ${server_log}." >&2
    tail -80 "$server_log" >&2 || true; exit 1
fi

echo "[pp-timeline] sending request"
if command -v timeout >/dev/null 2>&1; then timeout --signal=TERM "${request_timeout_s}s" bash -lc "$request_command"; else bash -lc "$request_command"; fi
echo "[pp-timeline] stopping server"
kill -INT "$server_pid" 2>/dev/null || kill -TERM "$server_pid" 2>/dev/null || true
for ((i=0; i<shutdown_timeout_s*10; i++)); do
    kill -0 "$server_pid" 2>/dev/null || break
    sleep 0.1
done

if ! compgen -G "${trace_dir}/pp_rank_*.jsonl" >/dev/null; then
    echo "No PP trace files produced; check that the model uses PipelineParallelMixin." >&2; exit 1
fi
python "$TIMELINE_TOOL" --trace-dir "$trace_dir" --output "${result_dir}/timeline" --bin-us "$bin_us"
[[ "$keep_server_log" -eq 1 ]] || rm -f "$server_log"
trap - EXIT INT TERM
echo "[pp-timeline] complete: ${result_dir}"
