#!/usr/bin/env bash
# Runs on the TPU VM as root (started detached by scripts/run_baseline.sh).
#
#   vm_baseline.sh smoke        Qwen3-0.6B, TP=1, one small benchmark
#   vm_baseline.sh tp8_ep       Qwen3-30B-A3B, TP=8 + EP=8           (8 chips)
#   vm_baseline.sh dp2_tp4_ep   Qwen3-30B-A3B, 2 replicas x TP=4, EP=8 (8 chips)
#   vm_baseline.sh tp4_ep       Qwen3-30B-A3B, TP=4 + EP=4           (4 chips)
#
# Each finished benchmark is copied to the bucket right away, so a VM that hits
# its max run duration mid-sweep still leaves the finished results behind.
# Writes $VM_WORKDIR/logs/<mode>.status as RUNNING, DONE, or FAILED.
set -Eeuo pipefail
source "$(dirname "$0")/config.sh"

MODE=${1:?usage: vm_baseline.sh smoke|tp8_ep|dp2_tp4_ep|tp4_ep}
W=$VM_WORKDIR
STATUS_FILE=$W/logs/$MODE.status
CONTAINER=afd-vllm
PORT=8000
mkdir -p "$W/logs"
echo RUNNING > "$STATUS_FILE"
trap 'echo FAILED > "$STATUS_FILE"; docker rm -f $CONTAINER >/dev/null 2>&1 || true' ERR

# Shared benchmark workload (course plan): random tokens, 1024 in, 256 out.
INPUT_LEN=1024
OUTPUT_LEN=256

# Settings shared by every Qwen3-30B-A3B layout. This is the v6e recipe from
# the vllm-torchtpu docs, except:
#  - bf16 KV cache (no --kv-cache-dtype fp8): the CUSTOM attention backend with
#    fp8 KV and one KV head per chip decodes garbage (bug #2435). bf16 KV still
#    leaves several times the KV space the c=128 sweep needs.
#  - --max-num-seqs 128 (per replica): covers c=128 and skips compiling the
#    256/512 batch buckets.
MOE_ARGS=(--enable-expert-parallel --attention-backend CUSTOM
  --max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 128
  --gpu-memory-utilization 0.92 --async-scheduling --no-enable-prefix-caching
  --profiler-config.profiler torch --profiler-config.torch_profiler_dir /profile
  --profiler-config.ignore_frontend true)
MOE_CONCURRENCIES="${CONCURRENCIES:-1 4 8 16 32 64 128}"

case "$MODE" in
  smoke)
    MODEL=Qwen/Qwen3-0.6B
    # Default attention backend on purpose: Qwen3-0.6B has 8 KV heads on one
    # chip, a layout that bug #2435 breaks in the CUSTOM backend.
    SERVE_ARGS=(--tensor-parallel-size 1 --max-model-len 4096
      --max-num-batched-tokens 4096 --max-num-seqs 16
      --gpu-memory-utilization 0.90 --async-scheduling --no-enable-prefix-caching)
    CONCURRENCIES="4"
    NUM_PROMPTS=20
    WARMUPS=""
    PROFILE_CONCURRENCY=""
    ;;
  tp8_ep|dp2_tp4_ep|tp4_ep)
    MODEL=Qwen/Qwen3-30B-A3B
    case "$MODE" in
      tp8_ep)     SERVE_ARGS=(--tensor-parallel-size 8 "${MOE_ARGS[@]}") ;;
      # One API server: vLLM otherwise starts one per DP rank.
      dp2_tp4_ep) SERVE_ARGS=(--data-parallel-size 2 --tensor-parallel-size 4
                    --api-server-count 1 "${MOE_ARGS[@]}") ;;
      tp4_ep)     SERVE_ARGS=(--tensor-parallel-size 4 "${MOE_ARGS[@]}") ;;
    esac
    CONCURRENCIES=$MOE_CONCURRENCIES
    NUM_PROMPTS="${NUM_PROMPTS:-200}"
    # Discarded passes that exercise small and large batches before measuring.
    WARMUPS="1:4 128:128"
    PROFILE_CONCURRENCY=16
    ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac

RUN_NAME=$([ "$MODE" = smoke ] && echo smoke || echo "colocated/$MODE")
RESULTS=$W/results/$RUN_NAME
PROFILE_DIR=$W/profile/$MODE
mkdir -p "$RESULTS" "$PROFILE_DIR"
SERVER_LOG=$W/logs/server_$MODE.log
DEST=$RUNS_URI/$(vm_run_id)

upload_results() {
  timeout 600 gcloud storage rsync -r "$W/results/" "$DEST/results/" --quiet || echo "WARN: results upload failed"
  timeout 600 gcloud storage rsync -r "$W/logs/" "$DEST/logs/" --quiet || echo "WARN: log upload failed"
}

docker rm -f $CONTAINER >/dev/null 2>&1 || true
echo "$(date -Is) starting $MODEL"
printf '%q ' vllm serve "$MODEL" "${SERVE_ARGS[@]}" --port $PORT > "$RESULTS/serve_cmd.txt"
docker run -d --name $CONTAINER \
  --privileged --net=host \
  -v /dev/shm:/dev/shm --shm-size 150gb \
  -v "$W/cache:/cache" -v "$W/hf:/hf" -v "$W/results:/results" -v "$PROFILE_DIR:/profile" \
  -e HF_HOME=/hf \
  -e TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=/cache/torch_tpu_tier3 \
  -e VLLM_CACHE_ROOT=/cache/vllm \
  -e PYTHONUNBUFFERED=1 \
  --entrypoint vllm "$IMAGE" \
  serve "$MODEL" "${SERVE_ARGS[@]}" --port $PORT >/dev/null
docker logs -f $CONTAINER > "$SERVER_LOG" 2>&1 &

# Cold start downloads weights and compiles; allow up to 75 minutes.
start=$(date +%s)
until curl -sf --max-time 10 localhost:$PORT/health >/dev/null; do
  if [ "$(docker inspect -f '{{.State.Running}}' $CONTAINER 2>/dev/null)" != true ]; then
    echo "Server container exited during startup. Last log lines:" >&2
    tail -50 "$SERVER_LOG" >&2
    false
  fi
  if [ $(( $(date +%s) - start )) -gt 4500 ]; then
    echo "Server not healthy after 75 minutes." >&2
    false
  fi
  sleep 15
done
echo "$(date -Is) server healthy after $(( $(date +%s) - start ))s" | tee "$RESULTS/startup.txt"

# Record exact versions.
{
  echo "image: $IMAGE"
  docker image inspect "$IMAGE" --format 'image id: {{.Id}}'
  docker exec $CONTAINER pip list 2>/dev/null | grep -iE '^(vllm|torch|jax|jaxlib|libtpu|transformers)' || true
} > "$RESULTS/versions.txt"

# Correctness check: greedy replies with known answers. The #2435 failure mode
# is a correct first token followed by garbage, so check long outputs too.
python3 - "$MODEL" "$PORT" "$RESULTS/sanity.json" <<'PY'
import json, sys, urllib.request
model, port, out = sys.argv[1], sys.argv[2], sys.argv[3]
cases = [
    ("What is the capital of France? Answer in one word.", "Paris"),
    ("What is 17 * 23? Reply with just the number.", "391"),
    ("Count from 1 to 30, separated by commas and spaces. Output only the numbers.", "27, 28, 29, 30"),
    ("Explain in three sentences why the sky is blue.", "scatter"),
]
results, ok = [], True
for prompt, expect in cases:
    body = {"model": model, "temperature": 0, "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    reply = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
    passed = expect.lower() in reply.lower()
    ok &= passed
    results.append({"prompt": prompt, "expect": expect, "reply": reply, "pass": passed})
    print(("PASS" if passed else "FAIL"), repr(prompt), "->", repr(reply[:300]))
json.dump({"all_pass": ok, "cases": results}, open(out, "w"), indent=2)
sys.exit(0 if ok else 1)
PY

bench() {  # bench CONCURRENCY NUM_PROMPTS RESULT_NAME [extra args...]
  local c=$1 n=$2 name=$3; shift 3
  timeout 2700 docker exec $CONTAINER vllm bench serve \
    --backend vllm --model "$MODEL" --host localhost --port $PORT \
    --dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
    --num-prompts "$n" --max-concurrency "$c" --request-rate inf \
    --ignore-eos --temperature 0 --seed 42 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "/results/$RUN_NAME" --result-filename "$name.json" "$@"
}

for w in $WARMUPS; do
  echo "$(date -Is) warmup c=${w%%:*} n=${w##*:} (discarded)"
  bench "${w%%:*}" "${w##*:}" "warmup_c${w%%:*}" > "$W/logs/${MODE}_warmup_c${w%%:*}.log" 2>&1
  rm -f "$RESULTS/warmup_c${w%%:*}.json"
done

for c in $CONCURRENCIES; do
  echo "$(date -Is) benchmark c=$c n=$NUM_PROMPTS"
  bench "$c" "$NUM_PROMPTS" "colocated_c$c" 2>&1 | tee "$W/logs/${MODE}_c$c.log"
  upload_results
done

if [ -n "$PROFILE_CONCURRENCY" ]; then
  # Separate, shorter run so profiler overhead never touches measured numbers.
  c=$PROFILE_CONCURRENCY
  echo "$(date -Is) profiled run c=$c"
  bench "$c" $((2 * c)) "profile_c$c" --profile 2>&1 | tee "$W/logs/${MODE}_profile_c$c.log"
  sleep 20  # let ranks finish writing traces
  timeout 900 gcloud storage rsync -r "$W/profile/" "$DEST/profile/" --quiet || echo "WARN: profile upload failed"
fi

docker rm -f $CONTAINER >/dev/null
upload_results
echo DONE > "$STATUS_FILE"
echo "$(date -Is) $MODE done"
