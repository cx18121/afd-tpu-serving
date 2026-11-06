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
HERE=$(cd "$(dirname "$0")" && pwd)

MODE=${1:?usage: vm_baseline.sh smoke|tp8_ep|dp2_tp4_ep|tp4_ep}
W=$VM_WORKDIR
STATUS_FILE=$W/logs/$MODE.status
CONTAINER=afd-vllm
PORT=8000
mkdir -p "$W/logs"
echo RUNNING > "$STATUS_FILE"
trap 'echo FAILED > "$STATUS_FILE"; docker rm -f $CONTAINER >/dev/null 2>&1 || true' ERR

# Benchmark protocol (see BASELINE.md). Defaults are the protocol all results
# so far use: random tokens, exactly 1024 in / 256 out, 200 prompts per point,
# plus GSM8K accuracy. Options for a later, fuller round:
#   WORKLOADS="1024:256 1024:1024"  more input:output pairs
#   LENGTHS=range                   lengths uniform in [0.8*L, L] (InferenceX)
#   NUM_PROMPTS=auto                max(5*c, 40) prompts, ~5 full waves
#   SHAREGPT=1                      one real-text point at c=64
#   SWEEP=0                         skip the sweep (e.g. GSM8K-only sessions)
WORKLOADS="${WORKLOADS:-1024:256}"
LENGTHS="${LENGTHS:-exact}"

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
    WORKLOADS="1024:256"
    WARMUPS=""
    PROFILE_CONCURRENCY=""
    ;;
  tp8_ep|dp2_tp4_ep|tp4_ep|afd_dp2)
    MODEL=Qwen/Qwen3-30B-A3B
    case "$MODE" in
      tp8_ep)     SERVE_ARGS=(--tensor-parallel-size 8 "${MOE_ARGS[@]}") ;;
      # One API server: vLLM otherwise starts one per DP rank.
      dp2_tp4_ep) SERVE_ARGS=(--data-parallel-size 2 --tensor-parallel-size 4
                    --api-server-count 1 "${MOE_ARGS[@]}") ;;
      tp4_ep)     SERVE_ARGS=(--tensor-parallel-size 4 "${MOE_ARGS[@]}") ;;
      # AFD (afd/plugin/afd_tpu): same layout as dp2_tp4_ep, but DP replica 0
      # runs attention for every request and replica 1 holds the experts.
      afd_dp2)    SERVE_ARGS=(--data-parallel-size 2 --tensor-parallel-size 4
                    --api-server-count 1 "${MOE_ARGS[@]}")
                  EXTRA_DOCKER=(-v "$HERE/plugin:/afd_plugin:ro"
                    -e AFD_ROLE_SPLIT=1 -e AFD_ATTN_DP_RANK=0
                    -e PYTHONPATH=/afd_plugin:/workspace/vllm-torchtpu/src:/workspace/vllm-torchtpu)
                  # Every request goes to the attention replica.
                  PIN_HEADER="X-data-parallel-rank=0" ;;
    esac
    CONCURRENCIES=$MOE_CONCURRENCIES
    NUM_PROMPTS="${NUM_PROMPTS:-200}"
    RUN_EXTRAS=1                     # GSM8K (+ ShareGPT if SHAREGPT=1)
    # Discarded passes that exercise small and large batches before measuring.
    WARMUPS="${WARMUPS-1:4 128:128}"
    PROFILE_CONCURRENCY=16
    ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac

case "$MODE" in
  smoke) RUN_NAME=smoke ;;
  afd_*) RUN_NAME="afd/$MODE" ;;
  *)     RUN_NAME="colocated/$MODE" ;;
esac
EXTRA_DOCKER=(${EXTRA_DOCKER[@]+"${EXTRA_DOCKER[@]}"})
PIN_HEADER=${PIN_HEADER:-}
RESULTS=$W/results/$RUN_NAME
PROFILE_DIR=$W/profile/$MODE
mkdir -p "$RESULTS" "$PROFILE_DIR"
SERVER_LOG=$W/logs/server_$MODE.log
DEST=$RUNS_URI/$(vm_run_id)

upload_results() {
  timeout 600 gcloud storage rsync -r "$W/results/" "$DEST/results/" --quiet || echo "WARN: results upload failed"
  timeout 600 gcloud storage rsync -r "$W/logs/" "$DEST/logs/" --quiet || echo "WARN: log upload failed"
}

# Test hook (tests/run_local_test.sh): AFD_MOCK_SERVER is a stand-in server
# script; the container runs it instead of vllm serve, so everything else in
# this file runs unchanged without a TPU.
if [ -n "${AFD_MOCK_SERVER:-}" ]; then
  # Keep torch and vLLM from loading torch_tpu, whose libraries need AVX,
  # which emulated amd64 on Apple silicon lacks.
  SERVER_CMD=(-e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e VLLM_PLUGINS=none
    -v "$AFD_MOCK_SERVER:/mock_server.py:ro"
    --entrypoint python3 "$IMAGE" /mock_server.py)
else
  SERVER_CMD=(--entrypoint vllm "$IMAGE" serve "$MODEL" "${SERVE_ARGS[@]}" --port $PORT)
fi

docker rm -f $CONTAINER >/dev/null 2>&1 || true
# One compile cache per mode: AFD changes the expert count per chip (16 -> 32),
# and a cache shared with dp2_tp4_ep served AFD a stale 16-expert executable
# ("RuntimeProgramInputMismatch"), so the cache key does not cover weight shapes.
mkdir -p "$W/cache/$MODE"
echo "$(date -Is) starting $MODEL"
printf '%q ' vllm serve "$MODEL" "${SERVE_ARGS[@]}" --port $PORT > "$RESULTS/serve_cmd.txt"
docker run -d --name $CONTAINER \
  --privileged --net=host \
  -v /dev/shm:/dev/shm --shm-size 150gb \
  -v "$W/cache:/cache" -v "$W/hf:/hf" -v "$W/results:/results" -v "$PROFILE_DIR:/profile" \
  -e HF_HOME=/hf \
  -e TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=/cache/$MODE/torch_tpu_tier3 \
  -e VLLM_CACHE_ROOT=/cache/$MODE/vllm \
  -e PYTHONUNBUFFERED=1 \
  ${EXTRA_DOCKER[@]+"${EXTRA_DOCKER[@]}"} \
  "${SERVER_CMD[@]}" >/dev/null
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
python3 - "$MODEL" "$PORT" "$RESULTS/sanity.json" "$PIN_HEADER" "$RESULTS/greedy.json" <<'PY'
import json, sys, urllib.request
model, port, out, pin, greedy_out = sys.argv[1:6]
headers = {"Content-Type": "application/json"}
if pin:
    k, v = pin.split("=", 1)
    headers[k] = v
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
                                 json.dumps(body).encode(), headers)
    reply = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
    # Ignore spacing and case: "28,29,30" and "28, 29, 30" are both right.
    squash = lambda t: "".join(t.lower().split())
    passed = squash(expect) in squash(reply)
    ok &= passed
    results.append({"prompt": prompt, "expect": expect, "reply": reply, "pass": passed})
    print(("PASS" if passed else "FAIL"), repr(prompt), "->", repr(reply[:300]))
json.dump({"all_pass": ok, "cases": results}, open(out, "w"), indent=2)

# Greedy reference: the first 32 tokens of fixed raw prompts, compared
# token by token across layouts (AFD must match colocated serving).
prompts = [
    "The three laws of thermodynamics are",
    "def fibonacci(n):",
    "In 1969, the Apollo 11 mission",
    "A mixture-of-experts language model routes each token to",
]
greedy = []
for p in prompts:
    body = {"model": model, "prompt": p, "temperature": 0, "max_tokens": 32,
            "logprobs": 1}
    req = urllib.request.Request(f"http://localhost:{port}/v1/completions",
                                 json.dumps(body).encode(), headers)
    choice = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]
    greedy.append({"prompt": p, "text": choice["text"],
                   "tokens": (choice.get("logprobs") or {}).get("tokens")})
json.dump(greedy, open(greedy_out, "w"), indent=2)
sys.exit(0 if ok else 1)
PY

# GSM8K accuracy (lm-evaluation-harness, in a throwaway Python container).
if [ "${RUN_EXTRAS:-0}" = 1 ] && [ "${GSM8K:-1}" = 1 ]; then
  echo "$(date -Is) GSM8K"
  timeout 2400 docker run --rm --net=host \
    -v "$HERE/gsm8k_eval.py:/gsm8k_eval.py:ro" -v "$RESULTS:/out" python:3.12-slim \
    bash -c "pip install -q 'lm-eval[api]==0.4.13' && python /gsm8k_eval.py \
      --port $PORT --model $MODEL --out /out/gsm8k.json ${PIN_HEADER:+--header $PIN_HEADER} \
      ${GSM8K_LIMIT:+--limit $GSM8K_LIMIT}" \
    > "$W/logs/${MODE}_gsm8k.log" 2>&1 || echo "WARN: GSM8K failed (see logs/${MODE}_gsm8k.log)"
  grep '^GSM8K' "$W/logs/${MODE}_gsm8k.log" || true
  upload_results
fi

bench() {  # bench ISL OSL CONCURRENCY NUM_PROMPTS RESULT_NAME [extra args...]
  local isl=$1 osl=$2 c=$3 n=$4 name=$5; shift 5
  local len_args
  if [ "$LENGTHS" = range ]; then
    # Uniform in [0.8*L, L]: vLLM samples [(1-r)*m, (1+r)*m], so m = 0.9*L, r = 1/9.
    len_args=(--random-input-len $(( isl * 9 / 10 )) --random-output-len $(( osl * 9 / 10 ))
      --random-range-ratio 0.1111)
  else
    len_args=(--random-input-len "$isl" --random-output-len "$osl")
  fi
  timeout 2700 docker exec $CONTAINER vllm bench serve \
    --backend vllm --model "$MODEL" --host localhost --port $PORT \
    --dataset-name random "${len_args[@]}" \
    --num-prompts "$n" --max-concurrency "$c" --request-rate inf \
    --ignore-eos --temperature 0 --seed 42 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "/results/$RUN_NAME" --result-filename "$name.json" \
    ${PIN_HEADER:+--header "$PIN_HEADER"} "$@"
}

first_wl=${WORKLOADS%% *}
for w in $WARMUPS; do
  echo "$(date -Is) warmup c=${w%%:*} n=${w##*:} (discarded)"
  bench "${first_wl%%:*}" "${first_wl##*:}" "${w%%:*}" "${w##*:}" "warmup_c${w%%:*}" \
    > "$W/logs/${MODE}_warmup_c${w%%:*}.log" 2>&1
  rm -f "$RESULTS/warmup_c${w%%:*}.json"
done

# Repeat the whole sweep (not each point back to back) so slow drift shows up
# as spread between repeats instead of hiding inside one point.
RUNS=$([ "$MODE" = smoke ] && echo 1 || echo "${REPEATS:-1}")
[ "${SWEEP:-1}" = 0 ] && RUNS=0
for r in $(seq 1 "$RUNS"); do
  for wl in $WORKLOADS; do
    isl=${wl%%:*} osl=${wl##*:}
    for c in $CONCURRENCIES; do
      if [ "$NUM_PROMPTS" = auto ]; then n=$(( 5 * c > 40 ? 5 * c : 40 )); else n=$NUM_PROMPTS; fi
      name="w${isl}x${osl}$([ "$LENGTHS" = range ] || echo e)_c${c}_r$r"
      echo "$(date -Is) benchmark $name n=$n repeat $r/$RUNS"
      bench "$isl" "$osl" "$c" "$n" "$name" 2>&1 | tee "$W/logs/${MODE}_${name}.log"
      upload_results
    done
  done
done

# One real-text point: ShareGPT conversations (routing on real text can load
# experts differently from random tokens). Output lengths from the dataset.
if [ "${RUN_EXTRAS:-0}" = 1 ] && [ "${SHAREGPT:-0}" = 1 ] && [ "${SWEEP:-1}" != 0 ]; then
  if docker exec $CONTAINER python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('anon8231489123/ShareGPT_Vicuna_unfiltered',
                'ShareGPT_V3_unfiltered_cleaned_split.json', repo_type='dataset',
                local_dir='/hf/sharegpt')" > "$W/logs/${MODE}_sharegpt_download.log" 2>&1; then
    echo "$(date -Is) ShareGPT c=64"
    timeout 2700 docker exec $CONTAINER vllm bench serve \
      --backend vllm --model "$MODEL" --host localhost --port $PORT \
      --dataset-name sharegpt --dataset-path /hf/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
      --num-prompts 320 --max-concurrency 64 --request-rate inf --temperature 0 --seed 42 \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
      --save-result --result-dir "/results/$RUN_NAME" --result-filename sharegpt_c64_r1.json \
      ${PIN_HEADER:+--header "$PIN_HEADER"} 2>&1 | tee "$W/logs/${MODE}_sharegpt_c64.log"
    upload_results
  else
    echo "WARN: ShareGPT download failed; skipping the real-text point"
  fi
fi

if [ -n "$PROFILE_CONCURRENCY" ] && [ "${SWEEP:-1}" != 0 ]; then
  # Separate, shorter run so profiler overhead never touches measured numbers.
  c=$PROFILE_CONCURRENCY
  echo "$(date -Is) profiled run c=$c"
  bench "${first_wl%%:*}" "${first_wl##*:}" "$c" $((2 * c)) "profile_c$c" --profile \
    2>&1 | tee "$W/logs/${MODE}_profile_c$c.log"
  sleep 20  # let ranks finish writing traces
  timeout 900 gcloud storage rsync -r "$W/profile/" "$DEST/profile/" --quiet || echo "WARN: profile upload failed"
fi

docker rm -f $CONTAINER >/dev/null
upload_results
echo DONE > "$STATUS_FILE"
echo "$(date -Is) $MODE done"
