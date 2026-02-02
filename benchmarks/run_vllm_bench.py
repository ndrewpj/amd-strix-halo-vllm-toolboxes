#!/usr/bin/env python3
import subprocess, time, json, sys, os, requests, argparse
from pathlib import Path


# =========================
# ⚙️ GLOBAL SETTINGS
# =========================

try:
    import models
except ImportError:
    # If running locally and models.py is in ../scripts?
    # Or if running in /opt where models.py is alongside.
    # We will try adding current dir to path just in case
    sys.path.append(os.getcwd())
    try:
        import models
    except ImportError:
        # Fallback for local structure: assuming this is in benchmarks/ and models is in scripts/
        sys.path.append(str(Path(__file__).parent.parent / "scripts"))
        import models

# Import from shared config
MODEL_TABLE = models.MODEL_TABLE
MODELS_TO_RUN = models.MODELS_TO_RUN
GPU_UTIL = models.GPU_UTIL
OFF_NUM_PROMPTS = models.OFF_NUM_PROMPTS
OFF_FORCED_OUTPUT = models.OFF_FORCED_OUTPUT
DEFAULT_BATCH_TOKENS = models.DEFAULT_BATCH_TOKENS

# Fallbacks
FALLBACK_INPUT_LEN  = 1024
FALLBACK_OUTPUT_LEN = 512

RESULTS_DIR = Path("benchmark_results")
RESULTS_DIR.mkdir(exist_ok=True)


# =========================
# UTILS
# =========================

def log(msg): print(f"\n[BENCH] {msg}")

def get_gpu_count():
    try:
        # Using rocm-smi --showid to list GPUs. 
        # Output format: "GPU[0] : Device Name: ..."
        res = subprocess.run(["rocm-smi", "--showid"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            return 1 # Force return 1 for Strix Halo APU
        else:
            log("rocm-smi failed, defaulting to 1 GPU (Hardcoded Fallback)")
            return 1
    except Exception as e:
        log(f"Error detecting GPUs: {e}, defaulting to 1 GPU")
        return 1

def kill_vllm():
    subprocess.run("pgrep -f 'vllm serve' | xargs -r kill -9", 
                   shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)

def nuke_vllm_cache():
    cache = Path.home() / ".cache" / "vllm"
    if cache.exists():
        try:
            subprocess.run(["rm", "-rf", str(cache)], check=True)
            cache.mkdir(parents=True, exist_ok=True)
            time.sleep(2)
        except: pass

def get_dataset():
    data_path = Path("ShareGPT_V3_unfiltered_cleaned_split.json")
    if data_path.exists(): return str(data_path)
    
    log("Downloading ShareGPT dataset...")
    url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    try:
        r = requests.get(url, stream=True, timeout=15)
        r.raise_for_status()
        with open(data_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        return str(data_path)
    except Exception as e:
        log(f"WARNING: ShareGPT download failed ({e}). using RANDOM.")
        return None



def get_model_args(model, tp_size):
    config = MODEL_TABLE.get(model, {"max_num_seqs": "32"})
    
    # Allow per-model GPU utilization override
    util = config.get("gpu_util", GPU_UTIL)

    cmd = [
        "--model", model,
        "--gpu-memory-utilization", util,
        "--dtype", "auto",
        "--tensor-parallel-size", str(tp_size),
        "--max-num-seqs", config["max_num_seqs"]
    ]
    
    # Optional: if a model really needs a hard limit, we can still support "ctx" in config,
    # but by default we rely on auto.
    if "ctx" in config:
        cmd.extend(["--max-model-len", config["ctx"]])
        
    if config.get("trust_remote"): cmd.append("--trust-remote-code")
    if config.get("enforce_eager"): cmd.append("--enforce-eager")
    
    return cmd

def run_throughput(model, tp_size, backend_name="Default", output_dir=RESULTS_DIR, extra_env=None):
    if tp_size not in MODEL_TABLE[model]["valid_tp"]: return
    
    model_safe = model.replace("/", "_")
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir_path / f"{model_safe}_tp{tp_size}_throughput.json"
    
    if output_file.exists():
        log(f"SKIP {model} (TP={tp_size} | {backend_name})")
        return

    dataset_path = get_dataset()
    dataset_args = ["--dataset-name", "sharegpt", "--dataset-path", dataset_path] if dataset_path else ["--input-len", "1024"]
    
    # Retrieve Model-Specific Batch Tokens
    batch_tokens = MODEL_TABLE[model].get("max_tokens", DEFAULT_BATCH_TOKENS)

    log(f"START {model} (TP={tp_size} | {backend_name}) [Batch: {batch_tokens}]...")
    kill_vllm()
    nuke_vllm_cache()

    cmd = ["vllm", "bench", "throughput"] + get_model_args(model, tp_size)
    cmd.extend([
        "--num-prompts", str(OFF_NUM_PROMPTS),
        "--max-num-batched-tokens", batch_tokens,
        "--output-len", OFF_FORCED_OUTPUT,
        "--output-json", str(output_file),
        "--disable-log-stats"
    ])
    cmd.extend(dataset_args)

    # Force Attention Backend via CLI if ROCm-Attn
    if backend_name == "ROCm-Attn":
        cmd.extend(["--attention-backend", "ROCM_ATTN"])

    # ENV Setup: Global + Model Specific
    env = os.environ.copy()
    
    # Inject model specific env vars (e.g. for AWQ)
    model_env = MODEL_TABLE[model].get("env", {})
    env.update(model_env)
    
    # Extra Env
    if extra_env:
        env.update(extra_env)

    try: 
        subprocess.run(cmd, check=True, env=env)
    except: 
        log(f"ERROR: Failed {model} [{backend_name}]")


def print_summary(tps):
    print(f"\n{'MODEL':<40} | {'TP':<2} | {'Triton':<8} | {'ROCm':<8}")
    print("-" * 75)
    
    for m in MODELS_TO_RUN:
        msafe = m.replace("/", "_")
        for tp in tps:
            if tp not in MODEL_TABLE[m]["valid_tp"]: continue
            
            # Default
            try: 
                p1 = RESULTS_DIR / f"{msafe}_tp{tp}_throughput.json"
                d1 = json.loads(p1.read_text())
                val1 = f"{d1.get('tokens_per_second', 0):.1f}"
            except: val1 = "N/A"
            
            # ROCm
            try:
                p2 = Path("benchmark_results_rocm") / f"{msafe}_tp{tp}_throughput.json"
                d2 = json.loads(p2.read_text())
                val2 = f"{d2.get('tokens_per_second', 0):.1f}"
            except: val2 = "N/A"

            name_cell = m.split('/')[-1]
            print(f"{name_cell:<40} | {tp:<2} | {val1:<8} | {val2:<8}")
    print("-" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, nargs="+", default=[1])
    args = parser.parse_args()
    
    gpu_count = get_gpu_count()
    log(f"Detected {gpu_count} AMD GPU(s)")
    
    valid_tp_args = [t for t in args.tp if t <= gpu_count]
    if not valid_tp_args:
        log(f"Requested TP={args.tp} but only {gpu_count} GPU(s) detected. Nothing to run.")
        sys.exit(0)

    kill_vllm()
    for tp in valid_tp_args:
        for m in MODELS_TO_RUN:
            # 1. Default (Triton)
            run_throughput(m, tp, "Default", RESULTS_DIR)
            
            # 2. ROCm Attention 
            # We force this via CLI argument --attention-backend ROCM_ATTN below
            # No specific env vars needed if forcing backend.
            rocm_env = {}
            print(f"[DEBUG] Forcing ROCm Env: {rocm_env} + CLI: --attention-backend ROCM_ATTN")
            run_throughput(m, tp, "ROCm-Attn", "benchmark_results_rocm", rocm_env)
            
    print_summary(valid_tp_args)
