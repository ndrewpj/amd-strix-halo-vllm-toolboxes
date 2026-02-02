#!/usr/bin/env python3
import json
import math
from pathlib import Path

# Config
RESULTS_FILE = Path(__file__).parent.parent / "benchmarks/max_context_results.json"

ORDER = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "google/gemma-3-12b-it",
    "openai/gpt-oss-20b",
    "Qwen/Qwen3-14B-AWQ",
    "btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-4bit",
    "btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-8bit",
    "dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16",
    "openai/gpt-oss-120b",
    "zai-org/GLM-4.7-Flash"
]

def format_tokens(n):
    if n >= 1024:
        return f"{int(n/1024)}k"
    return str(n)

def main():
    if not RESULTS_FILE.exists():
        print(f"Error: {RESULTS_FILE} not found.")
        return

    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    # Organize data: model -> tp -> requests -> result
    models = {}
    
    for entry in data:
        if entry["status"] != "success":
            continue
            
        model = entry["model"]
        tp = entry["tp"]
        seqs = entry["max_seqs"]
        util = float(entry["util"])
        ctx = entry["max_context_1_user"]
        
        if model not in models:
            models[model] = {}
        
        if tp not in models[model]:
            models[model][tp] = {}
            
        # Store tuple (ctx, util)
        # If multiple entries for same seqs (e.g. diff utils), pick standard logic?
        # The JSON usually has the best working one or we filter.
        # Assuming unique best entry per seqs/tp tuple from the finder script behavior.
        models[model][tp][seqs] = (ctx, util)

    # Generate Table
    print("| Model | TP | 1 Req | 4 Reqs | 8 Reqs | 16 Reqs |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")

    for model_name in ORDER:
        if model_name not in models:
             # Identify if there's a different naming or just missing
             continue
        
        tps = sorted(models[model_name].keys())
        for tp in tps:
            row = [f"**`{model_name}`**", str(tp)]
            
            for req in [1, 4, 8, 16]:
                val = models[model_name][tp].get(req)
                if val:
                    ctx, util = val
                    row.append(f"{format_tokens(ctx)} ({util:.2f})")
                else:
                    row.append("-")
            
            print("| " + " | ".join(row) + " |")

if __name__ == "__main__":
    main()
