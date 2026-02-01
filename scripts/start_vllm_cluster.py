#!/usr/bin/env python3
import sys
import os
import json
import shutil
import tempfile
import subprocess
import time
from pathlib import Path

# Add benchmarks dir to path to import config
SCRIPT_DIR = Path(__file__).parent.resolve()
BENCH_DIR = SCRIPT_DIR.parent / "benchmarks"
OPT_DIR = Path("/opt")

# Check /opt first (Container), then local fallback
if (OPT_DIR / "run_vllm_bench.py").exists():
    sys.path.append(str(OPT_DIR))
else:
    sys.path.append(str(BENCH_DIR))

try:
    from run_vllm_bench import MODEL_TABLE, MODELS_TO_RUN
except ImportError:
    print("Error: Could not import run_vllm_bench.py config.")
    sys.exit(1)

if (OPT_DIR / "max_context_results.json").exists():
    RESULTS_FILE = OPT_DIR / "max_context_results.json"
else:
    RESULTS_FILE = BENCH_DIR / "max_context_results.json"

HOST = os.getenv("HOST", "0.0.0.0")
PORT = os.getenv("PORT", "8000")

def get_discovered_models():
    """
    Overrides the hardcoded MODELS_TO_RUN by looking at what we actually have results for.
    """
    if not RESULTS_FILE.exists():
        return MODELS_TO_RUN
        
    try:
        with open(RESULTS_FILE, "r") as f:
            data = json.load(f)
            
        verified_models = set()
        for r in data:
            if r.get("status") == "success":
                verified_models.add(r["model"])
        
        final_list = []
        for m in sorted(list(verified_models)):
            if m in MODEL_TABLE:
                final_list.append(m)
                
        if final_list:
            return final_list
            
    except Exception as e:
        print(f"Warning: Model discovery failed ({e}). Using default list.")
        
    return MODELS_TO_RUN

# Refresh the list of models to run based on what we found
MODELS_TO_RUN = get_discovered_models()

def check_dependencies():
    missing = []
    if not shutil.which("dialog"):
        missing.append("dialog")
    if not shutil.which("ssh"):
        missing.append("ssh")
    if not shutil.which("ray"):
        missing.append("ray")
        
    if missing:
        print(f"Error: Missing dependencies: {', '.join(missing)}.")
        print("Please install them (e.g., sudo dnf install dialog openssh-clients).")
        print("Ensure 'ray' is in your PATH (pip install ray).")
        sys.exit(1)

def run_dialog(args):
    """Runs dialog and returns stderr (selection)."""
    with tempfile.NamedTemporaryFile(mode="w+") as tf:
        cmd = ["dialog"] + args
        try:
            subprocess.run(cmd, stderr=tf, check=True)
            tf.seek(0)
            return tf.read().strip()
        except subprocess.CalledProcessError:
            return None # User cancelled

def show_info(title, msg):
    run_dialog(["--title", title, "--msgbox", msg, "12", "60"])

def get_subnet_from_ip(ip):
    """Accurately gets the /24 subnet string for the given IP."""
    parts = ip.split('.')
    return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"

def setup_ips_dialog(current_head, current_worker):
    """
    Uses dialog --form to let user edit Head and Worker IPs simultaneously.
    Returns (new_head, new_worker) or None if cancelled.
    """
    # Layout:
    # Label 1 (Head) at 1,1
    # Input 1 at 1,20
    # Label 2 (Worker) at 2,1
    # Input 2 at 2,20
    
    cmd = [
        "dialog",
        "--title", "Configure Cluster IPs",
        "--form", "Edit the IP addresses for the Cluster nodes:",
        "10", "60", "2",
        "Head Node IP:", "1", "1", current_head, "1", "20", "20", "0",
        "Worker Node IP:", "2", "1", current_worker, "2", "20", "20", "0"
    ]
    
    try:
        # dialog --form outputs to stderr: "field1\nfield2\n..."
        res = subprocess.run(cmd, stderr=subprocess.PIPE, check=True, text=True)
        lines = res.stderr.strip().split('\n')
        if len(lines) >= 2:
            return lines[0], lines[1]
    except subprocess.CalledProcessError:
        return None
    return None

def setup_worker_node(worker_ip, head_ip): 
    subnet = get_subnet_from_ip(worker_ip)
    
    # Script to run on worker
    script = f"""
    source /etc/profile
    # Silece the kill command
    ray stop --force > /dev/null 2>&1 || true
    export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
    export RAY_memory_monitor_refresh_ms=0
    export VLLM_HOST_IP={worker_ip}
    export RDMA_IFACE=$(ip -o addr show to {subnet} | awk '{{print $2}}' | head -n1)
    export NCCL_SOCKET_IFNAME=$RDMA_IFACE
    export GLOO_SOCKET_IFNAME=$RDMA_IFACE
    echo "Starting Ray Worker on {worker_ip} connecting to {head_ip}..."
    ray start --address='{head_ip}:6379' --num-gpus=1 --num-cpus=8 --disable-usage-stats
    """
    
    print(f"Setting up Worker Node ({worker_ip})...")
    
    # Use bash -s to read script from stdin
    # Command: ssh user@host "toolbox run -c vllm -- bash -s"
    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", worker_ip, 
        "toolbox run -c vllm -- bash -s"
    ]
    
    try:
        subprocess.run(ssh_cmd, input=script.encode(), check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to setup worker: {e}")
        return False

def setup_head_node(head_ip):
    subnet = get_subnet_from_ip(head_ip)
    
    print(f"Setting up Head Node ({head_ip})...")
    
    script = f"""
    # Silence the kill command
    ray stop --force > /dev/null 2>&1 || true
    export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
    export RAY_memory_monitor_refresh_ms=0
    export VLLM_HOST_IP={head_ip}
    export RDMA_IFACE=$(ip -o addr show to {subnet} | awk '{{print $2}}' | head -n1)
    export NCCL_SOCKET_IFNAME=$RDMA_IFACE
    export GLOO_SOCKET_IFNAME=$RDMA_IFACE
    echo "Starting Ray Head on {head_ip}..."
    ray start --head --port=6379 --node-ip-address={head_ip} --num-gpus=1 --num-cpus=8 --disable-usage-stats
    """
    
    try:
        # Run locally
        subprocess.run(["bash", "-s"], input=script.encode(), check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to setup head: {e}")
        return False

def check_ray_status():
    """Returns (active_nodes, total_gpus) parsing 'ray status' output roughly."""
    try:
        res = subprocess.run(["ray", "status"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            return 0, 0
            
        output = res.stdout
        active_nodes = 0
        in_active_section = False
        for line in output.splitlines():
            if "Active:" in line:
                in_active_section = True
                continue
            if "Pending:" in line or "Recent failures:" in line:
                in_active_section = False
            
            if in_active_section and line.strip().startswith("1 node_"):
                active_nodes += 1
                
        return active_nodes, 2 # Assume 2 GPUs as per success criteria
    except:
        return 0, 0

def wait_for_cluster():
    print("Waiting for Ray cluster to initialize (expecting 2 nodes)...")
    for i in range(30):
        nodes, gpus = check_ray_status()
        print(f"Check {i+1}/30: Active Nodes={nodes}")
        if nodes >= 2:
            print("Cluster is Ready!")
            time.sleep(2)
            return True
        time.sleep(2)
        
    print("Timeout waiting for cluster.")
    return False

def nuke_vllm_cache():
    """Removes vLLM cache directory."""
    cache = Path.home() / ".cache" / "vllm"
    if cache.exists():
        try:
            print(f"Clearing vLLM cache at {cache}...", end="", flush=True)
            subprocess.run(["rm", "-rf", str(cache)], check=True)
            cache.mkdir(parents=True, exist_ok=True)
            print(" Done.")
            time.sleep(1)
        except Exception as e:
            print(f" Failed: {e}")

def get_verified_config(model_id, tp_size, max_seqs):
    """Reads max_context_results.json."""
    default_config = {
        "ctx": int(MODEL_TABLE.get(model_id, {}).get("ctx", 8192)),
        "util": 0.90
    }
    
    if not RESULTS_FILE.exists():
        return default_config

    try:
        with open(RESULTS_FILE, "r") as f:
            data = json.load(f)
            
        matches = [r for r in data 
                  if r["model"] == model_id 
                  and r["tp"] == tp_size 
                  and r["max_seqs"] == max_seqs 
                  and r["status"] == "success"]
        
        if not matches:
            return default_config
            
        matches.sort(key=lambda x: (float(x["util"]), x["max_context_1_user"]), reverse=True)
        best = matches[0]
        return {
            "ctx": best["max_context_1_user"],
            "util": float(best["util"])
        }
    except:
        return default_config

def configure_and_launch_vllm(model_idx, head_ip):
    model_id = MODELS_TO_RUN[model_idx]
    config = MODEL_TABLE[model_id]
    name = model_id.split("/")[-1]
    
    # Defaults
    current_tp = 2 # Forced default for Cluster
    current_seqs = 1
    
    # Lookup Config
    verified = get_verified_config(model_id, current_tp, current_seqs if isinstance(current_seqs, int) else 1)
    current_ctx = verified["ctx"]
    current_util = verified["util"]
    
    clear_cache = False
    use_eager = True # Default True for cluster as per request ("enforce-eager")
    trust_remote = True # Default True as per request
    
    while True:
        cache_status = "YES" if clear_cache else "NO"
        eager_status = "YES" if use_eager else "NO"
        trust_status = "YES" if trust_remote else "NO"
        
        menu_args = [
            "--clear", "--backtitle", f"AMD VLLM CLUSTER Launcher (Head: {head_ip})",
            "--title", f"Configuration: {name}",
            "--menu", "Customize Launch Parameters:", "22", "65", "9",
            "1", f"Tensor Parallelism:   {current_tp} (Fixed)",
            "2", f"Concurrent Requests:  {current_seqs}",
            "3", f"Context Length:       {current_ctx}",
            "4", f"GPU Utilization:      {current_util}",
            "5", f"Trust Remote Code:    {trust_status}",
            "6", f"Erase vLLM Cache:     {cache_status}",
            "7", f"Force Eager Mode:     {eager_status}",
            "8", "LAUNCH SERVER"
        ]
        
        choice = run_dialog(menu_args)
        if not choice: return False
        
        if choice == "1":
            # TP Selection - Allow change but warn?
             new_tp = run_dialog([
                "--title", "Tensor Parallelism",
                "--rangebox", "Set TP Size:", "10", "40", "1", "8", str(current_tp)
            ])
             if new_tp: current_tp = int(new_tp)
             
        elif choice == "2":
            new_seqs = run_dialog([
                "--title", "Concurrent Requests",
                "--inputbox", "Enter Max Concurrent Requests (or 'auto'):", "10", "40", str(current_seqs)
            ])
            if new_seqs: 
                if new_seqs.lower().strip() == "auto":
                    current_seqs = "auto"
                else:
                    try:
                        current_seqs = int(new_seqs)
                    except ValueError:
                        pass
            
        elif choice == "3":
            new_ctx = run_dialog([
                "--title", "Context Length",
                "--inputbox", f"Enter Context Length (or 'auto'):", "10", "40", str(current_ctx)
            ])
            if new_ctx:
                if new_ctx.lower().strip() == "auto":
                    current_ctx = "auto"
                else:
                    try:
                        current_ctx = int(new_ctx)
                    except ValueError:
                        pass

        elif choice == "4":
             new_util = run_dialog([
                "--title", "GPU Utilization",
                "--inputbox", "Enter GPU Utilization (0.1 - 1.0):", "10", "40", str(current_util)
            ])
             if new_util: current_util = float(new_util)
             
        elif choice == "5":
            trust_remote = not trust_remote
             
        elif choice == "6":
            clear_cache = not clear_cache
            
        elif choice == "7":
            use_eager = not use_eager
            
        elif choice == "8":
            break
            
    # Build Command
    subprocess.run(["clear"])
    
    if clear_cache:
        nuke_vllm_cache()
    
    # Environment Setup
    # We need to set these variables in the current process before exec or pass them in env
    subnet = get_subnet_from_ip(head_ip)
    
    # Compute RDMA IFACE dynamically
    # Note: we need to run logical command to get the iface name
    try:
        iface_cmd = f"ip -o addr show to {subnet} | awk '{{print $2}}' | head -n1"
        rdma_iface = subprocess.check_output(iface_cmd, shell=True, text=True).strip()
    except:
        rdma_iface = "eth0" # Fallback
        print("Warning: Could not detect RDMA IFACE, defaulting to eth0")

    print(f"Detected RDMA Interface: {rdma_iface}")
    
    env = os.environ.copy()
    env["RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES"] = "1"
    env["VLLM_HOST_IP"] = head_ip
    env["NCCL_SOCKET_IFNAME"] = rdma_iface
    env["NCCL_IB_GID_INDEX"] = "1"
    env["NCCL_IB_DISABLE"] = "0"
    env["NCCL_NET_GDR_LEVEL"] = "0"
    
    # Also need this for Ray backend?
    # vLLM usually handles ray connection if we pass --distributed-executor-backend ray
    
    cmd = [
        "vllm", "serve", model_id,
        "--host", HOST,
        "--port", PORT,
        "--tensor-parallel-size", str(current_tp),
        "--gpu-memory-utilization", str(current_util),
        "--distributed-executor-backend", "ray",
        "--dtype", "auto"
    ]
    
    if str(current_seqs) != "auto":
        cmd.extend(["--max-num-seqs", str(current_seqs)])
        
    if str(current_ctx) != "auto":
        cmd.extend(["--max-model-len", str(current_ctx)])
    
    if trust_remote: cmd.append("--trust-remote-code")
    if use_eager: cmd.append("--enforce-eager")
    
    print("\n" + "="*60)
    print(f" Launching VLLM Cluster on Head: {head_ip}")
    print(f" Model:     {name}")
    print(f" Config:    TP={current_tp} | Seqs={current_seqs} | Ctx={current_ctx}")
    print(f" Command:   {' '.join(cmd)}")
    print("="*60 + "\n")
    
    # Exec
    os.execvpe("vllm", cmd, env)

def main():
    check_dependencies()
    
    # Default IPs
    head_ip = "192.168.100.1"
    worker_ip = "192.168.100.2"
    
    while True:
        # Main Menu
        # 1. Configure IPs
        # 2. Start Cluster (Ray)
        # 3. Start VLLM
        # 4. Exit
        
        choice = run_dialog([
            "--clear", "--backtitle", "AMD VLLM RCCL Cluster Manager",
            "--title", "Main Menu",
            "--menu", "Select Action:", "15", "60", "5",
            "1", f"Configure IPs (Head: {head_ip}, Worker: {worker_ip})",
            "2", "Start Ray Cluster",
            "3", "Ray Cluster Status",
            "4", "Launch VLLM Serve",
            "5", "Exit"
        ])
        
        if not choice or choice == "5":
            subprocess.run(["clear"])
            sys.exit(0)
            
        if choice == "1":
            res = setup_ips_dialog(head_ip, worker_ip)
            if res:
                head_ip, worker_ip = res
            
        elif choice == "2":
            subprocess.run(["clear"])
            print("= Starting Ray Cluster Setup =")
            # 1. Start Head
            if setup_head_node(head_ip):
                print("Head node started successfully. Waiting 5s before worker connection...")
                time.sleep(5)
                # 2. Start Worker
                if setup_worker_node(worker_ip, head_ip):
                    # 3. Wait for full cluster
                    wait_for_cluster()
            input("Press Enter to continue...")

        elif choice == "3":
            subprocess.run(["clear"])
            print("= Ray Cluster Status =")
            subprocess.run(["ray", "status"])
            input("\nPress Enter to continue...")
            
        elif choice == "4":
            # Select Model
            menu_items = []
            for i, m_id in enumerate(MODELS_TO_RUN):
                name = m_id.split("/")[-1]
                menu_items.extend([str(i), name])
                
            m_choice = run_dialog([
                "--title", "Select Model",
                "--menu", "Choose a model to serve:", "20", "60", "10"
            ] + menu_items)
            
            if m_choice:
                configure_and_launch_vllm(int(m_choice), head_ip)
                # Note: execvpe replaces process, so we won't return here.

if __name__ == "__main__":
    main()
