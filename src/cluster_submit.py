"""
Submit cross-alloy TL experiments to TUHH HPC cluster.

One SLURM job per (alloy, fold, setting, model) combination, 1 core each.
75 combos x 5 models = 375 jobs. With cluster parallelism, finishes in
under an hour.

Usage:
    C:/Users/mbusc/miniconda3/envs/xtb_env/python.exe cluster_submit.py
"""
import sys, os, json, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paramiko

from data_loader import TARGET_ALLOYS, K_FOLD, SETTINGS

LOCAL_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RESULTS = LOCAL_ROOT / "results"
LOCAL_RESULTS.mkdir(exist_ok=True)
LOCAL_SRC = LOCAL_ROOT / "src"

# Cluster config
HOSTNAME = "hpclogin.rz.tuhh.de"
USERNAME = os.environ.get("TL_CLUSTER_USERNAME", "cmb1565")
PASSWORD = (os.environ.get("TL_CLUSTER_PASSWORD")
            or os.environ.get("TUHH_CLUSTER_PASSWORD")
            or "")  # set via env var or prompt
REMOTE_PYTHON = "~/xtb_env/bin/python"
REMOTE_BASE = "~/tl_crossalloy"


def get_password():
    global PASSWORD
    if not PASSWORD:
        import getpass
        PASSWORD = getpass.getpass(f"Cluster password for {USERNAME}@{HOSTNAME}: ")
    return PASSWORD

# SLURM time per (model, setting)
TIME_MAP = {
    ("RF", "exact"): "00:30:00",
    ("RF", "close_unfilt"): "01:00:00",
    ("RF", "close_filt"): "01:00:00",
    ("RF", "far_unfilt"): "02:00:00",
    ("RF", "far_filt"): "02:00:00",
    ("GBR", "exact"): "00:30:00",
    ("GBR", "close_unfilt"): "02:00:00",
    ("GBR", "close_filt"): "02:00:00",
    ("GBR", "far_unfilt"): "06:00:00",
    ("GBR", "far_filt"): "06:00:00",
    ("MLP", "exact"): "00:30:00",
    ("MLP", "close_unfilt"): "01:30:00",
    ("MLP", "close_filt"): "01:30:00",
    ("MLP", "far_unfilt"): "04:00:00",
    ("MLP", "far_filt"): "04:00:00",
    ("kNN_Tan", "exact"): "00:15:00",
    ("kNN_Tan", "close_unfilt"): "00:30:00",
    ("kNN_Tan", "close_filt"): "00:30:00",
    ("kNN_Tan", "far_unfilt"): "01:00:00",
    ("kNN_Tan", "far_filt"): "01:00:00",
    ("ChemProp", "exact"): "01:00:00",
    ("ChemProp", "close_unfilt"): "04:00:00",
    ("ChemProp", "close_filt"): "04:00:00",
    ("ChemProp", "far_unfilt"): "10:00:00",
    ("ChemProp", "far_filt"): "10:00:00",
}

MAX_QUEUED = 500
BATCH_SIZE = 100


def ssh_connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOSTNAME, username=USERNAME, password=get_password())
    return ssh


def download_predictions():
    """Pull every predictions/*.json from the cluster into local results/predictions/."""
    ssh = ssh_connect()
    out, _ = exec_cmd(ssh, f"echo {REMOTE_BASE}")
    remote_base = out.strip()
    sftp = ssh.open_sftp()
    local_dir = LOCAL_RESULTS / "predictions"
    local_dir.mkdir(exist_ok=True)
    out, _ = exec_cmd(ssh, f"ls {remote_base}/predictions/")
    files = [f.strip() for f in out.splitlines() if f.strip().endswith(".json")]
    print(f"Found {len(files)} prediction JSONs on cluster.")
    pulled = 0
    for f in files:
        local_f = local_dir / f
        if local_f.exists():
            continue
        sftp.get(f"{remote_base}/predictions/{f}", str(local_f))
        pulled += 1
        if pulled % 50 == 0:
            print(f"  pulled {pulled}/{len(files)}")
    sftp.close()
    ssh.close()
    print(f"Downloaded {pulled} new files (total local: {len(list(local_dir.glob('*.json')))}).")


def exec_cmd(ssh, cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    return stdout.read().decode(), stderr.read().decode()


def count_queued(ssh):
    out, _ = exec_cmd(ssh, f"squeue -u {USERNAME} -h | wc -l")
    return int(out.strip())


def wait_for_slots(ssh, target_free=100):
    while True:
        n = count_queued(ssh)
        if n < MAX_QUEUED - target_free:
            return n
        print(f"  Queue has {n} jobs, waiting...", flush=True)
        time.sleep(30)


def main():
    ssh = ssh_connect()
    out, _ = exec_cmd(ssh, f"echo {REMOTE_BASE}")
    remote_base = out.strip()

    # Setup remote dirs. Writing predictions into a NEW folder so we
    # don't overwrite the existing summary-only JSONs in results/jobs/.
    exec_cmd(ssh, f"mkdir -p {remote_base}/src {remote_base}/predictions "
                  f"{remote_base}/slurm {remote_base}/logs")

    # Upload Python sources (run_experiments.py now also packs y_test/y_pred
    # into the result dict so the cluster JSON includes molecule-level data).
    sftp = ssh.open_sftp()
    for fname in ["models.py", "data_loader.py", "run_one_combo.py",
                  "run_experiments.py"]:
        local = LOCAL_SRC / fname
        if local.exists():
            sftp.put(str(local), f"{remote_base}/src/{fname}")
            print(f"Uploaded {fname}")

    # Upload data file
    data_file = LOCAL_ROOT / "data" / "ExCorrDatasetClean.csv"
    sftp.put(str(data_file), f"{remote_base}/src/ExCorrDatasetClean.csv")
    print("Uploaded ExCorrDatasetClean.csv")

    # Models to run
    models = ["RF", "GBR", "MLP", "kNN_Tan", "ChemProp"]

    # Build full task list
    tasks = []
    for alloy in TARGET_ALLOYS:
        for fold in range(K_FOLD):
            for setting in SETTINGS:
                for model in models:
                    tasks.append((alloy, fold, setting, model))

    print(f"\nTotal tasks: {len(tasks)}")

    # Sort: classical first, ChemProp last (so quick wins fill quickly)
    model_order = {"kNN_Tan": 0, "RF": 1, "MLP": 2, "GBR": 3, "ChemProp": 4}
    setting_order = {"exact": 0, "close_filt": 1, "close_unfilt": 2,
                     "far_filt": 3, "far_unfilt": 4}
    tasks.sort(key=lambda t: (model_order[t[3]], setting_order[t[2]], t[0], t[1]))

    submitted = 0
    skipped = 0
    for i, (alloy, fold, setting, model) in enumerate(tasks):
        if i % BATCH_SIZE == 0 and i > 0:
            wait_for_slots(ssh)

        key = f"{alloy}_f{fold}_{setting}_{model}"
        result_file = f"{remote_base}/predictions/{key}.json"

        # Skip if result exists
        out, _ = exec_cmd(ssh, f"test -f {result_file} && echo exists || echo no")
        if out.strip() == "exists":
            skipped += 1
            continue

        time_limit = TIME_MAP.get((model, setting), "02:00:00")
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name=tl_{key[:20]}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --time={time_limit}
#SBATCH --output={remote_base}/logs/{key}.out
#SBATCH --error={remote_base}/logs/{key}.err

cd {remote_base}/src
{REMOTE_PYTHON} -u run_one_combo.py \
    --alloy {alloy} --fold {fold} --setting {setting} --model {model} \
    --out {result_file}
"""
        slurm_path = f"{remote_base}/slurm/{key}.slurm"
        with sftp.open(slurm_path, "w") as f:
            f.write(slurm_script)

        out, err = exec_cmd(ssh, f"sbatch {slurm_path}")
        if "Submitted" in out:
            submitted += 1
            if submitted % 25 == 0 or submitted == 1:
                print(f"  [{submitted}] {key} -> {out.strip().split()[-1]}")
        else:
            print(f"  FAIL {key}: {out} {err}")

    print(f"\nSubmitted {submitted}, skipped {skipped} existing.")
    sftp.close()
    ssh.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download_predictions()
    else:
        main()
