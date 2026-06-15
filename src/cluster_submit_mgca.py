"""Submit MgCa contrast experiment to cluster.
   4 folds * 5 settings * 5 models = 100 jobs.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paramiko
from pathlib import Path

from data_loader import SETTINGS

LOCAL_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = LOCAL_ROOT / "src"

HOSTNAME = "hpclogin.rz.tuhh.de"
USERNAME = os.environ.get("TL_CLUSTER_USERNAME", "your-cluster-username")
PASSWORD = os.environ.get("TL_CLUSTER_PASSWORD", "")  # set via env var
REMOTE_PYTHON = "~/xtb_env/bin/python"
REMOTE_BASE = "~/tl_crossalloy_mgca"

K_FOLD = 4
MODELS = ["kNN_Tan", "RF", "MLP", "GBR", "ChemProp"]

TIME_MAP = {
    ("RF", "exact"): "00:30:00",      ("RF", "close_unfilt"): "01:00:00",
    ("RF", "close_filt"): "01:00:00", ("RF", "far_unfilt"): "02:00:00",
    ("RF", "far_filt"): "02:00:00",
    ("GBR", "exact"): "00:30:00",     ("GBR", "close_unfilt"): "02:00:00",
    ("GBR", "close_filt"): "02:00:00",("GBR", "far_unfilt"): "06:00:00",
    ("GBR", "far_filt"): "06:00:00",
    ("MLP", "exact"): "00:30:00",     ("MLP", "close_unfilt"): "01:30:00",
    ("MLP", "close_filt"): "01:30:00",("MLP", "far_unfilt"): "04:00:00",
    ("MLP", "far_filt"): "04:00:00",
    ("kNN_Tan", "exact"): "00:15:00", ("kNN_Tan", "close_unfilt"): "00:30:00",
    ("kNN_Tan", "close_filt"): "00:30:00", ("kNN_Tan", "far_unfilt"): "01:00:00",
    ("kNN_Tan", "far_filt"): "01:00:00",
    ("ChemProp", "exact"): "01:00:00",("ChemProp", "close_unfilt"): "04:00:00",
    ("ChemProp", "close_filt"): "04:00:00", ("ChemProp", "far_unfilt"): "10:00:00",
    ("ChemProp", "far_filt"): "10:00:00",
}


def ssh_connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOSTNAME, username=USERNAME, password=PASSWORD)
    return ssh


def exec_cmd(ssh, cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    return stdout.read().decode(), stderr.read().decode()


def main():
    ssh = ssh_connect()
    out, _ = exec_cmd(ssh, f"echo {REMOTE_BASE}")
    rb = out.strip()
    exec_cmd(ssh, f"mkdir -p {rb}/src {rb}/results {rb}/jobs {rb}/slurm {rb}/logs")

    sftp = ssh.open_sftp()
    # Re-upload latest sources (and the wrapper)
    for fname in ["models.py", "data_loader.py", "run_experiments.py",
                  "mgca_run_one.py"]:
        local = LOCAL_SRC / fname
        if local.exists():
            sftp.put(str(local), f"{rb}/src/{fname}")
            print(f"Uploaded {fname}")

    data_file = LOCAL_ROOT / "data" / "ExCorrDatasetClean.csv"
    sftp.put(str(data_file), f"{rb}/src/ExCorrDatasetClean.csv")

    submitted, skipped = 0, 0
    for fold in range(K_FOLD):
        for setting in SETTINGS:
            for model in MODELS:
                key = f"MgCa_f{fold}_{setting}_{model}"
                rfile = f"{rb}/results/{key}.json"
                out, _ = exec_cmd(ssh, f"test -f {rfile} && echo y || echo n")
                if out.strip() == "y":
                    skipped += 1
                    continue
                tlim = TIME_MAP.get((model, setting), "02:00:00")
                slurm = f"""#!/bin/bash
#SBATCH --job-name=mgca_{key[:18]}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --time={tlim}
#SBATCH --output={rb}/logs/{key}.out
#SBATCH --error={rb}/logs/{key}.err

cd {rb}/src
{REMOTE_PYTHON} -u mgca_run_one.py --fold {fold} --setting {setting} --model {model} --out {rfile}
"""
                spath = f"{rb}/slurm/{key}.slurm"
                with sftp.open(spath, "w") as f:
                    f.write(slurm)
                out, err = exec_cmd(ssh, f"sbatch {spath}")
                if "Submitted" in out:
                    submitted += 1
                    if submitted % 20 == 0 or submitted == 1:
                        print(f"  [{submitted}] {key} -> {out.strip().split()[-1]}")
                else:
                    print(f"  FAIL {key}: {out.strip()} {err.strip()}")

    print(f"\nSubmitted {submitted}, skipped {skipped} existing.")
    sftp.close(); ssh.close()


if __name__ == "__main__":
    main()
