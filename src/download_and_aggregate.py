"""Download per-job JSON results from cluster and aggregate to CSV."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paramiko
import pandas as pd
from pathlib import Path

LOCAL_RESULTS = Path(__file__).resolve().parents[1] / "results"
LOCAL_RESULTS.mkdir(exist_ok=True)
JSON_DIR = LOCAL_RESULTS / "jobs"
JSON_DIR.mkdir(exist_ok=True)

HOSTNAME = "hpclogin.rz.tuhh.de"
USERNAME = os.environ.get("TL_CLUSTER_USERNAME", "your-cluster-username")
PASSWORD = os.environ.get("TL_CLUSTER_PASSWORD", "")  # set via env var
REMOTE_BASE = "~/tl_crossalloy"


def connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOSTNAME, username=USERNAME, password=PASSWORD)
    return ssh


def download_jobs():
    ssh = connect()
    sftp = ssh.open_sftp()
    stdin, stdout, _ = ssh.exec_command(f"echo {REMOTE_BASE}")
    rb = stdout.read().decode().strip()

    remote_results = f"{rb}/results"
    files = sftp.listdir(remote_results)
    json_files = [f for f in files if f.endswith(".json")]
    existing = {p.name for p in JSON_DIR.glob("*.json")}
    new_files = [f for f in json_files if f not in existing]

    print(f"Remote: {len(json_files)}  local: {len(existing)}  new: {len(new_files)}")
    for i, f in enumerate(new_files):
        try:
            sftp.get(f"{remote_results}/{f}", str(JSON_DIR / f))
        except Exception as e:
            try: sftp.close(); ssh.close()
            except: pass
            time.sleep(2)
            ssh = connect()
            sftp = ssh.open_sftp()
            try: sftp.get(f"{remote_results}/{f}", str(JSON_DIR / f))
            except Exception: pass
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(new_files)}")

    # queue status
    stdin, stdout, _ = ssh.exec_command(
        f"squeue -u {USERNAME} -o '%T' | sort | uniq -c | sort -rn")
    print(f"\nCluster queue:\n{stdout.read().decode()}")
    sftp.close()
    ssh.close()


def aggregate():
    rows = []
    for f in JSON_DIR.glob("*.json"):
        try:
            with open(f) as fh:
                rows.append(json.load(fh))
        except Exception:
            pass
    if not rows:
        print("No results to aggregate.")
        return
    df = pd.DataFrame(rows)
    out = LOCAL_RESULTS / "experiment_results.csv"
    df.to_csv(out, index=False)
    print(f"\nAggregated {len(df)} rows to {out}")

    # Quick summary
    print("\nCoverage:")
    cov = df.groupby(["model", "setting"]).size().unstack(fill_value=0)
    print(cov.to_string())


if __name__ == "__main__":
    download_jobs()
    aggregate()
