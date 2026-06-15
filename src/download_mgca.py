"""Download MgCa contrast results from cluster."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paramiko, json
import pandas as pd
from pathlib import Path

LOCAL = Path(__file__).resolve().parents[1] / "results" / "mgca_contrast"
LOCAL.mkdir(parents=True, exist_ok=True)

HOSTNAME = "hpclogin.rz.tuhh.de"
USERNAME = os.environ.get("TL_CLUSTER_USERNAME", "your-cluster-username")
PASSWORD = os.environ.get("TL_CLUSTER_PASSWORD", "")  # set via env var


def connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOSTNAME, username=USERNAME, password=PASSWORD)
    return ssh


ssh = connect()
stdin, stdout, _ = ssh.exec_command("echo ~/tl_crossalloy_mgca/results")
remote = stdout.read().decode().strip()
sftp = ssh.open_sftp()
files = [f for f in sftp.listdir(remote) if f.endswith(".json")]
existing = {p.name for p in LOCAL.glob("*.json")}
new_files = [f for f in files if f not in existing]

print(f"Remote: {len(files)}, local: {len(existing)}, new: {len(new_files)}")
for i, f in enumerate(new_files):
    try:
        sftp.get(f"{remote}/{f}", str(LOCAL / f))
    except Exception:
        time.sleep(2)
        try: sftp.close(); ssh.close()
        except: pass
        ssh = connect()
        sftp = ssh.open_sftp()
        try: sftp.get(f"{remote}/{f}", str(LOCAL / f))
        except Exception: pass
    if (i+1) % 25 == 0:
        print(f"  {i+1}/{len(new_files)}")

sftp.close()
ssh.close()

# Aggregate MgCa
rows = []
for f in LOCAL.glob("*.json"):
    try:
        with open(f) as fh:
            rows.append(json.load(fh))
    except Exception:
        pass
df = pd.DataFrame(rows)
out = LOCAL.parent / "mgca_contrast.csv"
df.to_csv(out, index=False)
print(f"\nAggregated {len(df)} MgCa rows to {out}")
print("Coverage:")
print(df.groupby(["model", "setting"]).size().unstack(fill_value=0).to_string())
