#!/usr/bin/env python3
"""
combined_run_status.py

 • Reads a launcher-file ( produced by your bash wrapper ) and determines the
   hashed output-directory for every planned run (convert_argparse_to_hash_path)
 • Builds a *live* map <hashed_dir -> {state,progress}> by
     - scanning `squeue -u $USER -l`
     - finding the newest log for every running job
     - parsing the first ~30 lines for a line that starts with the banner
       “Creating output directory hash using params: { … }” and recreating the
       same hashed_dir with `generate_hashed_dir_name`
 • Finally prints a table, combining the “static” checkpoint info from
   get_run_status.py with the live info when available
"""

from __future__ import annotations
import argparse, ast, json, os, re, subprocess, textwrap
from pathlib import Path
import torch
import time
import datetime
from typing import List
# ---------------------------------------------------------------------------
# your helper modules
from efficient_tokenization.utils              import get_latest_checkpoint
from efficient_tokenization.benchmarking_utils import (
        parse_args_from_file,
        convert_argparse_to_hash_path,          # <‑‑ hashed dir for launcher‑args
        parse_args_from_line
)


LOG_DIR   = Path("log")        # adjust if different
OUTDIR_RE = re.compile(r"Output directory:\s*(\S+)")

TAIL_LINES = 25                # how many last lines to scan for “cancelled” footer
PROGRESS_RE = re.compile(r"\s*(\d+)%\|.*?\|\s*(\d+)\s*/\s*(\d+)")
# --- regexes ---------------------------------------------------------------
# tqdm line with a loss metric, e.g.
#  20%|██        | 1000/5000 [6:03:32<8:18:03,  7.47s/it, mixed_loss=0.604]
TQDM_LOSS_RE = re.compile(
    r"""
    ^\s*                 # possible leading spaces
    (?P<pct>\d+)%\|      # percentage
    .*?                  # bar
    (?P<step>\d+)/(?P<tot>\d+)   # step / total
    \s*\[(?P<elapsed>[^<]+)<(?P<eta>[^,]+),   # elapsed and ETA
    .*?                 # speed
    (?P<metric>\w*_loss)=(?P<loss>[\d\.eE+-]+) # actual loss
    """,
    re.VERBOSE,
)
PREEMPT_RE = re.compile(
    r"JOB\s+(?P<jobid>\d+)\s+ON\s+(?P<node>\S+)\s+CANCELLED\s+AT\s+(?P<ts>[0-9T:\-]+)"
)

GPU_TYPE_RE = re.compile(
    r"accelerator distributed type \S+, num_processes \d+ on (?P<gpu>.+)"
)

array_re = re.compile(r"(?P<base>\d+)_\[(?P<inner>[^\]]+)\]")

def has_safetensors_files(directory):
    """Check if any files in the directory end with '.safetensors'."""
    try:
        files = os.listdir(directory)
        for file in files:
            if file.endswith('.safetensors'):
                return True
        return False
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
        return False


def get_run_status(output_dir: str) -> str:
    if os.path.exists(os.path.join(output_dir, "final_model")):
        directory_path = os.path.join(output_dir, "final_model")
        if has_safetensors_files(directory_path):
            return "Final model exists"
        else:
            print(f"No files ending with '.safetensors' found in the directory {directory_path}.")

    if os.path.exists(os.path.join(output_dir, "checkpoints")):
        latest_checkpoint, _ = get_latest_checkpoint(output_dir, recursively_check=False)
        if latest_checkpoint is not None:
            state_dict_path = os.path.join(latest_checkpoint, "checkpoint_meta.pt")
            if not os.path.exists(state_dict_path):
                return "Training corrupted"
            train_info = torch.load(state_dict_path)
            return f"Step: {train_info['update_step']}"
    return "NONE"


def expand_job_id(token: str) -> list[str]:
    """
    Expand a SLURM array‑notation jobid like 4127721_[1-2,4] into
    ['4127721_1', '4127721_2', '4127721_4'].

    Returns [token] unchanged if no brackets are present.
    """
    m = array_re.fullmatch(token)
    if not m:
        return [token]

    base = m.group("base")
    inner = m.group("inner")               # e.g. "1-2,4,6-8"

    parts = []
    for chunk in inner.split(","):
        if "-" in chunk:                   # range e.g. 1‑3
            lo, hi = map(int, chunk.split("-"))
            parts.extend([f"{base}_{i}" for i in range(lo, hi + 1)])
        else:                              # single index
            parts.append(f"{base}_{int(chunk)}")

    return parts

def run(cmd: list[str]) -> str:
    """Run a shell command and return *stdout* (text)."""
    return subprocess.check_output(cmd, text=True)

def parse_myq() -> list[dict]:
    """Return list of {jobid, state} from `squeue -u $USER -l`."""
    user_name = os.environ.get("USER", "astein0")
    out = run(["squeue", "-u", user_name, "-l"])

    jobs = []
    for ln in out.splitlines():
        cols = ln.split()
        if not cols:
            continue                      # blank line
        if cols[0] == "JOBID":            # header row
            continue
        if not cols[0][0].isdigit():      # date/banner or malformed row  ← NEW
            continue
        if cols[2] == "zsh":              # helper shell job
            continue

        for jid in expand_job_id(cols[0]):    # handle array notation
            jobs.append({"jobid": jid, "state_slurm": cols[4], "node": cols[8], "partition": cols[1]})

    return jobs

def newest_log(jobid: str) -> Path | None:
    """Return the latest log file whose name contains *jobid* (or None)."""
    cands = [p for p in LOG_DIR.glob(f"*{jobid}*.log")]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)   # newest by mtime

def parse_log(log_path: str | Path):
    """
    Inspect *log_path* and return:

    #   status        : "running" | "queued" | "no_log"
      progress      : dict or None      ─ latest tqdm-loss line
      cancelled     : bool              ─ cancellation ever occurred
      cancel_reason : str | None        ─ reason on last line, if queued
    """
    out = {
        # "status": "running",
        "progress": None,
        "cancelled": False,
        "cancel_reason": None,
        "node"     : None,   # <‑ new
        # "ts"       : None,   # <‑ new
        "output_dir": None,
        # "jid"      : None,
    }

    log_path = Path(log_path)
    if not log_path.is_file():
        # out["status"] = "no_log"
        return out

    with open(log_path, "r", errors="ignore") as f:
        lines = f.read().splitlines()

    # Scan all lines for GPU type info
    for line in reversed(lines):
        m = GPU_TYPE_RE.search(line)
        if m:
            out["gpu"] = m.group("gpu").split()[-1]
            break

    if not lines:                       # empty log
        return out

    # Add last updated timestamp
    try:
        mtime = log_path.stat().st_mtime
        out["last_updated"] = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        out["last_updated"] = "-"


    for line in lines[:40]:
        m = OUTDIR_RE.search(line)
        if m:
            out["output_dir"] = m.group(1)
            break

    # ---------------- progress (scan from the bottom)
    for line in reversed(lines):
        m = TQDM_LOSS_RE.match(line)
        if m:
            out["progress"] = {
                "pct":   int(m["pct"]),
                "step":  int(m["step"]),
                "total": int(m["tot"]),
                "metric": m["metric"],
                "loss": float(m["loss"]),
                "eta":  m["eta"].strip(),
            }
            break

    last_non_blank = next((l for l in reversed(lines) if l.strip()), "")
    m_cancel = PREEMPT_RE.search(last_non_blank)
    if m_cancel:
        # out["status"] = "queued"
        out["cancelled"] = True
        out["cancel_reason"] = last_non_blank.strip()
        out["node"] = m_cancel.group("node")
        out["ts"] = m_cancel.group("ts")
        out["jid"] = m_cancel.group("jobid")
    else:
        # cancellation existed but not last -> job restarted
        if any(PREEMPT_RE.search(l) for l in lines):
            out["cancelled"] = True

    return out

# --------------------------------------------------------------------------- helpers
def run_cmd(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True)

# --------------------------- merge_state -----------------------------------
def merge_state(job_slurm_state: str,
                live_info: dict | None) -> tuple[str, str]:
    """
    Decide (state, progress_text).

    • Slurm state (RUNNING / PENDING / COMPLETING …) always wins.
    • 'cancelled' flag from log only refines a PENDING job.
    • progress comes from live_info if present.
    """
    if live_info:
        # ----------- STATE -----------
        slurm_lower = job_slurm_state.lower()
        if slurm_lower == "running":
            state = f"RUNNING" #({live_info['node']})"                       # nothing overrides RUNNING
        elif slurm_lower == "pending":
            state = "PENDING" + (" (preempt)" if live_info.get("cancelled") else "")
        elif slurm_lower == "completing":
            state = "COMPLETING" + (" (preempt)" if live_info.get("cancelled") else "")
        elif slurm_lower == "finished":
            state = "FINISHED"
        elif slurm_lower == "queued":
            state = "QUEUED"
        else:                                       # COMPLETING, etc.
            state = slurm_lower
        # ----------- PROGRESS --------
        p = live_info.get("progress", None)
        if p:
            progress = (f"{p['pct']:>3d}% "
                        f"{p['step']}/{p['total']}"
                        # f"{p['metric']}={p['loss']:.3f}"
                        )
        else:
            progress = "-"
    else:
        # we have no live info: just echo Slurm state
        state     = job_slurm_state.lower()
        progress  = "-"

    return state, progress

def find_jobs_from_jobid_index(job_ids, log_dir="/nfshomes/astein0/.launch/") -> dict[str, List[str]]:
    job_ids_to_sub_map = {}
    for jobid in job_ids:
        parent_id = jobid.split("_")[0]
        sub_ids = jobid.split("_")[-1]
        if parent_id not in job_ids_to_sub_map:
            job_ids_to_sub_map[parent_id] = []
        job_ids_to_sub_map[parent_id].append(sub_ids)

    log_dir = Path(log_dir)
    if not log_dir.exists():
        print(f"Log directory {log_dir} not found.")
        return {}

    job_index_file = log_dir / "jobid_index.txt"
    if not job_index_file.exists():
        print(f"Job index file {job_index_file} not found.")
        return {}

    job_hashs = {}
    with open(job_index_file, "r") as f:
        line_read = 0
        for line in reversed(f.readlines()):
            if line_read > 100: # limit the number of lines to read
                break
            if len(job_hashs) == len(job_ids_to_sub_map):  # stop if we have all the job ids
                break
            jobid, authkey, qos, idx = line.strip().split(",")
            if jobid in job_ids_to_sub_map:
                job_hashs[jobid] = authkey
            line_read += 1
    
    all_job_lines = {}          
    if len(job_hashs) != len(job_ids_to_sub_map):
        missing_job_ids = set(job_ids_to_sub_map.keys()) - set(job_hashs.keys())
        # print(f"Could not find job hash for {missing_job_ids}")
        for jobid in missing_job_ids:
            for job_no in job_ids_to_sub_map[jobid]:
                full_job_id = f"{jobid}_{job_no}"
                all_job_lines[full_job_id] = None

    for jobid, hash_id in job_hashs.items():
        job_file_name = f"job_list_{hash_id}.temp.sh"
        job_file_path = log_dir / job_file_name
        if not job_file_path.exists():
            print(f"Job file {job_file_path} not found.")
            continue
        job_no = 1
        with open(job_file_path, "r") as f:
            for line in f:
                if str(job_no) in job_ids_to_sub_map[jobid]:
                    full_job_id = f"{jobid}_{job_no}"
                    all_job_lines[full_job_id] = line
                job_no += 1

    return all_job_lines




def live_status_map() -> tuple[dict[str,dict], set[str]]:
    mapping: dict[str,dict] = {}

    missing_job_ids = set()
    seen_logs = set()
    job_map = {}
    for job in parse_myq():
        job_map[job["jobid"]] = job

        jid   = job["jobid"]
        state = job["state_slurm"]
        partition = job["partition"]
        node = None
        if state != "PENDING":
            node = job["node"]

        log = newest_log(jid)
        if log is None:
            missing_job_ids.add(jid)
            continue

        seen_logs.add(log.name)

        log_info = parse_log(log) 

        hashed_dir = log_info["output_dir"]
        # TODO return the live info for things without logs to get pending run info
        if not hashed_dir:
            continue
        
        log_info["state"] = state.lower()
        log_info["log"] = f"{LOG_DIR.name}/{log.name}"
        log_info["jid"] = jid
        log_info["node"] = node
        log_info["partition"] = partition

        mapping[hashed_dir] = log_info
        
    # ------------------------------------------------------------------
    # Phase 2 – picks up jobs that have not started yet, gets them from launch files
    #           Will only be jobs that are "missing" (meaning that they dont have a log yet)
    # ------------------------------------------------------------------

    missing_job_lines = find_jobs_from_jobid_index(list(missing_job_ids))
    for jobid, line in missing_job_lines.items():
        if line is None:
            pass
            # print(f"Could not find job hash for {jobid}")
        else:
            file_args, pre_args = parse_args_from_line(line, print_warnings=False)
            if file_args is None:
                continue

            hashed_dir = convert_argparse_to_hash_path(
                file_args, accelerate_args=pre_args, output_folder=args.output_root
            )
            log_info = {}
            job_info = job_map[jobid]
            log_info["state"] = "QUEUED"
            log_info["jid"] = jobid
            log_info["partition"] = job_info["partition"]
            log_info["progress"] = None

        mapping[hashed_dir] = log_info
    # ------------------------------------------------------------------
    # Phase 3 – pick up *recent* logs that don’t belong to any running / pending
    #           Slurm job (e.g. pre‑empted or crashed runs).
    # ------------------------------------------------------------------
    newest_logs = sorted(
        LOG_DIR.glob("*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )[:100]

    for log_path in newest_logs:
        # skip if we already have it from an active Slurm id
        if log_path.name in seen_logs:
            # print(f"Skipping {log_path.name} because it is an active job (already has info stored)")
            continue

        info = parse_log(log_path)
        hashed_dir = info["output_dir"]
        if hashed_dir in mapping:
            # print(f"Skipping {hashed_dir} because a later run with same output directory has already been seen")
            continue

        if not hashed_dir:
            continue
        
        # log_info["state"] = state.lower()
        info["log"] = f"{LOG_DIR.name}/{log_path.name}"
        info["jid"] = info.get("jid")
        info["node"] = info.get("node")
        info["state"] = "-" # derive a synthetic state

        mapping[hashed_dir] = info

    return mapping


def fmt_progress(p: dict|None) -> str:
    if not p:
        return "-"
    return (f"{p['pct']:>3d}%  "
            f"{p['step']}/{p['total']}  "
            # f"{p['metric']}={p['loss']:.3f}"
            )

# --------------------------------------------------------------------------- main
def main(args: argparse.Namespace) -> None:


    # Part A – live map
    live = live_status_map()

    # Part B – go through every planned run
    file_args_list, pre_args_list, _ = parse_args_from_file(args.launcher_file)

    rows = []
    jobs_by_partition = {}
    for file_args, pre_args in zip(file_args_list, pre_args_list):
        hashed_dir = convert_argparse_to_hash_path(
            file_args, accelerate_args=pre_args, output_folder=args.output_root
        )

        # 1) static checkpoint info
        static_status = get_run_status(hashed_dir)
        live_info = live.get(hashed_dir, None)

        if live_info is None:
            live_info = {}

        if "--num_processes" in pre_args:
            live_info["num_processes"] = pre_args['--num_processes']


        if static_status == "Final model exists":
            if live_info.get("state", None) == "running":
                slurm_state = "COMPLETING"
            else:
                slurm_state = "FINISHED"
        elif live_info:
            slurm_state = live_info.get("state", "-")
        else:
            slurm_state = "-"

        if args.incomplete_only and slurm_state == "FINISHED":
            continue

        # 2) live overlay (if any)
        state, progress = merge_state(
            slurm_state,
            live_info
        )
        if live_info is not None:
            live_info["slurm_state"] = slurm_state

        if live_info is not None and live_info.get('partition') is not None:
            if live_info['partition'] not in jobs_by_partition:
                jobs_by_partition[live_info['partition']] = []
            jobs_by_partition[live_info['partition']].append(live_info)

        logfile = live_info.get("log", "-") if live_info else "-"
        info_list = []
        eta = "-"
        if live_info:
            if live_info.get('node') is not None:
                info_list.append(f"{live_info['node']:5s}")
            if "num_processes" in live_info:
                info_list.append(f"#gpus={live_info['num_processes']:2s}")
            if 'ts' in live_info and live_info['ts'] is not None:
                info_list.append(f"update={live_info['ts']:10s}")
            if "gpu" in live_info:
                info_list.append(f"gpu={live_info['gpu']}")
            if live_info.get('progress', None) is not None and slurm_state == "running":
                eta = live_info['progress']['eta']
            if live_info.get('partition') is not None:
                info_list.append(f"({live_info['partition']})")
            if logfile == "-" and "jid" in live_info:
                info_list.append(f"jid={live_info['jid']}")
        info_str = " ".join(info_list)
        last_updated = live_info.get("last_updated", "-") if live_info else "-"
        rows.append((hashed_dir, static_status, state, progress, logfile, last_updated, eta, info_str))

    # nice output
    if len(rows) == 0:
        print("No runs found")
        return
    hdr   = ["RUN_DIR","CHECKPOINT","SLURM","PROGRESS","LOG","UPDATED", "ETA", "INFO"]
    col_w = [max(max(len(str(r[i])), len(hdr[i])) for r in rows) for i in range(len(hdr))]
    output_str = f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    output_str += " | ".join(h.ljust(col_w[i]) for i, h in enumerate(hdr)) + "\n"
    output_str += "-+-".join("-"*w for w in col_w) + "\n"
    for r in rows:
        output_str += " | ".join(str(r[i]).ljust(col_w[i]) for i in range(len(hdr))) + "\n"

    if len(jobs_by_partition) > 0:
        output_str += "\nPending jobs by Partition:\n"
        for partition, jobs in jobs_by_partition.items():
            gpus = 0
            count = 0
            queued_gpus = 0
            queued_count = 0
            for job_info in jobs:
                if not job_info.get('slurm_state').lower() in ["running", "pending", "queued", "completing"]:
                    continue

                if job_info.get('slurm_state').lower() == "queued":
                    queued_gpus += int(job_info['num_processes'])
                    queued_count += 1
                else:
                    gpus += int(job_info['num_processes'])
                    count += 1

            part = f"{partition}:"
            output_str += f"{part:10s} {count:1d} jobs ({queued_count:1d} queued), {gpus:2d} gpus ({queued_gpus:2d} queued)\n"
    return output_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("launcher_file",
                        help="file that contains the sbatch / accelerate lines")
    parser.add_argument("-o","--output-root", default="output",
                        help="root directory that holds run folders")
    parser.add_argument("--incomplete-only", action="store_true",
                        help="only show incomplete runs")
    parser.add_argument("--watch", type=int, metavar="SECONDS",
                    help="re-run and refresh every N seconds")
    args = parser.parse_args()
    if args.watch:
        try:
            while True:
                table = main(args)
                os.system("clear")
                print(table)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        table = main(args)
        print(table)