"""
Roda um comando e o encerra se a RAM disponível da máquina cair abaixo de um limite, ou se outros processos
passarem a usar a GPU, para não afetar outros processos (por exemplo, os treinos do quati-ia). Só começa quando há
RAM suficiente e a GPU está praticamente livre de outros processos.

Uso: python rodar_com_vigia.py [--min-inicio-gb 12] [--limite-gb 8] [--log arquivo] -- comando args...
Saída: código do comando; 3 se o vigia encerrou.
"""
import argparse
import os
import signal
import subprocess
import sys
import time

import psutil

ap = argparse.ArgumentParser()
ap.add_argument("--min-inicio-gb", type=float, default=12.0)
ap.add_argument("--limite-gb", type=float, default=8.0)
ap.add_argument("--log", default=None)
ap.add_argument("--gpu-outros-max-gb", type=float, default=2.0,
                help="só começa (e continua) se os outros processos usarem no máximo isto de GPU")
ap.add_argument("--gpu-livre-min-gb", type=float, default=0.5, help="encerra se a GPU ficar com menos que isto livre")
ap.add_argument("cmd", nargs=argparse.REMAINDER)
a = ap.parse_args()
cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd


def disp_gb():
    return psutil.virtual_memory().available / 2**30


def gpu_estado(meus_pids):
    """(GB de GPU usados por outros processos, GB livres na GPU)."""
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10).stdout
        livre = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10).stdout
        outros = sum(float(m) for p, m in (l.split(",") for l in apps.strip().splitlines() if l.strip())
                     if int(p) not in meus_pids) / 1024
        return outros, float(livre.strip().splitlines()[0]) / 1024
    except Exception:
        return 0.0, 99.0


def meus_pids(pid):
    try:
        return {pid} | {c.pid for c in psutil.Process(pid).children(recursive=True)}
    except psutil.Error:
        return {pid}


while disp_gb() < a.min_inicio_gb or gpu_estado(set())[0] > a.gpu_outros_max_gb:
    time.sleep(30)

out = open(a.log, "a") if a.log else None
proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT if out else None, start_new_session=True)
pico = 0.0
while proc.poll() is None:
    try:
        rss = sum(p.memory_info().rss for p in [psutil.Process(proc.pid)] + psutil.Process(proc.pid).children(recursive=True))
        pico = max(pico, rss / 2**30)
    except psutil.Error:
        pass
    motivo = None
    if disp_gb() < a.limite_gb:
        motivo = f"RAM disponível < {a.limite_gb} GB"
    elif int(time.time()) % 3 == 0:
        outros, livre = gpu_estado(meus_pids(proc.pid))
        if outros > a.gpu_outros_max_gb:
            motivo = f"outros processos usando {outros:.1f} GB de GPU"
        elif livre < a.gpu_livre_min_gb:
            motivo = f"GPU com só {livre:.2f} GB livres"
    if motivo:
        os.killpg(proc.pid, signal.SIGTERM)
        time.sleep(10)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        print(f"[vigia] {motivo}: comando encerrado (pico de RSS {pico:.1f} GB)", flush=True)
        sys.exit(3)
    time.sleep(1)
print(f"[vigia] fim, código {proc.returncode}, pico de RSS {pico:.1f} GB", flush=True)
sys.exit(proc.returncode)
