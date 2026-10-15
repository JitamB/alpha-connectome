"""Run train/train_gru.py on a Colab GPU via the colab CLI, mirroring checkpoints back to runs/<run>/.

  python tools/colab_train.py RUN [--session wunder] [--gpu T4] [--keep-session] -- <train_gru.py args>

Cycle: ensure session -> upload code (+ local last.pt to resume) -> background job on the VM that
fetches train.parquet (if missing) and trains -> poll every few minutes, downloading log/checkpoints after
each epoch -> stop the session when training ends. If the VM is reclaimed, it is re-created and
training resumes from the last mirrored checkpoint.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/content/wunder'
CODE = ['src/data.py', 'src/evaluate.py', 'src/features.py', 'train/train_gru.py', 'train/gpu_features.py',
        'train/eval_torch.py',
        'tools/fetch_data.py', 'splits/holdout.json', 'wnn_connectome_starterpack/utils.py',
        'models/mask_model.py', 'models/mask_mlp.pt']
CKPT_FILES = ['last.pt', 'best.pt', 'ema.pt', 'history.json', 'config.json']


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def colab(*args, code=None, timeout=None, check=True, wait=600, retries=3):
    """Run a colab CLI command; a hung call (seen in practice) is killed after `wait` s and retried."""
    cmd = ['colab', *args]
    if timeout:
        cmd += ['--timeout', str(timeout)]
        wait = max(wait, timeout + 120)
    for attempt in range(1, retries + 1):
        try:
            r = subprocess.run(cmd, input=code, capture_output=True, text=True, timeout=wait)
            if r.returncode != 0 and any(e in r.stderr for e in ('ReadTimeout', 'ConnectionError', 'Connection was lost')):
                log(f'colab {args[0]} transient failure (attempt {attempt}/{retries})')  # seen right after `colab new`
                time.sleep(30)
                continue
            break
        except subprocess.TimeoutExpired:
            log(f'colab {args[0]} hung for {wait}s (attempt {attempt}/{retries})')
    else:
        raise RuntimeError(f'colab {" ".join(args)} hung {retries} times')
    if check and r.returncode != 0:
        raise RuntimeError(f'colab {" ".join(args)} failed: {r.stderr.strip()[-500:]}')
    return r.stdout


def remote_py(session, code, timeout=60):
    return colab('exec', '-s', session, code=code, timeout=timeout)


def session_alive(session):
    r = subprocess.run(['colab', 'status', '-s', session], capture_output=True, text=True)
    return r.returncode == 0 and 'Hardware' in r.stdout


def setup(session, gpu, run, local_run: Path):
    if not session_alive(session):
        log(f'creating session {session} ({gpu})')
        colab('new', '-s', session, '--gpu', gpu)
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / 'code.tar.gz'
        with tarfile.open(bundle, 'w:gz') as tar:
            for f in CODE:
                tar.add(ROOT / f, arcname=f)
        colab('upload', '-s', session, str(bundle), '/content/code.tar.gz')
    remote_py(session, f"import tarfile, os; os.makedirs('{REMOTE}', exist_ok=True); "
                       f"tarfile.open('/content/code.tar.gz').extractall('{REMOTE}'); print('code ok')")
    colab('install', '-s', session, 'onnx', 'onnxruntime')
    for f in ['last.pt', 'config.json', 'history.json']:
        if (local_run / f).exists():
            remote_py(session, f"import os; os.makedirs('{REMOTE}/runs/{run}', exist_ok=True)")
            colab('upload', '-s', session, str(local_run / f), f'{REMOTE}/runs/{run}/{f}')
            log(f'uploaded {f} for resume')


def launch(session, run, train_args):
    """Background job on the VM: fetch train/valid parquet if missing (CRC-checked), train, then evaluate V1/V2/V3."""
    import shlex
    kit = f'{REMOTE}/wnn_connectome_starterpack'
    fetch = ''.join(f'if [ "$(stat -c %s {kit}/datasets/{f} 2>/dev/null)" != "{size}" ]; then '
                    f'python -u tools/fetch_data.py {kit} datasets/{f} || exit 1; fi; '
                    for f, size in [('train.parquet', 29240219011), ('valid.parquet', 5615509128)])
    train = shlex.join(['python', '-u', 'train/train_gru.py', '--out', f'runs/{run}', '--device', 'cuda', *train_args])
    evaluate = shlex.join(['python', '-u', 'train/eval_torch.py', f'runs/{run}', '--name', run, '--batch', '32'])
    evaluate_ema = shlex.join(['python', '-u', 'train/eval_torch.py', f'runs/{run}', '--name', f'{run}_ema', '--weights', 'ema.pt',
                               '--batch', '32'])
    script = f'{fetch}{train} && {evaluate} && if [ -f runs/{run}/ema.pt ]; then {evaluate_ema}; fi'
    if '--dist-weight' in train_args:  # also score the decoded distributional head and the average of both heads
        for out in ('dist', 'blend'):
            script += ' && ' + shlex.join(['python', '-u', 'train/eval_torch.py', f'runs/{run}', '--name', f'{run}_{out}',
                                           '--output', out, '--batch', '32'])
    remote_py(session, f"""
import subprocess, os
pidfile = '{REMOTE}/runs/{run}.pid'
if os.path.exists(pidfile) and os.path.exists('/proc/' + open(pidfile).read().strip()):
    raise SystemExit('job already running; not launching a second copy')
os.makedirs('{REMOTE}/runs', exist_ok=True)
logf = open('{REMOTE}/runs/{run}.log', 'a')
proc = subprocess.Popen(['bash', '-c', {script!r}], cwd='{REMOTE}', stdout=logf, stderr=subprocess.STDOUT,
                        start_new_session=True)
open('{REMOTE}/runs/{run}.pid', 'w').write(str(proc.pid))
print('launched', proc.pid)
""")
    log(f'launched fetch+train: {train_args}')


def poll(session, run):
    out = remote_py(session, f"""
import os, json
pid = int(open('{REMOTE}/runs/{run}.pid').read())
alive = os.path.exists(f'/proc/{{pid}}') and open(f'/proc/{{pid}}/stat').read().split()[2] != 'Z'
h = '{REMOTE}/runs/{run}/history.json'
epochs = len(json.load(open(h))) if os.path.exists(h) else 0
tail = open('{REMOTE}/runs/{run}.log').read().strip().splitlines()[-1:] or ['']
print(json.dumps({{'alive': alive, 'epochs': epochs, 'tail': tail[0][-200:]}}))
""")
    return json.loads(out.strip().splitlines()[-1])


def mirror(session, run, local_run: Path):
    local_run.mkdir(parents=True, exist_ok=True)
    colab('download', '-s', session, f'{REMOTE}/runs/{run}.log', str(local_run.parent / f'{run}.colab.log'), check=False)
    for f in CKPT_FILES:
        colab('download', '-s', session, f'{REMOTE}/runs/{run}/{f}', str(local_run / f), check=False)
    (ROOT / 'results').mkdir(exist_ok=True)
    for name in (run, f'{run}_ema', f'{run}_dist', f'{run}_blend'):
        for ext in ('npz', 'json'):
            colab('download', '-s', session, f'{REMOTE}/results/{name}.{ext}', str(ROOT / 'results' / f'{name}.{ext}'), check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--session', default='wunder')
    ap.add_argument('--gpu', default='T4')
    ap.add_argument('--poll-minutes', type=float, default=5)
    ap.add_argument('--max-restarts', type=int, default=3)
    ap.add_argument('--keep-session', action='store_true', help='leave the VM running (data stays) for a follow-up run')
    argv = sys.argv[1:]
    split = argv.index('--') if '--' in argv else len(argv)
    args = ap.parse_args(argv[:split])
    train_args = argv[split + 1:]
    local_run = ROOT / 'runs' / args.run
    restarts, seen_epochs = 0, -1
    setup(args.session, args.gpu, args.run, local_run)
    launch(args.session, args.run, train_args)
    while True:
        time.sleep(args.poll_minutes * 60)
        try:
            state = poll(args.session, args.run)
        except Exception as e:  # noqa: BLE001 - VM reclaimed or transient CLI failure
            log(f'poll failed: {e}')
            if session_alive(args.session):
                continue
            restarts += 1
            if restarts > args.max_restarts:
                sys.exit('too many restarts, giving up')
            log(f'session lost; restart {restarts}/{args.max_restarts}, resuming from mirrored checkpoint')
            setup(args.session, args.gpu, args.run, local_run)
            launch(args.session, args.run, train_args)
            continue
        log(f"alive={state['alive']} epochs={state['epochs']} | {state['tail']}")
        if state['epochs'] != seen_epochs:
            mirror(args.session, args.run, local_run)
            seen_epochs = state['epochs']
        if not state['alive']:
            mirror(args.session, args.run, local_run)
            if args.keep_session:
                log('job ended; keeping session for the next run')
            else:
                log('job ended; stopping session')
                colab('stop', '-s', args.session, check=False)
            break


if __name__ == '__main__':
    main()
