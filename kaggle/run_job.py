"""Kaggle job runner (private notebook): fetch data, train one run per GPU in parallel, evaluate, save to /kaggle/working.

The code is shipped as the private dataset <user>/wunder-code (code.tgz built by kaggle/push.sh); JOBS below is
rewritten by push.sh for each job. Only checkpoints (best.pt, config.json, history.json), results/*.npz|json and logs
are kept in /kaggle/working; the 35 GB of data lives in /tmp.
"""
import glob
import os
import shlex
import shutil
import subprocess
import tarfile

JOBS = []  # [(run_name, [train_gru args]), ...]; one per GPU
WORK, OUT = '/tmp/wunder', '/kaggle/working'


def sh(cmd, **kw):
    print(f'$ {cmd}', flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


src = glob.glob('/kaggle/input/**/code.tgz', recursive=True)
os.makedirs(WORK, exist_ok=True)
if src:
    tarfile.open(src[0]).extractall(WORK)
else:  # Kaggle may have auto-extracted the archive
    shutil.copytree(os.path.dirname(glob.glob('/kaggle/input/**/train_gru.py', recursive=True)[0]) + '/..', WORK, dirs_exist_ok=True)
os.chdir(WORK)
sh('pip install -q onnx onnxruntime')
sh('python -u tools/fetch_data.py wnn_connectome_starterpack datasets/train.parquet datasets/valid.parquet')
procs = []
for gpu, (run, args) in enumerate(JOBS):
    if run.startswith('eval:'):  # evaluation only: args are eval_torch arguments (runs shipped via EXTRA)
        run = run[5:]
        cmd = shlex.join(['python', '-u', 'train/eval_torch.py', *args, '--name', run, '--batch', '32'])
    else:
        train = shlex.join(['python', '-u', 'train/train_gru.py', '--out', f'runs/{run}', '--device', 'cuda', *args])
        evaluate = shlex.join(['python', '-u', 'train/eval_torch.py', f'runs/{run}', '--name', run, '--batch', '32'])
        cmd = f'{train} && {evaluate}'
    log = open(f'{OUT}/{run}.log', 'w')
    procs.append((run, subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)})))
for run, p in procs:
    print(run, 'exit', p.wait(), flush=True)
    os.makedirs(f'{OUT}/runs/{run}', exist_ok=True)
    for f in ('best.pt', 'config.json', 'history.json'):
        if os.path.exists(f'runs/{run}/{f}'):
            shutil.copy(f'runs/{run}/{f}', f'{OUT}/runs/{run}/{f}')
    os.makedirs(f'{OUT}/results', exist_ok=True)
    for f in glob.glob(f'results/{run}*'):
        shutil.copy(f, f'{OUT}/results/')
