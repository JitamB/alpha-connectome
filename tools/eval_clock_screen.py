"""Kaggle-only paired verdict for the explicit causal state-clock screen."""
import subprocess
import sys


cmd = [sys.executable, 'tools/judge.py', '--base', 'lr7e4_full', 'lr7e4_full_seed1',
       '--new', 'clock_s0', 'clock_s1']
print('$', ' '.join(cmd), flush=True)
subprocess.run(cmd, check=True)
