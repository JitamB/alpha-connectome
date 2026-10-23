"""Kaggle-only paired verdict for the class-3 target-innovation screen."""
import subprocess
import sys


cmd = [sys.executable, 'tools/judge.py', '--base', 'lr7e4_full', 'lr7e4_full_seed1',
       '--new', 'delta_c3_s0', 'delta_c3_s1']
print('$', ' '.join(cmd), flush=True)
subprocess.run(cmd, check=True)
