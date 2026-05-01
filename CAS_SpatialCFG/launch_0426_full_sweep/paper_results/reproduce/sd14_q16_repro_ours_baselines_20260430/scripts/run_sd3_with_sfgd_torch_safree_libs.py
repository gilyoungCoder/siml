#!/usr/bin/env python3
import os, sys, runpy
os.environ.setdefault('PYTHONNOUSERSITE', '1')
os.environ.setdefault('TRANSFORMERS_NO_TORCHVISION', '1')
os.environ.setdefault('TRANSFORMERS_NO_VISION', '1')
import torch
try:
    import triton, triton.backends
except Exception as e:
    print(f'[WRAP WARN] triton preload failed: {e}')
# Do not import torchvision; transformers image paths are disabled above.
safree_site = '/mnt/home3/yhgil99/.conda/envs/safree/lib/python3.10/site-packages'
if safree_site not in sys.path:
    sys.path.insert(0, safree_site)
script = sys.argv[1]
sys.argv = sys.argv[1:]
print(f'[WRAP] torch={torch.__version__} cuda={torch.version.cuda} arch_tail={torch.cuda.get_arch_list()[-3:] if torch.cuda.is_available() else None}')
runpy.run_path(script, run_name='__main__')
