import sys
from pytorch_fid.fid_score import calculate_fid_given_paths
pairs=sys.argv[1:]
for spec in pairs:
    name,a,b=spec.split('=',1)[0],*spec.split('=',1)[1].split('|')
    v=calculate_fid_given_paths([a,b], batch_size=64, device='cuda', dims=2048, num_workers=4)
    print(f'{name}: {v:.4f}', flush=True)
