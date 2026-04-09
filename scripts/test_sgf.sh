#!/usr/bin/env bash
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/mnt/home3/yhgil99/.conda/envs/sfgd/lib:${LD_LIBRARY_PATH:-}
cd /mnt/home3/yhgil99/unlearning/SGF/nudity_sdv1
/mnt/home3/yhgil99/.conda/envs/sfgd/bin/python3.10 -c "
import torch.onnx.symbolic_opset9 as o
if not hasattr(o,'_cast_Long'): o._cast_Long=lambda g,i,n:g.op('Cast',i,to_i=7)
import sys; sys.path.insert(0,'.')
from main_utils import load_yaml; print('1. main_utils OK')
from models.textuals_visual.modified_sld_pipeline_sgf import ModifiedSLDPipeline_Rep; print('2. SGF OK')
print('ALL GOOD!')
"
