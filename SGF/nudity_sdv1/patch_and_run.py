"""Wrapper: monkey-patch onnx issue then run the actual generation script."""
import sys, os, importlib, types

# Compat for older torchvision expecting torch._six
import torch
if not hasattr(torch, '_six'):
    torch._six = types.SimpleNamespace(string_classes=(str, bytes))

# Fix onnx _cast_Long
import torch.onnx.symbolic_opset9 as _op9
if not hasattr(_op9, '_cast_Long'):
    _op9._cast_Long = lambda g, i, n: g.op("Cast", i, to_i=7)

# The actual script to run is passed as first argument
script = sys.argv[1]
sys.argv = sys.argv[1:]  # shift so the script sees its own args

# Execute the script
with open(script) as f:
    exec(compile(f.read(), script, 'exec'))
