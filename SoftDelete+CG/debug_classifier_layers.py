"""
Debug script to print all available layers in the classifier
"""

import torch
from geo_models.classifier.classifier import load_discriminator

# Load classifier
CLASSIFIER_CKPT = "./work_dirs/nudity_three_class/checkpoint/step_11800/classifier.pth"

print("Loading classifier...")
classifier = load_discriminator(
    ckpt_path=CLASSIFIER_CKPT,
    condition=None,
    eval=True,
    channel=4,
    num_classes=3
)

print("\n" + "="*80)
print("ALL LAYERS IN CLASSIFIER")
print("="*80 + "\n")

for name, module in classifier.named_modules():
    # Print all layers
    print(f"{name:60s} {module.__class__.__name__}")

print("\n" + "="*80)
print("CANDIDATE LAYERS FOR GRAD-CAM (ResBlock and Attention)")
print("="*80 + "\n")

candidates = []
for name, module in classifier.named_modules():
    module_type = module.__class__.__name__
    # Look for ResBlock or similar
    if any(x in module_type.lower() for x in ['resblock', 'residual', 'attention', 'upsample', 'downsample']):
        if 'encoder_model' in name:  # Only show layers inside encoder_model
            candidates.append(name)
            print(f"✓ {name}")

print("\n" + "="*80)
print("RECOMMENDED LAYERS FOR INTERPRETATION")
print("="*80 + "\n")

# Filter to find good candidates
for name in candidates:
    if 'output' in name or ('input' in name and any(str(i) in name for i in [8, 9, 10, 11])):
        print(f"  👉 {name}")
