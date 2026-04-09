import torch

d16 = torch.load("/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/sd14/clip_exemplar_embeddings.pt", map_location="cpu")
d32 = torch.load("/mnt/home3/yhgil99/unlearning/CAS_SpatialCFG/exemplars/sd14/clip_exemplar_full_nudity.pt", map_location="cpu")

print("16-image pt:", d16["target_clip_features"].shape)
print("  config:", d16["config"])
print()
print("32-image pt:", d32["target_clip_features"].shape)
print("  config:", d32["config"])
