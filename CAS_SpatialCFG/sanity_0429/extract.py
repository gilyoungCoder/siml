#!/usr/bin/env python3
"""Print TC/AC from a pack — one shell line per pair (`TC=...;AC=...`)."""
import sys, torch
pack = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
fm = pack['family_metadata']
tc = [f.replace('_',' ') for f in fm.keys()]
ac = [fm[f].get('anchor_words', ['safe'])[0] for f in fm.keys()]
print('TC=' + '|'.join(tc))
print('AC=' + '|'.join(ac))
