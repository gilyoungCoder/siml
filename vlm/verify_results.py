#!/usr/bin/env python3
"""Verify all GPT-4o, NudeNet, and Q16 results by reading actual files."""

import os
import re
from pathlib import Path

def parse_gpt4o_file(filepath):
    """Parse GPT-4o result file and extract distribution."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        result = {
            'concept': None,
            'total': None,
            'distribution': {},
            'harmful_rate': None
        }

        # Extract concept
        match = re.search(r'Concept:\s*(\w+)', content)
        if match:
            result['concept'] = match.group(1)

        # Extract total images
        match = re.search(r'Total images:\s*(\d+)', content)
        if match:
            result['total'] = int(match.group(1))

        # Extract distribution - handle both NotRelevant and NotPeople
        for label in ['NotRelevant', 'NotPeople', 'Safe', 'Partial', 'Full']:
            match = re.search(rf'-\s*{label}:\s*(\d+)\s*\((\d+\.?\d*)%\)', content)
            if match:
                result['distribution'][label] = {
                    'count': int(match.group(1)),
                    'percent': float(match.group(2))
                }

        # Extract harmful rate
        match = re.search(r'Harmful Rate.*?(\d+)/(\d+)\s*\((\d+\.?\d*)%\)', content)
        if match:
            result['harmful_rate'] = float(match.group(3))

        return result
    except Exception as e:
        return {'error': str(e)}

def parse_nudenet_file(filepath):
    """Parse NudeNet result file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        result = {}

        # Extract unsafe rate
        match = re.search(r'Unsafe Rate:\s*[\d.]+\s*\((\d+\.?\d*)%\)', content)
        if match:
            result['unsafe_rate'] = float(match.group(1))

        # Extract total images
        match = re.search(r'Total Images:\s*(\d+)', content)
        if match:
            result['total'] = int(match.group(1))

        return result
    except Exception as e:
        return {'error': str(e)}

def parse_q16_file(filepath):
    """Parse Q16 result file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        result = {}

        # Extract inappropriate rate
        match = re.search(r'Inappropriate Rate:\s*[\d.]+\s*\((\d+\.?\d*)%\)', content)
        if match:
            result['inappropriate_rate'] = float(match.group(1))

        # Extract total images
        match = re.search(r'Total Images:\s*(\d+)', content)
        if match:
            result['total'] = int(match.group(1))

        return result
    except Exception as e:
        return {'error': str(e)}

def main():
    base_dirs = [
        '/mnt/home/yhgil99/unlearning/SoftDelete+CG/SDbaseline/i2p',
        '/mnt/home/yhgil99/unlearning/SAFREE/safree_outputs/i2p',
        '/mnt/home/yhgil99/unlearning/SAFREE/results',
        '/mnt/home/yhgil99/unlearning/SoftDelete+CG/scg_outputs',
    ]

    all_results = []

    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                filepath = os.path.join(root, f)
                rel_path = os.path.relpath(filepath, '/mnt/home/yhgil99/unlearning')

                if f.startswith('results_gpt4o_'):
                    result = parse_gpt4o_file(filepath)
                    result['type'] = 'gpt4o'
                    result['path'] = rel_path
                    all_results.append(result)
                elif f == 'results_nudenet.txt':
                    result = parse_nudenet_file(filepath)
                    result['type'] = 'nudenet'
                    result['path'] = rel_path
                    all_results.append(result)
                elif f == 'results_q16.txt':
                    result = parse_q16_file(filepath)
                    result['type'] = 'q16'
                    result['path'] = rel_path
                    all_results.append(result)

    # Print organized results
    print("=" * 100)
    print("GPT-4o Results Summary")
    print("=" * 100)

    # Group by method type
    baseline_results = []
    safree_results = []
    safree_ours_results = []
    ours_skip_results = []
    ours_noskip_results = []
    other_results = []

    for r in all_results:
        if r['type'] != 'gpt4o':
            continue
        path = r['path']
        if 'SDbaseline' in path:
            baseline_results.append(r)
        elif 'safree_outputs' in path:
            safree_results.append(r)
        elif 'grid_search_safree' in path:
            safree_ours_results.append(r)
        elif '_skip' in path:
            ours_skip_results.append(r)
        elif 'scg_outputs' in path:
            ours_noskip_results.append(r)
        else:
            other_results.append(r)

    def print_gpt4o_results(results, label):
        print(f"\n### {label} ###")
        for r in sorted(results, key=lambda x: x.get('concept', '')):
            if 'error' in r:
                print(f"  ERROR: {r['path']} - {r['error']}")
                continue
            dist = r.get('distribution', {})
            safe = dist.get('Safe', {}).get('percent', 'N/A')
            partial = dist.get('Partial', {}).get('percent', 'N/A')
            full = dist.get('Full', {}).get('percent', 'N/A')
            not_rel = dist.get('NotRelevant', dist.get('NotPeople', {})).get('percent', 'N/A')
            harmful = r.get('harmful_rate', 'N/A')
            total = r.get('total', 'N/A')
            concept = r.get('concept', 'Unknown')
            print(f"  [{concept}] Total={total} | Safe={safe}% Partial={partial}% Full={full}% NotRel={not_rel}% | Harmful={harmful}%")
            print(f"    Path: {r['path']}")

    print_gpt4o_results(baseline_results, "SD 1.4 Baseline")
    print_gpt4o_results(safree_results, "SAFREE")
    print_gpt4o_results(safree_ours_results, "SAFREE+Ours")
    print_gpt4o_results(ours_skip_results, "Ours (skip)")
    print_gpt4o_results(ours_noskip_results, "Ours (noskip)")

    # NudeNet results
    print("\n" + "=" * 100)
    print("NudeNet Results Summary")
    print("=" * 100)
    nudenet_results = [r for r in all_results if r['type'] == 'nudenet']
    for r in sorted(nudenet_results, key=lambda x: x['path']):
        if 'error' in r:
            print(f"  ERROR: {r['path']} - {r['error']}")
            continue
        unsafe = r.get('unsafe_rate', 'N/A')
        total = r.get('total', 'N/A')
        print(f"  Unsafe={unsafe}% Total={total}")
        print(f"    Path: {r['path']}")

    # Q16 results
    print("\n" + "=" * 100)
    print("Q16 Results Summary")
    print("=" * 100)
    q16_results = [r for r in all_results if r['type'] == 'q16']
    for r in sorted(q16_results, key=lambda x: x['path']):
        if 'error' in r:
            print(f"  ERROR: {r['path']} - {r['error']}")
            continue
        inapp = r.get('inappropriate_rate', 'N/A')
        total = r.get('total', 'N/A')
        print(f"  Inappropriate={inapp}% Total={total}")
        print(f"    Path: {r['path']}")

if __name__ == '__main__':
    main()
