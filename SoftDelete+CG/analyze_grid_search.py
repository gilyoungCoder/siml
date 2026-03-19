#!/usr/bin/env python3
"""
Grid Search 결과 분석 스크립트
각 실험의 guidance 적용률, harmful 감지률 등을 비교
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd


def parse_experiment_name(name):
    """
    실험 이름에서 파라미터 추출
    예: gs7.0_hs1.25_st0.3_ws4.0-1.0_ts0.0--2.0
    """
    parts = name.split('_')
    params = {}

    for part in parts:
        if part.startswith('gs'):
            params['guidance_scale'] = float(part[2:])
        elif part.startswith('hs'):
            params['harmful_scale'] = float(part[2:])
        elif part.startswith('st'):
            params['spatial_threshold'] = float(part[2:])
        elif part.startswith('ws'):
            # Weight schedule: ws3.0-0.5
            schedule_str = part[2:]
            if '-' in schedule_str:
                parts_ws = schedule_str.split('-')
                params['weight_start'] = float(parts_ws[0])
                params['weight_end'] = float(parts_ws[1])
            else:
                params['weight_start'] = float(schedule_str)
                params['weight_end'] = float(schedule_str)
        elif part.startswith('ts'):
            # Threshold schedule: ts0.0--2.0 or ts-0.5--2.5
            schedule_str = part[2:]

            # Split by first occurrence of double dash
            if '--' in schedule_str:
                idx = schedule_str.index('--')
                start_str = schedule_str[:idx]
                end_str = schedule_str[idx+2:]

                params['threshold_start'] = float(start_str) if start_str else 0.0
                params['threshold_end'] = -float(end_str) if end_str else 0.0
            elif '-' in schedule_str and schedule_str[0] == '-':
                # Starts with negative: -0.5--2.5
                # Find second dash
                second_dash = schedule_str.find('-', 1)
                if second_dash > 0:
                    start_str = schedule_str[:second_dash]
                    end_str = schedule_str[second_dash+1:]
                    if end_str.startswith('-'):
                        end_str = end_str[1:]
                    params['threshold_start'] = float(start_str)
                    params['threshold_end'] = -float(end_str) if end_str else 0.0
                else:
                    params['threshold_start'] = float(schedule_str)
                    params['threshold_end'] = float(schedule_str)
            else:
                params['threshold_start'] = float(schedule_str)
                params['threshold_end'] = float(schedule_str)

    return params


def analyze_experiment(exp_dir):
    """
    단일 실험 디렉토리 분석
    """
    exp_path = Path(exp_dir)

    if not exp_path.exists():
        return None

    # 이미지 개수
    images = list(exp_path.glob("*.png"))
    num_images = len(images)

    # Visualization 개수
    vis_dir = exp_path / "visualizations"
    num_visualizations = len(list(vis_dir.glob("*.png"))) if vis_dir.exists() else 0

    # results.txt에서 분류 결과 파싱
    results_file = exp_path / "results.txt"
    not_relevant = 0
    safe = 0
    partial = 0
    full = 0

    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # 예: " - NotRelevant: 38개 (76.0%)" or " - NotPeople: 37개 (74.0%)"
            import re
            for line in content.split('\n'):
                # Handle both NotRelevant (violence) and NotPeople (nudity)
                if 'NotRelevant:' in line or 'NotPeople:' in line:
                    match = re.search(r'(\d+)개', line)
                    if match:
                        not_relevant = int(match.group(1))
                elif 'Safe:' in line:
                    match = re.search(r'(\d+)개', line)
                    if match:
                        safe = int(match.group(1))
                elif 'Partial:' in line:
                    match = re.search(r'(\d+)개', line)
                    if match:
                        partial = int(match.group(1))
                elif 'Full:' in line:
                    match = re.search(r'(\d+)개', line)
                    if match:
                        full = int(match.group(1))

    # 비율 계산
    total = not_relevant + safe + partial + full
    if total > 0:
        full_not_relevant_ratio = (full + not_relevant) / total
        safe_partial_ratio = (safe + partial) / total  # Safe + Partial 비율 (높을수록 좋음)
        safe_ratio = safe / total
        partial_ratio = partial / total
    else:
        full_not_relevant_ratio = 0.0
        safe_partial_ratio = 0.0
        safe_ratio = 0.0
        partial_ratio = 0.0

    # 통계 정보
    stats = {
        'num_images': num_images,
        'num_visualizations': num_visualizations,
        'not_relevant': not_relevant,
        'safe': safe,
        'partial': partial,
        'full': full,
        'total': total,
        'safe_partial_ratio': safe_partial_ratio,  # 높을수록 좋음
        'full_not_relevant_ratio': full_not_relevant_ratio,
        'safe_ratio': safe_ratio,
        'partial_ratio': partial_ratio,
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze grid search results")
    parser.add_argument('base_dir', type=str,
                        help="Base directory containing grid search results")
    parser.add_argument('--output', type=str, default='grid_search_analysis.csv',
                        help="Output CSV file")

    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    if not base_dir.exists():
        print(f"❌ Error: Directory not found: {base_dir}")
        return

    print("="*80)
    print(f"🔍 Analyzing Grid Search Results")
    print("="*80)
    print(f"Base directory: {base_dir}")
    print()

    # 모든 실험 디렉토리 찾기
    experiment_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('gs')]

    if not experiment_dirs:
        print("❌ No experiment directories found!")
        return

    print(f"Found {len(experiment_dirs)} experiments")
    print()

    # 각 실험 분석
    results = []

    for exp_dir in sorted(experiment_dirs):
        exp_name = exp_dir.name
        print(f"Analyzing: {exp_name} ... ", end='')

        # 파라미터 파싱
        params = parse_experiment_name(exp_name)

        # 실험 분석
        stats = analyze_experiment(exp_dir)

        if stats is None:
            print("SKIP (not found)")
            continue

        # 결과 저장
        result = {
            'experiment': exp_name,
            **params,
            **stats
        }
        results.append(result)

        print(f"✓ ({stats['num_images']} images)")

    print()
    print("="*80)

    # DataFrame 생성
    df = pd.DataFrame(results)

    # 정렬: Safe+Partial 비율 높은 순 (좋은 성능)
    df = df.sort_values('safe_partial_ratio', ascending=False)

    # 저장
    output_path = Path(args.output)
    df.to_csv(output_path, index=False)

    print(f"✅ Results saved to: {output_path}")
    print()

    # 요약 통계
    print("="*80)
    print("📊 Summary Statistics")
    print("="*80)
    print()

    print(f"Total experiments: {len(df)}")
    print(f"Total images generated: {df['num_images'].sum()}")
    print()

    # 파라미터별 그룹 통계
    print("By Guidance Scale:")
    print(df.groupby('guidance_scale')['num_images'].describe())
    print()

    print("By Harmful Scale:")
    print(df.groupby('harmful_scale')['num_images'].describe())
    print()

    print("By Weight Schedule (start):")
    print(df.groupby('weight_start')['num_images'].describe())
    print()

    # Top 10 실험 (Safe+Partial 비율 높은 순)
    print("="*80)
    print("🏆 Top 10 Best Experiments (highest Safe+Partial ratio)")
    print("="*80)
    print("Higher ratio = Better performance (more Safe/Partial)")
    print()
    top_10 = df.head(10)[['experiment', 'guidance_scale', 'harmful_scale', 'spatial_threshold',
                           'safe', 'partial', 'full', 'not_relevant', 'safe_partial_ratio']]
    # 비율을 백분율로 표시
    top_10_display = top_10.copy()
    top_10_display['safe_partial_ratio'] = top_10_display['safe_partial_ratio'].apply(lambda x: f"{x*100:.1f}%")
    print(top_10_display.to_string(index=False))
    print()

    print("="*80)
    print("✅ Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
