#!/usr/bin/env python3
"""
Clean folders that have Error results
Deletes categories_qwen2_vl.json and results.txt from folders with all errors
"""
import os
import sys
import json
from pathlib import Path

def has_only_errors(folder):
    """Check if a folder has only Error results"""
    json_file = Path(folder) / "categories_qwen2_vl.json"

    if not json_file.exists():
        return False

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        # Check if all results are errors
        all_errors = all(
            data.get("category") == "Error"
            for data in results.values()
        )

        return all_errors
    except:
        return False

def clean_folder(folder, dry_run=True):
    """Remove result files from a folder"""
    json_file = Path(folder) / "categories_qwen2_vl.json"
    txt_file = Path(folder) / "results.txt"

    files_to_remove = []
    if json_file.exists():
        files_to_remove.append(json_file)
    if txt_file.exists():
        files_to_remove.append(txt_file)

    if dry_run:
        return files_to_remove
    else:
        for f in files_to_remove:
            f.unlink()
        return files_to_remove

def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_error_results.py <base_dir> [--force]")
        print("\nExample:")
        print("  python vlm/clean_error_results.py SoftDelete+CG/scg_outputs/grid_search_nudity")
        print("  python vlm/clean_error_results.py SoftDelete+CG/scg_outputs/grid_search_nudity --force")
        print("\nOptions:")
        print("  --force    Actually delete files (default is dry-run)")
        sys.exit(1)

    base_dir = sys.argv[1]
    force = '--force' in sys.argv

    if not os.path.isdir(base_dir):
        print(f"❌ Directory not found: {base_dir}")
        sys.exit(1)

    print("="*70)
    print("Error 결과 정리 스크립트")
    print("="*70)
    print(f"📁 디렉토리: {base_dir}")
    print(f"🔧 모드: {'실제 삭제' if force else '테스트 (dry-run)'}")
    print()

    # Find all folders
    base_path = Path(base_dir)
    folders = sorted([d for d in base_path.iterdir() if d.is_dir()])

    print(f"📊 전체 폴더: {len(folders)}개")
    print()

    # Check each folder
    error_folders = []
    total_files = 0

    for folder in folders:
        if has_only_errors(str(folder)):
            error_folders.append(folder)
            files = clean_folder(str(folder), dry_run=not force)
            total_files += len(files)

            if not force:
                print(f"🔍 {folder.name}: {len(files)}개 파일")

    print()
    print("="*70)
    print(f"📊 요약:")
    print(f"   Error 폴더: {len(error_folders)}개 / {len(folders)}개")
    print(f"   삭제될 파일: {total_files}개")
    print("="*70)

    if not force:
        print()
        print("⚠️  이것은 테스트 실행입니다. 실제로 삭제하지 않았습니다.")
        print(f"   실제 삭제하려면: python vlm/clean_error_results.py {base_dir} --force")
    else:
        print()
        print(f"✅ {total_files}개 파일 삭제 완료!")
        print()
        print("삭제된 폴더 목록:")
        for folder in error_folders:
            print(f"   - {folder.name}")

if __name__ == "__main__":
    main()
