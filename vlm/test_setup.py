#!/usr/bin/env python3
"""
Test setup script - Verify everything is ready for batch evaluation
"""
import os
import sys
from pathlib import Path

def check_env():
    """Check environment setup"""
    print("🔍 Checking environment setup...\n")

    # Check OPENAI_API_KEY
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✅ OPENAI_API_KEY is set (length: {len(api_key)})")
    else:
        print("❌ OPENAI_API_KEY is NOT set")
        print("   Please set it: export OPENAI_API_KEY='your-key-here'")
        return False

    return True

def check_paths():
    """Check required paths exist"""
    print("\n🔍 Checking paths...\n")

    paths_to_check = {
        "Nudity images": "SoftDelete+CG/scg_outputs/grid_search_nudity",
        "Violence images": "SoftDelete+CG/scg_outputs/grid_search_violence",
        "GPT nudity script": "vlm/gpt.py",
        "GPT violence script": "vlm/gpt_violence.py",
        "Batch evaluator": "vlm/batch_evaluate.py",
        "Quick summary": "vlm/quick_summary.py",
        "Launch script": "run_batch_evaluation.sh"
    }

    all_exist = True
    for name, path in paths_to_check.items():
        if Path(path).exists():
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} NOT FOUND")
            all_exist = False

    return all_exist

def check_folders():
    """Check image folders"""
    print("\n🔍 Checking image folders...\n")

    nudity_dir = Path("SoftDelete+CG/scg_outputs/grid_search_nudity")
    violence_dir = Path("SoftDelete+CG/scg_outputs/grid_search_violence")

    for category, base_dir in [("Nudity", nudity_dir), ("Violence", violence_dir)]:
        if not base_dir.exists():
            print(f"❌ {category} directory not found: {base_dir}")
            continue

        folders = [d for d in base_dir.iterdir() if d.is_dir()]
        print(f"📁 {category}: {len(folders)} folders found")

        # Check a sample folder
        if folders:
            sample = folders[0]
            images = list(sample.glob("*.png")) + list(sample.glob("*.jpg")) + \
                     list(sample.glob("*.jpeg")) + list(sample.glob("*.webp"))
            print(f"   Sample folder: {sample.name}")
            print(f"   Images in sample: {len(images)}")

    print()

def check_dependencies():
    """Check Python dependencies"""
    print("🔍 Checking Python dependencies...\n")

    required_modules = ["openai", "json", "base64"]
    all_ok = True

    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} NOT INSTALLED")
            all_ok = False

    return all_ok

def estimate_cost():
    """Estimate API cost and time"""
    print("\n💰 Cost and Time Estimation...\n")

    nudity_dir = Path("SoftDelete+CG/scg_outputs/grid_search_nudity")
    violence_dir = Path("SoftDelete+CG/scg_outputs/grid_search_violence")

    total_folders = 0
    total_images = 0

    for base_dir in [nudity_dir, violence_dir]:
        if base_dir.exists():
            folders = [d for d in base_dir.iterdir() if d.is_dir()]
            total_folders += len(folders)

            # Sample first folder to estimate images per folder
            if folders:
                sample = folders[0]
                images = list(sample.glob("*.png")) + list(sample.glob("*.jpg")) + \
                         list(sample.glob("*.jpeg")) + list(sample.glob("*.webp"))
                total_images += len(images) * len(folders)

    # GPT-4o pricing (approximate, check OpenAI pricing for accuracy)
    # Vision API costs more than text
    cost_per_image = 0.01  # Very rough estimate, actual cost depends on image size

    estimated_cost = total_images * cost_per_image
    estimated_time_hours = (total_images * 3) / 3600  # ~3 seconds per image

    print(f"📊 Estimated totals:")
    print(f"   Total folders: {total_folders}")
    print(f"   Total images: {total_images} (estimated)")
    print(f"   Estimated cost: ${estimated_cost:.2f} (ROUGH ESTIMATE)")
    print(f"   Estimated time: {estimated_time_hours:.1f} hours")
    print(f"\n⚠️  NOTE: Actual costs may vary significantly based on:")
    print(f"   - Image sizes")
    print(f"   - OpenAI pricing changes")
    print(f"   - Network conditions")
    print(f"\n   Please check OpenAI pricing page for accurate costs!")

def main():
    print("="*70)
    print("VLM Batch Evaluation - Setup Check")
    print("="*70)

    all_checks_passed = True

    # Run all checks
    if not check_env():
        all_checks_passed = False

    if not check_paths():
        all_checks_passed = False

    check_folders()

    if not check_dependencies():
        all_checks_passed = False

    estimate_cost()

    print("\n" + "="*70)
    if all_checks_passed:
        print("✅ All checks passed! You're ready to run batch evaluation.")
        print("\nTo start:")
        print("  ./run_batch_evaluation.sh nudity background")
        print("  ./run_batch_evaluation.sh violence background")
        print("  ./run_batch_evaluation.sh both background")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)
    print("="*70)

if __name__ == "__main__":
    main()
