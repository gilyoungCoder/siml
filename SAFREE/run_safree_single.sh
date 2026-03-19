#!/bin/bash

################################################################################
# SAFREE Single Generation Script
# Based on ICLR 2025 paper default settings
################################################################################

################################################################################
# ★★★ 여기를 수정하세요 (USER CONFIGURATION) ★★★
################################################################################

# === 필수 설정 ===
TXT_FILE="../SoftDelete+CG/prompts/coco_30.txt"    # 프롬프트가 담긴 텍스트 파일 경로
SAVE_DIR="./results/safree_single/coco"          # 결과 저장 폴더
CATEGORY="nudity"                            # 제거할 카테고리 (단일 또는 콤마로 구분)
                                             # 예시: "nudity" / "violence" / "nudity,violence"
                                             #      "artists-VanGogh" / "artists-VanGogh,artists-KellyMcKernan"
DEVICE="cuda:2"                              # GPU 디바이스: cuda:0, cuda:1, ... 또는 cpu

# === 모델 설정 ===
MODEL_ID="CompVis/stable-diffusion-v1-4"    # SD 모델 ID
# MODEL_ID="stabilityai/stable-diffusion-xl-base-1.0"  # SDXL 사용 시

# === Erase 방법 (선택사항) ===
ERASE_ID="std"                               # std (기본) / esd / sld
# ERASE_CHECKPOINT=""                        # ESD 체크포인트 경로 (esd 사용 시)

# === 생성 파라미터 ===
NUM_SAMPLES=1          # 프롬프트당 생성할 이미지 개수
NUM_STEPS=50           # Inference steps (논문 기본값: 50)
GUIDANCE=7.5           # Guidance scale (논문 기본값: 7.5)
SEED=42                # Random seed
IMAGE_SIZE=512         # 이미지 크기 (512 for SD v1.4, 1024 for SDXL)

# === SAFREE 기능 ON/OFF ===
SAFREE=true            # SAFREE 활성화 (false로 설정 시 일반 SD 생성)
SVF=true               # Self-Validation Filter
LRA=true               # Latent Re-Attention (FreeU)

################################################################################
# SAFREE 고급 하이퍼파라미터 (논문 기본값, 일반적으로 수정 불필요)
################################################################################
SF_ALPHA=0.01                    # α: Toxic token detection sensitivity (Eq. 4)
RE_ATTN_T="-1,1001"              # Re-attention timestep range
UP_T=10                          # Update threshold
FREEU_HYP="1.0-1.0-0.9-0.2"      # FreeU parameters (b1-b2-s1-s2)

################################################################################
# 아래 코드는 수정하지 마세요 (DO NOT MODIFY BELOW)
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse command line arguments (커맨드라인 옵션이 위 설정을 덮어씀)
while [[ $# -gt 0 ]]; do
    case $1 in
        --txt)
            TXT_FILE="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --model)
            MODEL_ID="$2"
            shift 2
            ;;
        --category)
            CATEGORY="$2"
            shift 2
            ;;
        --steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --guidance)
            GUIDANCE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --no-safree)
            SAFREE=false
            shift
            ;;
        --no-svf)
            SVF=false
            shift
            ;;
        --no-lra)
            LRA=false
            shift
            ;;
        --erase-id)
            ERASE_ID="$2"
            shift 2
            ;;
        --erase-ckpt)
            ERASE_CHECKPOINT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --txt FILE              Path to text file with prompts (required)"
            echo "  --save-dir DIR          Output directory"
            echo "  --model MODEL_ID        Model ID"
            echo "  --category CAT          Category to remove (single or comma-separated)"
            echo "                          Examples: nudity, violence, nudity,violence"
            echo "                                   artists-VanGogh,artists-KellyMcKernan"
            echo "  --steps N               Number of inference steps (default: 50)"
            echo "  --guidance G            Guidance scale (default: 7.5)"
            echo "  --seed S                Random seed (default: 42)"
            echo "  --num-samples N         Images per prompt (default: 1)"
            echo "  --device DEVICE         Device (default: cuda:0)"
            echo "  --no-safree             Disable SAFREE (baseline generation)"
            echo "  --no-svf                Disable Self-Validation Filter"
            echo "  --no-lra                Disable Latent Re-Attention (FreeU)"
            echo "  --erase-id ID           Erase method: std|esd|sld (default: std)"
            echo "  --erase-ckpt PATH       Path to ESD checkpoint (for esd mode)"
            echo ""
            echo "Examples:"
            echo "  # Single concept removal"
            echo "  $0 --txt prompts.txt --category nudity"
            echo ""
            echo "  # Multi-concept removal (nudity + violence)"
            echo "  $0 --txt prompts.txt --category nudity,violence"
            echo ""
            echo "  # Multiple artists removal"
            echo "  $0 --txt prompts.txt --category artists-VanGogh,artists-KellyMcKernan"
            echo ""
            echo "  # Baseline SD generation (no SAFREE)"
            echo "  $0 --txt prompts.txt --no-safree"
            echo ""
            echo "  # Using ESD checkpoint"
            echo "  $0 --txt prompts.txt --erase-id esd --erase-ckpt ./checkpoints/esd_nudity.pt"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ ! -f "$TXT_FILE" ]]; then
    echo "Error: Text file not found: $TXT_FILE"
    echo "Please specify a valid text file with --txt or modify TXT_FILE in the script"
    exit 1
fi

# Build configs (following original SAFREE script style)
# Note: Values with commas must be quoted
configs="--txt $TXT_FILE \
    --save-dir $SAVE_DIR \
    --model_id $MODEL_ID \
    --category $CATEGORY \
    --num-samples $NUM_SAMPLES \
    --num_inference_steps $NUM_STEPS \
    --guidance_scale $GUIDANCE \
    --seed $SEED \
    --image_length $IMAGE_SIZE \
    --device $DEVICE \
    --erase-id $ERASE_ID \
    --sf_alpha $SF_ALPHA \
    --re_attn_t \"$RE_ATTN_T\" \
    --up_t $UP_T \
    --freeu_hyp \"$FREEU_HYP\""

# Add SAFREE flags
if [[ "$SAFREE" == "true" ]]; then
    configs="$configs --safree"
fi

if [[ "$SVF" == "true" ]]; then
    configs="$configs -svf"
fi

if [[ "$LRA" == "true" ]]; then
    configs="$configs -lra"
fi

# Add ESD checkpoint if specified
if [[ -n "$ERASE_CHECKPOINT" ]]; then
    configs="$configs --erase_concept_checkpoint $ERASE_CHECKPOINT"
fi

# Print configuration
echo "=========================================="
echo "SAFREE Single Image Generation"
echo "=========================================="
echo "Input file:       $TXT_FILE"
echo "Output dir:       $SAVE_DIR"
echo "Model:            $MODEL_ID"
echo "Category:         $CATEGORY"
echo "Device:           $DEVICE"
echo "Erase method:     $ERASE_ID"
echo ""
echo "Generation settings:"
echo "  Steps:          $NUM_STEPS"
echo "  Guidance:       $GUIDANCE"
echo "  Seed:           $SEED"
echo "  Images/prompt:  $NUM_SAMPLES"
echo "  Image size:     ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo ""
echo "SAFREE settings (ICLR 2025 defaults):"
echo "  Enabled:        $SAFREE"
echo "  SVF:            $SVF"
echo "  LRA (FreeU):    $LRA"
echo "  Alpha (α):      $SF_ALPHA"
echo "  Re-attn steps:  $RE_ATTN_T"
echo "  Up threshold:   $UP_T"
echo "  FreeU params:   $FREEU_HYP"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$SAVE_DIR"

# Run generation (following original SAFREE script style)
echo "Starting generation..."
echo ""
echo "Command: python gen_safree_single.py $configs"
echo ""

python gen_safree_single.py $configs 2>&1

# Check exit status
if [[ $? -eq 0 ]]; then
    echo ""
    echo "=========================================="
    echo "Generation completed successfully!"
    echo "Results saved to: $SAVE_DIR"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Generation failed with error code: $?"
    echo "=========================================="
    exit 1
fi
