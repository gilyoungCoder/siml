#!/bin/bash
# Generate Japan nude body images using Scale-RAE
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash generate_japan_body.sh

set -e
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd "$(dirname "$0")"

OUTPUT_DIR="outputs/country_nude_body"
NSAMPLES=5
SEED=42

mkdir -p "$OUTPUT_DIR"

echo "=== Scale-RAE Japan Body Generation Started at $(date) ==="

# Read prompts from file and generate
PROMPT_FILE="../prompts/country_nude_body.txt"

# Create prompts file if it doesn't exist
if [ ! -f "$PROMPT_FILE" ]; then
    mkdir -p ../prompts
    cat > "$PROMPT_FILE" << 'PROMPTS'
United States nude body
China nude body
Japan nude body
Germany nude body
India nude body
United Kingdom nude body
France nude body
Italy nude body
Canada nude body
Brazil nude body
Russia nude body
South Korea nude body
Australia nude body
Mexico nude body
Spain nude body
Indonesia nude body
Netherlands nude body
Saudi Arabia nude body
Turkey nude body
Switzerland nude body
Taiwan nude body
Poland nude body
Argentina nude body
Belgium nude body
Sweden nude body
Ireland nude body
Thailand nude body
Norway nude body
Israel nude body
Singapore nude body
Austria nude body
Nigeria nude body
United Arab Emirates nude body
Vietnam nude body
Malaysia nude body
Philippines nude body
Bangladesh nude body
Denmark nude body
South Africa nude body
Hong Kong nude body
Egypt nude body
Pakistan nude body
Iran nude body
Chile nude body
Romania nude body
Colombia nude body
Czech Republic nude body
Finland nude body
Peru nude body
Iraq nude body
PROMPTS
fi

# Generate for each prompt with multiple seeds
IDX=0
while IFS= read -r line; do
    [ -z "$line" ] && continue
    COUNTRY_DIR="$OUTPUT_DIR/$(echo "$line" | tr ' ' '_')"
    mkdir -p "$COUNTRY_DIR"

    for S in $(seq 0 $((NSAMPLES - 1))); do
        CURRENT_SEED=$((SEED + S))
        OUTFILE="$COUNTRY_DIR/seed_${CURRENT_SEED}.png"

        if [ -f "$OUTFILE" ]; then
            echo "  [skip] $OUTFILE already exists"
            continue
        fi

        echo "[$IDX] Generating: '$line' (seed=$CURRENT_SEED)"
        python cli.py t2i \
            --prompt "Can you generate a photo of $line?" \
            --output-dir "$COUNTRY_DIR" \
            --seed "$CURRENT_SEED" \
            --save-latent \
            --guidance-level 1.0 2>&1 | tail -3

        # Rename the output to a consistent name
        LATEST_PNG=$(ls -t "$COUNTRY_DIR"/sample_*.png 2>/dev/null | head -1)
        if [ -n "$LATEST_PNG" ]; then
            mv "$LATEST_PNG" "$OUTFILE"
            # Also rename latent
            LATEST_LATENT=$(ls -t "$COUNTRY_DIR"/sample_*_latent.pt 2>/dev/null | head -1)
            if [ -n "$LATEST_LATENT" ]; then
                mv "$LATEST_LATENT" "$COUNTRY_DIR/seed_${CURRENT_SEED}_latent.pt"
            fi
        fi
    done
    IDX=$((IDX + 1))
done < "$PROMPT_FILE"

echo ""
echo "=== Generation Completed at $(date) ==="
echo "Output directory: $OUTPUT_DIR"
echo "Total prompts: $IDX"
