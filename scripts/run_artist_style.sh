#!/bin/bash
# Artist style removal experiments on siml-01 GPU 0-7
set -e
cd /mnt/home3/yhgil99/unlearning

eval "$(/usr/local/anaconda3/bin/conda shell.bash hook)"
conda activate sdd_copy
export PYTHONPATH="/mnt/home3/yhgil99/unlearning/SafeGen:$PYTHONPATH"

EXEMPLAR="CAS_SpatialCFG/exemplars/concepts_v2"
OUT="CAS_SpatialCFG/outputs/v2_experiments/artist"
mkdir -p $OUT

# Artist prompts - we'll generate them
PROMPTS_DIR="CAS_SpatialCFG/prompts/artists"
mkdir -p $PROMPTS_DIR

# Create artist prompt files (60 prompts each)
python3 -c "
artists = {
    'vangogh': 'in the style of Van Gogh',
    'picasso': 'in the style of Picasso',
    'monet': 'in the style of Monet',
    'rembrandt': 'in the style of Rembrandt',
    'warhol': 'in the style of Andy Warhol',
    'hopper': 'in the style of Edward Hopper',
}

subjects = [
    'a landscape with mountains and a river',
    'a portrait of a young woman',
    'a still life with flowers in a vase',
    'a city street at night',
    'a sunset over the ocean',
    'a field of sunflowers',
    'a bridge over a pond',
    'a cafe terrace at evening',
    'a self portrait of an artist',
    'a bedroom interior',
    'a starry night sky',
    'a garden with trees and paths',
    'a boat on a lake',
    'a church in a village',
    'a woman reading a book',
    'a man sitting in a chair',
    'a bowl of fruit on a table',
    'a horse in a meadow',
    'a dancer on stage',
    'a winter scene with snow',
    'a seaside village',
    'a park bench under trees',
    'a windmill in a field',
    'a portrait of an old man',
    'a woman with a hat',
    'a rainy day in the city',
    'a peaceful countryside',
    'a lighthouse by the sea',
    'a forest path in autumn',
    'a market scene with vendors',
]

for artist, style in artists.items():
    with open(f'$PROMPTS_DIR/{artist}.txt', 'w') as f:
        for subj in subjects:
            f.write(f'{subj}, {style}\n')
    print(f'{artist}: {len(subjects)} prompts')
"

echo "=== Artist Style START $(date) ==="

run_artist() {
    local gpu=$1 artist=$2 probe=$3 how=$4 fam=$5 cas=$6 ss=$7

    local suffix="${probe}_${how}"
    [ "$fam" = "1" ] && suffix="${suffix}_fam" || suffix="${suffix}_single"
    suffix="${suffix}_cas${cas}_ss${ss}"
    local outdir="$OUT/${artist}_${suffix}"

    [ -d "$outdir" ] && [ "$(ls $outdir/*.png 2>/dev/null | wc -l)" -gt 5 ] && return 0

    # Artist-specific target/anchor concepts
    local tgt="" anc=""
    case $artist in
        vangogh) tgt="Van Gogh thick impasto swirling brushstrokes" anc="smooth digital art flat colors photorealistic" ;;
        picasso) tgt="Picasso cubism geometric angular fragmented" anc="realistic proportional smooth classical" ;;
        monet)   tgt="Monet impressionism soft broken brushstrokes hazy" anc="sharp detailed crisp digital photography" ;;
        rembrandt) tgt="Rembrandt chiaroscuro dark dramatic lighting" anc="bright even lighting modern portrait" ;;
        warhol)  tgt="Warhol pop art bold colors screen print" anc="traditional painting natural colors classical" ;;
        hopper)  tgt="Hopper isolation geometric shadow urban" anc="warm cozy populated cheerful scene" ;;
    esac

    echo "[GPU$gpu] $artist $suffix"
    PYTHONPATH=/mnt/home3/yhgil99/unlearning/SafeGen:$PYTHONPATH \
    CUDA_VISIBLE_DEVICES=$gpu python3 -m safegen.generate_family \
        --prompts "$PROMPTS_DIR/${artist}.txt" --outdir "$outdir" \
        --probe_mode $probe --how_mode $how \
        --cas_threshold $cas --safety_scale $ss \
        --target_concepts $tgt --anchor_concepts $anc \
        --steps 50 --seed 42 --cfg_scale 7.5 2>&1 | tail -3
    echo "[DONE GPU$gpu] $artist $suffix ($(ls $outdir/*.png 2>/dev/null | wc -l) imgs)"
}

# Also generate baselines
gen_baseline() {
    local gpu=$1 artist=$2
    local outdir="$OUT/${artist}_baseline"
    [ -d "$outdir" ] && [ "$(ls $outdir/*.png 2>/dev/null | wc -l)" -gt 5 ] && return 0
    echo "[GPU$gpu] Baseline $artist"
    PYTHONPATH=/mnt/home3/yhgil99/unlearning/SafeGen:$PYTHONPATH \
    CUDA_VISIBLE_DEVICES=$gpu python3 -m safegen.generate_baseline \
        --prompts "$PROMPTS_DIR/${artist}.txt" --outdir "$outdir" \
        --steps 50 --seed 42 2>&1 | tail -3
    echo "[DONE GPU$gpu] baseline $artist"
}

# GPU 0: Van Gogh (grid)
(
gen_baseline 0 vangogh
for probe in text both; do
    for ss in 0.8 1.0 1.2; do
        run_artist 0 vangogh $probe anchor_inpaint 0 0.4 $ss
    done
done
echo "[GPU0 COMPLETE]"
) &

# GPU 1: Picasso (grid)
(
gen_baseline 1 picasso
for probe in text both; do
    for ss in 0.8 1.0 1.2; do
        run_artist 1 picasso $probe anchor_inpaint 0 0.4 $ss
    done
done
echo "[GPU1 COMPLETE]"
) &

# GPU 2: Monet (grid)
(
gen_baseline 2 monet
for probe in text both; do
    for ss in 0.8 1.0 1.2; do
        run_artist 2 monet $probe anchor_inpaint 0 0.4 $ss
    done
done
echo "[GPU2 COMPLETE]"
) &

# GPU 3: Rembrandt (grid)
(
gen_baseline 3 rembrandt
for probe in text both; do
    for ss in 0.8 1.0 1.2; do
        run_artist 3 rembrandt $probe anchor_inpaint 0 0.4 $ss
    done
done
echo "[GPU3 COMPLETE]"
) &

# GPU 4: Warhol (grid)
(
gen_baseline 4 warhol
for probe in text both; do
    for ss in 0.8 1.0 1.2; do
        run_artist 4 warhol $probe anchor_inpaint 0 0.4 $ss
    done
done
echo "[GPU4 COMPLETE]"
) &

# GPU 5: Hopper (grid)
(
gen_baseline 5 hopper
for probe in text both; do
    for ss in 0.8 1.0 1.2; do
        run_artist 5 hopper $probe anchor_inpaint 0 0.4 $ss
    done
done
echo "[GPU5 COMPLETE]"
) &

# GPU 6-7: hybrid mode for top artists
(
for artist in vangogh picasso monet; do
    run_artist 6 $artist text hybrid 0 0.4 1.0
    run_artist 6 $artist both hybrid 0 0.4 1.0
done
echo "[GPU6 COMPLETE]"
) &

(
for artist in rembrandt warhol hopper; do
    run_artist 7 $artist text hybrid 0 0.4 1.0
    run_artist 7 $artist both hybrid 0 0.4 1.0
done
echo "[GPU7 COMPLETE]"
) &

wait
echo "=== Artist Style COMPLETE $(date) ==="
