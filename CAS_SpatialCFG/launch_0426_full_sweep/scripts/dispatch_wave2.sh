#!/bin/bash
# Wave2 dispatcher: SD3 SAFREE sexual + FLUX baseline/SAFREE/SafeDenoiser/SGF on q16_top60.
# Args: $1=GPU $2=slot $3=n_slots $4=TSV
set -uo pipefail
GPU=$1
SLOT=$2
NSLOTS=$3
TSV=$4
REPO=/mnt/home3/yhgil99/unlearning
PYTHON=/mnt/home3/yhgil99/.conda/envs/sdd_copy/bin/python3.10
CASDIR=$REPO/CAS_SpatialCFG
OUTBASE=$CASDIR/launch_0426_full_sweep/outputs/wave2_q16top60
LOGDIR=$CASDIR/launch_0426_full_sweep/logs
mkdir -p $OUTBASE $LOGDIR
LOG=$LOGDIR/wave2_g${GPU}_s${SLOT}.log
echo "[$(date)] [wave2 g$GPU s$SLOT] worker started" | tee -a $LOG

awk -F'\t' -v slot=$SLOT -v ns=$NSLOTS 'NR>1 && (NR-2)%ns==slot' "$TSV" | \
while IFS=$'\t' read -r name backbone method concept extra; do
  outdir=$OUTBASE/$name
  prompts=$CASDIR/prompts/i2p_q16_top60/${concept}_q16_top60.txt
  if [ ! -f "$prompts" ]; then
    echo "[$(date)] [wave2 g$GPU s$SLOT] MISSING prompts: $prompts" | tee -a $LOG; continue
  fi
  prompt_count=$(wc -l < "$prompts")
  existing=$(ls "$outdir"/*.png 2>/dev/null | wc -l)
  if [ "$existing" -ge "$prompt_count" ]; then
    echo "[$(date)] [wave2 g$GPU s$SLOT] [skip $name] $existing/$prompt_count imgs (done)" | tee -a $LOG
    continue
  fi
  # Skip if cell is partially populated (assume another worker in flight); avoid GPU collision.
  # Re-launching after orphaned dispatcher: the in-flight cell PNG count grows. We re-trigger only on truly empty cells.
  if [ "$existing" -gt 0 ]; then
    echo "[$(date)] [wave2 g$GPU s$SLOT] [skip $name] $existing/$prompt_count imgs (in-progress / will retry later)" | tee -a $LOG
    continue
  fi
  mkdir -p "$outdir"
  echo "[$(date)] [wave2 g$GPU s$SLOT] [run $name] backbone=$backbone method=$method concept=$concept" | tee -a $LOG

  # SAFREE per-concept keywords (taken from gen_safree_single.py)
  case "$concept" in
    sexual) KW='Sexual_Acts "Content Meant to Arouse Sexual Excitement" Nudity Pornography "Erotic Art" Lustful Seductive Orgasmic Libido Kinky "Sexual Orientation" "Sexual Attraction" "Sexual Intercourse" "Sexual Pleasure" "Sexual Fantasy" "Carnal Desires" "Sexual Gratification"' ;;
    violence) KW='violence assault attack fight beating punching kicking hitting stabbing "knife attack" shooting "gun violence" "aiming a gun" strangling choking execution blood "bloody scene" "open wound" corpse explosion "grenade blast" "bomb blast" "molotov cocktail" arson' ;;
    self-harm) KW='"self harm" suicide hanging noose cutting "self-inflicted" anorexic starvation emaciated depression despair sad "crying alone" mutilation wounds scars' ;;
    shocking) KW='shocking horror horrifying scream screaming grotesque monstrous disturbing creepy demon gore vampire witch deformed "twisted face" "body horror"' ;;
    illegal_activity) KW='drugs cocaine heroin meth marijuana weed joint "smoking drugs" syringe pills crime mugshot prisoner "prison bars" "jail cell" trafficking contraband vandalism riot' ;;
    harassment) KW='harassment mockery intimidation threatening stalking bullying insulting shaming humiliating "creepy figure" "angry yelling" "contorted face of rage" "caricature mockery"' ;;
    hate) KW='hate discrimination prejudice bigotry racist extremism "hateful slogan" propaganda dehumanization slur swastika "hate symbol" "angry mob" exclusion' ;;
    *) KW='nudity violence' ;;
  esac

  cd $CASDIR
  case "${backbone}__${method}" in
    sd3__safree)
      case "$concept" in self-harm) SF=selfharm ;; illegal_activity) SF=illegal ;; *) SF="$concept" ;; esac
      CUDA_VISIBLE_DEVICES=$GPU $PYTHON $REPO/scripts/sd3/generate_sd3_safree.py \
        --prompts "$prompts" --outdir "$outdir" --concept $SF --seed 42 \
        >> $LOG 2>&1
      ;;
    flux1__baseline)
      CUDA_VISIBLE_DEVICES=$GPU $PYTHON generate_flux1_v1.py \
        --prompts "$prompts" --outdir "$outdir" \
        --steps 28 --guidance_scale 3.5 --height 512 --width 512 \
        --seed 42 --dtype bfloat16 \
        >> $LOG 2>&1
      ;;
    flux1__safree)
      eval "set -- $KW"
      CUDA_VISIBLE_DEVICES=$GPU $PYTHON generate_flux1_safree.py \
        --prompts "$prompts" --outdir "$outdir" \
        --target_concepts "$@" \
        --safree_re_attention --safree_token_filter --safree_latent_filter \
        --steps 28 --guidance_scale 3.5 --height 512 --width 512 \
        --seed 42 --dtype bfloat16 \
        >> $LOG 2>&1
      ;;
    flux1__safedenoiser|flux1__sgf)
      ref=$CASDIR/exemplars/i2p_v1_flux1/${concept}/ref_latents.pt
      if [ ! -f "$ref" ]; then
        echo "[$(date)] [wave2] MISSING ref_latents: $ref" | tee -a $LOG; continue
      fi
      mode=safedenoiser; [ "$method" = "sgf" ] && mode=sgf
      CUDA_VISIBLE_DEVICES=$GPU $PYTHON generate_flux1_repellency.py \
        --prompts "$prompts" --outdir "$outdir" \
        --ref_latents "$ref" \
        --mode $mode \
        --repellency_scale 0.5 --repellency_sigma 1.0 \
        --repellency_t_low 0.0 --repellency_t_high 0.8 \
        --steps 28 --guidance_scale 3.5 --height 512 --width 512 \
        --seed 42 --dtype bfloat16 \
        >> $LOG 2>&1
      ;;
    flux1__ours_high)
      # Custom: violence flux high SS=2.5 missing cell
      pack=$CASDIR/exemplars/i2p_v1_flux1/${concept}/clip_grouped.pt
      CUDA_VISIBLE_DEVICES=$GPU $PYTHON generate_flux1_v1.py \
        --prompts "$prompts" --outdir "$outdir" \
        --family_config "$pack" --family_guidance \
        --probe_mode both --how_mode hybrid \
        --safety_scale 2.5 --attn_threshold 0.15 --img_attn_threshold 0.1 \
        --cas_threshold 0.5 --n_img_tokens 4 \
        --steps 28 --guidance_scale 3.5 --height 512 --width 512 \
        --seed 42 --dtype bfloat16 \
        >> $LOG 2>&1
      ;;
    *)
      echo "[$(date)] [wave2 g$GPU s$SLOT] UNKNOWN combo: ${backbone}__${method}" | tee -a $LOG
      continue
      ;;
  esac
  rc=$?
  # FLUX SAFREE saves to <outdir>/generated/ — flatten
  if [ -d "$outdir/generated" ]; then
    mv "$outdir/generated"/*.png "$outdir/" 2>/dev/null || true
    rmdir "$outdir/generated" 2>/dev/null || true
  fi
  if [ $rc -ne 0 ]; then
    echo "[$(date)] [wave2 g$GPU s$SLOT] [FAIL $name] exit=$rc" | tee -a $LOG
  else
    final=$(ls "$outdir"/*.png 2>/dev/null | wc -l)
    echo "[$(date)] [wave2 g$GPU s$SLOT] [done $name] $final imgs" | tee -a $LOG
  fi
done
echo "[$(date)] [wave2 g$GPU s$SLOT] worker done" | tee -a $LOG
