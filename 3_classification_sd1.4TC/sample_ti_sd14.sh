# 1) <clothed>만 사용 (긍정에 넣기)
python sample_ti_sd14.py \
  --clothed_embed_dir work_dirs/ti_clothed \
  --prompt "a portrait of a person, <clothed>, studio lighting" \
  --negative "<nudity>, nude, naked" \
  --steps 30 --cfg 7.5 --out out_clothed.png \
  --use_adaptive_cfg --clothed_boost 0.2

# # 2) <clothed> + <nudity> (네거티브에 병행)
# python sample_ti_sd14.py \
#   --clothed_embed_dir work_dirs/ti_clothed \
#   --nudity_embed_dir  work_dirs/ti_nudity \
#   --prompt "a portrait of a person, <clothed>, studio lighting" \
#   --negative "<nudity>, nude, naked" \
#   --steps 30 --cfg 7.5 --out out_both.png \
#   --use_adaptive_cfg --clothed_boost 0.2
