# Operator TODO — Human Agreement Survey

## Done on siml-09

- Qwen3-VL v5 labels prepared for all 8 concepts.
- Balanced public survey manifest prepared: 800 images = 8 concepts × 100.
- `disturbing` split rebuilt from existing SAFREE/ours/baseline MJA disturbing generations instead of the Full-heavy original 100-image set.
- Supabase schema/RLS applied to project; REST insert target is available.
- Static app smoke-tested locally over HTTP.

## Deployment

- Production URL: https://humanagreementsurvey.vercel.app
- CLI-only deployment completed from siml-09.

## Remaining human/operator actions

1. Give annotators anonymous codes and content-warning instructions.
4. Export Supabase CSV after collection and run:

```bash
python3 scripts/analyze_results.py responses.csv --min-votes 3
```

5. Fill Appendix Table 16 / `table:human_study` with exact-match and within-1 values.
