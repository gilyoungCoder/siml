# Human Agreement Survey for Appendix Table 16

This is a low-cost static survey app for the Qwen3-VL-8B human-majority agreement study. It is designed for **Vercel static hosting + Supabase free-tier storage**.

## What it does

- Shows one generated image at a time with a concept-specific four-class rubric.
- Each browser/nickname receives 80 random items, sampled as 10 per concept from the 400-item pool, and resumes locally from the last unfinished item.
- Human annotator chooses exactly one of `Full`, `Partial`, `Safe`, `NotRelevant`.
- Browser writes only the human response to Supabase.
- Public manifest **does not include Qwen labels**; model labels are kept in `data/qwen_labels_private.json` for offline analysis.
- `scripts/analyze_results.py` joins exported human votes with the private Qwen labels and prints Table 16 exact-match / within-1 rows.

## Current data status

Prepared from:

`CAS_SpatialCFG/outputs/launch_0425_sdxl_lightning_human_eval`

Current manifest contains **400 items**: 50 each for 8 concepts. The `sexual` concept pool includes selected positionmap examples labeled by Qwen3-VL v5.

All labels are Qwen3-VL v5 eval outputs. `disturbing` is intentionally a balanced split built from existing MJA disturbing generations (baseline for Full, SAFREE/ours for Safe/Partial, and collapsed/noisy ours outputs for NotRelevant) because the original 100-image `mja_disturbing` set was too Full-heavy for a human agreement survey. Rerun data prep after changing sources:

```bash
cd /mnt/home3/yhgil99/unlearning
python3 human_agreement_survey/scripts/prepare_manifest.py
```

## Production deployment

Production URL: https://humanagreementsurvey.vercel.app

Dev viewer with Qwen labels: https://humanagreementsurvey.vercel.app/dev.html

## Local preview

```bash
cd /mnt/home3/yhgil99/unlearning/human_agreement_survey
cp public/config.example.js public/config.js
# Fill public/config.js with Supabase URL + anon key before real submissions.
python3 -m http.server 5173 -d public
# open http://<server>:5173
```

Without a valid Supabase config the app loads but blocks annotation start.

## Supabase setup (free tier)

1. Create a Supabase account/project.
2. In **SQL Editor**, run `supabase/schema.sql` from this folder.
3. In **Project Settings → API**, copy:
   - Project URL
   - `anon public` key
4. Create `public/config.js` from `public/config.example.js`:

```js
window.SURVEY_CONFIG = {
  supabaseUrl: "https://YOUR-PROJECT.supabase.co",
  supabaseAnonKey: "YOUR_SUPABASE_ANON_KEY",
  studyId: "qwen3vl_human_agreement_v1"
};
```

RLS policy allows public browser clients to `INSERT` annotations only. It intentionally does **not** allow public `SELECT`, so annotators cannot download study responses.

## Vercel deployment (free tier)

Use CLI-only deployment; do **not** use Git integration for this project. Ensure `public/config.js` exists before deploy; `scripts/deploy_vercel_cli.sh` writes it from `.key`.

Minimal Vercel build alternative:

```bash
cat > public/config.js <<EOF2
window.SURVEY_CONFIG = {
  supabaseUrl: "$SUPABASE_URL",
  supabaseAnonKey: "$SUPABASE_ANON_KEY",
  studyId: "qwen3vl_human_agreement_v1"
};
EOF2
```

Then set Vercel env vars `SUPABASE_URL` and `SUPABASE_ANON_KEY`.

## Annotation assignment

The manifest assigns 20 round-robin batches (`b01`–`b20`). Recommended:

- Ask each annotator to enter a name/nickname. The browser assigns 80 random items on first start and stores progress locally for that nickname.
- Assign each annotator either `All available items` or a batch.
- For the paper protocol (~20 annotators over all 800 images), use **All available items** if workload is acceptable; otherwise use batches and collect enough overlapping votes per image for majority vote.

## Export and Table 16 analysis

After collection:

1. Supabase Table Editor → `human_agreement_annotations` → Export CSV.
2. Put CSV in this folder, e.g. `responses.csv`.
3. Run:

```bash
cd /mnt/home3/yhgil99/unlearning/human_agreement_survey
python3 scripts/analyze_results.py responses.csv --min-votes 3
```

Output gives per-concept and pooled:

- exact-match agreement: Qwen label equals human majority label
- within-1 accuracy: Qwen label is at most one rubric step away from human majority under `NotRelevant < Safe < Partial < Full`

## Files

- `public/index.html` — static survey UI.
- `public/data/items.json` — public stimuli manifest, no Qwen labels.
- `public/assets/` — compressed survey images.
- `data/qwen_labels_private.json` — private model labels for analysis; do not publish if you want to minimize leakage.
- `data/prepare_report.json` — data-prep audit report.
- `scripts/prepare_manifest.py` — rebuilds manifest/assets from CAS_SpatialCFG outputs.
- `scripts/analyze_results.py` — computes Table 16 metrics from Supabase CSV export.
- `supabase/schema.sql` — table and RLS policy.

## CLI-only Vercel deployment (no Git)

This project is intended to be deployed from the command line, not through Git integration.

1. Keep credentials in `human_agreement_survey/.key` (already gitignored, chmod 600).
2. Fill `SUPABASE_URL` in `.key` after checking Supabase Project Settings → API.
3. Install Vercel CLI once if needed:

```bash
npm i -g vercel
```

4. Deploy directly from the folder:

```bash
cd /mnt/home3/yhgil99/unlearning/human_agreement_survey
./scripts/deploy_vercel_cli.sh
```

The script writes `public/config.js` locally from `.key`, then runs `vercel deploy --prod --yes --token "$VERCEL_TOKEN"`. The Vercel AI Gateway key is not a deploy token; create an Account Token in Vercel and set `VERCEL_TOKEN` in `.key`.

## Supabase CLI reference for this project

Project URL/ref now stored in `.key`:

- `SUPABASE_URL=https://ndmxewvxyqqlvpbcjasq.supabase.co`
- `SUPABASE_PROJECT_REF=ndmxewvxyqqlvpbcjasq`

If you have Supabase CLI installed locally, use CLI-only setup:

```bash
cd /mnt/home3/yhgil99/unlearning/human_agreement_survey
supabase login
supabase init   # already represented by supabase/config.toml here; safe if CLI wants to refresh files
supabase link --project-ref ndmxewvxyqqlvpbcjasq
supabase db push
```

If `supabase db push` does not pick up `supabase/schema.sql`, paste `supabase/schema.sql` into Supabase SQL Editor once. The web app only needs the table/RLS schema applied; it does not need Git integration.
