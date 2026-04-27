-- Human agreement survey storage. Run this in Supabase SQL Editor or via scripts/apply_supabase_schema.py.
create table if not exists public.human_agreement_annotations (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  study_id text not null default 'qwen3vl_human_agreement_v1',
  annotator_code text not null,
  item_id text not null,
  concept text not null,
  label text not null check (label in ('Full','Partial','Safe','NotRelevant')),
  response_ms integer,
  batch_id text,
  user_agent text,
  client_meta jsonb default '{}'::jsonb,
  unique (study_id, annotator_code, item_id)
);

alter table public.human_agreement_annotations enable row level security;

drop policy if exists "public_insert_annotations" on public.human_agreement_annotations;
create policy "public_insert_annotations"
on public.human_agreement_annotations
for insert
to anon
with check (
  study_id = 'qwen3vl_human_agreement_v1'
  and label in ('Full','Partial','Safe','NotRelevant')
  and length(annotator_code) between 1 and 80
);

-- Duplicate submissions are rejected by the unique constraint; the browser treats duplicates as already saved.
-- No SELECT policy on purpose: browser clients can submit but cannot read annotations.
