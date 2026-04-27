export default async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  if (req.method !== 'POST') return res.status(405).json({ error: 'POST only' });
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SECRET_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) return res.status(500).json({ error: 'Missing Supabase server credentials.' });
  const allowed = new Set(['Full', 'Partial', 'Safe', 'NotRelevant']);
  const p = req.body || {};
  if (!allowed.has(p.label)) return res.status(400).json({ error: 'Invalid label' });
  if (!p.annotator_code || String(p.annotator_code).length > 80) return res.status(400).json({ error: 'Invalid name/nickname' });
  if (!p.item_id || !p.concept) return res.status(400).json({ error: 'Missing item_id or concept' });
  const row = {
    study_id: p.study_id || 'qwen3vl_human_agreement_v1',
    annotator_code: String(p.annotator_code).slice(0, 80),
    item_id: p.item_id,
    concept: p.concept,
    label: p.label,
    batch_id: p.batch_id || null,
    response_ms: Number.isFinite(p.response_ms) ? p.response_ms : null,
    user_agent: p.user_agent || '',
    client_meta: p.client_meta || {}
  };
  const endpoint = `${url.replace(/\/$/, '')}/rest/v1/human_agreement_annotations?on_conflict=study_id,annotator_code,item_id`;
  const r = await fetch(endpoint, {
    method: 'POST',
    headers: {
      apikey: key,
      Authorization: `Bearer ${key}`,
      'Content-Type': 'application/json',
      Prefer: 'resolution=merge-duplicates,return=minimal'
    },
    body: JSON.stringify(row)
  });
  const text = await r.text();
  if (!r.ok) return res.status(r.status).send(text);
  return res.status(200).json({ ok: true });
}
