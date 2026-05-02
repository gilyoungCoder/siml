export default async function handler(req, res) {
  res.setHeader('Cache-Control', 'no-store');
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SECRET_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) {
    return res.status(500).json({ error: 'Missing SUPABASE_URL or SUPABASE_SECRET_KEY on Vercel.' });
  }
  const endpoint = `${url.replace(/\/$/, '')}/rest/v1/human_agreement_annotations?select=annotator_code,item_id,concept,label,batch_id,response_ms,created_at&order=created_at.desc&limit=10000`;
  const r = await fetch(endpoint, { headers: { apikey: key, Authorization: `Bearer ${key}` } });
  const text = await r.text();
  if (!r.ok) return res.status(r.status).send(text);
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  return res.status(200).send(text);
}
