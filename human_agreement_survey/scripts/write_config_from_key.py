#!/usr/bin/env python3
from pathlib import Path
import os, shlex

root = Path(__file__).resolve().parents[1]
key_path = root / ".key"
vals = {}
if key_path.exists():
    for line in key_path.read_text().splitlines():
        line=line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k,v=line.split("=",1)
        vals[k]=shlex.split(v)[0] if v else ""
url = os.environ.get("SUPABASE_URL") or vals.get("SUPABASE_URL", "")
pub = os.environ.get("SUPABASE_PUBLISHABLE_KEY") or vals.get("SUPABASE_PUBLISHABLE_KEY", "")
if not url or not pub:
    raise SystemExit("SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY are required. Fill .key first.")
(root / "public" / "config.js").write_text(f'''window.SURVEY_CONFIG = {{\n  supabaseUrl: "{url}",\n  supabaseAnonKey: "{pub}",\n  studyId: "qwen3vl_human_agreement_v1"\n}};\n''')
print("wrote public/config.js")
