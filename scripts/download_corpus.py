"""
Download the USCIS immigration-form corpus (form PDFs + their instructions).

Two pathways are covered, since they share several forms (I-485, I-765, I-131):
  - Family-based green card  (I-130 -> I-485 chain)
  - Employment-based green card (I-140 -> I-485 chain)

USCIS serves these as static PDFs at a stable URL pattern:
    https://www.uscis.gov/sites/default/files/document/forms/<slug>.pdf
where the instruction booklet is "<form>instr". The instruction booklets carry
the procedural prose ("submit Form I-693 with your application", eligibility
conditions, filing locations) that the typed-relation extractor turns into
`requires` / `filed_with` / `references` edges.

Run:
    .venv/Scripts/python.exe scripts/download_corpus.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

BASE = "https://www.uscis.gov/sites/default/files/document/forms"
OUT = Path(__file__).resolve().parent.parent / "data" / "sources" / "immigration"

# slug -> human label + pathway tags. We download both the blank form and, when
# it exists, the instruction booklet (<slug>instr). Instructions are the priority.
FORMS: dict[str, dict] = {
    # ── Shared adjustment-of-status core ─────────────────────────────────────
    "i-485":  {"label": "Application to Register Permanent Residence or Adjust Status", "pathways": ["family", "employment"]},
    "i-693":  {"label": "Report of Medical Examination and Vaccination Record",          "pathways": ["family", "employment"]},
    "i-765":  {"label": "Application for Employment Authorization (EAD)",                 "pathways": ["family", "employment"]},
    "i-131":  {"label": "Application for Travel Document (Advance Parole)",               "pathways": ["family", "employment"]},
    "i-864":  {"label": "Affidavit of Support Under Section 213A",                        "pathways": ["family", "employment"]},

    # ── Family-based petition ────────────────────────────────────────────────
    "i-130":  {"label": "Petition for Alien Relative",                                    "pathways": ["family"]},
    "i-130a": {"label": "Supplemental Information for Spouse Beneficiary",                "pathways": ["family"], "no_instr": True},

    # ── Employment-based petition ────────────────────────────────────────────
    "i-140":  {"label": "Immigrant Petition for Alien Worker",                            "pathways": ["employment"]},
    "i-907":  {"label": "Request for Premium Processing Service",                         "pathways": ["employment"]},
    "i-129":  {"label": "Petition for a Nonimmigrant Worker (H-1B etc.)",                 "pathways": ["employment"]},
}

UA = {"User-Agent": "Mozilla/5.0 (compatible; GraphRAG-corpus-fetch/1.0)"}


def fetch(slug: str, client: httpx.Client) -> bool:
    url = f"{BASE}/{slug}.pdf"
    dest = OUT / f"{slug}.pdf"
    try:
        r = client.get(url, headers=UA, follow_redirects=True, timeout=60)
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {slug:12} network error: {e}")
        return False
    ctype = r.headers.get("content-type", "")
    if r.status_code == 200 and "pdf" in ctype.lower():
        dest.write_bytes(r.content)
        print(f"  [ok] {slug:12} {len(r.content):>8,} bytes")
        return True
    print(f"  [FAIL] {slug:12} HTTP {r.status_code} ({ctype})")
    return False


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Downloading USCIS corpus -> {OUT}\n")
    ok = 0
    total = 0
    with httpx.Client() as client:
        for slug, meta in FORMS.items():
            print(f"{slug}  ({', '.join(meta['pathways'])}) — {meta['label']}")
            total += 1
            ok += fetch(slug, client)
            if not meta.get("no_instr"):
                total += 1
                ok += fetch(f"{slug}instr", client)
            time.sleep(0.4)  # be polite to USCIS
    print(f"\nDone: {ok}/{total} files downloaded into {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
