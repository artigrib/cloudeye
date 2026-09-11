#!/usr/bin/env python3
"""Shot 6 acceptance: the "Download for Isaac Sim" button must deliver the nvblox USD.

Takes 1 and 2 both recorded this shot against production on :8000 - a hand-started vite
defaults its proxy there - and downloaded prod's 15,409,996-byte legacy collision export
instead. This checks the bytes that actually land on disk from a real browser download
event through the :5173 proxy, and compares them to `<scene_dir>/usd/nvblox.usd` by md5.

    start the dev stack (docs/INSTALL.md) - the dev API on :8010, never prod :8000
    xvfb-run -a uv run --with playwright python -u tests/acceptance/probe_shot6_export.py --headless
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, tempfile
from pathlib import Path

DEFAULT_BASE_URL = "http://127.0.0.1:5173"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hero_scene import HERO_PROJECT_ID as DEFAULT_PROJECT_ID, HERO_SCENE_ID as DEFAULT_SCENE_ID  # noqa: E402
EXPECTED_BYTES = 3_669_032
LEGACY_PROD_BYTES = 15_409_996


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    ap.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    ap.add_argument("--upload-dir", default=os.environ.get("UPLOAD_DIR", "var/uploads"))
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args(argv)

    from playwright.sync_api import sync_playwright

    reference = Path(a.upload_dir) / "scenes" / a.scene_id / "usd" / "nvblox.usd"
    ref_md5 = md5(reference) if reference.is_file() else None
    ref_size = reference.stat().st_size if reference.is_file() else None

    url = f"{a.base_url}/workspaces/{a.project_id}/scenes/{a.scene_id}"
    variant = None

    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        browser = p.chromium.launch(headless=a.headless)
        ctx = browser.new_context(viewport={"width": 1400, "height": 900}, accept_downloads=True)
        ctx.add_init_script("localStorage.setItem('cloudeye:captureGuideSeen', '1')")
        page = ctx.new_page()

        def on_response(r):
            nonlocal variant
            if "/usd" in r.url and r.headers.get("x-usd-variant"):
                variant = r.headers["x-usd-variant"]

        page.on("response", on_response)
        page.goto(url)
        page.wait_for_timeout(4000)
        # The export is a link on the toolbar now - no disclosure to open first. Located
        # by testid, not by its text: the text was "Download for Isaac Sim" when this
        # probe was written and has been "Export to Isaac Sim" since the toolbar was
        # re-composed, so the by-name lookup had been timing out for 30 s and failing this
        # gate on a label rather than on the file. The comment above had already been
        # updated for that re-layout; the selector under it had not.
        with page.expect_download(timeout=180000) as dl:
            page.locator('[data-testid="download-usd"]').click()
        download = dl.value
        saved = Path(tmp) / download.suggested_filename
        download.save_as(str(saved))
        got_size, got_md5 = saved.stat().st_size, md5(saved)
        name = download.suggested_filename
        browser.close()

    print(f"скачанное имя:        {name}")
    print(f"скачано байт:         {got_size:,}   (ожидается {EXPECTED_BYTES:,})")
    print(f"md5 скачанного:       {got_md5}")
    print(f"md5 usd/nvblox.usd:   {ref_md5}   ({ref_size:,} байт)" if ref_md5 else "  эталон не найден")
    print(f"X-Usd-Variant:        {variant}")
    if got_size == LEGACY_PROD_BYTES:
        print("  ВНИМАНИЕ: это прод-экспорт с :8000, запись шла мимо dev-API")

    ok = got_size == EXPECTED_BYTES and ref_md5 is not None and got_md5 == ref_md5 and variant == "nvblox"

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"filename": name, "bytes": got_size, "md5": got_md5,
                       "reference_md5": ref_md5, "reference_bytes": ref_size,
                       "usd_variant": variant, "ok": ok}, f, indent=1, ensure_ascii=False)
        print(f"json -> {a.json_out}")

    print("ВЕРДИКТ:", "ПРОЙДЕНО" if ok else "НЕ ПРОЙДЕНО")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
