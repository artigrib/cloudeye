#!/usr/bin/env python3
"""The hero scene's six acceptance numbers, read out of a real browser's DOM.

The numbers are the contract for every re-layout of the scene screen: they describe the
room and the robots, not the design, so a layout change that moves them is a layout
change that broke something.

    TurtleBot3 Burger  13 of 17 reachable   (picker, and the Compare all row)
    Unitree Go2        13 of 17 reachable   (picker, and the Compare all row)
    Husky A200         10 of 17 reachable   (picker, and the Compare all row)
    goto desk, TurtleBot   total_length_m 3.922
    goto desk, Husky       total_length_m 3.505
    Download for Isaac Sim 3 669 032 B, md5 0f568f4e...   (tests/acceptance/probe_shot6_export.py)

Everything except the last line is checked here; the export has its own probe because it
needs a real download event and a byte comparison against the file on disk.

Read from the DOM, not from a screenshot: a picture cannot distinguish 13/17 from 13/17
recorded before the fix, and cannot be diffed. The route lengths are additionally
cross-checked against the command row the API recorded, because the panel rounds to two
decimals and 3.922 vs 3.925 must not both read as "3.92 m".

    start the dev stack (docs/INSTALL.md) - the dev API on :8010, never prod :8000
    xvfb-run -a uv run --with playwright python -u tests/acceptance/probe_hero_numbers.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:5173"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hero_scene import HERO_PROJECT_ID as PROJECT_ID, HERO_SCENE_ID as SCENE_ID  # noqa: E402

# Re-baselined 2026-09-10 when navigation moved to the band layer (layers/backfill/) -
# every moved number, old -> new, with the reason:
#
#   go2 verdict/compare  13 -> 10   BY DESIGN. 534 cells the old 0.1-0.5 m histogram called
#                                   UNKNOWN the band calls OBSTACLE; inflated by go2's
#                                   0.2496 m (5 cells of dilation against burger's 2) they
#                                   seal the corridor to the bed. Not the robot's height:
#                                   0.1847 -> 0.40 m moves this count by zero.
#   burger goto desk  3.922 -> 4.066  a detour, +0.144 m: the band adds table tops the
#                                   histogram could not see, and the route goes around one.
#   husky  goto desk  3.505 -> 2.198  the START moved, not the route. Husky's start is
#                                   1.58 m from the camera anchor now (status "moved") -
#                                   at 0.5528 m nothing near the anchor has clearance once
#                                   tall surfaces are obstacles - and it moved TOWARD the
#                                   desk. Start displacement 1.58 m, route change 1.31 m:
#                                   same order, same sign, and the drop is smaller than the
#                                   displacement, which is what a partly-constrained path
#                                   does. Still a real route, not the 0.37 m stub shot 5
#                                   once recorded. (2.198 for one run in between, while the
#                                   command path was still planning on the DEFAULT
#                                   platform's map - see the one-route-one-number check
#                                   below, which is what caught it.)
EXPECT_VERDICT = {"burger": "13 of 17", "go2": "10 of 17", "husky": "10 of 17"}
EXPECT_COMPARE = {"burger": "13/17", "go2": "10/17", "husky": "10/17"}
EXPECT_ROUTE_M = {"burger": 4.066, "husky": 2.216}
ROUTE_TOLERANCE_M = 0.01

# Step 1's honest footnote: the reachable count above is partly carried by ground the scan
# never covered, and the bar says by how much - IN METRES (cells x the grid's resolution,
# one decimal), because "139 cells" is a fact about the grid and nobody reading a verdict
# about a robot knows what a cell is worth. Unknown is crossed at 1.5x rather than
# blocked - blocking it measures 0/17 for EVERY robot on this scene - so this number is
# reporting, not a bar; what is asserted is that it is PRESENT and agrees with the API.
EXPECT_UNOBSERVED_SHOWN = {"burger": True, "go2": True, "husky": True}
GOTO_OBJECT = "desk"

# The verdict's bottleneck comes from the fit-probability audit's corridor_width_p5_m and
# from nowhere else. Husky A200 fits in 0 of 100 trials on this scene, so there is no
# successful trial to take a percentile over and the audit writes null - which must reach
# the screen as the shared formatter's em dash, not as "0.00 m".
EXPECT_GAP = {"burger": "tightest gap 0.28 m", "husky": "tightest gap —"}

# Exactly the strings app/services/pathfinding.py can put on the wire. A row may print
# one of these verbatim and nothing else - no paraphrase, and above all no invented gap
# measurement, since none is taken on this path.
BACKEND_REASONS = {
    "outside_grid",
    "disconnected",
    "no_free_cell_within_2m",
    "no_visible_cell_within_2m",
    "robot_does_not_fit",
}


def api(base_url: str, path: str):
    with urllib.request.urlopen(f"{base_url}/api{path}") as r:
        return json.load(r)


def latest_command(base_url: str) -> dict:
    """The newest command row the backend recorded for this scene, whatever the UI
    happens to be rendering - the authoritative length for the tolerance check."""
    res = api(base_url, f"/scenes/{SCENE_ID}/commands?skip=0&limit=200")
    items = res["items"] if isinstance(res, dict) else res
    return max(items, key=lambda c: c["created_at"]) if items else {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--settle", type=float, default=30.0)
    ap.add_argument("--after-click", type=float, default=14.0)
    ap.add_argument("--headed", action="store_true")
    a = ap.parse_args(argv)

    from playwright.sync_api import sync_playwright

    failures: list[str] = []
    observed: dict[str, object] = {}

    def check(name: str, got, want) -> None:
        observed[name] = got
        if got != want:
            failures.append(f"{name}: got {got!r}, expected {want!r}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not a.headed)
        ctx = browser.new_context(viewport={"width": 1400, "height": 900})
        ctx.add_init_script("localStorage.setItem('cloudeye:captureGuideSeen', '1')")
        page = ctx.new_page()
        errs: list[str] = []
        page.on("pageerror", lambda e: errs.append(str(e)[:160]))
        page.goto(f"{a.base_url}/workspaces/{PROJECT_ID}/scenes/{SCENE_ID}")
        page.wait_for_timeout(int(a.settle * 1000))

        def verdict() -> str:
            return (page.locator('[data-testid="verdict-count"]').first.inner_text() or "").strip()

        def check_unobserved(platform_id: str) -> None:
            """"via N unobserved cells" is on screen, and N is the API's own number.

            Asserted against the API rather than against a constant, because N is
            reporting - how much of this answer crosses ground nobody scanned - and a
            constant would turn a footnote into a bar. What must not happen is the bar
            silently going missing, or printing a number the response does not carry.
            """
            want_shown = EXPECT_UNOBSERVED_SHOWN[platform_id]
            el = page.locator('[data-testid="unobserved-cells"]')
            shown = el.count() > 0
            check(f"unobserved bar shown {platform_id}", shown, want_shown)
            if not shown:
                return
            api_resp = api(a.base_url, f"/scenes/{SCENE_ID}/reachability?robot_id={platform_id}")
            n = api_resp["unknown_cells_on_route"]
            # The bar prints METRES; the API carries cells. The conversion is the grid's
            # own resolution and nothing else, so the probe does it the same way rather
            # than hard-coding a string - a cell count is a fact about the grid, and a
            # verdict about a robot has to be in units somebody can picture.
            res = api(a.base_url, f"/scenes/{SCENE_ID}/map?robot_id={platform_id}")["resolution"]
            check(f"unobserved metres {platform_id}",
                  (el.first.inner_text() or "").strip(),
                  f"via {n * res:.1f} m unobserved")
            # The gate the owner set: obstacles on a route are a hard fail, unknown is not.
            check(f"obstacle cells on route {platform_id}",
                  api_resp["obstacle_cells_on_route"], 0)

        def select_platform(display_name: str) -> None:
            page.locator('[aria-label="Robot platform"]').first.click()
            page.wait_for_timeout(300)
            page.get_by_role("option", name=re.compile(display_name, re.I)).first.click()
            page.wait_for_timeout(int(a.after_click * 1000))

        # --- the picker's two platforms -------------------------------------------
        check("picker default label", page.locator('[aria-label="Robot platform"]').first.inner_text().strip(),
              "TurtleBot3 Burger")
        check("verdict burger", verdict(), EXPECT_VERDICT["burger"])
        check("tightest gap burger",
              (page.locator('[data-testid="tightest-gap"]').first.inner_text() or "").strip(),
              EXPECT_GAP["burger"])
        check_unobserved("burger")

        # --- every platform, including the ones the picker no longer offers --------
        page.locator('[data-testid="compare-all"]').click()
        page.wait_for_timeout(int(a.after_click * 1000))
        for platform_id, want in EXPECT_COMPARE.items():
            row = page.locator(f'tr[data-platform-id="{platform_id}"] td').last
            check(f"compare {platform_id}", (row.inner_text() or "").strip(), want)
        page.locator('[data-testid="compare-all"]').click()
        page.wait_for_timeout(1500)

        # --- every platform in EXPECT_VERDICT, through the picker -------------------
        # Go2 is read here, not only off the Compare all row: the picker is what a user
        # actually drives, and "13 of 17, same as TurtleBot" is a fact about this room
        # that the product should be able to state, whatever one demo shot chose to show.
        select_platform("Unitree Go2")
        check("verdict go2", verdict(), EXPECT_VERDICT["go2"])
        check_unobserved("go2")

        # --- Husky ----------------------------------------------------------------
        select_platform("Husky A200")
        check("verdict husky", verdict(), EXPECT_VERDICT["husky"])
        check("tightest gap husky",
              (page.locator('[data-testid="tightest-gap"]').first.inner_text() or "").strip(),
              EXPECT_GAP["husky"])
        check_unobserved("husky")

        # --- per-row verdicts ------------------------------------------------------
        # Every unreachable row carries one of the backend's own labels and nothing else;
        # every reachable row carries a length that agrees with /reachability's
        # path_length_m for the same object. Same-named objects are collapsed into
        # groups, so every group is expanded first - otherwise this inspects the two or
        # three rows that happen to be singletons and calls that "every row".
        for _ in range(4):
            opened = page.evaluate("""() => {
              let n = 0
              for (const b of document.querySelectorAll('button[data-object-group]')) {
                if (b.getAttribute('aria-expanded') === 'false') { b.click(); n++ }
              }
              return n
            }""")
            page.wait_for_timeout(600)
            if not opened:
                break
        # `el` FIRST, then a descendant. The verdict used to be a child span of a
        # full-width row; on a chip it is an attribute of the chip itself. Reading only the
        # descendant made this whole cross-check pass while inspecting nothing - 17 rows,
        # 0 reasons, 0 lengths, PASS - which is why the two counts below are asserted.
        rows = page.evaluate("""() => {
          const attr = (el, name) =>
            el.getAttribute(name) ?? el.querySelector('[' + name + ']')?.getAttribute(name) ?? null
          return Array.from(document.querySelectorAll('[data-object-id]')).map(el => ({
            id: el.getAttribute('data-object-id'),
            reason: attr(el, 'data-object-reason'),
            length: attr(el, 'data-object-length'),
            text: (el.textContent || '').trim(),
          }))
        }""")
        # BY robot_id, not by radius. Those two stopped being interchangeable on
        # 2026-09-10: radius still decides inflation, but the GRID is now derived from the
        # layer's obstacle-height map at the platform's own roof, and a radius-only request
        # falls back to the DEFAULT platform's height. `?radius=0.5528` therefore answers
        # about Husky's width on a TurtleBot's map - close enough to look right (same
        # counts, same reasons) and wrong in the third decimal of every route length, which
        # is exactly how this probe caught it. HANDOFF's note that the two are
        # byte-identical was true before heights existed and is not any more.
        reach = api(a.base_url, f"/scenes/{SCENE_ID}/reachability?robot_id=husky")
        lengths = reach["path_length_m"]
        reasons = reach["unreachable_reasons"]
        observed["rows inspected"] = len(rows)
        if len(rows) != len(lengths) + len(reasons):
            failures.append(
                f"rows inspected {len(rows)}, expected {len(lengths) + len(reasons)} "
                f"({len(lengths)} reachable + {len(reasons)} unreachable)"
            )
        n_reason = sum(1 for r in rows if r["reason"])
        n_length = sum(1 for r in rows if r["length"])
        observed["rows with a backend reason"] = f"{n_reason} (API: {len(reasons)})"
        observed["rows with a route length"] = f"{n_length} (API: {len(lengths)})"
        # A cross-check that inspects nothing passes for free. These two say it inspected
        # what the API says is there.
        if n_reason != len(reasons):
            failures.append(f"{n_reason} rows carry a reason, the API gives {len(reasons)}")
        if n_length != len(lengths):
            failures.append(f"{n_length} rows carry a route length, the API gives {len(lengths)}")
        for row in rows:
            oid = row["id"]
            if row["reason"] is not None:
                if row["reason"] not in BACKEND_REASONS:
                    failures.append(f"row {oid}: reason {row['reason']!r} is not one the backend writes")
                if reasons.get(oid) != row["reason"]:
                    failures.append(f"row {oid}: shows {row['reason']!r}, API says {reasons.get(oid)!r}")
                # "no invented gap number": an unreachable row's text is the object name
                # and the label, and carries no measurement of its own.
                if re.search(r"\d+\.\d+\s*m", row["text"]):
                    failures.append(f"row {oid}: unreachable row carries a measurement: {row['text']!r}")
            elif row["length"] is not None and oid in lengths:
                if abs(float(row["length"]) - lengths[oid]) > 1e-6:
                    failures.append(f"row {oid}: length {row['length']} != API {lengths[oid]}")

        # --- the two routes -------------------------------------------------------
        # Planning always continues from wherever the robot stands, so each route is
        # preceded by a reset to the SELECTED platform's own start - this is the trap
        # that once turned Husky's 3.505 m into a 0.37 m stub (HANDOFF 5b).
        def route_for(display_name: str, platform_id: str) -> None:
            select_platform(display_name)
            page.locator('[data-testid="reset-robot"]').click()
            page.wait_for_timeout(3000)
            rows = page.locator("button[data-object-id]").filter(has_text=re.compile(GOTO_OBJECT, re.I))
            if rows.count() == 0:
                page.locator("button[data-object-group]").filter(
                    has_text=re.compile(GOTO_OBJECT, re.I)
                ).first.click()
                page.wait_for_timeout(1200)
                rows = page.locator("button[data-object-id]").filter(has_text=re.compile(GOTO_OBJECT, re.I))
            rows.first.dblclick(timeout=30000)
            page.wait_for_timeout(int(a.after_click * 1000))

            command = latest_command(a.base_url)
            length = command.get("total_length_m")
            observed[f"route {platform_id} total_length_m"] = length

            # ONE ROUTE, ONE NUMBER. The object list's row length and the command's own
            # total describe the same journey and must agree; `path_length_m`'s whole
            # contract is that the panel and the list never print two lengths for one
            # route. They came apart on 2026-09-10 - the list asked by robot_id and the
            # command did not, so the command planned on the default platform's map -
            # and every constant in this file still matched, because each was checked
            # against its own source. This compares the two sources to each other.
            api_len = api(a.base_url, f"/scenes/{SCENE_ID}/reachability?robot_id={platform_id}")
            desk_ids = [oid for oid, v in api_len["path_length_m"].items()
                        if abs(v - (length or -1)) <= ROUTE_TOLERANCE_M]
            observed[f"route {platform_id} agrees with the object list"] = bool(desk_ids)
            if not desk_ids:
                failures.append(
                    f"route {platform_id}: the command says {length} m but no object in "
                    f"/reachability?robot_id={platform_id} has that length - the list and "
                    f"the panel are describing two different routes"
                )

            want = EXPECT_ROUTE_M[platform_id]
            if length is None or abs(length - want) > ROUTE_TOLERANCE_M:
                failures.append(f"route {platform_id}: total_length_m {length}, expected {want} +/- {ROUTE_TOLERANCE_M}")
            if command.get("status") != "done":
                failures.append(f"route {platform_id}: command status {command.get('status')!r}, expected 'done'")
            # And the panel must be showing that same number, rounded the way the shared
            # formatter rounds it - a route the backend planned but the UI never drew is
            # exactly the failure the demo recorder was written to catch.
            shown = (page.locator('[data-testid="path-status"]').first.inner_text() or "").strip()
            observed[f"route {platform_id} path-status"] = shown
            if length is not None and f"{length:.2f} m" not in shown:
                failures.append(f"route {platform_id}: path status {shown!r} does not carry {length:.2f} m")

        route_for("TurtleBot3 Burger", "burger")
        route_for("Husky A200", "husky")

        observed["pageerrors"] = errs or "none"
        if errs:
            failures.append(f"pageerrors: {errs}")
        browser.close()

    for key, value in observed.items():
        print(f"  {key:34s} {value}")
    for f in failures:
        print(f"MISS: {f}")
    print(f"\n{'PASS' if not failures else 'FAIL'} - {len(failures)} mismatch(es)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
