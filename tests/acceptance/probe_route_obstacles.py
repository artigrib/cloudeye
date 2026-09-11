#!/usr/bin/env python3
"""THE ROUTE GATE: no planned route may cross a cell the layer calls an obstacle.

    OBSTACLE cells on any planned route = 0   -> hard fail
    UNKNOWN  cells on a route               -> allowed, counted, printed

Settled by the owner on 2026-09-10 (docs/DECISIONS.md). Unknown stays traversable at 1.5x
because blocking it measures 0/17 reachable for every robot on hero-74 - 43% of that grid
was never directly observed, so a robot forbidden from crossing unknown cannot leave the
pocket it starts in. Obstacles are the line that does not move.

WHAT THIS ACTUALLY CATCHES, since a planner cannot route through infinite cost by
construction. Two grids are involved and they must stay the same grid:

  * the INFLATED cost grid a route is planned on, and
  * the tri-state grid `GET /map?robot_id=...` serves, which is what a user looks at.

`obstacle_cells_on_route` counts the first against the second. That pair is exactly what
came apart under gpu/stage_occupancy.py's 0.1-0.5 m density histogram, which could not see
a table top at 0.9 m: hero-74 drew 54 (burger) / 64 (go2) / 31 (husky) route cells straight
through furniture, and own_0901_161054 drew route through it too. Nothing in the UI said
so. This is the assertion that would have.

No browser. Reads the API only, writes nothing.

    start the dev stack (docs/INSTALL.md) - the dev API on :8010, never prod :8000
    uv run python tests/acceptance/probe_route_obstacles.py
    uv run python tests/acceptance/probe_route_obstacles.py --scene <scene-uuid>   # just one
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_API = "http://127.0.0.1:8010"
# Every registered platform, so the gate covers the whole radius/height range rather than
# the two the demo happens to record. app/robots.py is the source; these are its ids.
PLATFORMS = ["burger", "limo", "waffle_pi", "turtlebot4", "jackal", "go2", "husky"]


def get(url: str, timeout: float = 180.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:300].decode(errors="replace")
    except Exception as e:  # noqa: BLE001 - a probe reports, it does not raise
        return -1, repr(e)[:300]


def done_scenes(api: str) -> list[str]:
    """Every `done` scene, through the API rather than the DB.

    There is no scene-list route, but a project detail carries its scenes with their
    status, so projects -> detail is the whole catalogue. Going through the API keeps this
    probe runnable against any deployment without knowing its database URL - and it is the
    same surface the thing under test uses.
    """
    projects: list[dict] = []
    skip, page = 0, 200          # 200 is the endpoint's own ceiling, so page, don't guess
    while True:
        # include_archived, or the gate quietly covers a third of the fleet: 28 of this
        # deployment's 38 projects are archived and they hold 26 of the 40 done scenes.
        # A gate that skips most of its subjects and still prints PASS is worse than none.
        code, body = get(f"{api}/api/projects?skip={skip}&limit={page}&include_archived=true")
        if code != 200 or not isinstance(body, dict):
            print(f"could not list projects: HTTP {code} {str(body)[:160]}")
            return []
        items = body.get("items", [])
        projects += items
        skip += len(items)
        if len(items) < page or skip >= body.get("total", skip):
            break
    out: list[str] = []
    for project in projects:
        code, detail = get(f"{api}/api/projects/{project['id']}")
        if code != 200 or not isinstance(detail, dict):
            continue
        out += [s["id"] for s in detail.get("scenes", []) if s.get("status") == "done"]
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--scene", action="append", default=None,
                    help="scene uuid; repeatable. Default: every done scene.")
    ap.add_argument("--platforms", nargs="+", default=PLATFORMS)
    a = ap.parse_args()

    scenes = a.scene or done_scenes(a.api)
    if not scenes:
        print("no done scenes found - is the dev API up? (start the dev stack (docs/INSTALL.md) - the dev API on :8010, never prod :8000)")
        return 1

    print(f"route gate: {len(scenes)} scene(s) x {len(a.platforms)} platform(s) "
          f"against {a.api}\n")

    failures: list[str] = []
    non_200: list[str] = []
    no_layer = 0
    checked = 0
    worst_unknown = (0, "")

    for sid in scenes:
        for rid in a.platforms:
            code, body = get(f"{a.api}/api/scenes/{sid}/reachability?robot_id={rid}")
            if code != 200 or not isinstance(body, dict):
                non_200.append(f"{sid[:8]} {rid}: HTTP {code} {str(body)[:120]}")
                continue
            if not body.get("layer_available", True):
                no_layer += 1
                continue
            checked += 1
            obstacles = body.get("obstacle_cells_on_route", 0)
            unknown = body.get("unknown_cells_on_route", 0)
            reach = len(body.get("reachable_object_ids", []))
            if obstacles:
                failures.append(
                    f"{sid[:8]} {rid}: {obstacles} route cell(s) cross OBSTACLE "
                    f"({reach} reachable)"
                )
            if unknown > worst_unknown[0]:
                worst_unknown = (unknown, f"{sid[:8]} {rid} ({reach} reachable)")

    print(f"answered with a verdict : {checked}")
    print(f"answered 'layer not available' : {no_layer}")
    print(f"non-200 : {len(non_200)}")
    for line in non_200[:10]:
        print(f"    {line}")
    print(f"\nmost unknown cells on one platform's routes: {worst_unknown[0]}"
          f"{' - ' + worst_unknown[1] if worst_unknown[1] else ''}"
          "   (reported, not a bar)")
    print(f"\nOBSTACLE cells on a route: {len(failures)} platform(s) over 0")
    for line in failures[:20]:
        print(f"    {line}")

    # A non-200 is its own failure: the contract is 200-with-a-verdict or
    # 200-with-layer-not-available, and a probe that only counted obstacles would pass
    # while every scene 500'd.
    ok = not failures and not non_200
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
