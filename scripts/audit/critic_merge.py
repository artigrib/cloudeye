"""Scene critic, part 3: merges critic_rules.json (scripts/audit/critic_rules.py) +
critic_vlm.json (scripts/audit/critic_vlm.py) into one `critic.json` (machine-
readable) and `<n>_critic.md` (human-readable), for one scene. Report-only, same
guarantee as the other two modules - reads its two input JSON files and writes only
the two outputs it's asked to write.

"Overlap" between a rule finding and a VLM finding (task spec: "same object id, or
same region within ~0.5 m, and compatible issue text") is computed textually, not
geometrically: VLM findings are free-text JSON with no coordinates attached (the
model was asked for `object_id_or_region`/`issue` strings, not pixel or world
coordinates - matching the task's own output schema), so "same region within ~0.5 m"
isn't literally computable from a VLM response without a separate text-to-geometry
grounding step this task doesn't ask for. What IS computed, and documented here so
the overlap count means something concrete rather than an invented number:
  - "same object id": the VLM's `object_id_or_region` string contains the rule
    finding's `object_id` (case-insensitive), OR
  - "same region + compatible issue text": the VLM string contains the rule
    finding's object CLASS/label (e.g. "bed" for bed_0) AND the VLM's `issue` text
    contains at least one keyword from CHECK_KEYWORDS for that rule check (e.g.
    "yaw"/"rotat"/"diagonal"/"angle" for `yaw_vs_wall`).
See `find_overlaps()`.

**The rules engine is the gate; the VLM is advisory only** (PM decision,
2026-09-07). Every finding critic_rules.py produces carries `"source": "rule"`,
`"unverified": False`; every finding critic_vlm.py produces carries `"source":
"vlm"`, `"unverified": True` - see critic_rules.py's module docstring. This
module doesn't change that tagging, but reports it prominently: `merge_critic`'s
output carries a `vlm_advisory` block, and `render_markdown` states corroboration
in the PM's own wording, e.g. "23 VLM findings, 1 corroborated" - "corroborated"
here means the SAME thing as `overlap.n_overlap`, just named for what it actually
tells a reader (a rule finding the VLM independently also raised), not a generic
count.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CHECK_KEYWORDS: dict[str, list[str]] = {
    "yaw_vs_wall": ["yaw", "rotat", "diagonal", "orient", "angle", "tilt", "skew", "sideways", "crooked"],
    "bbox_vs_class_limit": ["size", "large", "big", "oversiz", "long", "huge", "stretch"],
    "centroid_outside_room": ["outside", "beyond", "wall", "outdoor", "exterior", "off the", "out of"],
    "asset_centroid_vs_hull": ["shift", "offset", "misplace", "position", "mismatch", "moved", "wrong place"],
    "missing_support_reason": ["float", "support", "hover", "unsupported", "suspend"],
}


def _norm(s: Any) -> str:
    return str(s or "").strip().lower()


def find_overlaps(
    rule_findings: list[dict[str, Any]],
    vlm_findings: list[dict[str, Any]],
    objects_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    overlaps = []
    for rf in rule_findings:
        rf_id = _norm(rf.get("object_id"))
        label = _norm(objects_by_id.get(rf.get("object_id"), {}).get("label"))
        keywords = CHECK_KEYWORDS.get(rf.get("check", ""), [])
        for vf in vlm_findings:
            region = _norm(vf.get("object_id_or_region"))
            issue = _norm(vf.get("issue"))
            if not region:
                continue
            id_match = bool(rf_id) and rf_id in region
            label_match = bool(label) and label in region
            keyword_match = any(k in issue for k in keywords)
            if id_match or (label_match and keyword_match):
                overlaps.append(
                    {
                        "rule_finding": rf,
                        "vlm_finding": vf,
                        "match_basis": "object_id" if id_match else "label+keyword",
                    }
                )
    return overlaps


def diagonal_bed_acceptance(rule_findings: list[dict[str, Any]], vlm_findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Task acceptance criterion: "the diagonal-bed case is flagged by BOTH the
    rules (the >15 deg yaw-vs-wall check) and the VLM. If the VLM misses it, say so
    plainly and show its raw output." Looks specifically for a bed_0 (or any
    "bed"-labelled object) `yaw_vs_wall` rule finding, and any VLM finding whose
    region mentions "bed" and whose issue text is orientation-flavored."""
    bed_rule_findings = [
        f for f in rule_findings if f.get("check") == "yaw_vs_wall" and "bed" in _norm(f.get("object_id"))
    ]
    yaw_keywords = CHECK_KEYWORDS["yaw_vs_wall"]
    bed_vlm_findings = [
        f
        for f in vlm_findings
        if "bed" in _norm(f.get("object_id_or_region")) and any(k in _norm(f.get("issue")) for k in yaw_keywords)
    ]
    return {
        "rules_flagged": bool(bed_rule_findings),
        "vlm_flagged": bool(bed_vlm_findings),
        "both_flagged": bool(bed_rule_findings) and bool(bed_vlm_findings),
        "rule_findings": bed_rule_findings,
        "vlm_findings": bed_vlm_findings,
    }


def merge_critic(
    *,
    scene_label: str,
    rules_result: dict[str, Any],
    vlm_result: dict[str, Any] | None,
    objects_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rule_findings = rules_result.get("findings", [])
    vlm_findings = vlm_result.get("findings", []) if vlm_result else []
    overlaps = find_overlaps(rule_findings, vlm_findings, objects_by_id)
    return {
        "scene": scene_label,
        "rules": rules_result,
        "vlm": vlm_result,
        "overlap": {
            "n_overlap": len(overlaps),
            "pairs": overlaps,
            "definition": (
                "same object id (VLM region string contains the rule finding's object_id), OR "
                "same object class + compatible issue keywords (see CHECK_KEYWORDS in critic_merge.py) "
                "- not a literal ~0.5 m geometric distance, since VLM findings carry no coordinates."
            ),
        },
        "diagonal_bed_acceptance": diagonal_bed_acceptance(rule_findings, vlm_findings),
        "summary": {
            "n_rule_findings": len(rule_findings),
            "n_vlm_findings": len(vlm_findings),
            "n_overlap": len(overlaps),
        },
        # PM decision (2026-09-07): make "rules gate, VLM is advisory only"
        # explicit and machine-readable, not just implied by the source/unverified
        # tags already on every individual finding (critic_rules.py's docstring).
        "vlm_advisory": {
            "gates": False,
            "note": "VLM findings are advisory only and never gate; only rule findings (source=rule, unverified=False) do.",
            "corroboration_sentence": f"{len(vlm_findings)} VLM findings, {len(overlaps)} corroborated",
        },
    }


def render_markdown(critic: dict[str, Any]) -> str:
    lines = [f"# Scene critic: {critic['scene']}", ""]
    rules = critic["rules"]
    vlm = critic.get("vlm")
    summary = critic["summary"]
    advisory = critic.get("vlm_advisory")
    lines += [
        "## Summary",
        "",
        "**The rules engine is the gate; the VLM is advisory only and never gates.**",
        "",
        f"- Rule findings (gating): {summary['n_rule_findings']}",
    ]
    if advisory:
        lines.append(f"- **{advisory['corroboration_sentence']}** (a VLM finding independently matching a rule finding - see `overlap.definition` in critic.json; not gating either way)")
    else:
        lines.append(f"- VLM findings: {summary['n_vlm_findings']} (corroborated: {summary['n_overlap']})")
    lines += [
        f"- Objects checked: {rules.get('n_objects')} (source: {rules.get('objects_source')})",
        f"- GLB used: `{rules.get('glb_used')}`",
    ]
    if vlm:
        lines.append(f"- VLM model: `{vlm.get('model')}`, temperature={vlm.get('temperature')}, seed={vlm.get('seed')}, pairs={vlm.get('n_pairs')}")
    lines.append("")

    if rules.get("notes"):
        lines += ["## Notes (rules)", ""]
        lines += [f"- {n}" for n in rules["notes"]]
        lines.append("")

    lines += ["## Rule findings", ""]
    if rule_findings := rules.get("findings"):
        lines.append("| check | object_id | severity | measured | threshold | detail |")
        lines.append("|---|---|---|---|---|---|")
        for f in rule_findings:
            detail = str(f.get("detail", "")).replace("|", "\\|")
            lines.append(f"| {f['check']} | {f['object_id']} | {f['severity']} | {f.get('measured')} | {f.get('threshold')} | {detail} |")
    else:
        lines.append("(none)")
    lines.append("")

    lines += ["## VLM findings (advisory only - unverified, never gating)", ""]
    if vlm and vlm.get("findings"):
        lines.append("| unverified | view | object_id_or_region | severity | issue |")
        lines.append("|---|---|---|---|---|")
        for f in vlm["findings"]:
            issue = str(f.get("issue", "")).replace("|", "\\|")
            lines.append(f"| {f.get('unverified', True)} | {f.get('view')} | {f.get('object_id_or_region')} | {f.get('severity')} | {issue} |")
    elif vlm:
        lines.append("(none)")
    else:
        lines.append("(VLM pass not run for this scene)")
    lines.append("")

    dba = critic["diagonal_bed_acceptance"]
    lines += ["## Diagonal-bed acceptance check", ""]
    lines.append(f"- Rules flagged: **{dba['rules_flagged']}**")
    lines.append(f"- VLM flagged: **{dba['vlm_flagged']}**")
    lines.append(f"- Both flagged: **{dba['both_flagged']}**")
    if dba["rule_findings"]:
        lines.append("")
        lines.append("Rule finding(s):")
        for f in dba["rule_findings"]:
            lines.append(f"- `{f['object_id']}`: {f['detail']}")
    if dba["vlm_findings"]:
        lines.append("")
        lines.append("VLM finding(s):")
        for f in dba["vlm_findings"]:
            lines.append(f"- ({f.get('view')}) `{f.get('object_id_or_region')}`: {f.get('issue')}")
    elif vlm is not None:
        lines.append("")
        lines.append("**VLM did not report a bed-orientation finding.** Raw VLM responses for every pair, for inspection:")
        for call in vlm.get("calls", []):
            raw = call["raw_response_text"].strip()
            if raw.startswith("```"):
                # Model already fenced its own response as ```json ... ``` - strip
                # that outer fence so this doesn't nest one fence inside another
                # (renders oddly in markdown).
                raw = raw.strip("`").removeprefix("json").strip()
            lines.append(
                f"\n<details><summary>{call['view']} (n_findings={call['n_findings']}, error={call['error']})</summary>\n\n```\n{raw}\n```\n</details>"
            )
    lines.append("")

    if critic["overlap"]["pairs"]:
        lines += ["## Corroboration detail (rule findings the VLM independently also raised)", ""]
        for pair in critic["overlap"]["pairs"]:
            rf, vf = pair["rule_finding"], pair["vlm_finding"]
            lines.append(f"- **{rf['object_id']}** ({rf['check']}, {pair['match_basis']}): rule says \"{rf['detail']}\"; VLM ({vf.get('view')}) says \"{vf.get('issue')}\"")
        lines.append("")

    return "\n".join(lines)


def write_critic(
    *,
    scene_label: str,
    out_dir: Path,
    rules_json_path: Path,
    vlm_json_path: Path | None,
    progress_md_path: Path,
    progress_json_path: Path,
) -> dict[str, Any]:
    rules_result = json.loads(Path(rules_json_path).read_text())
    vlm_result = json.loads(Path(vlm_json_path).read_text()) if vlm_json_path and Path(vlm_json_path).is_file() else None

    objects_json_path = Path(out_dir) / "objects.json"
    objects_by_id = {}
    if objects_json_path.is_file():
        objects_json = json.loads(objects_json_path.read_text())
        objects_by_id = {o["id"]: o for o in objects_json.get("objects", []) if "id" in o}

    critic = merge_critic(scene_label=scene_label, rules_result=rules_result, vlm_result=vlm_result, objects_by_id=objects_by_id)

    critic_json_path = Path(out_dir) / "critic.json"
    critic_json_path.write_text(json.dumps(critic, indent=2))
    Path(progress_json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(progress_json_path).write_text(json.dumps(critic, indent=2))

    markdown = render_markdown(critic)
    Path(progress_md_path).parent.mkdir(parents=True, exist_ok=True)
    Path(progress_md_path).write_text(markdown)

    return critic


def combine_scenes(scenes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Combines several already-merged per-scene critic dicts (each from
    `merge_critic`/`write_critic`) into one top-level report - used for the task's
    `progress/17_critic.md` + `progress/17_critic.json`, which cover every scene the
    critic was run against in one place, alongside each scene's own single-scene
    `critic.json` living in its own out dir (task spec: "in the scene's out dir and
    copied to progress/17_critic.json"). `scenes` keys are short scene labels (e.g.
    "hero", "rejected")."""
    return {
        "scenes": scenes,
        "totals": {
            "n_rule_findings": sum(s["summary"]["n_rule_findings"] for s in scenes.values()),
            "n_vlm_findings": sum(s["summary"]["n_vlm_findings"] for s in scenes.values()),
            "n_overlap": sum(s["summary"]["n_overlap"] for s in scenes.values()),
        },
    }


def render_combined_markdown(combined: dict[str, Any]) -> str:
    lines = [
        "# Scene critic report",
        "",
        "**The rules engine is the gate; the VLM is advisory only and never gates.**",
        "",
        "## Totals across all scenes",
        "",
    ]
    totals = combined["totals"]
    lines.append(f"- Rule findings (gating): {totals['n_rule_findings']}")
    lines.append(f"- **{totals['n_vlm_findings']} VLM findings, {totals['n_overlap']} corroborated** (advisory only, not gating)")
    lines.append("")
    for label, critic in combined["scenes"].items():
        lines.append(f"---\n")
        lines.append(render_markdown(critic))
        lines.append("")
    return "\n".join(lines)


def write_combined_critic(scenes: dict[str, dict[str, Any]], *, progress_md_path: Path, progress_json_path: Path) -> dict[str, Any]:
    combined = combine_scenes(scenes)
    Path(progress_json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(progress_json_path).write_text(json.dumps(combined, indent=2))
    Path(progress_md_path).parent.mkdir(parents=True, exist_ok=True)
    Path(progress_md_path).write_text(render_combined_markdown(combined))
    return combined


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("scene_label")
    parser.add_argument("--rules-json", type=Path, default=None)
    parser.add_argument("--vlm-json", type=Path, default=None)
    parser.add_argument("--progress-md", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path, required=True)
    args = parser.parse_args()
    rules_json = args.rules_json or (args.out_dir / "critic_rules.json")
    vlm_json = args.vlm_json or (args.out_dir / "critic_vlm.json")
    critic = write_critic(
        scene_label=args.scene_label,
        out_dir=args.out_dir,
        rules_json_path=rules_json,
        vlm_json_path=vlm_json,
        progress_md_path=args.progress_md,
        progress_json_path=args.progress_json,
    )
    print(f"wrote {args.out_dir / 'critic.json'} and {args.progress_md}: {critic['summary']}")


if __name__ == "__main__":
    _main()
