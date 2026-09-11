"""Unit tests for scripts/audit/critic_merge.py."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.audit import critic_merge as cm


def _rule_finding(check="yaw_vs_wall", object_id="bed_0", **kwargs):
    base = {"check": check, "object_id": object_id, "severity": "high", "measured": 25.0, "threshold": 15.0, "detail": "bed yaw off"}
    base.update(kwargs)
    return base


def _vlm_finding(region="bed_0", issue="the bed looks rotated diagonally", **kwargs):
    base = {"object_id_or_region": region, "issue": issue, "severity": "medium", "view": "view_000"}
    base.update(kwargs)
    return base


class TestFindOverlaps:
    def test_exact_object_id_match(self):
        overlaps = cm.find_overlaps([_rule_finding()], [_vlm_finding(region="bed_0")], {})
        assert len(overlaps) == 1
        assert overlaps[0]["match_basis"] == "object_id"

    def test_label_plus_keyword_match_without_exact_id(self):
        objects_by_id = {"bed_0": {"label": "bed"}}
        vlm = _vlm_finding(region="the bed near the window", issue="it appears rotated at an angle")
        overlaps = cm.find_overlaps([_rule_finding()], [vlm], objects_by_id)
        assert len(overlaps) == 1
        assert overlaps[0]["match_basis"] == "label+keyword"

    def test_label_present_but_no_keyword_is_not_an_overlap(self):
        objects_by_id = {"bed_0": {"label": "bed"}}
        vlm = _vlm_finding(region="the bed", issue="the color looks off")
        assert cm.find_overlaps([_rule_finding()], [vlm], objects_by_id) == []

    def test_unrelated_finding_is_not_an_overlap(self):
        vlm = _vlm_finding(region="television_0", issue="tv is missing")
        assert cm.find_overlaps([_rule_finding()], [vlm], {}) == []

    def test_no_findings_yields_no_overlaps(self):
        assert cm.find_overlaps([], [], {}) == []


class TestDiagonalBedAcceptance:
    def test_both_flagged(self):
        result = cm.diagonal_bed_acceptance(
            [_rule_finding(check="yaw_vs_wall", object_id="bed_0")],
            [_vlm_finding(region="bed_0", issue="bed appears rotated diagonally")],
        )
        assert result["both_flagged"] is True
        assert result["rules_flagged"] is True
        assert result["vlm_flagged"] is True

    def test_vlm_misses_it(self):
        result = cm.diagonal_bed_acceptance(
            [_rule_finding(check="yaw_vs_wall", object_id="bed_0")],
            [_vlm_finding(region="lamp_0", issue="lamp missing")],
        )
        assert result["rules_flagged"] is True
        assert result["vlm_flagged"] is False
        assert result["both_flagged"] is False

    def test_non_yaw_rule_finding_does_not_count(self):
        result = cm.diagonal_bed_acceptance(
            [_rule_finding(check="bbox_vs_class_limit", object_id="bed_0")],
            [_vlm_finding(region="bed_0", issue="bed too big")],
        )
        assert result["rules_flagged"] is False


class TestMergeCriticAndMarkdown:
    def test_merge_produces_summary_and_markdown_renders_without_error(self):
        rules_result = {
            "out_dir": "/tmp/x",
            "objects_source": "objects.json",
            "n_objects": 1,
            "glb_used": "/tmp/x/scene.glb",
            "notes": ["a note"],
            "findings": [_rule_finding()],
            "n_findings": 1,
        }
        vlm_result = {
            "model": "google/gemma-4-31b-it",
            "temperature": 0.0,
            "seed": 0,
            "n_pairs": 1,
            "pairs": [],
            "calls": [{"view": "view_000", "prompt_hash": "abc", "usage": None, "error": None, "n_findings": 1, "raw_response_text": "{}"}],
            "findings": [_vlm_finding()],
            "n_findings": 1,
        }
        objects_by_id = {"bed_0": {"label": "bed"}}

        critic = cm.merge_critic(scene_label="test-scene", rules_result=rules_result, vlm_result=vlm_result, objects_by_id=objects_by_id)

        assert critic["summary"] == {"n_rule_findings": 1, "n_vlm_findings": 1, "n_overlap": 1}
        assert critic["diagonal_bed_acceptance"]["both_flagged"] is True

        markdown = cm.render_markdown(critic)
        assert "test-scene" in markdown
        assert "bed_0" in markdown
        assert "Both flagged: **True**" in markdown

    def test_vlm_advisory_block_and_corroboration_sentence(self):
        """PM decision (2026-09-07): rules gate, VLM is advisory only - critic.json
        must say so explicitly (`vlm_advisory`), and the markdown must state
        corroboration in the PM's own wording, e.g. "23 VLM findings, 1
        corroborated"."""
        rules_result = {
            "out_dir": "/tmp/x", "objects_source": "objects.json", "n_objects": 1, "glb_used": None, "notes": [],
            "findings": [_rule_finding()], "n_findings": 1,
        }
        vlm_result = {
            "model": "google/gemma-4-31b-it", "temperature": 0.0, "seed": 0, "n_pairs": 2, "pairs": [], "calls": [],
            "findings": [_vlm_finding(region="bed_0", issue="bed rotated diagonally"), _vlm_finding(region="lamp_0", issue="lamp missing")],
            "n_findings": 2,
        }
        critic = cm.merge_critic(
            scene_label="hero", rules_result=rules_result, vlm_result=vlm_result, objects_by_id={"bed_0": {"label": "bed"}}
        )

        assert critic["vlm_advisory"]["gates"] is False
        assert critic["vlm_advisory"]["corroboration_sentence"] == "2 VLM findings, 1 corroborated"

        markdown = cm.render_markdown(critic)
        assert "2 VLM findings, 1 corroborated" in markdown
        assert "rules engine is the gate" in markdown
        assert "advisory only" in markdown

    def test_corroboration_sentence_matches_pm_example_numbers(self):
        # PM's own numbers for the hero: "23 VLM findings, 1 corroborated".
        rules_result = {
            "out_dir": "/tmp/x", "objects_source": "objects.json", "n_objects": 1, "glb_used": None, "notes": [],
            "findings": [_rule_finding()], "n_findings": 1,
        }
        vlm_findings = [_vlm_finding(region="bed_0", issue="bed rotated diagonally")] + [
            _vlm_finding(region=f"other_{i}", issue="something else") for i in range(22)
        ]
        vlm_result = {"model": "m", "temperature": 0.0, "seed": 0, "n_pairs": 7, "pairs": [], "calls": [], "findings": vlm_findings, "n_findings": 23}
        critic = cm.merge_critic(
            scene_label="hero", rules_result=rules_result, vlm_result=vlm_result, objects_by_id={"bed_0": {"label": "bed"}}
        )
        assert critic["vlm_advisory"]["corroboration_sentence"] == "23 VLM findings, 1 corroborated"

    def test_merge_with_no_vlm_result(self):
        rules_result = {"out_dir": "/tmp/x", "objects_source": "objects.json", "n_objects": 0, "glb_used": None, "notes": [], "findings": [], "n_findings": 0}
        critic = cm.merge_critic(scene_label="no-vlm", rules_result=rules_result, vlm_result=None, objects_by_id={})
        assert critic["vlm"] is None
        assert critic["summary"]["n_vlm_findings"] == 0
        markdown = cm.render_markdown(critic)
        assert "VLM pass not run" in markdown

    def test_markdown_shows_raw_vlm_output_when_bed_is_missed(self):
        rules_result = {
            "out_dir": "/tmp/x", "objects_source": "objects.json", "n_objects": 1, "glb_used": None, "notes": [],
            "findings": [_rule_finding()], "n_findings": 1,
        }
        vlm_result = {
            "model": "google/gemma-4-31b-it", "temperature": 0.0, "seed": 0, "n_pairs": 1, "pairs": [],
            "calls": [{"view": "view_000", "prompt_hash": "abc", "usage": None, "error": None, "n_findings": 0, "raw_response_text": '{"findings": []}'}],
            "findings": [], "n_findings": 0,
        }
        critic = cm.merge_critic(scene_label="miss", rules_result=rules_result, vlm_result=vlm_result, objects_by_id={"bed_0": {"label": "bed"}})
        markdown = cm.render_markdown(critic)
        assert "VLM did not report a bed-orientation finding" in markdown
        assert '{"findings": []}' in markdown


class TestWriteCritic:
    def test_writes_critic_json_and_markdown_and_progress_copy(self, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "objects.json").write_text(json.dumps({"objects": [{"id": "bed_0", "label": "bed"}]}))
        rules_json = out_dir / "critic_rules.json"
        rules_json.write_text(
            json.dumps(
                {
                    "out_dir": str(out_dir), "objects_source": "objects.json", "n_objects": 1,
                    "glb_used": None, "notes": [], "findings": [_rule_finding()], "n_findings": 1,
                }
            )
        )
        vlm_json = out_dir / "critic_vlm.json"
        vlm_json.write_text(
            json.dumps(
                {
                    "model": "google/gemma-4-31b-it", "temperature": 0.0, "seed": 0, "n_pairs": 1, "pairs": [],
                    "calls": [], "findings": [_vlm_finding()], "n_findings": 1,
                }
            )
        )
        progress_md = tmp_path / "progress" / "17_critic.md"
        progress_json = tmp_path / "progress" / "17_critic.json"

        critic = cm.write_critic(
            scene_label="hero", out_dir=out_dir, rules_json_path=rules_json, vlm_json_path=vlm_json,
            progress_md_path=progress_md, progress_json_path=progress_json,
        )

        assert (out_dir / "critic.json").is_file()
        assert progress_md.is_file()
        assert progress_json.is_file()
        assert json.loads((out_dir / "critic.json").read_text()) == critic
        assert json.loads(progress_json.read_text()) == critic

class TestCombineScenes:
    def test_totals_sum_across_scenes(self):
        rules_result = {"out_dir": "/tmp/a", "objects_source": "objects.json", "n_objects": 1, "glb_used": None, "notes": [], "findings": [_rule_finding()], "n_findings": 1}
        vlm_result = {"model": "m", "temperature": 0.0, "seed": 0, "n_pairs": 1, "pairs": [], "calls": [], "findings": [_vlm_finding()], "n_findings": 1}
        hero = cm.merge_critic(scene_label="hero", rules_result=rules_result, vlm_result=vlm_result, objects_by_id={"bed_0": {"label": "bed"}})
        rejected = cm.merge_critic(scene_label="rejected", rules_result=rules_result, vlm_result=None, objects_by_id={})

        combined = cm.combine_scenes({"hero": hero, "rejected": rejected})

        assert combined["totals"]["n_rule_findings"] == 2
        assert combined["totals"]["n_vlm_findings"] == 1
        assert combined["totals"]["n_overlap"] == 1

    def test_combined_markdown_includes_both_scenes(self):
        rules_result = {"out_dir": "/tmp/a", "objects_source": "objects.json", "n_objects": 1, "glb_used": None, "notes": [], "findings": [], "n_findings": 0}
        hero = cm.merge_critic(scene_label="hero-scene", rules_result=rules_result, vlm_result=None, objects_by_id={})
        rejected = cm.merge_critic(scene_label="rejected-scene", rules_result=rules_result, vlm_result=None, objects_by_id={})

        markdown = cm.render_combined_markdown(cm.combine_scenes({"hero": hero, "rejected": rejected}))

        assert "hero-scene" in markdown
        assert "rejected-scene" in markdown

    def test_write_combined_critic_writes_both_files(self, tmp_path):
        rules_result = {"out_dir": "/tmp/a", "objects_source": "objects.json", "n_objects": 0, "glb_used": None, "notes": [], "findings": [], "n_findings": 0}
        hero = cm.merge_critic(scene_label="hero", rules_result=rules_result, vlm_result=None, objects_by_id={})

        progress_md = tmp_path / "17_critic.md"
        progress_json = tmp_path / "17_critic.json"
        combined = cm.write_combined_critic({"hero": hero}, progress_md_path=progress_md, progress_json_path=progress_json)

        assert progress_md.is_file()
        assert progress_json.is_file()
        assert json.loads(progress_json.read_text()) == combined


class TestWriteCriticMissingVlm:
    def test_missing_vlm_json_degrades_gracefully(self, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        rules_json = out_dir / "critic_rules.json"
        rules_json.write_text(json.dumps({"out_dir": str(out_dir), "objects_source": "none", "n_objects": 0, "glb_used": None, "notes": [], "findings": [], "n_findings": 0}))

        critic = cm.write_critic(
            scene_label="rejected", out_dir=out_dir, rules_json_path=rules_json, vlm_json_path=out_dir / "does_not_exist.json",
            progress_md_path=tmp_path / "17_critic.md", progress_json_path=tmp_path / "17_critic.json",
        )

        assert critic["vlm"] is None
