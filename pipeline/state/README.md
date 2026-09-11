# pipeline/state

Small mutable facts the pipeline updates by itself, kept out of the code so a box change is
not a code change.

- `nvblox_box_id` — the instance `start_or_create()` tries first. Rewritten whenever a
  replacement box is created, so the next run starts from the box that last worked.
- `retired_box_ids` — ids that were replaced and are **candidates for the owner's `destroy`**.
  Nothing here is ever destroyed automatically; `destroy` is denied to the agent in
  `.claude/settings.json` and stays a human act.
