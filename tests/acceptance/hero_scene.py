"""The hero scene the acceptance probes read, named once instead of pasted as literals.

These identify a specific scene in a specific database - the one the numbers in README.md
were measured on. They are constants rather than UUID literals scattered through three
files so that pointing the probes at your own scene is one environment variable, not a
search-and-replace:

    HERO_PROJECT_ID=<uuid> HERO_SCENE_ID=<uuid> python3 tests/acceptance/probe_hero_numbers.py

A scene id that does not exist in the database the probe is pointed at makes every request
404; that is a configuration mistake, not a failed assertion, and the probes say so.
"""

from __future__ import annotations

import os

#: The workspace (project) the hero scene belongs to.
HERO_PROJECT_ID = os.environ.get("HERO_PROJECT_ID", "b54210ab-5935-4935-ad5f-bc76a4a054d1")

#: The scene itself - a 15 fps walkthrough, 17 detected objects.
HERO_SCENE_ID = os.environ.get("HERO_SCENE_ID", "7ccaa75d-cc66-4d09-9182-d6810c2f341f")

#: Where the probes look for a running stack. The dev stack, never production on :8000.
DEFAULT_API_URL = os.environ.get("HERO_API_URL", "http://127.0.0.1:8010")
DEFAULT_WEB_URL = os.environ.get("HERO_WEB_URL", "http://127.0.0.1:5173")
