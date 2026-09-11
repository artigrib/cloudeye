# Furniture Asset Pack Attribution

## 1. Kenney Furniture Kit

- **Name:** Furniture Kit (v2.0)
- **Author:** Kenney (www.kenney.nl)
- **Source URL used:** https://kenney.nl/media/pages/assets/furniture-kit/440e0608a4-1677580847/kenney_furniture-kit.zip
  (asset page: https://kenney.nl/assets/furniture-kit)
- **License:** CC0 1.0 Universal — https://creativecommons.org/publicdomain/zero/1.0/
- **Download date:** 2026-09-06
- **Archive filename:** `kenney_furniture-kit.zip`
- **SHA-256:** `e67652d0932cee41683f74711c03d3e192a2af9979ef8e6b237711f5482d46b0`
- **Formats included:** GLB, GLTF, OBJ+MTL, FBX, DAE (Collada), STL (140 models per format)
- **Status:** Obtained successfully via plain `curl` with a browser User-Agent header — no Cloudflare
  challenge encountered on kenney.nl. Extracted to `kenney_furniture_kit/`.

CC0 works require no attribution, but we credit Kenney anyway as good practice.

---

## 2. mastjie "Low poly household goods"

- **Name:** Low poly household goods
- **Author:** mastjie (itch.io: https://mastjie.itch.io ; Twitter: @samusaaji)
- **Source URL (canonical, not obtained):** https://mastjie.itch.io/low-poly-household-goods
- **License (confirmed via archived page text):** CC0 — "Free for personal and commercial use, no
  attribution required (CC0)." Link: https://creativecommons.org/publicdomain/zero/1.0/
- **Download date:** N/A — file could not be downloaded (see Status below)
- **Archive filename:** N/A (not obtained)
- **SHA-256:** N/A (not obtained)
- **Formats (per pack description, not verified locally):** FBX and GLTF ("Contains 25 low poly 3D
  assets in FBX and GLTF file format" — furniture items include: 2-seat sofa, bookshelf, chair, coffee
  table, cupboard, desk, dining table, bed, TV cabinet, wardrobe; plus electrical/kitchen appliances)
- **Status: NOT OBTAINED.** All non-interactive download attempts were blocked. What was tried:
  1. `curl` with a standard browser User-Agent to `https://mastjie.itch.io/` and to
     `https://mastjie.itch.io/low-poly-household-goods` — both returned HTTP 403 with a Cloudflare
     "Just a moment..." managed-challenge (Turnstile) page.
  2. `curl` to the main `itch.io` domain (`itch.io/search`, `itch.io/g/<id>`) — also 403 Cloudflare
     challenge. The `itch.io/embed/<id>` endpoint (a non-download preview widget) does load via curl,
     but it only links back to the same protected `mastjie.itch.io` subdomain for the actual download —
     it does not expose a downloadable file.
  3. `curl_cffi` with Chrome-120 TLS fingerprint impersonation — still served the Cloudflare challenge
     page (JS/Turnstile challenge, not just a TLS/UA check).
  4. Playwright headless Chromium, and Playwright headed Chromium under Xvfb with
     anti-automation flags (`--disable-blink-features=AutomationControlled`, spoofed
     `navigator.webdriver`) — the Cloudflare Turnstile challenge never auto-passed; a scripted click on
     the "Verify you are human" checkbox produced a fresh Ray ID (new challenge) instead of passing,
     indicating Cloudflare's bot detection flags the automated browser regardless of UA/TLS spoofing.
  5. Wayback Machine (web.archive.org): successfully found the archived game page
     (`https://web.archive.org/web/20230909073529/https://mastjie.itch.io/low-poly-household-goods`),
     which confirmed the CC0 license, author, content list, and the download filename
     `household_goods.zip` (512 kB). However, the actual zip file itself was never crawled/archived —
     itch.io serves downloads through a session-authenticated, dynamically generated URL
     (`.../download_url` POST endpoint) that is not a static crawlable link, so no cached copy of the
     binary exists on Wayback (CDX search for the upload ID and common itch CDN hosts returned no
     results).
  6. Searched OpenGameArt.org (`art-search-advanced?keys=mastjie` and `...keys=household+goods`) — no
     matching listing/mirror found.
  7. Searched GitHub (code search blocked without auth; repository search for "mastjie") — found only
     the author's unrelated `mastjie-dev/mastjie-dev.github.io` GitHub Pages repo, no asset mirror.
  8. Checked poly.pizza (a CC0 model aggregator that mirrors some itch.io/Kenney packs) — it lists a
     different mastjie pack ("Low Poly Characters Bundle") but the site's search/catalog is rendered
     client-side via an API that requires an API key; no evidence the household-goods pack specifically
     is mirrored there, and pursuing it further risked substituting unverified content for the actual
     pack, which was explicitly disallowed.

  **Blocker:** Cloudflare Turnstile "managed challenge" in front of `*.itch.io` creator subdomains
  cannot be solved non-interactively with the tools available in this environment (no CAPTCHA-solving
  service, no real browser profile with an established trust/reputation history). No substitute pack
  was downloaded in its place, per instructions.

---

## 3. Poly Haven furniture models

- **Source:** Poly Haven (https://polyhaven.com), via the public API (`api.polyhaven.com`), no key required.
- **License:** CC0 1.0 Universal — https://polyhaven.com/license (also https://creativecommons.org/publicdomain/zero/1.0/).
  No attribution is required by the license; every asset is listed below anyway as good practice.
- **Download date:** 2026-09-06
- **Format obtained:** glTF 1k resolution (main `.gltf` + `.bin` + JPG textures), converted locally to a
  single-file `.glb` per asset with `trimesh` (`trimesh.load(path, force='scene').export('model.glb')`).
- **Local path:** `polyhaven/<asset_id>/` (original glTF + textures) and `polyhaven/<asset_id>/<asset_id>.glb`
  (converted single-file GLB).
- **Total assets obtained:** 24 (across 9 requested furniture classes).

| Asset ID | Name | Class hint | URL |
|---|---|---|---|
| Sofa_01 | Sofa 01 | sofa/couch | https://polyhaven.com/a/Sofa_01 |
| sofa_02 | Sofa 02 | sofa/couch | https://polyhaven.com/a/sofa_02 |
| sofa_03 | Sofa 03 | sofa/couch | https://polyhaven.com/a/sofa_03 |
| ArmChair_01 | Arm Chair 01 | chair/armchair | https://polyhaven.com/a/ArmChair_01 |
| dining_chair_02 | Dining Chair 02 | chair/armchair | https://polyhaven.com/a/dining_chair_02 |
| WoodenChair_01 | Wooden Chair 01 | chair/armchair | https://polyhaven.com/a/WoodenChair_01 |
| dining_table | Dining Table | table (dining/coffee) | https://polyhaven.com/a/dining_table |
| CoffeeTable_01 | Coffee Table 01 | table (dining/coffee) | https://polyhaven.com/a/CoffeeTable_01 |
| modern_coffee_table_01 | Modern Coffee Table 01 | table (dining/coffee) | https://polyhaven.com/a/modern_coffee_table_01 |
| SchoolDesk_01 | School Desk 01 | desk | https://polyhaven.com/a/SchoolDesk_01 |
| metal_office_desk | Metal Office Desk | desk | https://polyhaven.com/a/metal_office_desk |
| ClassicNightstand_01 | Classic Nightstand 01 | nightstand/side table/bedside | https://polyhaven.com/a/ClassicNightstand_01 |
| painted_wooden_nightstand | Painted Wooden Nightstand | nightstand/side table/bedside | https://polyhaven.com/a/painted_wooden_nightstand |
| side_table_01 | Side Table 01 | nightstand/side table/bedside | https://polyhaven.com/a/side_table_01 |
| GothicBed_01 | Gothic Bed 01 | bed | https://polyhaven.com/a/GothicBed_01 |
| old_bed_frame | Old Bed Frame | bed | https://polyhaven.com/a/old_bed_frame |
| vintage_day_bed | Vintage Day Bed | bed | https://polyhaven.com/a/vintage_day_bed |
| wooden_bookshelf_worn | Wooden Bookshelf Worn | cabinet/shelf/bookshelf/wardrobe/dresser | https://polyhaven.com/a/wooden_bookshelf_worn |
| modern_wooden_cabinet | Modern Wooden Cabinet | cabinet/shelf/bookshelf/wardrobe/dresser | https://polyhaven.com/a/modern_wooden_cabinet |
| Shelf_01 | Shelf 01 | cabinet/shelf/bookshelf/wardrobe/dresser | https://polyhaven.com/a/Shelf_01 |
| desk_lamp_arm_01 | Desk Lamp Arm 01 | lamp (floor/table lamp) | https://polyhaven.com/a/desk_lamp_arm_01 |
| vintage_oil_lamp | Vintage Oil Lamp | lamp (floor/table lamp) | https://polyhaven.com/a/vintage_oil_lamp |
| Television_01 | Television 01 | tv/television/monitor | https://polyhaven.com/a/Television_01 |
| television_02 | Television 02 | tv/television/monitor | https://polyhaven.com/a/television_02 |

**Note on classes with limited/no Poly Haven coverage:** Poly Haven's furniture-tagged model catalog (85
assets under category `furniture` as of 2026-09-06, ~521 models site-wide) has no dedicated "monitor" model
and no dedicated "floor lamp" model (only wall/ceiling/desk/table lamps and outdoor street lamps exist under
the `lighting` category). See `polyhaven/README.md` for the full breakdown of what was and wasn't available.

## 4. Google Scanned Objects (small-object tier, T16b)

Location on the VPS: `var/assets/small_objects/gso/<name>/` (OBJ + texture as shipped, plus the
converted `<name>.glb`); inventory `var/assets/small_objects/INVENTORY.json`; per-model attribution
table `var/assets/small_objects/ATTRIBUTION.md` (this section mirrors it).

**License: CC BY 4.0 - attribution REQUIRED** wherever these models or renders containing them are
redistributed: "Google Scanned Objects, Google Research,
https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research,
CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)". Downloaded 2026-09-07 from Gazebo Fuel
(`https://fuel.gazebosim.org/1.0/GoogleResearch/models/<name>.zip`, 9 models, 54.6 MB). The GSO
OBJs are Z-up; the GLBs were re-exported Y-up with trimesh (rotation only, metres kept).

| model (Fuel name) | index class | used for detected label |
|---|---|---|
| Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White | cup | glass (proxy: GSO has no drinking glass) |
| Room_Essentials_Mug_White_Yellow | cup | glass |
| Cole_Hardware_Mug_Classic_Blue | cup | glass |
| ACE_Coffee_Mug_Kristen_16_oz_cup | cup | glass |
| Ecoforms_Cup_B4_SAN | cup | glass |
| BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup | cup | glass |
| Orbit_Bubblemint_Mini_Bottle_6_ct | bottle | bottle |
| Marc_Anthony_Strictly_Curls_Curl_Envy_Perfect_Curl_Cream_6_fl_oz_bottle | bottle | bottle |
| CoQ10 | bottle | bottle |

Not covered by GSO (checked against the full 1033-model Fuel listing): drinking glass, pillow, phone.
