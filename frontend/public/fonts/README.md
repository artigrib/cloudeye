# JetBrains Mono (vendored)

Latin + Cyrillic subsets, weights 400/500, WOFF2. Sourced from Google Fonts
(SIL Open Font License 1.1) rather than JetBrains' own release, since upstream
ships TTF/OTF only.

To refresh (e.g. for a new weight or subset), fetch the CSS for that weight
individually — a combined `wght@400;500` request has been observed to return
the same file hash for both weights — then download the `url(...)` from the
`latin` and `cyrillic` blocks:

```
curl -sL "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400&display=swap" \
  -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
```

The `-A` (modern desktop UA) is required — without it Google serves woff/ttf
instead of woff2.

## Inter (2026-09-09)

`inter-{400,500,600}-{latin,cyrillic}.woff2`, copied verbatim from
`@fontsource/inter@5.3.0`'s `files/inter-<subset>-<weight>-normal.woff2` (SIL Open Font
License 1.1, same licence family as JetBrains Mono above). Vendored rather than loaded
from a CDN for the same reason: this app is served from one box and must not depend on a
third party being reachable to render its own text.

Latin and Cyrillic are separate files with their own `unicode-range`, so a Latin-only page
never fetches the Cyrillic glyphs - the pattern the JetBrains Mono faces already follow.
Three weights: 400 body/caption, 500 headings, 600 the display number.

To refresh: `npm i -D @fontsource/inter@<version>` and re-copy those six files.

## Licences

Both families are SIL Open Font License 1.1. The upstream licence text ships beside the
subsets, as the OFL requires of any redistribution:

- `OFL-Inter.txt` — https://raw.githubusercontent.com/rsms/inter/master/LICENSE.txt
- `OFL-JetBrainsMono.txt` — https://raw.githubusercontent.com/JetBrains/JetBrainsMono/master/OFL.txt

Both fetched verbatim 2026-09-11. See `THIRD_PARTY.md` for the full record.
