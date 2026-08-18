# Riftbound: Vendetta — Legend Dashboard

Interactive dashboard for Riftbound's *Vendetta* metagame, scraped from
[riftDecks.com](https://riftdecks.com), covering tournaments with 200+
registered players since the expansion's release (2026-07-31).

**Live site:** enable GitHub Pages on this repo (Settings → Pages → deploy
from `main` / root) and it will serve `index.html` directly.

## What's in the dashboard

- **Legends played** — times played, total games, win rate per legend across
  the qualifying tournaments.
- **Matchups** — a per-legend win-rate "wheel" against every other legend,
  with a picker and General / Super Nexus Night (Bo1 proxy) tabs.
- **Expected opponent** — ranked by a Wilson-score lower bound (penalizes
  small-sample noise), including an estimated mirror-match rate.
- **Barcelona 2026 projection** — a pure statistical projection (not reused
  tournament data) for the Barcelona Regional Qualifier (2,210 players,
  8 Swiss rounds on day 1): expected copies of each legend in the room, and
  expected encounters across your 8 rounds under a win-rate-weighted
  "bracket drift" model (if you're also winning, Swiss increasingly pairs
  you against players piloting above-average-win-rate decks).

## Reproducing the data

```bash
pip install requests beautifulsoup4
python scripts/scrape_vendetta_legends.py       # sitewide + 256+-player legend stats
python scripts/scrape_tournament_breakdowns.py  # per-tournament deck breakdowns
python scripts/scrape_matchups.py               # legend-vs-legend matchup matrices
python scripts/generate_chart.py                # builds index.html from the data/ above
```

Each script's docstring documents exactly which riftDecks.com filters it
uses and why (including a couple of scraping gotchas around that site's
inconsistent query-parameter handling across pages).

## Data notes / caveats

- "Barcelona" and "Super Nexus Night" scenarios are clearly labeled
  projections/proxies — neither tournament had real Vendetta-era data in
  riftDecks as of this scrape (Barcelona hadn't happened yet; no Vendetta
  Super Nexus Night was found).
- Mirror-match rates are estimated from metagame share, since riftDecks
  excludes mirrors from its own win-rate matrix.
- Data reflects a snapshot from 2026-08-18 and will drift as riftDecks
  ingests more tournament results — re-run the scripts to refresh.
