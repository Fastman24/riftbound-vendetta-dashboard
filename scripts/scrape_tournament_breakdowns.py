"""
Fixes a scoping bug in scrape_vendetta_legends.py: riftDecks.com's /legends
(Metashare) page has NO "minimum_players" filter at all (only metagame_id and
relevance), so "times_played" there was silently the WHOLE Vendetta metagame
across tournaments of any size -- not scoped to the 200+/256+ player events
like the rest of this dashboard claims.

Each tournament's own page has a "/breakdown" sub-page with a genuine
per-tournament legend -> deck-count table. Summing that across the 4 target
tournaments gives a correctly-scoped times_played.

Writes tournament_breakdown_legends.json: [{legend, times_played}], summed
across the 4 tournaments, sorted by times_played desc.
"""
import json
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}
OUT_DIR = Path(__file__).parent

# The 4 Vendetta tournaments with 200+ (256+) registered players, found via
# scrape_vendetta_legends.py's tournament-list scrape.
TOURNAMENT_SLUGS = [
    "riftbound-showdown-series-germany-speyer-tournament-decks-13992",
    "riftbound-showdown-ottawa-tournament-decks-13733",
    "10k-showdown-auckland-card-show-tournament-decks-13965",
    "riftatlas-convergence-2-top-16-tournament-decks-13986",
]


def fetch_breakdown(slug: str):
    url = f"https://riftdecks.com/riftbound-tournaments/{slug}/breakdown"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    counts = {}
    for row in soup.select("tr[data-href^='/legends/constructed/']"):
        name_el = row.select_one("strong.d-none.d-md-inline")
        decks_td = row.select_one("td.sort-totaldecks")
        if not (name_el and decks_td and decks_td.get("data-totaldecks")):
            continue
        name = name_el.get_text(strip=True)
        counts[name] = int(decks_td["data-totaldecks"])
    return counts


def main():
    per_tournament = {}
    for slug in TOURNAMENT_SLUGS:
        print(f"Fetching breakdown for {slug}...", file=sys.stderr)
        counts = fetch_breakdown(slug)
        per_tournament[slug] = counts
        print(f"  {len(counts)} legends, {sum(counts.values())} decks", file=sys.stderr)

    # Raw per-tournament data, so other scripts can re-aggregate different subsets
    # (e.g. just the 512+-player events) without re-fetching.
    raw_path = OUT_DIR / "tournament_breakdown_raw.json"
    raw_path.write_text(json.dumps(per_tournament, indent=2, ensure_ascii=False), encoding="utf-8")

    totals = {}
    for counts in per_tournament.values():
        for name, n in counts.items():
            totals[name] = totals.get(name, 0) + n

    rows = sorted(
        ({"legend": name, "times_played": n} for name, n in totals.items()),
        key=lambda r: -r["times_played"],
    )

    out_path = OUT_DIR / "tournament_breakdown_legends.json"
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    total_decks = sum(r["times_played"] for r in rows)
    print(f"\nWrote {raw_path} and {out_path}")
    print(f"Total decks across the 4 tournaments: {total_decks}")
    print("\nTop 10:")
    for r in rows[:10]:
        print(f"  {r['legend']:<35} {r['times_played']}")


if __name__ == "__main__":
    main()
