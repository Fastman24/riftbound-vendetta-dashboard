"""
Scrapes riftDecks.com for Riftbound: Vendetta legend stats.

Produces, per legend:
  - times_played  -> "Total Decks" from the Metashare page. /legends has no
                      minimum_players filter, so this is scoped to ALL Vendetta
                      tournaments of any size, all time -- generate_chart.py
                      overrides this with the 256+-player-only figure (from
                      tournament_breakdown_legends.json) for the main chart,
                      but keeps this sitewide figure for the Barcelona
                      projection, which deliberately wants the broadest,
                      largest-sample population estimate available.
  - total_games   -> "matches" from the Winrate Matrix page's OVERALL column,
                      restricted to 256+-player tournaments (the closest
                      bucket to ">200 players").
  - win_rate_pct  -> overall win rate (%), same 256+ restriction.

Writes vendetta_legends.csv and vendetta_legends.json next to this script.
"""
import csv
import json
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://riftdecks.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}
# metagame_id=4 -> Vendetta Metagame. minimum_players=256 -> closest available
# bucket to "more than 200 players" (site offers 16/32/64/128/256/512/1024).
WINRATE_PARAMS = {"metagame_id": "4", "minimum_players": "256", "date_range": "all"}
# /legends has NO minimum_players filter at all, and its date_range accepts
# different values than /stats/winrate ("all" isn't one of them) -- passing
# WINRATE_PARAMS here silently falls back to some narrow default window
# instead of erroring, undercounting every legend by ~3-4x. Only metagame_id
# is safe to pass; that alone reproduces the site's own "All Time" default.
LEGENDS_PARAMS = {"metagame_id": "4"}

OUT_DIR = Path(__file__).parent


def fetch(path: str, params: dict) -> BeautifulSoup:
    resp = requests.get(f"{BASE}{path}", headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def scrape_metashare():
    """Legend -> times_played (Total Decks) across ALL Vendetta tournaments, any size, all time."""
    soup = fetch("/legends", LEGENDS_PARAMS)
    out = {}
    for row in soup.select("tr[data-href^='/legends/constructed/']"):
        name_el = row.select_one("strong.d-none.d-md-inline")
        decks_td = row.select_one("td.sort-totaldecks")
        winrate_td = row.select_one("td.sort-winrate")
        if not (name_el and decks_td):
            continue
        name = name_el.get_text(strip=True)
        times_played = int(decks_td.get("data-totaldecks", "0"))
        win_rate = winrate_td.get("data-winrate") if winrate_td else None
        out[name] = {
            "times_played": times_played,
            "win_rate_pct_fallback": float(win_rate) if win_rate else None,
        }
    return out


def scrape_winrate_matrix():
    """Legend -> total_games (OVERALL matches), win_rate_pct (precise), 256+ players only."""
    soup = fetch("/stats/winrate", WINRATE_PARAMS)
    out = {}
    for row in soup.select("tr.item[data-name]"):
        name = row.get("data-name", "").strip()
        matches = row.get("data-matches")
        winrate = row.get("data-winrate")
        if not name or matches is None:
            continue
        out[name] = {
            "total_games": int(matches),
            "win_rate_pct": round(float(winrate) * 100, 1) if winrate else None,
        }
    return out


def main():
    print("Fetching Metashare (times played)...", file=sys.stderr)
    metashare = scrape_metashare()
    print(f"  {len(metashare)} legends found", file=sys.stderr)

    print("Fetching Winrate Matrix (total games, win rate)...", file=sys.stderr)
    winrate = scrape_winrate_matrix()
    print(f"  {len(winrate)} legends found", file=sys.stderr)

    names = sorted(set(metashare) | set(winrate))
    rows = []
    for name in names:
        ms = metashare.get(name, {})
        wr = winrate.get(name, {})
        times_played = ms.get("times_played")
        total_games = wr.get("total_games")  # None if below the matrix's sample threshold
        win_rate = wr.get("win_rate_pct")  # None if below the matrix's sample threshold
        if not times_played and not total_games:
            continue
        rows.append(
            {
                "legend": name,
                "times_played": times_played or 0,
                "total_games": total_games,  # may be None: insufficient sample size
                "win_rate_pct": win_rate,  # may be None: insufficient sample size
            }
        )

    rows.sort(key=lambda r: r["times_played"], reverse=True)

    csv_path = OUT_DIR / "vendetta_legends.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["legend", "times_played", "total_games", "win_rate_pct"])
        writer.writeheader()
        writer.writerows(rows)

    json_path = OUT_DIR / "vendetta_legends.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(rows)} legends to {csv_path} and {json_path}")
    print("\nTop 10 by times played:")
    for r in rows[:10]:
        games = r["total_games"] if r["total_games"] is not None else "n/a"
        wr_ = f"{r['win_rate_pct']}%" if r["win_rate_pct"] is not None else "n/a"
        print(f"  {r['legend']:<35} played={r['times_played']:<5} games={games:<6} winrate={wr_}")


if __name__ == "__main__":
    main()
