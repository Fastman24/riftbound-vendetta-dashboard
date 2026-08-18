"""
Scrapes the full legend-vs-legend matchup matrix (win rate + matches per pair)
from riftDecks.com's Winrate Matrix page, once for the combined "General"
population (all 4 tournaments, minimum_players=256 -- the closest bucket to
"more than 200 players") and once per INDIVIDUAL tournament (via the site's
event_ids[] filter, discovered from the page's "Tournaments" multiselect
field: /stats/winrate?...&event_ids[]=<id>). Each of those 5 scopes is
scraped across three player-slice tiers:

  - any:   top_percent="" (every player)
  - top25: top_percent="25"
  - top10: top_percent="10"

riftDecks does not expose an absolute "Top 64" / "Top 8" cut, only percentages
relative to each tournament's own size, so top_percent=10/25 are the closest
available approximation, flagged as such in the dashboard UI.

Per-tournament event IDs were found via the lazy multiselect's backing
endpoint, /stats/filter-events?metagame_id=4&minimum_players=256&format_id=1
-- note that endpoint only listed 3 of the 4 tournaments (RiftAtlas
Convergence #2 was missing), but querying event_ids[]=<its id> directly still
returns real match data, so all 4 are included here regardless of whether
that dropdown surfaces them.

(An earlier version of this script also scraped "barcelona" and "nexusnight"
proxy scenarios at minimum_players=512, reusing 2 of these tournaments as a
stand-in for events that don't have real Vendetta data yet. Those were
replaced by a proper statistical projection -- see generate_chart.py's
Barcelona section -- and dropped here as they added no real signal.)

Writes matchups.json: { "{scope}_{tier}": { pilot_legend: { opponent_legend: {win_rate, matches} } } }
and legends_order.json: [legend names in the matrix's row/column order]
"""
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
OUT_DIR = Path(__file__).parent

TIERS = {"any": "", "top25": "25", "top10": "10"}

# scope_key -> extra query params identifying the population (on top of
# metagame_id=4 + top_percent, added in fetch_matrix).
SCOPES = {
    "general": {"minimum_players": "256"},
    "germany": {"event_ids[]": "13992"},
    "ottawa": {"event_ids[]": "13733"},
    "auckland": {"event_ids[]": "13965"},
    "riftatlas": {"event_ids[]": "13986"},
}


def fetch_matrix(scope_params: dict, top_percent: str):
    params = {"metagame_id": "4", "date_range": "all", "top_percent": top_percent, **scope_params}
    resp = requests.get(f"{BASE}/stats/winrate", headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Column order: read from the FIRST <thead> only -- the page duplicates the
    # whole header (a hidden/sticky mirror for horizontal scroll), so selecting
    # "table thead th" across both copies double-counts and desyncs the zip below.
    first_thead = soup.find("thead")
    header_ths = first_thead.select("th") if first_thead else []
    col_legends = []
    for th in header_ths:
        span = th.select_one("span[title]")
        if span:
            col_legends.append(span["title"].strip())

    matrix = {}
    for row in soup.select("tr.item[data-name]"):
        pilot = row.get("data-name", "").strip()
        cells = row.select("td.winrate-cell")
        # cells[0] is the OVERALL column, cells[1:] align with col_legends
        opponent_cells = cells[1:]
        opp_data = {}
        for opp_name, cell in zip(col_legends, opponent_cells):
            wr_attr = cell.get("data-winrate")
            matches_el = cell.select_one(".matches-number")
            if wr_attr is None or matches_el is None:
                continue  # mirror match ("--") or zero-sample cell
            m = re.search(r"([\d,]+)\s*matches", matches_el.get_text())
            matches = int(m.group(1).replace(",", "")) if m else None
            opp_data[opp_name] = {"win_rate": int(wr_attr), "matches": matches}
        matrix[pilot] = opp_data

    return col_legends, matrix


def main():
    all_matrices = {}
    legend_order = None
    for scope_key, scope_params in SCOPES.items():
        for tier_key, top_percent in TIERS.items():
            scenario_key = f"{scope_key}_{tier_key}"
            print(f"Fetching matchup matrix for '{scenario_key}' ({scope_params}, top_percent={top_percent or 'any'})...",
                  file=sys.stderr)
            cols, matrix = fetch_matrix(scope_params, top_percent)
            all_matrices[scenario_key] = matrix
            if legend_order is None or len(cols) > len(legend_order):
                legend_order = cols
            print(f"  {len(matrix)} pilots parsed", file=sys.stderr)

    matchups_path = OUT_DIR / "matchups.json"
    matchups_path.write_text(json.dumps(all_matrices, indent=2, ensure_ascii=False), encoding="utf-8")

    order_path = OUT_DIR / "legends_order.json"
    order_path.write_text(json.dumps(legend_order, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote {matchups_path} and {order_path}")


if __name__ == "__main__":
    main()
