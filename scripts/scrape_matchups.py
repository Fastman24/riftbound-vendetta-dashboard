"""
Scrapes the full legend-vs-legend matchup matrix (win rate + matches per pair)
from riftDecks.com's Winrate Matrix page, for the 4 tournaments covered by
scrape_tournament_breakdowns.py (minimum_players=256, the closest bucket to
"more than 200 players"), across three player-slice tiers the site offers:

  - general_any:   top_percent="" (every player)
  - general_top25: top_percent="25"
  - general_top10: top_percent="10"

riftDecks does not expose an absolute "Top 64" / "Top 8" cut, only percentages
relative to each tournament's own size, so top_percent=10/25 are the closest
available approximation, flagged as such in the dashboard UI.

(An earlier version of this script also scraped "barcelona" and "nexusnight"
proxy scenarios at minimum_players=512, reusing these same 2 tournaments as a
stand-in for events that don't have real Vendetta data yet. Those were
replaced by a proper statistical projection -- see generate_chart.py's
Barcelona section -- and dropped here as they added no real signal.)

Writes matchups.json: { scenario_key: { pilot_legend: { opponent_legend: {win_rate, matches} } } }
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

SCENARIOS = {
    "general_any": {"minimum_players": "256", "top_percent": ""},
    "general_top25": {"minimum_players": "256", "top_percent": "25"},
    "general_top10": {"minimum_players": "256", "top_percent": "10"},
}


def fetch_matrix(minimum_players: str, top_percent: str):
    params = {
        "metagame_id": "4",
        "minimum_players": minimum_players,
        "date_range": "all",
        "top_percent": top_percent,
    }
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
    for scenario_key, params in SCENARIOS.items():
        print(f"Fetching matchup matrix for scenario '{scenario_key}' "
              f"(minimum_players={params['minimum_players']}, top_percent={params['top_percent'] or 'any'})...",
              file=sys.stderr)
        cols, matrix = fetch_matrix(params["minimum_players"], params["top_percent"])
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
