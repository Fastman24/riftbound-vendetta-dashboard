"""
Scrapes the full legend-vs-legend matchup matrix (win rate + matches per pair)
from riftDecks.com's Winrate Matrix page, for several named "scenarios" -- each
a (minimum_players, top_percent) pair the site actually supports:

  - general_any / general_top25 / general_top10:
      minimum_players=256 (closest bucket to "more than 200 players"), across
      Any / Top 25% / Top 10% of finishers per tournament. This is the 4
      tournaments covered by scrape_tournament_breakdowns.py.

  - barcelona: minimum_players=512, top_percent=10 -- a PROXY for the
      Barcelona Regional Qualifier (2,210 players, cuts to Top 64 on day 2),
      which hasn't happened yet as of this scrape and so has no real data of
      its own. 512+ is the largest bucket we have actual matches for (only
      Germany-Speyer 626p and Ottawa 594p qualify), and top_percent=10 is
      used as the closest available stand-in for "the players who are
      actually competing for the cut" -- NOT a real Barcelona top-64 cut.

  - nexusnight: minimum_players=512, top_percent="" (any) -- a PROXY for a
      ~500-player "Super Nexus Night" in the Vendetta metagame. No such event
      exists yet in riftDecks' Vendetta data (the one found, Lille, was
      pre-Vendetta and a different metagame), so this reuses the same 512+
      bucket (Germany-Speyer + Ottawa) but on the full field (no cut), since
      Nexus Night is a single Bo1 Swiss night with no elimination top cut.

riftDecks does not expose an absolute "Top 64" / "Top 8" cut (only percentages
relative to each tournament's own size), and it has no data at all for a
2,210-player or ~500-player Vendetta event -- both approximations above are
flagged clearly in the dashboard UI, not presented as real results.

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
    "barcelona": {"minimum_players": "512", "top_percent": "10"},
    "nexusnight": {"minimum_players": "512", "top_percent": ""},
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
