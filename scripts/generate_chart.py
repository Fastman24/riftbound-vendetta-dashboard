"""
Builds a self-contained HTML chart from vendetta_legends.json,
tournament_breakdown_legends.json, tournaments_200plus.json, matchups.json and
legends_order.json (all produced by the scrape_*.py scripts in this folder).
Writes chart.html next to this script.
"""
import json
import math
from pathlib import Path


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n. Returns (phat, lo, hi)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    return phat, max(0.0, center - margin), min(1.0, center + margin)

DIR = Path(__file__).parent

legends_raw = json.loads((DIR / "vendetta_legends.json").read_text(encoding="utf-8"))
breakdown = json.loads((DIR / "tournament_breakdown_legends.json").read_text(encoding="utf-8"))
tournaments = json.loads((DIR / "tournaments_200plus.json").read_text(encoding="utf-8"))
matchups = json.loads((DIR / "matchups.json").read_text(encoding="utf-8"))
legend_order = json.loads((DIR / "legends_order.json").read_text(encoding="utf-8"))

# vendetta_legends.json's "times_played" comes from riftDecks' sitewide /legends
# page, which has NO minimum_players filter -- it silently covers the whole
# Vendetta metagame, not just these 4 tournaments. times_played here instead
# comes from summing each tournament's own /breakdown page (a real per-event
# legend->deck count), so it's actually scoped to the 200+ player events this
# dashboard is about. total_games / win_rate_pct are kept from vendetta_legends.json
# since those DO come from the correctly-filtered winrate matrix (minimum_players=256).
games_by_legend = {l["legend"]: (l["total_games"], l["win_rate_pct"]) for l in legends_raw}
legends = []
for r in breakdown:
    games, wr = games_by_legend.get(r["legend"], (None, None))
    legends.append({
        "legend": r["legend"],
        "times_played": r["times_played"],
        "total_games": games,
        "win_rate_pct": wr,
    })
legends.sort(key=lambda l: -l["times_played"])

total_players = sum(t["players"] for t in tournaments)
total_decks = sum(l["times_played"] for l in legends)
total_games = sum(l["total_games"] or 0 for l in legends)
max_played = max(l["times_played"] for l in legends)

# Picker order: legends that actually have matchup rows, sorted by times_played desc.
played_rank = {l["legend"]: l["times_played"] for l in legends}
picker_legends = sorted((n for n in legend_order if n in matchups.get("general_any", {})),
                         key=lambda n: -played_rank.get(n, 0))

# Individual-tournament matchup scopes, matched to scrape_matchups.py's SCOPES
# by riftDecks event id. Order matches tournaments_200plus.json (players desc).
TOURNAMENT_SCOPE_BY_NAME = {
    "Riftbound Showdown Series Germany - Speyer": ("germany", "Alemania (Speyer)"),
    "Riftbound Showdown Ottawa": ("ottawa", "Ottawa"),
    "10K Showdown - Auckland Card Show": ("auckland", "Auckland"),
    "RiftAtlas Convergence #2": ("riftatlas", "RiftAtlas"),
}
tournament_scopes = [
    {"key": TOURNAMENT_SCOPE_BY_NAME[t["name"]][0], "label": f'{TOURNAMENT_SCOPE_BY_NAME[t["name"]][1]} ({t["players"]:,}p)'}
    for t in tournaments if t["name"] in TOURNAMENT_SCOPE_BY_NAME
]

# ---- Barcelona 2026 projection: pure math, no reused-tournament proxy ----
# Population estimate = ALL Vendetta tournaments, any size, all time (the
# broadest sample riftDecks has, via vendetta_legends.json's sitewide /legends
# scrape) -- not just our 4 headline 200+-player events, since more data means
# a tighter, more reliable estimate of each legend's true population share,
# and there's no evidence the metagame differs systematically by tournament size.
BARCELONA_PLAYERS = 2210
BARCELONA_DAY1_ROUNDS = 8
sitewide_total_decks = sum(l["times_played"] for l in legends_raw)

barcelona_rows = []
for l in legends_raw:
    k = l["times_played"]
    p, lo, hi = wilson(k, sitewide_total_decks)
    barcelona_rows.append({
        "legend": l["legend"],
        "share_pct": round(p * 100, 2),
        "share_lo": round(lo * 100, 2),
        "share_hi": round(hi * 100, 2),
        "expected_count": round(p * BARCELONA_PLAYERS, 1),
        "count_lo": round(lo * BARCELONA_PLAYERS, 1),
        "count_hi": round(hi * BARCELONA_PLAYERS, 1),
        "prob8_pct": round((1 - (1 - p) ** BARCELONA_DAY1_ROUNDS) * 100, 1),
        "prob8_lo": round((1 - (1 - lo) ** BARCELONA_DAY1_ROUNDS) * 100, 1),
        "prob8_hi": round((1 - (1 - hi) ** BARCELONA_DAY1_ROUNDS) * 100, 1),
        "n": k,
    })


def poisson_binomial_pmf(probs):
    """P(exactly k successes) for independent Bernoulli trials with different p's, via DP convolution."""
    pmf = [1.0]
    for p in probs:
        new_pmf = [0.0] * (len(pmf) + 1)
        for k, prob_k in enumerate(pmf):
            new_pmf[k] += prob_k * (1 - p)
            new_pmf[k + 1] += prob_k * p
        pmf = new_pmf
    return pmf


# Round-by-round "bracket drift" model: if you also keep winning, Swiss pairs
# you against other winners each round, and a deck's win rate is exactly what
# determines how much of that deck survives into the winners' bracket. Model
# each deck's representation as decaying/compounding by its own win rate every
# round (relative to the field, renormalized so it's a valid distribution each
# round): weight_i(r) = share_i * win_rate_i^(r-1). Decks without a reliable
# win rate (too few recorded matches) fall back to 0.5 (neutral: no assumed drift).
# This is a first-order approximation of real Swiss bracket segregation --  it
# ignores rematch avoidance, precise match-point buckets, and assumes YOU are
# also winning every round; it is not an exact simulation of the pairing algorithm.
win_rate_by_legend = {l["legend"]: (l["win_rate_pct"] / 100 if l["win_rate_pct"] is not None else 0.5)
                       for l in legends_raw}
share_by_legend = {l["legend"]: l["times_played"] / sitewide_total_decks for l in legends_raw}

round_probs_by_legend = {}  # legend -> [p(round1), ..., p(round8)]
for r in range(1, BARCELONA_DAY1_ROUNDS + 1):
    weights = {name: share_by_legend[name] * (win_rate_by_legend[name] ** (r - 1)) for name in share_by_legend}
    total_w = sum(weights.values())
    for name, w in weights.items():
        round_probs_by_legend.setdefault(name, []).append(w / total_w)

for row in barcelona_rows:
    probs = round_probs_by_legend[row["legend"]]
    pmf = poisson_binomial_pmf(probs)
    row["expected_times_drift"] = round(sum(probs), 2)
    row["p_at_least_1_drift"] = round((1 - pmf[0]) * 100, 1)
    row["p_at_least_2_drift"] = round((1 - pmf[0] - pmf[1]) * 100, 1)
    row["win_rate_pct_display"] = round(win_rate_by_legend[row["legend"]] * 100, 1)

# share/count/prob8 are all monotonic transforms of the same p, so they share
# one ranking -- by Wilson lower bound, same anti-small-sample-noise logic as
# the "rival esperado" ranking elsewhere on this page.
barcelona_rows.sort(key=lambda r: -r["share_lo"])
max_share_pct = max(r["share_pct"] for r in barcelona_rows)

html = f"""<meta charset="utf-8">
<title>Riftbound Vendetta — Legends en torneos 200+ jugadores</title>
<style>
  /* Tokens live on :root (not .viz-root) so html/body can use them too --
     this page is served standalone by GitHub Pages (no wrapping skeleton),
     so without this, the default white/UA body background showed through
     as a visible band around .viz-root's centered, page-colored content. */
  :root {{
    color-scheme: light;
    --surface-1: #fcfcfb;
    --page: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --gridline: #e1e0d9;
    --baseline: #c3c2b7;
    --border: rgba(11,11,11,0.10);
    --div-blue: #2a78d6;
    --div-red: #e34948;
    --div-mid: #898781;
    --chip-bg: #f0efec;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) {{
      color-scheme: dark;
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --gridline: #2c2c2a;
      --baseline: #383835;
      --border: rgba(255,255,255,0.10);
      --div-blue: #3987e5;
      --div-red: #e66767;
      --div-mid: #898781;
      --chip-bg: #24231f;
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --surface-1: #1a1a19;
    --page: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --gridline: #2c2c2a;
    --baseline: #383835;
    --border: rgba(255,255,255,0.10);
    --div-blue: #3987e5;
    --div-red: #e66767;
    --div-mid: #898781;
    --chip-bg: #24231f;
  }}

  html, body {{ margin: 0; padding: 0; background: var(--page); min-height: 100%; }}
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--text-primary); }}

  .viz-root {{ padding: 32px 20px 48px; max-width: 900px; margin: 0 auto; background: var(--page); }}
  .viz-root * {{ box-sizing: border-box; }}
  .viz-root ::selection {{ background: color-mix(in oklab, var(--div-blue) 35%, transparent); }}
  .viz-h1 {{ font-size: 24px; font-weight: 700; margin: 0 0 4px; letter-spacing: -.01em; }}
  .viz-kicker {{ display: inline-flex; align-items: center; gap: 6px; font-size: 11px; font-weight: 650; text-transform: uppercase; letter-spacing: .08em; color: var(--div-blue); margin-bottom: 8px; }}
  .viz-kicker::before {{ content: ''; width: 6px; height: 6px; border-radius: 50%; background: var(--div-blue); }}
  .viz-sub {{ font-size: 13.5px; color: var(--text-secondary); margin: 0 0 20px; line-height: 1.55; max-width: 68ch; }}

  .stat-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-bottom: 24px; }}
  .stat-tile {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 13px 14px; transition: border-color .15s ease, transform .15s ease; }}
  .stat-tile:hover {{ border-color: color-mix(in oklab, var(--div-blue) 40%, var(--border)); transform: translateY(-1px); }}
  .stat-tile .label {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: .03em; }}
  .stat-tile .value {{ font-size: 21px; font-weight: 700; margin-top: 3px; letter-spacing: -.01em; }}

  .card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 14px; padding: 20px 22px 22px; margin-bottom: 20px; box-shadow: 0 1px 2px rgba(11,11,11,.04), 0 8px 24px -16px rgba(11,11,11,.12); }}
  .card h2 {{ font-size: 15px; font-weight: 700; margin: 0 0 10px; color: var(--text-primary); letter-spacing: -.005em; }}

  table.tourney {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  table.tourney th {{ text-align: left; color: var(--text-muted); font-weight: 550; font-size: 11.5px; text-transform: uppercase; letter-spacing: .02em; padding: 6px 8px; border-bottom: 1px solid var(--gridline); }}
  table.tourney td {{ padding: 7px 8px; border-bottom: 1px solid var(--gridline); color: var(--text-secondary); }}
  table.tourney td:first-child, table.tourney td.name {{ color: var(--text-primary); }}
  table.tourney td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}

  .legend-scale {{ display: flex; align-items: center; gap: 8px; font-size: 11.5px; color: var(--text-muted); margin: 2px 0 16px; }}
  .legend-scale .bar {{ flex: 1; height: 10px; border-radius: 5px; background: linear-gradient(to right, var(--div-red), var(--div-mid), var(--div-blue)); }}

  .toggle-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }}
  .toggle-btn {{ font: inherit; font-size: 12.5px; padding: 6px 12px; border-radius: 999px; border: 1px solid var(--border); background: var(--surface-1); color: var(--text-secondary); cursor: pointer; transition: background .15s ease, color .15s ease, border-color .15s ease; }}
  .toggle-btn:hover:not(.active) {{ background: var(--chip-bg); border-color: color-mix(in oklab, var(--div-blue) 30%, var(--border)); }}
  .toggle-btn.active {{ background: var(--text-primary); color: var(--surface-1); border-color: var(--text-primary); }}
  .toggle-btn:focus-visible, .picker-btn:focus-visible {{ outline: 2px solid var(--div-blue); outline-offset: 2px; }}

  .bar-row {{ display: grid; grid-template-columns: 168px 1fr 92px; align-items: center; gap: 10px; height: 20px; margin-bottom: 3px; position: relative; }}
  .bar-row .name {{ font-size: 12px; color: var(--text-secondary); text-align: right; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .bar-row .track {{ position: relative; height: 14px; background: var(--gridline); border-radius: 4px; overflow: hidden; }}
  .bar-row .fill {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px 0 0 4px; cursor: pointer; transition: filter .15s ease; }}
  .bar-row .fill:hover {{ filter: brightness(1.08); }}
  .bar-row .fill.insufficient {{ background: repeating-linear-gradient(45deg, var(--text-muted) 0 3px, transparent 3px 7px); opacity: .55; }}
  .bar-row .tail {{ font-size: 11px; color: var(--text-muted); font-variant-numeric: tabular-nums; }}

  .tooltip {{ position: fixed; pointer-events: none; background: var(--text-primary); color: var(--surface-1); font-size: 12px; padding: 8px 10px; border-radius: 8px; line-height: 1.5; box-shadow: 0 6px 20px rgba(0,0,0,.25); z-index: 50; display: none; white-space: nowrap; }}
  .tooltip b {{ font-weight: 650; }}

  table.datatable {{ width: 100%; border-collapse: collapse; font-size: 13px; display: none; }}
  table.datatable.active {{ display: table; }}
  #chartView.active {{ display: block; }}
  #chartView {{ display: none; }}
  table.datatable th {{ text-align: right; color: var(--text-muted); font-weight: 550; font-size: 11.5px; text-transform: uppercase; letter-spacing: .02em; padding: 8px; border-bottom: 1px solid var(--gridline); cursor: pointer; user-select: none; transition: color .15s ease; }}
  table.datatable th:hover {{ color: var(--text-primary); }}
  table.datatable th:first-child {{ text-align: left; }}
  table.datatable td {{ padding: 7px 8px; border-bottom: 1px solid var(--gridline); text-align: right; font-variant-numeric: tabular-nums; color: var(--text-secondary); }}
  table.datatable td:first-child {{ text-align: left; color: var(--text-primary); font-variant-numeric: normal; }}
  table.datatable th.sorted::after {{ content: " \\2193"; }}
  table.datatable th.sorted.asc::after {{ content: " \\2191"; }}
  table.datatable tbody tr {{ transition: background .1s ease; }}
  table.datatable tbody tr:hover {{ background: var(--chip-bg); }}
  table.tourney tbody tr {{ transition: background .1s ease; }}
  table.tourney tbody tr:hover {{ background: var(--chip-bg); }}
  table.freqtable tbody tr {{ transition: background .1s ease; }}
  table.freqtable tbody tr:hover {{ background: var(--chip-bg); }}

  .foot {{ font-size: 11.5px; color: var(--text-muted); margin-top: 8px; line-height: 1.6; }}
  .foot a {{ color: inherit; }}

  .picker {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; max-height: 148px; overflow-y: auto; padding: 2px; }}
  .picker-btn {{ font: inherit; font-size: 12px; padding: 5px 10px; border-radius: 999px; border: 1px solid var(--border); background: var(--chip-bg); color: var(--text-secondary); cursor: pointer; white-space: nowrap; transition: background .15s ease, color .15s ease, border-color .15s ease; }}
  .picker-btn:hover:not(.active) {{ background: color-mix(in oklab, var(--div-blue) 12%, var(--chip-bg)); border-color: color-mix(in oklab, var(--div-blue) 30%, var(--border)); }}
  .picker-btn.active {{ background: var(--text-primary); color: var(--surface-1); border-color: var(--text-primary); font-weight: 600; }}

  .rosco-wrap {{ display: flex; flex-direction: column; align-items: center; }}
  .rosco-wrap svg {{ width: 100%; max-width: 760px; height: auto; }}
  .rosco-node circle {{ cursor: pointer; transition: filter .12s ease; stroke: var(--surface-1); stroke-width: 2; }}
  .rosco-node:hover circle {{ filter: brightness(1.12); }}
  .rosco-node.insufficient circle {{ fill: var(--gridline) !important; stroke: var(--baseline); stroke-dasharray: 2 2; }}
  .rosco-node text {{ fill: var(--surface-1); font-weight: 650; pointer-events: none; }}
  .rosco-node .opp-name {{ fill: var(--text-secondary); font-weight: 600; }}
  .rosco-node.insufficient text.opp-name {{ fill: var(--text-muted); }}
  .rosco-node.insufficient text.wr-label {{ fill: var(--text-muted); }}
  .rosco-center circle {{ fill: var(--text-primary); }}
  .rosco-center text {{ fill: var(--surface-1); font-weight: 650; }}
  .rosco-spoke {{ stroke: var(--gridline); stroke-width: 1; }}

  .freq-table-wrap {{ overflow-x: auto; }}
  table.freqtable {{ width: 100%; border-collapse: collapse; font-size: 12.5px; min-width: 640px; }}
  table.freqtable th {{ text-align: left; color: var(--text-muted); font-weight: 550; font-size: 11px; text-transform: uppercase; letter-spacing: .02em; padding: 6px 10px; border-bottom: 1px solid var(--gridline); white-space: nowrap; }}
  table.freqtable td {{ padding: 6px 10px; border-bottom: 1px solid var(--gridline); color: var(--text-secondary); vertical-align: middle; white-space: nowrap; }}
  table.freqtable td.name {{ color: var(--text-primary); font-weight: 550; }}
  table.freqtable td b {{ color: var(--text-primary); font-variant-numeric: tabular-nums; }}
  table.freqtable td .ci {{ color: var(--text-muted); font-variant-numeric: tabular-nums; }}
  .mini-track {{ display: inline-block; width: 70px; height: 8px; background: var(--gridline); border-radius: 4px; overflow: hidden; vertical-align: middle; margin-right: 6px; }}
  .mini-fill {{ height: 100%; border-radius: 4px; }}
</style>

<div class="viz-root">
  <div class="viz-kicker">Riftbound · Vendetta</div>
  <div class="viz-h1">Leyendas en torneos grandes</div>
  <p class="viz-sub">
    Metagame Vendetta (desde el lanzamiento de la expansión, 2026-07-31), restringido a torneos con
    más de 200 jugadores registrados. El filtro disponible más cercano en la fuente son eventos de
    256+ jugadores — coincide exactamente con "más de 200" aquí, porque el siguiente torneo por
    debajo tiene solo 177 jugadores. Fuente: riftDecks.com.
  </p>

  <div class="stat-row">
    <div class="stat-tile"><div class="label">Torneos incluidos</div><div class="value">{len(tournaments)}</div></div>
    <div class="stat-tile"><div class="label">Jugadores (suma)</div><div class="value">{total_players:,}</div></div>
    <div class="stat-tile"><div class="label">Decks / veces jugadas</div><div class="value">{total_decks:,}</div></div>
    <div class="stat-tile"><div class="label">Partidas totales</div><div class="value">{total_games:,}</div></div>
  </div>

  <div class="card">
    <h2>Torneos analizados (200+ jugadores, Vendetta)</h2>
    <table class="tourney">
      <thead><tr><th>Fecha</th><th>Torneo</th><th style="text-align:right">Jugadores</th><th>País</th></tr></thead>
      <tbody>
        {"".join(f'<tr><td>{t["date"]}</td><td class="name">{t["name"]}</td><td class="num">{t["players"]:,}</td><td>{t["country"]}</td></tr>' for t in tournaments)}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Leyendas jugadas — veces jugada (barra), win rate (color), partidas totales (etiqueta)</h2>

    <div class="legend-scale">
      <span>Win rate bajo</span>
      <div class="bar"></div>
      <span>50%</span>
      <div class="bar" style="background:linear-gradient(to right, var(--div-mid), var(--div-blue));flex:1"></div>
      <span>Win rate alto</span>
    </div>

    <div class="toggle-row">
      <button class="toggle-btn active" id="btnChart">Gráfica</button>
      <button class="toggle-btn" id="btnTable">Tabla</button>
    </div>

    <div id="chartView" class="active">
      {"".join(
        f'<div class="bar-row" data-i="{i}">'
        f'<div class="name">{l["legend"]}</div>'
        f'<div class="track"><div class="fill{" insufficient" if l["total_games"] is None else ""}" '
        f'style="width:{(l["times_played"]/max_played*100):.1f}%; '
        + (
            f'background: color-mix(in oklab, {"var(--div-blue)" if (l["win_rate_pct"] or 0) >= 50 else "var(--div-red)"} '
            f'{abs((l["win_rate_pct"] or 50)-50)*2:.0f}%, var(--div-mid));'
            if l["win_rate_pct"] is not None else ""
          )
        + f'"></div></div>'
        f'<div class="tail">{l["times_played"]} · {(str(l["win_rate_pct"])+"%") if l["win_rate_pct"] is not None else "n/d"}</div>'
        f'</div>'
        for i, l in enumerate(legends)
      )}
    </div>

    <table class="datatable" id="dataTable">
      <thead>
        <tr>
          <th data-key="legend" data-type="str">Leyenda</th>
          <th data-key="times_played" data-type="num">Veces jugada</th>
          <th data-key="total_games" data-type="num">Partidas totales</th>
          <th data-key="win_rate_pct" data-type="num">Win rate</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>

    <div class="foot">
      Barras rayadas / "n/d" = sin partidas suficientes en la matriz de win rate de la fuente para ese cruce
      (la leyenda sí se jugó, pero por debajo del umbral de muestra de esa matriz). "Veces jugada" = mazos
      registrados con esa leyenda, sumados directamente del desglose propio de cada uno de los 4 torneos
      (no del contador general del sitio, que no se puede filtrar por tamaño de torneo). "Partidas totales" =
      número de partidas (games) registradas con esa leyenda como piloto.
      Datos obtenidos de <a href="https://riftdecks.com" target="_blank" rel="noopener">riftDecks.com</a>
      el 2026-08-17 mediante scraping automatizado (script Python incluido).
    </div>
  </div>

  <div class="card">
    <h2>Barcelona 2026 — proyección matemática (2.210 jugadores)</h2>
    <p class="viz-sub" style="margin-bottom:14px;">
      Esto NO reutiliza datos de otro torneo como aproximación — es un cálculo estadístico puro.
      Partimos de la cuota de metagame de cada leyenda calculada sobre <b>todos</b> los torneos Vendetta
      registrados en riftDecks, de cualquier tamaño, desde el lanzamiento de la expansión
      ({sitewide_total_decks:,} mazos analizados, muestra mucho mayor y más fiable que limitarnos a los
      4 torneos grandes). Con esa cuota — y su intervalo de confianza de Wilson al 95%, para no fingir
      más precisión de la que da la muestra — escalamos a los <b>2.210 jugadores</b> esperados en
      Barcelona para estimar cuántas copias de cada leyenda debería haber en la sala.
    </p>
    <p class="viz-sub" style="margin-bottom:14px;">
      <b>Campo plano (columna "≥1 vez, campo plano"):</b> probabilidad de cruzarte con esa leyenda al
      menos una vez en las <b>8 rondas suizas del día 1</b> si cada ronda te empareja con alguien al azar
      de todo el campo, sin importar récords.
      <br><br>
      <b>Con drift por win rate (columnas de la derecha):</b> eso no es realista — el Swiss empareja por
      récord, así que si tú vas ganando, cada ronda te enfrentas a otros jugadores que también van
      ganando, y quién sobrevive en ese grupo depende directamente del win rate de cada mazo. Aquí cada
      ronda reescalamos la representación de cada leyenda por su propio win rate elevado al número de
      rondas ganadas — así los mazos por encima del 50% se concentran más en tu grupo ronda a ronda (y
      los que están por debajo se diluyen), asumiendo que <b>tú también vas ganando</b> todas las rondas.
      Con eso calculamos cuántas veces esperamos cruzarnos con cada leyenda (puede ser más de una vez en
      8 rondas) y la probabilidad de que sea ≥1 vez o ≥2 veces. Es una aproximación de primer orden del
      Swiss real, no una simulación exacta del algoritmo de emparejamiento (que evita repeticiones y usa
      puntos de partida, no solo win/loss); las leyendas sin muestra suficiente para un win rate fiable
      usan 50% como valor neutro (sin drift asumido). Por esta complejidad no llevamos intervalo de
      confianza en las columnas de drift — son la mejor estimación puntual, no un rango.
    </p>

    <div class="freq-table-wrap">
      <table class="freqtable">
        <thead>
          <tr>
            <th>Leyenda</th>
            <th>Win rate</th>
            <th>Cuota de metagame (IC 95%)</th>
            <th>Copias esperadas / 2.210 (IC 95%)</th>
            <th>≥1 vez, campo plano</th>
            <th>Veces esperadas (drift WR)</th>
            <th>≥1 vez (drift WR)</th>
            <th>≥2 veces (drift WR)</th>
          </tr>
        </thead>
        <tbody>
          {"".join(
            f'<tr><td class="name">{r["legend"]}</td>'
            f'<td>{r["win_rate_pct_display"]}%</td>'
            f'<td><div class="mini-track"><div class="mini-fill" style="width:{r["share_pct"]/max_share_pct*100:.1f}%; '
            f'background:color-mix(in oklab, var(--div-blue) {r["share_pct"]/max_share_pct*100:.0f}%, var(--surface-1))"></div></div>'
            f'<b>{r["share_pct"]}%</b> <span class="ci">({r["share_lo"]}–{r["share_hi"]}%, n={r["n"]})</span></td>'
            f'<td><b>{r["expected_count"]:.0f}</b> <span class="ci">({r["count_lo"]:.0f}–{r["count_hi"]:.0f})</span></td>'
            f'<td>{r["prob8_pct"]}%</td>'
            f'<td><b>{r["expected_times_drift"]}</b></td>'
            f'<td>{r["p_at_least_1_drift"]}%</td>'
            f'<td>{r["p_at_least_2_drift"]}%</td>'
            f'</tr>'
            for r in barcelona_rows
          )}
        </tbody>
      </table>
    </div>

    <div class="foot">
      Lectura rápida: con Kennen en torno al {barcelona_rows[0]["share_pct"]}% de cuota, esperaríamos
      unas {barcelona_rows[0]["expected_count"]:.0f} copias suyas entre los 2.210 jugadores de Barcelona
      (IC 95%: {barcelona_rows[0]["count_lo"]:.0f}–{barcelona_rows[0]["count_hi"]:.0f}). En campo plano,
      un {barcelona_rows[0]["prob8_pct"]}% de probabilidad de cruzarte con uno en las 8 rondas; si tú
      también vas ganando, el modelo con drift dice que esperarías cruzarte con Kennen unas
      {barcelona_rows[0]["expected_times_drift"]} veces de media en esas 8 rondas ({barcelona_rows[0]["p_at_least_1_drift"]}%
      de al menos una vez, {barcelona_rows[0]["p_at_least_2_drift"]}% de dos o más). Esto no considera el
      espejo por separado — si juegas tú mismo Kennen, esa fila ES tu probabilidad de espejo. Tampoco
      modela el corte del día 2 (Top 64 de 2.210): esta tabla
      describe el campo completo del día 1, no quién sobrevive a él.
    </div>
  </div>

  <div class="card">
    <h2>Matchups — win rate de una leyenda contra cada otra</h2>
    <p class="viz-sub" style="margin-bottom:14px;">
      Elige una leyenda para ver su "rosco" de matchups: cada nodo alrededor es un rival, coloreado
      por win rate (mismo azul/rojo divergente centrado en 50%). Filtra por torneo individual o mira
      los 4 combinados ("General"), y dentro de cada uno por nivel de jugadores — la fuente no ofrece
      un corte exacto "Top 64" / "Top 8", solo porcentajes relativos al tamaño de cada torneo, así que
      Top 25% y Top 10% son la aproximación más cercana disponible.
    </p>

    <div class="toggle-row" id="tournamentToggle">
      <button class="toggle-btn active" data-scope="general">General (4 torneos)</button>
      {"".join(f'<button class="toggle-btn" data-scope="{s["key"]}">{s["label"]}</button>' for s in tournament_scopes)}
    </div>

    <div class="toggle-row" id="tierToggle">
      <button class="toggle-btn active" data-tier="any">Todos los jugadores</button>
      <button class="toggle-btn" data-tier="top25">Top 25% por torneo</button>
      <button class="toggle-btn" data-tier="top10">Top 10% por torneo (≈ Top 64)</button>
    </div>

    <div class="foot" id="matchupCoverage" style="margin-bottom:14px;"></div>

    <div class="picker" id="legendPicker"></div>

    <div class="legend-scale">
      <span>Desfavorable</span>
      <div class="bar"></div>
      <span>50%</span>
      <div class="bar" style="background:linear-gradient(to right, var(--div-mid), var(--div-blue));flex:1"></div>
      <span>Favorable</span>
    </div>

    <div class="rosco-wrap">
      <svg id="roscoSvg" viewBox="0 0 1300 1300"></svg>
    </div>

    <div class="foot" id="roscoFoot"></div>
  </div>
</div>

<div class="tooltip" id="tip"></div>

<script>
(function() {{
  const data = {json.dumps(legends, ensure_ascii=False)};
  const tip = document.getElementById('tip');
  const rows = document.querySelectorAll('.bar-row');

  rows.forEach(row => {{
    const i = +row.dataset.i;
    const d = data[i];
    row.addEventListener('mousemove', e => {{
      tip.style.display = 'block';
      tip.style.left = (e.clientX + 14) + 'px';
      tip.style.top = (e.clientY + 14) + 'px';
      const wr = d.win_rate_pct !== null ? d.win_rate_pct + '%' : 'muestra insuficiente';
      const games = d.total_games !== null ? d.total_games.toLocaleString('es-ES') : 'n/d';
      tip.innerHTML = '<b>' + d.legend + '</b><br>Veces jugada: ' + d.times_played +
        '<br>Partidas totales: ' + games + '<br>Win rate: ' + wr;
    }});
    row.addEventListener('mouseleave', () => tip.style.display = 'none');
  }});

  const btnChart = document.getElementById('btnChart');
  const btnTable = document.getElementById('btnTable');
  const chartView = document.getElementById('chartView');
  const table = document.getElementById('dataTable');

  function renderTable(sortKey, asc) {{
    const rowsData = [...data].sort((a, b) => {{
      let av = a[sortKey], bv = b[sortKey];
      if (av === null) av = -1;
      if (bv === null) bv = -1;
      if (typeof av === 'string') return asc ? av.localeCompare(bv) : bv.localeCompare(av);
      return asc ? av - bv : bv - av;
    }});
    const tbody = table.querySelector('tbody');
    tbody.innerHTML = rowsData.map(d => (
      '<tr><td>' + d.legend + '</td><td>' + d.times_played + '</td><td>' +
      (d.total_games !== null ? d.total_games.toLocaleString('es-ES') : 'n/d') + '</td><td>' +
      (d.win_rate_pct !== null ? d.win_rate_pct + '%' : 'n/d') + '</td></tr>'
    )).join('');
  }}

  let currentSort = {{ key: 'times_played', asc: false }};
  renderTable(currentSort.key, currentSort.asc);

  table.querySelectorAll('th').forEach(th => {{
    th.addEventListener('click', () => {{
      const key = th.dataset.key;
      const asc = currentSort.key === key ? !currentSort.asc : false;
      currentSort = {{ key, asc }};
      table.querySelectorAll('th').forEach(h => h.classList.remove('sorted', 'asc'));
      th.classList.add('sorted');
      if (asc) th.classList.add('asc');
      renderTable(key, asc);
    }});
  }});
  table.querySelector('th[data-key="times_played"]').classList.add('sorted');

  btnChart.addEventListener('click', () => {{
    btnChart.classList.add('active'); btnTable.classList.remove('active');
    chartView.classList.add('active'); table.classList.remove('active');
  }});
  btnTable.addEventListener('click', () => {{
    btnTable.classList.add('active'); btnChart.classList.remove('active');
    table.classList.add('active'); chartView.classList.remove('active');
  }});

  // ---- Matchup rosco ----
  const matchups = {json.dumps(matchups, ensure_ascii=False)};
  const legendOrder = {json.dumps(legend_order, ensure_ascii=False)};
  const pickerLegends = {json.dumps(picker_legends, ensure_ascii=False)};

  // Plain first-name is ambiguous when two legends share it (e.g. the two
  // Master Yi variants) -- disambiguate only those with their epithet.
  const shortNameBase = n => n.split(',')[0].trim();
  const shortNameCounts = {{}};
  legendOrder.forEach(n => {{
    const b = shortNameBase(n);
    shortNameCounts[b] = (shortNameCounts[b] || 0) + 1;
  }});
  const shortName = n => {{
    const base = shortNameBase(n);
    if (shortNameCounts[base] > 1) {{
      const epithet = n.split(',').slice(1).join(',').trim();
      return base + ' (' + epithet + ')';
    }}
    return base;
  }};

  const picker = document.getElementById('legendPicker');
  const svg = document.getElementById('roscoSvg');
  const roscoFoot = document.getElementById('roscoFoot');
  const tierToggle = document.getElementById('tierToggle');
  const tournamentToggle = document.getElementById('tournamentToggle');
  const matchupCoverage = document.getElementById('matchupCoverage');

  let currentTier = 'any';
  let currentScope = 'general';
  let currentPilot = pickerLegends[0];

  function currentScenarioKey() {{
    return currentScope + '_' + currentTier;
  }}

  picker.innerHTML = pickerLegends.map(n =>
    '<button class="picker-btn' + (n === currentPilot ? ' active' : '') + '" data-legend="' + n.replace(/"/g, '&quot;') + '">' +
    shortName(n) + '</button>'
  ).join('');

  function wrColor(wr) {{
    if (wr === null || wr === undefined) return 'var(--gridline)';
    const hue = wr >= 50 ? 'var(--div-blue)' : 'var(--div-red)';
    const pct = Math.min(Math.abs(wr - 50) * 2, 100);
    return 'color-mix(in oklab, ' + hue + ' ' + pct.toFixed(0) + '%, var(--div-mid))';
  }}

  function renderRosco() {{
    const opponents = legendOrder.filter(n => n !== currentPilot);
    const tierData = (matchups[currentScenarioKey()] || {{}})[currentPilot] || {{}};
    const N = opponents.length;
    // ringR sized so adjacent node circles never touch even at N=36-ish:
    // circumference / N must clear nodeR*2 + a gap. cx/cy leave enough margin
    // for the longest disambiguated label ("Master Yi (Wuju Bladesman)") to
    // fit past labelR without clipping the SVG viewBox on any side.
    const cx = 650, cy = 650, ringR = 380, nodeR = 23, centerR = 60, labelR = ringR + nodeR + 16;

    let svgHtml = '';
    // spokes first (so nodes render on top)
    opponents.forEach((opp, i) => {{
      const angle = (i / N) * 2 * Math.PI - Math.PI / 2;
      const x = cx + ringR * Math.cos(angle);
      const y = cy + ringR * Math.sin(angle);
      svgHtml += '<line class="rosco-spoke" x1="' + cx + '" y1="' + cy + '" x2="' + x.toFixed(1) + '" y2="' + y.toFixed(1) + '"></line>';
    }});

    svgHtml += '<g class="rosco-center"><circle cx="' + cx + '" cy="' + cy + '" r="' + centerR + '"></circle>' +
      '<text x="' + cx + '" y="' + (cy - 6) + '" text-anchor="middle" font-size="13">' + shortName(currentPilot) + '</text>' +
      '<text x="' + cx + '" y="' + (cy + 14) + '" text-anchor="middle" font-size="11" opacity="0.85">piloto</text></g>';

    opponents.forEach((opp, i) => {{
      const angle = (i / N) * 2 * Math.PI - Math.PI / 2;
      const x = cx + ringR * Math.cos(angle);
      const y = cy + ringR * Math.sin(angle);
      const d = tierData[opp];
      const insufficient = !d || d.matches === undefined || d.matches === null;
      const fill = insufficient ? '' : ('style="fill:' + wrColor(d.win_rate) + '"');
      const wrLabel = insufficient ? 'S/D' : (d.win_rate + '%');
      // Label sits further out along the SAME radius as the node (not just
      // "above" it) so it doesn't drift into neighboring nodes at the sides
      // of the circle -- the old fixed vertical offset only worked near the
      // top and broke down at 3/9 o'clock, which is what caused the overlap.
      const lx = cx + labelR * Math.cos(angle);
      const ly = cy + labelR * Math.sin(angle);
      const cosA = Math.cos(angle);
      const anchor = cosA > 0.15 ? 'start' : (cosA < -0.15 ? 'end' : 'middle');
      svgHtml += '<g class="rosco-node' + (insufficient ? ' insufficient' : '') + '" data-opp="' + opp.replace(/"/g, '&quot;') + '">' +
        '<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="' + nodeR + '" ' + fill + '></circle>' +
        '<text class="wr-label" x="' + x.toFixed(1) + '" y="' + (y + 4).toFixed(1) + '" text-anchor="middle" font-size="11.5">' + wrLabel + '</text>' +
        '<text class="opp-name" x="' + lx.toFixed(1) + '" y="' + (ly + 3.5).toFixed(1) + '" text-anchor="' + anchor + '" font-size="12">' + shortName(opp) + '</text>' +
        '</g>';
    }});

    svg.innerHTML = svgHtml;

    svg.querySelectorAll('.rosco-node').forEach(node => {{
      const opp = node.dataset.opp;
      const d = tierData[opp];
      node.addEventListener('mousemove', e => {{
        tip.style.display = 'block';
        tip.style.left = (e.clientX + 14) + 'px';
        tip.style.top = (e.clientY + 14) + 'px';
        if (!d) {{
          tip.innerHTML = '<b>' + shortName(currentPilot) + ' vs ' + shortName(opp) + '</b><br>Sin datos suficientes en este nivel';
        }} else {{
          tip.innerHTML = '<b>' + shortName(currentPilot) + ' vs ' + shortName(opp) + '</b><br>Win rate: ' + d.win_rate +
            '%<br>Partidas: ' + d.matches;
        }}
      }});
      node.addEventListener('mouseleave', () => tip.style.display = 'none');
    }});

    const withData = opponents.filter(o => tierData[o]).length;
    roscoFoot.textContent = shortName(currentPilot) + ': ' + withData + ' de ' + N + ' matchups con datos en este nivel. ' +
      'Pasa el ratón sobre un nodo para ver partidas exactas.';
  }}

  function updateCoverage() {{
    const scopeLabel = tournamentToggle.querySelector('.toggle-btn.active').textContent;
    const tierLabel = tierToggle.querySelector('.toggle-btn.active').textContent;
    const matrix = matchups[currentScenarioKey()] || {{}};
    const pilotCount = Object.keys(matrix).length;
    matchupCoverage.textContent = scopeLabel + ' · ' + tierLabel + ': ' + pilotCount +
      ' leyendas con muestra suficiente para matchups en este recorte.';
  }}

  picker.addEventListener('click', e => {{
    const btn = e.target.closest('.picker-btn');
    if (!btn) return;
    currentPilot = btn.dataset.legend;
    picker.querySelectorAll('.picker-btn').forEach(b => b.classList.toggle('active', b === btn));
    renderRosco();
  }});

  tierToggle.addEventListener('click', e => {{
    const btn = e.target.closest('.toggle-btn');
    if (!btn) return;
    currentTier = btn.dataset.tier;
    tierToggle.querySelectorAll('.toggle-btn').forEach(b => b.classList.toggle('active', b === btn));
    updateCoverage();
    renderRosco();
  }});

  tournamentToggle.addEventListener('click', e => {{
    const btn = e.target.closest('.toggle-btn');
    if (!btn) return;
    currentScope = btn.dataset.scope;
    tournamentToggle.querySelectorAll('.toggle-btn').forEach(b => b.classList.toggle('active', b === btn));
    updateCoverage();
    renderRosco();
  }});

  updateCoverage();
  renderRosco();
}})();
</script>
"""

out_path = DIR / "chart.html"
out_path.write_text(html, encoding="utf-8")
print(f"Wrote {out_path}")
