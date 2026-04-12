"""Generate equity curve chart for the final recommended strategy."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.sweep_v2 import run_v2


def main():
    # Config A: The recommended production config
    config = {
        "min_our_payout": 1.85,
        "base_frac": 0.06, "cap_bnb": 2.0, "floor_bnb": 0.10,
        "btc_agree_mult": 1.5, "btc_disagree_mult": 0.7,
        "evening_skip": None, "pool_confirm_thresh": None,
        "payout_sizing_mode": "linear",
        "payout_linear_base": 0.1, "payout_linear_slope": 1.0,
        "btc_only_signal": True,
        "btc_only_thresh": 0.0003, "btc_only_min_payout": 3.0, "btc_only_bet": 0.15,
        "allowed_hours": [h for h in range(24) if h not in [3, 4]],
    }

    net, segs, trades, hrs = run_v2(config, verbose=True)

    # Build equity curve (only bet events)
    equity_points = []
    bet_idx = 0
    for t in trades:
        if t[1] == "BET":
            bet_idx += 1
            equity_points.append({"x": bet_idx, "y": round(t[3] - 50.0, 4),
                                   "tier": t[4], "sig": t[5],
                                   "profit": round(t[2], 4)})

    # Build hour chart data
    hour_data = []
    for h in range(24):
        if h in hrs:
            w, l, pnl = hrs[h]
            hour_data.append({"hour": h, "wins": w, "losses": l, "pnl": round(pnl, 2)})
        else:
            hour_data.append({"hour": h, "wins": 0, "losses": 0, "pnl": 0})

    # Segment stats for annotation
    seg_labels = []
    for i, (nb, wr, pnl) in enumerate(segs):
        seg_labels.append(f"Seg{i+1}: {nb} bets, WR={wr:.1f}%, PnL={pnl:+.2f}")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>PancakeBot Equity Curve - Final Strategy</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; margin: 20px; }}
  .card {{ background: #161b22; border-radius: 12px; padding: 20px; margin-bottom: 20px; border: 1px solid #30363d; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
  .stat {{ text-align: center; }}
  .stat .value {{ font-size: 2em; font-weight: bold; }}
  .stat .label {{ font-size: 0.85em; color: #8b949e; }}
  .green {{ color: #3fb950; }}
  .red {{ color: #f85149; }}
  .yellow {{ color: #d29922; }}
  canvas {{ max-height: 400px; }}
  h1 {{ color: #58a6ff; }}
  h2 {{ color: #c9d1d9; margin-top: 0; }}
  .segments {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 15px; }}
  .seg {{ background: #21262d; border-radius: 8px; padding: 12px; text-align: center; }}
  .config {{ font-size: 0.8em; color: #8b949e; line-height: 1.6; }}
</style></head><body>
<h1>PancakeBot — Final Strategy Performance</h1>

<div class="card">
  <div class="grid">
    <div class="stat"><div class="value green">+{net:.2f}</div><div class="label">Net PnL (BNB)</div></div>
    <div class="stat"><div class="value">{sum(s[0] for s in segs)}</div><div class="label">Total Bets</div></div>
    <div class="stat"><div class="value">{sum(1 for t in trades if t[1]=='BET' and t[2]>0)/max(1,sum(1 for t in trades if t[1]=='BET'))*100:.1f}%</div><div class="label">Win Rate</div></div>
    <div class="stat"><div class="value">{net/max(1,sum(1 for t in trades if t[1]=='BET')):.3f}</div><div class="label">Avg PnL/Bet</div></div>
  </div>
</div>

<div class="card">
  <h2>Equity Curve (20k rounds)</h2>
  <canvas id="equityChart"></canvas>
</div>

<div class="card">
  <h2>Segment Performance (4 segments × 5k rounds)</h2>
  <div class="segments">
    {"".join(f'<div class="seg"><div class="value {"green" if s[2]>0 else "red"}">{s[2]:+.2f}</div><div class="label">{s[0]} bets | WR {s[1]:.1f}%</div></div>' for s in segs)}
  </div>
</div>

<div class="card">
  <h2>PnL by Hour (UTC)</h2>
  <canvas id="hourChart"></canvas>
</div>

<div class="card">
  <h2>Strategy Configuration</h2>
  <div class="config">
    <b>Signal:</b> BNB 1s accel (pairs: 7/10, 5/10, 5/7, thresh: 0.0002) + BTC confirmation (30s, thresh: 0.0003)<br>
    <b>Payout floor:</b> min_our_payout ≥ 1.85 (skip rounds where payout multiplier on our side &lt; 1.85)<br>
    <b>Sizing:</b> Linear payout-proportional: mult = 0.1 + 1.0 × (payout − 1.0), base=6% of pool, floor=0.10, cap=2.0<br>
    <b>BTC modifiers:</b> agree ×1.5, disagree ×0.7<br>
    <b>BTC contrarian:</b> On non-accel rounds, bet AGAINST BTC direction if our side payout ≥ 3.0 (fixed 0.15 BNB)<br>
    <b>Hour filter:</b> Skip hours 3-4 UTC (low-liquidity, sub-45% WR)<br>
    <b>No evening skip, no pool confirmation filter</b>
  </div>
</div>

<script>
const points = {json.dumps(equity_points)};
const hourData = {json.dumps(hour_data)};

// Equity curve
new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{
    datasets: [{{
      label: 'Cumulative PnL (BNB)',
      data: points.map(p => ({{x: p.x, y: p.y}})),
      borderColor: '#3fb950',
      backgroundColor: 'rgba(63,185,80,0.1)',
      fill: true,
      pointRadius: 0,
      borderWidth: 2,
      tension: 0.1
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ labels: {{ color: '#c9d1d9' }} }},
      tooltip: {{
        callbacks: {{
          label: ctx => `PnL: ${{ctx.parsed.y.toFixed(2)}} BNB (bet #${{ctx.parsed.x}})`
        }}
      }}
    }},
    scales: {{
      x: {{ type: 'linear', title: {{ display: true, text: 'Bet #', color: '#8b949e' }}, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
      y: {{ title: {{ display: true, text: 'Cumulative PnL (BNB)', color: '#8b949e' }}, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }}
    }}
  }}
}});

// Hour chart
new Chart(document.getElementById('hourChart'), {{
  type: 'bar',
  data: {{
    labels: hourData.map(h => h.hour + ':00'),
    datasets: [{{
      label: 'PnL (BNB)',
      data: hourData.map(h => h.pnl),
      backgroundColor: hourData.map(h => h.pnl >= 0 ? 'rgba(63,185,80,0.7)' : 'rgba(248,81,73,0.7)'),
      borderColor: hourData.map(h => h.pnl >= 0 ? '#3fb950' : '#f85149'),
      borderWidth: 1
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#c9d1d9' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
      y: {{ title: {{ display: true, text: 'PnL (BNB)', color: '#8b949e' }}, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }}
    }}
  }}
}});
</script>
</body></html>"""

    out = Path("var/backtest_chart_v2.html")
    out.write_text(html, encoding="utf-8")
    print(f"\nChart written to {out}")


if __name__ == "__main__":
    main()
