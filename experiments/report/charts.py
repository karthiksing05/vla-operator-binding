"""Charts drawn to scale as inline SVG, coloured by the page's theme tokens."""
from .base import esc


def methods_chart(rows):
    """Success rate per method on the stream's tasks (fresh start states), to scale 0-1; each bar labelled with its successes,
    the demonstrations it used and the bytes it stores.  rows: (label, successes, episodes, demos, bytes, ours)."""
    W, top, bh, gap, right = 640, 30, 18, 30, 150
    scale = (W - right)
    h = top + len(rows) * (bh + gap) + 20
    out = [f'<svg viewBox="0 0 {W} {h}" class="chart" role="img" aria-label="success rate per method with demonstrations and bytes">',
           '<rect x="0" y="4" width="12" height="10" class="bar ours"/><text x="18" y="13" class="tick">this scheme</text>',
           '<rect x="110" y="4" width="12" height="10" class="bar base"/><text x="128" y="13" class="tick">baselines</text>']
    y = top
    for lab, k, n, demos, nb, ours in rows:
        r = k / n if n else 0.0
        out.append(f'<text x="0" y="{y - 4}" class="lab">{esc(lab)} <tspan class="dim">({demos} demonstrations, '
                   f'{nb / 1e6:.1f} MB)</tspan></text>')
        out.append(f'<rect x="0" y="{y}" width="{max(r * scale, 2):.1f}" height="{bh}" rx="2" class="bar {"ours" if ours else "base"}">'
                   f'<title>{esc(lab)}: {k} of {n}</title></rect>')
        out.append(f'<text x="{r * scale + 6:.1f}" y="{y + bh - 4}" class="val">{r:.2f} <tspan class="dim">{k} of {n}</tspan></text>')
        y += bh + gap
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<text x="{t * scale:.1f}" y="{y + 2}" class="tick" text-anchor="{"start" if t == 0 else "middle"}">{t:.2f}</text>')
    out.append("</svg>")
    return "".join(out)


def dot_rows(rows, arms, xmax, label_w=250, unit="of 20"):
    """rows = [(label, {arm: value}, highlight)], arms = [(key, name, css)] -> a dot plot to scale (0..xmax)."""
    W, rh, top = 720, 22, 26
    plot = W - label_w - 20
    sx = lambda v: label_w + v / xmax * plot                                 # noqa: E731
    H = top + len(rows) * rh + 22
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="per task, successes of each arm">']
    for i, (key, name, css) in enumerate(arms):
        out.append(f'<circle cx="{label_w + i * 150 + 6}" cy="10" r="5" class="dot {css}"/><text x="{label_w + i * 150 + 16}" y="14" '
                   f'class="tick">{esc(name)}</text>')
    for t in range(0, xmax + 1, 5):
        out.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{top - 4}" y2="{H - 18}" class="gridl"/>'
                   f'<text x="{sx(t):.1f}" y="{H - 4}" class="tick" text-anchor="middle">{t}</text>')
    for j, (lab, vals, hl) in enumerate(rows):
        y = top + j * rh + rh / 2
        out.append(f'<text x="{label_w - 8}" y="{y + 4}" class="rl{" hl" if hl else ""}" text-anchor="end">{esc(lab)}</text>')
        xs = [sx(vals[k]) for k, _, _ in arms if k in vals]
        if xs:
            out.append(f'<line x1="{min(xs):.1f}" x2="{max(xs):.1f}" y1="{y}" y2="{y}" class="span"/>')
        for k, _, css in arms:
            if k in vals:
                out.append(f'<circle cx="{sx(vals[k]):.1f}" cy="{y}" r="5" class="dot {css}"><title>{esc(lab)}: {vals[k]} {unit}</title></circle>')
    out.append(f'<text x="{label_w + plot}" y="{H - 4}" class="tick" text-anchor="end">successes {unit}</text>')
    out.append("</svg>")
    return "".join(out)
