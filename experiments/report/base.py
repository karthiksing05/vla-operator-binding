"""Shared pieces of the results page: ledger access, figure and table numbering, media, and the page skeleton.  Every number on
the page is read from ledger/ through row()."""
import html
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from lifelong_ood import ledger as L                                       # noqa: E402

R = os.path.join(ROOT, "results", "robot")
SITE = os.path.join(R, "site")
MEDIA = json.load(open(os.path.join(SITE, "media", "media.json")))
CSS = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css"), encoding="utf-8").read()
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
esc = lambda s: html.escape(str(s))                                        # noqa: E731
FIG, TAB = {"n": 0}, {"n": 0}


def row(exp, tag):
    """The latest unretired ledger row of (experiment, tag) -> its metrics, or None."""
    rows = [r for r in L.load(exp, tag=tag) if not r.get("retired")]
    return rows[-1]["metrics"] if rows else None


def fig(body, caption, wide=False):
    FIG["n"] += 1
    return (f'<figure class="{"wide" if wide else ""}" id="fig{FIG["n"]}">{body}<figcaption><b>Figure {FIG["n"]}.</b> '
            f'{caption}</figcaption></figure>')


def table(head, rows, caption, num_from=1, cls=""):
    TAB["n"] += 1
    h = "".join(f'<th class="{"n" if i >= num_from else ""}">{esc(x)}</th>' for i, x in enumerate(head))
    b = "".join("<tr>" + "".join(f'<td class="{"n" if i >= num_from else ""}">{x}</td>' for i, x in enumerate(r)) + "</tr>"
                for r in rows)
    return (f'<div class="tbl {cls}"><p class="tcap"><b>Table {TAB["n"]}.</b> {caption}</p><div class="scroll"><table>'
            f'<thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div></div>')


def video(name):
    m = next(g[name] for g in MEDIA.values() if isinstance(g, dict) and name in g)
    return (f'<video src="media/{m["file"]}" poster="media/{m["file"][:-4]}.jpg" controls playsinline muted preload="metadata">'
            f'</video>')


def design_box(question, data, arms, states, test):
    """A section's experimental design, as a compact labelled box."""
    rows = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in (("Question", question), ("Data", data), ("Arms", arms),
                                                          ("Start states", states), ("Test", test)))
    return f'<aside class="design"><p class="pl">Experimental design</p><dl>{rows}</dl></aside>'


def pending(title, text):
    return f'<aside class="pending"><p class="pl">Running</p><p class="pt">{esc(title)}</p><p>{text}</p></aside>'


def pct(x):
    return f"{100 * x:.0f}%"


def page(title, description, header, parts):
    """parts: [(part title or None, [(id, number, title, html, collapsed)])] -> the page.  The contents rail lists every section;
    a collapsed section is a <details> block (closed at load, opened by its link)."""
    toc, body = [], []
    for ptitle, secs in parts:
        if ptitle:
            toc.append(f'<li class="part">{esc(ptitle)}</li>')
            body.append(f'<h2 class="part">{esc(ptitle)}</h2>')
        for sid, num, stitle, content, collapsed in secs:
            toc.append(f'<li><a href="#{sid}"><span class="num">{num}</span><span>{esc(stitle)}</span></a></li>')
            head = f'<span class="sn">{num}</span>{esc(stitle)}' if num else esc(stitle)
            if collapsed:
                body.append(f'<details class="sec" id="{sid}"><summary><h3>{head}</h3></summary>{content}</details>')
            else:
                body.append(f'<section id="{sid}"><h2>{head}</h2>{content}</section>')
    opener = ("<script>function openHash(){var h=location.hash.slice(1);var e=h&&document.getElementById(h);"
              "if(e&&e.tagName==='DETAILS'){e.open=true;e.scrollIntoView();}}window.addEventListener('hashchange',openHash);"
              "openHash();</script>")
    return f"""<title>{esc(title)}</title>
<meta name="description" content="{esc(description)}">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<style>{CSS}</style>
<div class="shell">
<nav class="toc" aria-label="Contents"><ol>{"".join(toc)}</ol></nav>
<main>
{header}
{"".join(body)}
<footer>Simulation only (LIBERO-90). Every number is read from <code>ledger/</code> at commit {esc(commit)} by
<code>experiments/paper_page.py</code>; the README's copies are re-derived by <code>experiments/audit_numbers.py</code>.</footer>
</main></div>
{opener}
"""
