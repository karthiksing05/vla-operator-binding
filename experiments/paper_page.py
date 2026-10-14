"""Build the results page (the artifact https://claude.ai/artifact/9TTYizddrmLnsA6GxuLeQk) -> results/robot/site/index.html.

Part I is the current program (experiments/report/current.py): the factored, hierarchical prototype library and its streams,
opening with an index of every read.  Part II is the earlier program (experiments/report/history.py), kept verbatim and collapsed.
Every number is read from ledger/.  Media (rollout clips, figures) come from experiments/site_media.py and experiments/fig_*.py.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/paper_page.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import base, current                                          # noqa: E402

part1 = [(sid, num, title, fn(), collapsed) for sid, num, title, fn, collapsed in current.SECTIONS]
from report import history                                                # noqa: E402  (built after Part I: figure numbers follow the page)

part2 = [(sid, num, title, content, True) for sid, num, title, content in history.SECTIONS]
part2.append(("h-limits", "", "Limitations of the earlier program", history.LIMITS + f"<p>{history.PROBE_NOTE}</p>", True))
header = ('<header class="title"><p class="kicker">lifelong-ood-learner · working paper</p>'
          "<h1>Composing a frozen robot policy's own prototypes: one example in, new tasks out</h1>"
          f'<p class="meta">pi0.5 on LIBERO-90 · every number from the project\'s ledger at commit {base.esc(base.commit)}; boxes marked '
          "<em>Running</em> fill in as their pre-registered reads land.</p></header>")
HTML = base.page("Read, Compose, Fold",
                 "A frozen robot policy takes on new LIBERO-90 tasks by composing object and goal prototypes learned from one example each.",
                 header, [("Part I · the current program", part1), ("Part II · the earlier program", part2)])
os.makedirs(base.SITE, exist_ok=True)
open(os.path.join(base.SITE, "index.html"), "w", encoding="utf-8").write(HTML)
print(os.path.join(base.SITE, "index.html"), len(HTML))
