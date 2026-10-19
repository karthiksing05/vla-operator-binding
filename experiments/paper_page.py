"""Build the results page (the artifact https://claude.ai/artifact/9TTYizddrmLnsA6GxuLeQk) -> results/robot/site/index.html.

The current program only (experiments/report/consolidated.py opens it: the comparison with Wang et al., the experimental design,
the consolidated results tables; then experiments/report/current.py).  The earlier program is no longer shown.
Every number is read from ledger/.  Media (rollout clips, figures) come from experiments/site_media.py and experiments/fig_*.py.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/paper_page.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import base, consolidated, current                            # noqa: E402

secs = list(current.SECTIONS)
i = next(k for k, x in enumerate(secs) if x[0] == "c-overview") + 1
secs[i:i] = [("c-iclr", "", "How this differs from Wang et al. (ICLR 2027)", consolidated.iclr_diff, False),
             ("c-design", "", "Experimental design", consolidated.design, False),
             ("c-results", "", "Results: every method on every benchmark", consolidated.results, False),
             ("c-scaling", "", "How space and time scale with the number of tasks", consolidated.scaling, False)]
part1 = [(sid, num, title, (current.DESIGN[sid]() if sid in current.DESIGN else "") + fn(), collapsed) for sid, num, title, fn, collapsed in secs]
header = ('<header class="title"><p class="kicker">lifelong-ood-learner · working paper</p>'
          "<h1>Out-of-distribution learning is composition: one system, iterated, on five frozen flow-matching policies</h1>"
          f'<p class="meta">pi0.5, pi0, SmolVLA, X-VLA and GR00T N1.5 on LIBERO-90, Li et al.\'s compositional OOD suites and '
          f"LIBERO-PRO · every number from the project\'s ledger at commit {base.esc(base.commit)}; boxes marked "
          "<em>Running</em> fill in as their pre-registered reads land.</p></header>")
HTML = base.page("OOD Learning Is Composition",
                 "A frozen flow-matching robot policy learns new tasks by composing concepts it holds or learns from one example, "
                 "on five architectures and three benchmarks.",
                 header, [(None, part1)])
os.makedirs(base.SITE, exist_ok=True)
open(os.path.join(base.SITE, "index.html"), "w", encoding="utf-8").write(HTML)
print(os.path.join(base.SITE, "index.html"), len(HTML))
