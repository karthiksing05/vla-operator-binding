"""Build the results page (the artifact https://claude.ai/artifact/9TTYizddrmLnsA6GxuLeQk) -> results/robot/site/index.html.

The body is the paper: paper/DRAFT.md, rendered section by section with its figures and rollout clips (experiments/report/paper.py);
every number in the draft is audited against ledger/ by experiments/audit_numbers.py.  The appendix keeps the ledger-driven tables of
experiments/report/consolidated.py (the comparison with Wang et al., the experimental design, every method on every benchmark) and
the verification notes.  Media come from experiments/site_media.py, experiments/fig_*.py and experiments/fig_plan_z.py.

    ~/flash/miniconda3/envs/chunks/bin/python experiments/paper_page.py
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import base, consolidated, current, paper                     # noqa: E402

for f in ("z_grades.png", "z_unsupervised.png", "z_blend.png"):            # plan Z's figures (experiments/fig_plan_z.py)
    shutil.copy(os.path.join(base.ROOT, "figures", "robot", f), os.path.join(base.SITE, "media", f))

title, meta, secs = paper.sections()
body = [(sid, num, name, html, False) for sid, num, name, html in secs]
appendix = [("a-iclr", "A", "How this differs from Wang et al. (ICLR 2027)", consolidated.iclr_diff(), True),
            ("a-design", "B", "Experimental designs", consolidated.design(), True),
            ("a-results", "C", "Every method on every benchmark (ledger tables)", consolidated.results(), True),
            ("a-verify", "D", "How the numbers are verified", current.verification(), True)]
header = (f'<header class="title"><p class="kicker">lifelong-ood-learner · working paper</p><h1>{base.esc(title)}</h1>'
          f'<p class="meta">pi0.5, pi0, SmolVLA, X-VLA, GR00T N1.5 and FLOWER on LIBERO-90, Li et al.\'s compositional OOD suites and '
          f"LIBERO-PRO · the text is <code>paper/DRAFT.md</code>, every number audited against the project's ledger at commit "
          f"{base.esc(base.commit)}.</p></header>")
HTML = base.page("OOD Learning Is Composition",
                 "A frozen flow-matching robot policy learns new tasks by composing what it holds and creating units only where a task "
                 "leaves a scene or part unseen: a supervised and an unsupervised scheme on a graded compositional split.",
                 header, [(None, body), ("Appendix", appendix)])
os.makedirs(base.SITE, exist_ok=True)
open(os.path.join(base.SITE, "index.html"), "w", encoding="utf-8").write(HTML)
print(os.path.join(base.SITE, "index.html"), len(HTML))
