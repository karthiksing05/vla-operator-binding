"""The paper page's body: paper/DRAFT.md (audited by experiments/audit_numbers.py) rendered section by section, with the figures and
rollout clips each section shows.  A small Markdown reader covers what the draft uses: ### subheadings, paragraphs, **bold**,
*italic*, `code`, [links](url), nested - / 1. lists with wrapped lines, and | tables |."""
import html
import os
import re

from . import base, diagrams

DRAFT = os.path.join(base.ROOT, "paper", "DRAFT.md")
LIST = re.compile(r"^(\s*)(- |\d+\. )(.*)$")


def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
    return s


def _table(lines):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines if not re.match(r"^\s*\|[-:| ]+\|\s*$", l)]
    head, body = rows[0], rows[1:]
    h = "".join(f"<th>{inline(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
    return f'<div class="tbl"><div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div></div>'


def _list(lines):
    """Nested lists from '- ' / '1. ' items; an indented line without a marker continues the previous item."""
    items = []                                                          # (depth indent, ordered, text)
    for l in lines:
        m = LIST.match(l)
        if m:
            items.append([len(m.group(1)), m.group(2) != "- ", m.group(3)])
        elif items:
            items[-1][2] += " " + l.strip()
    out, stack = [], []
    for ind, ordered, text in items:
        while stack and ind < stack[-1][0]:
            out.append("</li></%s>" % stack.pop()[1])
        if not stack or ind > stack[-1][0]:
            tag = "ol" if ordered else "ul"
            stack.append((ind, tag))
            out.append(f"<{tag}><li>{inline(text)}")
        else:
            out.append(f"</li><li>{inline(text)}")
    while stack:
        out.append("</li></%s>" % stack.pop()[1])
    return "".join(out)


def render(md):
    out, block = [], []

    def flush():
        if not block:
            return
        if all(l.lstrip().startswith("|") for l in block):
            out.append(_table(block))
        elif LIST.match(block[0]):
            out.append(_list(block))
        else:
            out.append(f"<p>{inline(' '.join(l.strip() for l in block))}</p>")
        block.clear()
    for line in md.splitlines():
        if line.startswith("### "):
            flush()
            out.append(f"<h3>{inline(line[4:].strip())}</h3>")
        elif not line.strip():
            flush()
        else:
            if block and (LIST.match(line) and not LIST.match(block[0]) or line.lstrip().startswith("|") != block[0].lstrip().startswith("|")):
                flush()
            block.append(line)
    flush()
    return "".join(out)


def img(name, alt):
    return f'<img src="media/{name}" alt="{html.escape(alt)}" style="width:100%">'


# figures per section, by the section's title prefix (after its number): [(html, caption)]
def figures():
    F = base.fig
    return {
        "The framework": [
            (diagrams.episode(), "<b>One composed step (supervised scheme).</b> The object unit acts from the start until the policy's own "
                                 "grasp; the goal unit from the grasp to the release. A held unit is a pointer to the policy's own field."),
            (diagrams.hierarchy(), "<b>The library as concepts.</b> Object types with their instance children (approach) and goal "
                                   "concepts (transport); task leaves sit below only where practice prefers them.")],
        "The supervised scheme": [
            (img("iterations.png", "success per iteration"), "<b>The iterations.</b> Each iteration of the system against its reference "
                                                             "on LIBERO-90 and on Li's OOD suites."),
            (base.video("fac_53_0"), "<b>The approach is where the policy fails.</b> Pick up the orange juice and put it in the basket: "
                                     "the object unit takes the policy to the named object."),
            (base.video("inst_14_0"), "<b>An instance child.</b> The middle black bowl: the type prototype reaches for the wrong bowl, "
                                      "the instance child for the named one."),
            (diagrams.stream(), "<b>The learner on one arrival.</b> A demonstration is requested only when neither the policy nor the "
                                "composition passes practice."),
            (img("i5_dendrogram.png", "dendrograms of the prototypes"), "<b>The library in weight space.</b> Object and goal prototypes "
                                                                         "cluster by concept.")],
        "The unsupervised scheme": [
            (img("z_unsupervised.png", "boundaries vs identities"), "<b>Likelihood names units; contact events cut them.</b> Left: grasps "
             "found within 10 frames by likelihood-based segmenters against random cuts at the same count (72 pick-and-place demos). "
             "Right: the share of contact segments whose lowest-loss training instruction holds the segment's concept, against "
             "chance (ledger <code>zu_read</code>)."),
            (img("z_blend.png", "soft blend held-out loss and donation"), "<b>What a soft blend of held fields donates.</b> Left: "
             "held-out flow loss on the approach (position) for each expert set; lower is better. Right: mean Shapley share of the "
             "blend's donation by the expert's relation to the task, LIBERO-90 (ledger <code>zi_blend</code>).")],
        "Tests of the system": [
            (img("four_policies.png", "Li's OOD suites on four policies"), "<b>Li's compositional OOD suites</b> on the first four "
             "policies: alone, Li's TLI with hand-chosen pairs, and the composition practice chooses. GR00T N1.5 is in Figure "
             "<i>results by grade</i> and in the text.")],
        "Results by grade": [
            (img("z_grades.png", "success by grade"), "<b>The grade decides the regime.</b> Left: pi0.5 by what each task leaves unseen; "
             "relations and bindings (Li's tasks) from G15, scenes and parts from LIBERO-90 (pointers: Z3's tasks; leaves: G8's). "
             "Right: Li's 20 tasks pooled over G1 and G2 on five policies (ledger <code>split_grades</code>).")],
        "What composition is": [
            (img("h1_factor_ood.png", "composed success predicted from component competences"), "<b>Success ≈ grasp × place.</b> "
             "Composed success predicted from the competence of each phase's source, out of sample."),
            (img("i7_paths.png", "end-effector paths per unit"), "<b>What each unit does to the motion.</b> Top-down end-effector "
                                                                  "paths per unit from one start state.")],
        "What composition costs": [
            (img("h2_pareto_libero90.png", "success against storage and episodes"), "<b>Success against storage and practice</b> on "
                                                                                     "LIBERO-90, every method."),
            (img("h3_space.png", "storage against number of tasks"), "<b>How storage scales</b> with the number of tasks: one unit "
                                                                      "per concept against one per task.")],
    }


def sections():
    """-> (title, meta line, [(id, number, title, html)]) from the draft."""
    md = open(DRAFT, encoding="utf-8").read()
    title = md.splitlines()[0].lstrip("# ").strip()
    parts = re.split(r"^## ", md, flags=re.M)
    meta = parts[0].split("\n", 1)[1].strip()
    figs = figures()
    out = []
    for p in parts[1:]:
        head, body = p.split("\n", 1)
        m = re.match(r"(\d+) (.*)", head.strip())
        num, name = (m.group(1), m.group(2)) if m else ("", head.strip())
        content = render(body)
        for key, fl in figs.items():
            if name.startswith(key):
                content += "".join(base.fig(b, c) for b, c in fl)
        out.append((f"s-{num or name.lower()}", num, name, content))
    return title, meta, out
