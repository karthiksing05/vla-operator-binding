"""Process diagrams of the current method, hand-drawn inline SVG themed by the page's tokens (classes .dg in style.css)."""
import json
import os

from .base import R, esc

ARROW = ('<defs><marker id="{id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
         '<path d="M0,0 L10,5 L0,10 z" class="ah"/></marker></defs>')


def _box(x, y, w, h, label, sub=None, cls="box"):
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" class="{cls}"/>'
    if sub is None:
        return out + f'<text x="{x + w / 2}" y="{y + h / 2 + 4}" text-anchor="middle" class="t">{esc(label)}</text>'
    return (out + f'<text x="{x + w / 2}" y="{y + h / 2 - 3}" text-anchor="middle" class="t">{esc(label)}</text>'
            f'<text x="{x + w / 2}" y="{y + h / 2 + 13}" text-anchor="middle" class="s">{esc(sub)}</text>')


def _line(x1, y1, x2, y2, mid, label=None, dx=6, dy=-6, anchor="start"):
    out = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="ln" marker-end="url(#{mid})"/>'
    if label:
        out += f'<text x="{(x1 + x2) / 2 + dx}" y="{(y1 + y2) / 2 + dy}" class="s" text-anchor="{anchor}">{esc(label)}</text>'
    return out


def episode():
    """One episode of a composed step: the object prototype until the policy's own grasp, the goal prototype until the release."""
    W, H, x0, g, r, x1 = 720, 250, 120, 380, 610, 700
    m = "ar-ep"
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart dg" role="img" aria-label="An episode of a composed step: the object prototype '
           f'acts from the start to the policy\'s own grasp, the goal prototype from the grasp to the release; the frozen policy runs '
           f'throughout.">', ARROW.format(id=m)]
    out.append(f'<text x="0" y="52" class="s">adapter on</text><text x="0" y="122" class="s">frozen policy</text>'
               f'<text x="0" y="182" class="s">gripper</text>')
    out.append(f'<rect x="{x0}" y="30" width="{g - x0}" height="36" rx="6" class="box hl"/>'
               f'<text x="{(x0 + g) / 2}" y="46" text-anchor="middle" class="t">OBJECT prototype</text>'
               f'<text x="{(x0 + g) / 2}" y="60" text-anchor="middle" class="s">put | the black bowl (| middle)</text>')
    out.append(f'<rect x="{g}" y="30" width="{r - g}" height="36" rx="6" class="box hl2"/>'
               f'<text x="{(g + r) / 2}" y="46" text-anchor="middle" class="t">GOAL prototype</text>'
               f'<text x="{(g + r) / 2}" y="60" text-anchor="middle" class="s">on | the plate</text>')
    out.append(f'<text x="{(r + x1) / 2}" y="52" text-anchor="middle" class="s">none</text>')
    out.append(f'<rect x="{x0}" y="100" width="{x1 - x0}" height="34" rx="6" class="box"/>'
               f'<text x="{(x0 + x1) / 2}" y="121" text-anchor="middle" class="t">pi0.5 generates every action (weights frozen)</text>')
    out.append(f'<polyline points="{x0},190 {g - 30},190 {g - 10},166 {r},166 {r + 20},190 {x1},190" class="ln"/>'
               f'<text x="{x0 + 4}" y="206" class="s">open</text><text x="{(g + r) / 2}" y="160" text-anchor="middle" class="s">closed</text>')
    for x in (g, r):
        out.append(f'<line x1="{x}" y1="22" x2="{x}" y2="222" class="ev"/>')
    out.append(f'<text x="{g}" y="240" text-anchor="middle" class="s">grasp (the policy\'s own gripper closes)</text>'
               f'<text x="{r}" y="240" text-anchor="start" class="s" dx="-40">release</text>')
    out.append(_line(x0, 84, g - 4, 84, m, "approach: which object", dx=-60, dy=-4)
               + _line(g + 4, 84, r - 4, 84, m, "transport: where it goes", dx=-60, dy=-4))
    return "".join(out) + "</svg>"


def hierarchy():
    """The library as concepts: object types with instance children, goal concepts; one task picks one object node and one goal."""
    plan = json.load(open(os.path.join(R, "e3_plan.json")))
    n_obj, n_inst, n_goal = {}, {}, {}
    for v in plan.values():
        n_obj[v["object_key"]] = len(v["obj_tasks"])
        n_goal[v["goal_key"]] = len(v["goal_tasks"])
        if v["instance_key"]:
            n_inst[v["instance_key"]] = len(v["object_tasks"])
    W, H = 720, 322
    m = "ar-hi"
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart dg" role="img" aria-label="The library as a hierarchy of concepts: object types '
           f'with instance children, and goal concepts; the task put the middle black bowl on the plate composes the middle child and '
           f'the plate goal.">', ARROW.format(id=m)]
    out.append('<text x="10" y="18" class="s">object concepts (approach) · demonstrations</text>'
               '<text x="492" y="18" class="s">goal concepts (transport)</text>')
    bb = n_obj.get("put|the black bowl", 0)
    out.append(_box(10, 32, 200, 40, "the black bowl", f"type · {bb} demos"))
    kids = [("middle", "put|the black bowl|middle"), ("back", "put|the black bowl|back"), ("front", "put|the black bowl|front")]
    for i, (lab, k) in enumerate(kids):
        x = 10 + i * 140
        out.append(f'<line x1="110" y1="72" x2="{x + 60}" y2="112" class="ln"/>')
        out.append(_box(x, 112, 120, 40, lab, f"instance · {n_inst.get(k, 0)}", cls="box hl" if lab == "middle" else "box"))
    out.append(_box(230, 32, 200, 40, "the ketchup", f"type · {n_obj.get('put|the ketchup', 0)} demos"))
    out.append(f'<text x="10" y="314" class="s">the library: {len(n_obj)} object types, {len(n_inst)} instance children, {len(n_goal)} goals</text>')
    goals = [("on the plate", "on|the plate"), ("in the basket", "in|the basket"), ("on top of the cabinet", "on top of|the cabinet"),
             ("in the tray", "in|the tray")]
    for i, (lab, k) in enumerate(goals):
        out.append(_box(492, 32 + i * 48, 220, 38, lab, f"{n_goal.get(k, 0)} demos", cls="box hl2" if i == 0 else "box"))
    out.append(_box(180, 236, 360, 44, "task: put the middle black bowl on the plate", "composed from two nodes", cls="box task"))
    out.append(_line(300, 236, 72, 156, m, "approach", dx=-40, dy=6, anchor="end"))
    out.append(_line(470, 236, 488, 52, m, "transport", dx=-8, dy=20, anchor="end"))
    return "".join(out) + "</svg>"


def stream():
    """The learner on one arrival, with the one decision where the stream rules differ."""
    W, H = 720, 330
    m = "ar-st"
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart dg" role="img" aria-label="The stream learner: compose the arrival from the '
           f'library, practise it against the policy; keep the better without a demonstration if it reaches 8 of 10, else request the '
           f'one demonstration, refit every concept on its path, practise again and keep the refit composition only if it beats the '
           f'policy.">', ARROW.format(id=m)]
    out.append(_box(10, 20, 150, 46, "arrival", "instruction + scene"))
    out.append(_box(200, 20, 170, 46, "compose", "instance or type + goal"))
    out.append(_box(410, 20, 170, 46, "practise ×10", "composition vs policy"))
    out.append(_line(160, 43, 198, 43, m) + _line(370, 43, 408, 43, m))
    cx, cy = 495, 130
    out.append(f'<polygon points="{cx},{cy - 38} {cx + 110},{cy} {cx},{cy + 38} {cx - 110},{cy}" class="box hl"/>'
               f'<text x="{cx}" y="{cy - 3}" text-anchor="middle" class="t">better of the two</text>'
               f'<text x="{cx}" y="{cy + 13}" text-anchor="middle" class="s">reaches 8 of 10?</text>')
    out.append(_line(495, 66, 495, 90, m))
    out.append(_box(620, 106, 90, 48, "keep it", "no demo"))
    out.append(_line(605, 130, 618, 130, m) + '<text x="602" y="112" class="s">yes</text>')
    out.append(_box(400, 196, 190, 46, "request the one demo", "its own scene"))
    out.append(_line(495, 168, 495, 194, m, "no", dx=8, dy=4))
    out.append(_box(180, 196, 180, 46, "refit its concepts", "instance, type, goal"))
    out.append(_line(398, 219, 362, 219, m))
    out.append(_box(10, 196, 130, 46, "library", "the concepts", cls="box store"))
    out.append(_line(178, 219, 142, 219, m, "writes", dx=0, dy=-8, anchor="middle"))
    out.append(f'<path d="M75,196 L75,100 L278,100 L285,68" class="ln dash" marker-end="url(#{m})"/>'
               f'<text x="82" y="94" class="s">reads</text>')
    out.append(_box(180, 272, 300, 46, "practise again", "keep it only if it beats the policy"))
    out.append(_line(270, 242, 270, 270, m))
    out.append('<text x="620" y="178" class="s">E4 asked</text><text x="620" y="192" class="s">instead: does</text>'
               '<text x="620" y="206" class="s">composition</text><text x="620" y="220" class="s">beat policy?</text>')
    return "".join(out) + "</svg>"


def lineage(rows):
    """How the unit was factored, approach by approach.  rows: [(name, [(start, end, label, cls)], result)] with start / end the
    fraction of an episode (approach before the grasp, transport after it); cls in {pol, tok, step, obj, inst, goal}."""
    W, top, rh, x0, x1 = 720, 34, 46, 182, 470
    H = top + len(rows) * rh + 8
    out = [f'<svg viewBox="0 0 {W} {H}" class="chart dg" role="img" aria-label="How the unit stored per task was factored, approach by '
           f'approach: one unit for the whole task, one per step, then a goal unit after the grasp, then an object unit before it, '
           f'then instance children; each row with its result.">']
    gx = x0 + 0.5 * (x1 - x0)
    out.append(f'<text x="{x0}" y="14" class="s">approach</text>'
               f'<text x="{x1}" y="14" class="s" text-anchor="end">transport</text>'
               f'<text x="{gx}" y="14" class="s" text-anchor="middle">grasp</text>'
               f'<line x1="{gx}" y1="20" x2="{gx}" y2="{H - 6}" class="ev"/>')
    for i, (name, segs, res) in enumerate(rows):
        y = top + i * rh
        out.append(f'<text x="0" y="{y + 19}" class="t">{esc(name)}</text>')
        for s0, s1, lab, cls in segs:
            xa, xb = x0 + s0 * (x1 - x0), x0 + s1 * (x1 - x0)
            out.append(f'<rect x="{xa + 1}" y="{y + 4}" width="{xb - xa - 2}" height="24" rx="4" class="seg {cls}"/>'
                       f'<text x="{(xa + xb) / 2}" y="{y + 20}" text-anchor="middle" class="s segt">{esc(lab)}</text>')
        for k, part in enumerate(res.split("; ")[:2]):
            out.append(f'<text x="{x1 + 12}" y="{y + 14 + 14 * k}" class="s">{esc(part)}</text>')
    return "".join(out) + "</svg>"
