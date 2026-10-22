"""Pre-registration of plan D's read D1, the ROLLOUT stage (ledger m3_prereg tag zd_rollouts), written after the offline action
breakdown (ledger zd_breakdown tags stage1 / stage1_step100 / stage2; docs/INTERPRETATION.md section 5) and before any D1 episode.
The rollout script (experiments/zd_rollouts.py) is not written yet; nothing here has been launched.

Sources as in experiments/zd_breakdown.py: a_policy (pi0.5 under the task's instruction), a_proto in {ptr, lib} (supervised) and
{prog, blend} (unsupervised), a_demo = a_proto + the residual rank-16 leaf fitted on demonstration 0 (results/robot/zd/
leaf_<task>_<src>_n1.pt; refitted with the same recipe and seed where the file is missing).  Episodes: Z3's 25 LIBERO-90 tasks,
set d, 20 per task, noise seed 13, the same initial states for every arm of a task (paired).  Step-0 arms reuse the existing paired
episodes where their protocol matches (pi0.5 alone: G8; ptr: Z3; lib: G8's library; blend: Z5; prog: served fresh where no matching arm exists).

(3a) Rollout learning curves.  On the 12 curve tasks (zd_breakdown.curve_tasks()), the leaf from each start in {policy, ptr, lib,
     blend} at n = 1 and 5 demonstrations and fit steps 0 / 200 / 800 (the 200- and 5-demo leaves refitted and saved).  The value of a
     prototype in success units: the number of demonstrations (log-linear between 1 and 5, censored at 5) the policy start needs to
     match the prototype start's 1-demo success, per task, median over tasks.
(3b) Causal swaps.  For src in {lib, blend} (the strongest supervised and the unsupervised donor), all 25 tasks: the 2^3 assignments
     of the component groups translation (dims 0-2) / rotation (3-5) / gripper (6) to a_proto or a_demo.  Served through BlendHooks
     with one-hot group weights over two rows per field, the field with the leaf off (a_proto) and on (a_demo); for the blend each
     of its experts carries both rows and the blend's own weights act within the leaf-off and leaf-on parts alike.  The gripper
     row is fixed by the assignment (not by BlendHooks' best-expert rule).  Assignment (proto, proto, proto) is a_proto and
     (demo, demo, demo) is a_demo; both double as a parity check against (3a)'s n = 1 step-800 arm and the step-0 arm (identical
     episodes required up to the sampler's bf16 noise; a success-rate gap above 2 episodes per task in more than 2 tasks halts the
     read).  Per-component Shapley value of "demo instead of proto" on success, exact over the 3 groups, per task and pooled.
"""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/lifelong-ood-learner")
from lifelong_ood import ledger

if __name__ == "__main__":
    print(ledger.record(
        "m3_prereg", bench="libero_90", tag="zd_rollouts", seed=13,
        metrics={"hypotheses": {
            "H1": "unsupervised value: at n = 1, 800 fit steps, the leaf from the blend > the leaf from the policy on Z3's 25 tasks "
                  "(per-task success, exact sign test, alpha 0.05)",
            "H2": "supervised value: at n = 1, 800 fit steps, the leaf from the library (lib) > the leaf from the policy (same test)",
            "H3": "the offline breakdown transfers to success: across the 3 groups, the pooled Shapley value of a_demo over a_proto "
                  "ranks the groups in the order of their offline taught share at 800 steps (stage1, all phases), for lib and blend",
            "descriptive": "ptr and prog starts; the 12-task learning curves (n = 1 / 5, steps 0 / 200 / 800) and the value in "
                           "demonstration units; Shapley values per grade and per task; the 2-way group interactions"},
            "protocol": "set d, 20 episodes per task and arm, noise seed 13, paired initial states; Z3's 25 LIBERO-90 tasks (3b and the "
                        "H1 / H2 arms), the 12 curve tasks (3a curves)",
            "a_demo": "the 800-step residual leaf (rank 16, lr 1e-4, batch 32, AdamW, seed 0) fitted on demonstration 0 through the "
                      "composed field; the 200-step and 5-demonstration leaves for 3a only",
            "parity": "before the read: swap (proto, proto, proto) and (demo, demo, demo) reproduce the plain a_proto and a_demo arms "
                      "(within 2 episodes per task in all but 2 tasks)",
            "written_before": "any D1 episode"},
        bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
        script="experiments/prereg/zd_rollouts.py", command="pre-registration of plan D's read D1 rollouts (curves and causal swaps)"))
