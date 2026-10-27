"""Pre-registration of plan X's read X2 (ledger m3_prereg tag x2_bridge_switch), written after the zero-shot baselines (ledger
x2_simpler_zs) and before any composition episode on SimplerEnv.  Policy: pi0 (open-pi-zero bridge_beta, float32;
lifelong_ood/policy/backbone_opz.py, parity 1.8e-7), stock SimplerEnv protocol (allenzren fork, stock scenes, 60 steps, 120 in the
sink scene), the driver experiments/x2_simpler_rollout.py.  The supervised scheme's grasp switch: prompt A until the policy's
gripper command is 'close' for 3 consecutive queries, prompt B after; nothing fitted."""
import sys
sys.path.insert(0, "/coc/flash10/ksingara3/vla-operator-binding")
from lifelong_ood import ledger
ARMS = {"X2G1GreenBlockOnPlate": ["stack the green block on the yellow block", "put carrot on plate"],
        "X2G1CarrotOnYellowBlock": ["put carrot on plate", "stack the green block on the yellow block"],
        "X2G1CarrotInBasket": ["put carrot on plate", "put eggplant into yellow basket"],
        "X2G4PepsiOnTowel": ["put can in pot", "put the spoon on the towel"],
        "X2G4RedBullOnPlate": ["put can in pot", "put carrot on plate"],
        "X2G4SevenUpOnTowel": ["put can in pot", "put the spoon on the towel"]}
print(ledger.record(
    "m3_prereg", bench="simpler_widowx", tag="x2_bridge_switch", seed=0,
    metrics={"arms": ARMS,
             "episodes": "24 stock layouts x 10 flow-noise seeds = 240 per task and arm, paired by (layout, seed) with the zero-shot run",
             "hypotheses": {
                 "H1": "on the 3 G1 relation tasks (both parts named in Bridge V2, never together), the grasp switch > pi0 alone under the "
                       "task's own instruction, pooled over 720 paired episodes (exact sign test on discordant pairs, alpha 0.05); "
                       "task-level difference reported with its bootstrap interval",
                 "G4": "DESCRIPTIVE: the 3 G4 can tasks (brands never named in Bridge) are at floor partly for physical reasons (the light "
                       "upright can tips and rolls); fixed BEFORE any episode of the variant: each can's density raised so its mass is "
                       "0.15 kg; pi0 alone and the grasp switch both run on that variant, 240 paired episodes each",
                 "descriptive": "grasped / moved-correct-object rates per arm; when the switch fired"},
             "written_before": "any composition episode on SimplerEnv; after the zero-shot baselines"},
    bytes=dict(experts=0, tree=0, codes=0, samples=0, teacher_copies=0, total=0), config={},
    script="experiments/prereg/x2_bridge_switch.py", command="pre-registration of plan X's read X2 (grasp switch on Bridge)"))
