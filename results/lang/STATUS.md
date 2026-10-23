# Plan L status (L0a / L0b / L1), docs/PLAN_LANG.md

- 2026-10-22: L1 pre-registered (ledger/m3_prereg tag pl1_predict, experiments/prereg/pl1_predict.py) BEFORE any grounding score.
  Zero-shot data located: arm zs in g4b_main (pi05) / g4c_main (pi0), g13_main, g15_main, g17_base; sets c-f, 80 episodes per task
  (pi05 596 / 1600, pi0 149 / 1600).
- next: write lifelong_ood/lang/audit.py, experiments/pl0_audit.py (render OOD first frames, libero venv; audit, GPU), pl0_trace.py, pl1_predict.py.

# Plan L status (L2: fast composition from language prototypes), docs/PLAN_LANG.md -- L2 agent

- 2026-10-22: started. Li's OOD tasks have NO demonstrations (docs/COST.md, INTERPRETATION.md), so L2 records its own:
  successful rollouts on a NEW start-state set fresh_ood_pl2d (BASE 3300000, 8 per task) by the OTHER policy's strongest
  earlier arm (cross-policy, so no scored operator generated its own demo); closed loop on fresh_ood_pl2 (BASE 3400000, 20).
  Job 4191019 (pl2-fresh) generates both sets. Files: lifelong_ood/lang/transplant.py, experiments/pl2_compose.py,
  experiments/prereg/pl2_compose.py, results/lang/pl2/. I do not touch audit.py / pl0_* / pl1_*.
- 2026-10-22 10:50: code written: lifelong_ood/lang/audit.py, experiments/pl0_render.py (done: results/lang/frames, 20 OOD tasks x 10
  first frames of fresh_ood_a), pl0_audit.py, pl0_trace.py, pl1_predict.py. Smoke OK (6 s / observation, 34 prompts).
  Running: 4191023/4191024 audit pi05 shards 0,1 of 2; 4191025 audit pi0 (1 shard); 4191026 trace pi05 (120 traces).
  All resumable: resubmit the same commands (see each script's docstring) if a job dies. Outputs results/lang/audit/<bb>/, trace/pi05_libero/.
  After: `$PY experiments/pl0_audit.py --read --record`, `pl0_trace.py --read --record`, `pl1_predict.py --record`.
- 2026-10-22 11:10 (L2): pre-registered (ledger m3_prereg tag pl2_compose, experiments/prereg/pl2_compose.py) before any L2 number.
  Dev tasks (sha1 split): goal:1,2,4,5,6,7, spatial:6,7. Smoke offline OK (l90:46, 586 s, 93 evals). Demo recording packs
  4191081-90 (lolpack, 3 lanes) running; pi05 LIBERO-90 offline 4191114. Shared-file edits (additive, atomic): launch_pack.py
  arm lt@<name>; serve_guided.py mode lt:<spec.json>.
- 11:15: pi05 shards crashed on a missing pose file (tr38_e1283_f0); fixed in scene_of; resubmitted 4191135/4191136. pi0 job 4191025 will hit the same; resubmit it after.
- 11:35: pi05 audit COMPLETE (800 obs); read works. pi0 audit 4191025 running (+4191186 queued afterany to finish past the pose crash). trace 4191026 ~40/120. L1 preview pi05-only: AUC(-min G) 0.505 (min G is almost always the verb).
- 12:50: pi0 audit COMPLETE; ledger pl0_audit (read_pi05, read_pi0) and pl1_predict (read) recorded. L1 pre-registered GO FAILS: pooled AUC(-min G) 0.459 [0.24, 0.67]; pi05 0.505, pi0 0.556; mean-G AUC 0.627. Trace 101/120 running (4191026).
- 13:10: ALL DONE. L0a (both policies, 800 obs each), L0b (120 traces), L1 recorded. Results section appended to docs/PLAN_LANG.md.
  Verdict: L1 KILL (pooled AUC 0.459); L0b: word leaves its positions after layer 1 into image tokens -> rank-1 edit KILL; token inversion / early-layer transplant remain.
- 2026-10-22 14:40 (L2): offline done for Li's 20 x {pi0.5, pi0} and pi0.5 LIBERO-90 16 (pi0 LIBERO-90 3/16, rest queued).
  Held-out demo-1 flow loss: l2sel beats zs on 16/20 (pi0.5) and 18/20 (pi0) tasks; l2sel beats both controls 19-20/20.
  l2sel picks TLI on 13/20 for both policies. Closed loop on 8 dev tasks running: pi0 packs 4192693-708, pi0.5 4193247-262.
- 2026-10-22 17:40 (L2): DONE. Ledger pl2_offline, pl2_dev recorded; docs/PLAN_LANG.md "L2 results"; figures/robot/pl2_offline.png,
  pl2_dev.png. Dev closed loop: pi0.5 l2sel 91/160 vs zs 63 (H1 pass) vs Li-TLI 84 (H2 pass, lb -0.031); pi0 l2sel 39 vs zs 11 (H1
  pass) vs Li-TLI 106 (H2 FAIL). Word transplants collapse on pi0 closed loop (2-3/160) despite offline wins. T8 pos control worst
  arm on pi0.5. Confirm read (12 tasks) not run.
- 2026-10-22 17:50 (L2 confirm, pi0.5 only): addendum prereg recorded BEFORE any confirm episode (ledger
  m3_prereg__libero_ood__pl2_confirm__s13__426df7fe, experiments/prereg/pl2_confirm.py): dev protocol unchanged on the 12 confirm
  tasks; frozen offline picks (sha256 in the row); 8 arms; fresh_ood_pl2 (confirm starts never rolled). pl2_compose.py gained
  --launch-confirm / --read-confirm (exp pl2_conf, ledger pl2_confirm). Packs 4194384-407 (overcap, 4 lanes).
- 2026-10-22 21:30 (L2 confirm): DONE, ledger pl2_confirm/pl2_confirm__libero_ood__read__s13__eb3a3b2f.json (payload
  results/lang/pl2/confirm_read.json, figure figures/robot/pl2_confirm.png). pi0.5, 12 tasks x 20: l2sel 128/240, zs 93, tli_li 169.
  H1 PASSES (discordant 77:42, sign p 0.0017; task-bootstrap diff +0.146 [-0.171, +0.442]). H2 FAILS (l2sel - tli_li -0.171
  [-0.375, +0.004]). Selector failures without a guard: spatial:0 tmpl 0/20 vs zs 20; spatial:3 tli_sel 7 vs zs 20. One pack
  instance (spatial:2 tmpl) OOM'd at model load on cheetah (foreign GPU process), resubmitted alone (4196320), same starts/seed.
