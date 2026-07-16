# SRP 2026 Presentation Briefing — Hypergraph Traffic Scene Learning

Paste this whole file into a chat with Claude to help build slides and rehearse
talking points. It contains only things that were actually run and measured —
no invented numbers. All five figures and both models' results are final as
of this version.

---

## 0. Read this first — how to frame the talk

This is a **progress / methods update**, not a "we beat the baseline" result.
Be upfront about that framing in the first 30 seconds — it's the difference
between a credible research talk and an overclaim a technical audience will
poke holes in.

What you legitimately have:
- A complete, correct, end-to-end pipeline (data → hypergraph → model → loss → eval), verified against real data at every stage.
- A **first real training run** (today, quick/reduced-scale) comparing the HGNN encoder against a vanilla GCN baseline on the same task.

What you do **not** have yet:
- A tuned, fully-converged model (this run was 40 epochs on 3 of 4 scenarios, sized to finish in ~15 minutes, not the eventual full run).
- A win for HGNN over GCN — see §5: **the GCN baseline currently outperforms
  the HGNN on every split**, by 0.03-0.04 macro-F1. Lead with this honestly;
  it's a normal, explainable early-stage result (only H1 is implemented so
  far), not a failed project. Val loss also rose for both models (overfitting
  on today's short run — expected, not a red flag, but say so if asked).
- H2–H5 hyperedges (only H1/proximity is implemented and used today).
- Any R4 (V2X, roadside+vehicle) results.

---

## 1. Project one-liner

Self-supervised hypergraph learning for traffic scene understanding. Roadside
LiDAR (TUMTraf R2/R4 datasets) → a learnable hypergraph over the objects in a
scene → trained with contrastive losses, no human interaction labels required.

**Team:** Liu Restrepo · Prof. Cheng · Hyun Lee · Chris Yi
**Program:** SRP 2026 · Trinity College

---

## 2. The dataset (real, computed from the actual label files)

TUMTraf R2 — roadside LiDAR, two sensors (north/south) per scenario, 4 scenarios:

| Scenario | Time of day | Frames (north+south) | Instances (north+south) |
|---|---|---|---|
| r2_s01 | DUSK | 282 + 282 | 3,869 + 5,178 |
| r2_s02 | DUSK | 282 + 282 | 4,299 + 4,953 |
| r2_s03 | DAY | 1,033 + 1,033 | 13,603 + 15,334 |
| r2_s04 | NIGHT | 563 + 563 | 3,820 + 5,018 |
| **Total** | | **4,320 frames** | **56,074 instances**, 503 tracks |

Class distribution (all of R2, computed directly from the labels):

CAR 59.4% · VAN 10.8% · TRAILER 9.2% · TRUCK 8.0% · PEDESTRIAN 5.9% · BUS 2.9% ·
MOTORCYCLE 1.8% · BICYCLE 1.3% · other/emergency 0.6%

→ This is figure **fig1**.

---

## 3. Method

**Five hyperedge types are designed** (proximity, tracklet, converging, VRU-proximity,
V2X-match) but **only H1 (proximity)** is implemented and used in today's run — say
this plainly if asked "what about the other four."

**Pipeline** (shared weights across the two views of a scene):

```
Graph pair (north view, south view)
    → HGNN encoder f (2 layers, hypergraph convolution, Feng et al. AAAI 2019)
    → mean pool over nodes → scene embedding z
    → projector g (2-layer MLP, 256 → 128)
    → NT-Xent contrastive loss (GraphCL, tau = 0.5)
```

**Ablation baseline:** a vanilla 2-layer GCN over the same node features, using
plain pairwise BEV-distance edges instead of the H1 hypergraph — isolates what
the hyperedge/hypergraph formulation buys over a standard graph.

**Node features:** position, heading, box size, derived velocity (vx, vy,
speed), LiDAR point count, occlusion level, 8-class one-hot.

**Proxy task:** speed-state classification (stopped / slow / moving / fast),
a pseudo-label derived from consecutive-frame displacement — no human
annotation needed. Worth knowing: the node feature vector includes velocity
directly, so this proxy task is partly solvable from raw input features, not
purely from structure. It's still useful for a first relative comparison
(same features feed both models), just don't oversell it as a hard task.

→ This is figure **fig2**.

**Worked example** — a single frame (frame 250, r2_s04 north) with 11 objects,
18 proximity edges, and 4 connected components (one 7-object cluster, one
2-object cluster, two isolated objects) — this is the exact structure H1
builds, shown concretely. → figure **fig3**.

---

## 4. What was verified before this run (pipeline correctness)

1. **Frame parsing sanity check** — loaded frame 250, confirmed 11 objects, 18
   BEV edges at ≤10m, correct connected-component structure. Passed.
2. **NT-Xent sanity check** — ran 10 real training batches, confirmed the
   contrastive loss decreases (2.65 → 2.25) rather than being stuck or NaN.
   Passed.

These are "does the code work" checks, not results — but worth one slide/line
to show the pipeline was validated before any numbers were trusted.

---

## 5. Today's real run — setup

- Train: r2_s01, r2_s02, r2_s03 (north/south positive pairs). Validate: r2_s04.
- 40 epochs, batch size 32, Adam lr 1e-3, encoder dim 256, projector dim 128.
- Both models (HGNN, GCN baseline) trained from the same code path, same
  features, same optimizer settings — only the connectivity structure (H1
  hypergraph vs. plain BEV graph) differs.
- Run on GPU (conda env `srp26`), ~11 minutes total for both models.

### Training curves → figure **fig4**

Both models' training loss drops similarly (3.5 → ~2.5 over 40 epochs).
**Validation loss rises for both** after roughly epoch 15–20 — classic
small-run overfitting (short training window, val is a single held-out
scenario). Framing for the talk: *"this is a first quick run to prove the
comparison pipeline works end-to-end; the next run will use more data and
early stopping / regularization."* Don't present the rising val loss as if it
weren't noticed — noticing it is a sign of a careful presenter.

### Linear-probe evaluation (frozen encoder → logistic regression on speed-state)

Overall macro-F1 by split, both models (frozen encoder → logistic-regression probe):

| Split | test n | HGNN | GCN baseline |
|---|---|---|---|
| Random 80/20 | 10,747 | 0.691 | **0.729** |
| Scenario holdout (s01–03 → s04) | 8,364 | 0.627 | **0.656** |
| Day/night holdout (→ NIGHT) | 8,364 | 0.627 | **0.656** |

(Scenario holdout and day/night holdout land on the same number because, in
R2, r2_s04 is *both* the held-out scenario *and* the only NIGHT scenario —
not a bug, just how this particular dataset splits.)

**The honest headline: the GCN baseline currently beats the HGNN on every
split and every stratified subset**, by 0.03-0.04 macro-F1. Say this plainly
— don't spin it. Plausible reasons, in order of how much they likely matter:

1. **Only H1 is implemented.** H1's hyperedges are connected components of
   the *same* proximity graph the GCN already uses — so HGNN's extra
   structure over GCN today is small (grouping via connected components
   instead of direct pairwise edges), not the richer H2-H5 structure the
   project is ultimately about. A real test of the hypergraph idea needs
   H2-H5 in the loop.
2. **Quick, untuned run.** 40 epochs, no hyperparameter search, no
   regularization — either model's number could move with real tuning.
3. **The proxy task is easy and feature-driven** (see §3) — a ceiling effect
   where more expressive structure has less room to show an advantage.

Stratified (random-split test set): VRU-only macro-F1 0.609 (HGNN) vs. 0.688
(GCN), n=935; low-point-count (≤5 pts) 0.698 vs. 0.737, n=1,060; occlusion
NOT/PARTIAL/FULL 0.704/0.615/0.660 (HGNN) vs. 0.741/0.673/0.657 (GCN). Same
pattern throughout — GCN ahead. VRU subset is **empty (n=0)** for the
scenario/day-night holdout splits — r2_s04 apparently has no pedestrian/
cyclist/motorcycle instances with a valid derived velocity. Flag this rather
than hide it if asked about VRU performance on those splits.

→ Comparison chart is figure **fig5**.

---

## 6. Figures — what each one is and where it goes

All in `hgraph/outputs/figures/`, 300 DPI PNG, ready to drop into slides at
full width (each is designed as a single-slide figure, not a multi-up).

| File | Shows | Suggested slide |
|---|---|---|
| `fig1_dataset_overview.png` | R2 frames/instances per scenario (north vs south) + full class distribution, VRU classes highlighted in red | **Slide 2-3: "The data"** — right after the one-line project intro |
| `fig2_pipeline_diagram.png` | Full pipeline schematic: two view-graphs → shared HGNN encoder → pool → projector → NT-Xent, plus the 5 hyperedge types as a legend strip | **Slide 4: "The method"** — the one diagram people should screenshot |
| `fig3_bev_scene_graph.png` | Real worked example: frame 250, 11 objects, BEV proximity edges, 4 hyperedges colored | **Slide 5: "What a hyperedge looks like"** — makes H1 concrete before showing the model |
| `fig4_training_curves.png` | HGNN vs GCN train/val NT-Xent loss over 40 epochs | **Slide 6-7: "First training run"** — pair with the honest overfitting caveat from §5 |
| `fig5_hgnn_vs_gcn.png` | HGNN vs GCN macro-F1, grouped by the 3 eval splits | **Slide 8: "Does the hypergraph help?"** — the closest thing to a headline result; state the caveats from §5 in the same breath |

Regenerate any figure with (from `hgraph/`):
```
conda run -n srp26 python make_figures.py        # figs 1-3, no training run needed
conda run -n srp26 python make_figures.py 4      # training curves (needs outputs/train_log_*.csv)
conda run -n srp26 python make_figures.py 5      # comparison chart (needs outputs/eval_*.txt)
```

---

## 7. Suggested slide outline (10-12 slides, ~8-10 min talk)

1. Title — project name, team, program
2. Motivation — why hypergraphs for traffic scenes (pairwise graphs miss multi-way interactions: a merge, a group crossing together, a V2X match)
3. Dataset — fig1
4. Method / pipeline — fig2
5. Concrete example of a hyperedge — fig3
6. What we verified before trusting any numbers — the two sanity checks (§4)
7. Today's experiment setup — train/val split, both models, same features
8. Training curves — fig4 + the honest overfitting note
9. Linear-probe comparison — fig5 + the numbers table from §5
10. Limitations / what's not done yet — H2-H5, R4/V2X, tuning (§0)
11. Next steps — see §8 below
12. Questions

## 8. Next steps (say these out loud — shows momentum, not just a status report)

- Implement H2 (tracklet), H3 (converging), H4 (VRU-proximity), H5 (V2X match) hyperedges — currently only H1.
- Wire in the STRL temporal loss (needs H2 tracklet pairs) alongside NT-Xent.
- Extend to R4 for the V2X (roadside + vehicle) cooperative-perception setting.
- Full-scale run: all scenarios, more epochs, early stopping against overfitting.
- A proxy task less entangled with the raw input features than speed-state.

---

## 9. Anticipated questions and honest answers

- **"Does the hypergraph actually help?"** → Not yet — in today's run the GCN
  baseline actually scores higher than the HGNN on every split (fig5). The
  most likely reason is that only H1 is implemented, and H1's structure
  (connected components of the same proximity graph GCN already sees) isn't
  that different from plain pairwise edges yet. This run wasn't tuned or run
  to convergence — it's a first apples-to-apples comparison to prove the eval
  pipeline is correct, and it also tells us H2-H5 are load-bearing for the
  hypothesis, not optional polish.
- **"Why does validation loss go up?"** → Short run (40 epochs), small val
  set (one scenario), no regularization/early stopping added yet — expected
  in a first pass, being fixed in the next run.
- **"Isn't speed leaking into the label from the features?"** → Yes, node
  features include derived velocity directly, and the proxy label is a
  bucketed function of that same velocity. It's a fair same-features
  comparison between HGNN and GCN, but not a hard downstream task by itself.
- **"What about H2-H5?"** → Designed (see the five-hyperedge-type table),
  not yet implemented. H1 (proximity) is the only one driving results today.
