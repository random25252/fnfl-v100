# Wide verify windows: design notes (2026-11-24, in progress)

The goal: let a lookup step verify more than the 4 proposals today's 5-row verify allows, for agent turns that copy
text from the context (edits). Measurements and the expected gain are in
[`decode-headroom.md`](decode-headroom.md#wide-verify-windows-for-edit-turns-quantified-2026-11-24). This file is the
build plan and what the code survey found.

## The refined estimate, from what is measured

Per-family cost of the extra rows on **today's kernel forms**, 16 rows against 4:

| Family | 4 rows (ms a step) | 16 rows, estimated | Source |
|---|---|---|---|
| Dense products | 4.19 | ~9.6 (2.3×) | `dp-bench wide`: the joint launch (one weight read, U lanes × R rows) |
| Experts | 3.51 | ~11 (+0.62 ms a row) | Nsight Systems of the `-np 2` joint step: 6.0 ms at 2 × 4 rows |
| HC | ~2.5 | ~4 | the work grows slowly (BF16 weights read once); the joint form's waits excluded |
| Exchanges | ~1.9 | ~2.5 | on the host routes |
| QSA, GDN | ~0.9 | ~2.4 | rows are sequential inside GDN; attention per row |
| Head, draft, rest | ~1.9 | ~2.5 | the head 234 → 544 µs a launch |
| **Step** | **~14.5** | **~32 (≈1.45 ms a row)** | |

- At ~1.45 ms a row the replay gives **+18% on edits at 8 proposals and +28% at 15–16** (`decode-headroom.md`'s table,
  slope 1.5).
- The experts' extra rows are mostly a bandwidth floor: each row adds up to 10 new experts a layer (~0.95 MB a card
  each, ~0.45 GB a row at most, less where rows share experts), so ~0.3–0.45 ms a row on any kernel.
- Tensor-core dense products (FP16 MMA on weights unpacked in registers) would take the dense family to ~flat; with
  that, ~0.7 ms a row and **~+50% on edits**. A second stage.

## Why the joint (multi-user) path matters

FMU2's joint round already runs the row-independent families over U lanes of R rows with one weight read, each row
bitwise its lone launch at B = R: `dense::product_joint`, the experts' `*_lanes` kernels, `hc::sub_joint`, the head's
joint logits, `xchg::send_half_joint` / `recv_add_joint`. It is slow today (two clients: 36.5 ms for 2 × 4 rows)
because its exchanges never got the host routes: `hc_sub_joint_kernel` 116 µs a call against 43, `recv_add_lanes`
45 µs against 2–8, the router 80 µs (waiting on X-HMIX). Its kernels' own work is close to the estimates above.

## What a wide round needs (survey of the 5-row limit)

`max_rows = 5` and `n_version = 6` appear in ~40 files. Grouped by what has to change:

- **Row-independent, lane forms exist:** dense (`dense.h` `product_joint`), experts (`experts.h` lanes), HC
  (`hc.h` `sub_joint`), head (`head.h` joint logits), X-A/X-F (`xchg.h` `*_joint`). A wide round can run these as
  lanes of one sequence, if the lane rings take the host routes.
- **Sequence-coupled, need a wide form for one sequence:**
  - GDN core (`gdn.h`): rows run in order inside the launch; records `[2][5][16,512 B]` grow to `[2][R][...]`
    (ME11's replay keeps the memory small: ~0.56 MB a layer and card at 17 rows). `round/version.h`'s
    `gdn_max_steps`.
  - The convolution history (`[6][5120][3]`, versions 1..B): `n_version` 6 → R + 1 (~1.1 MB a layer and card at 17).
    In the joint dense launch it is a per-lane epilogue; a wide round needs the history to flow across lanes
    (the standalone `gdn::conv_silu` over all rows, or a lane-chained epilogue).
  - QSA (`qsa.h`, `attend.h`): new cells and selection bodies of up to R rows; X-SELECT's slot.
  - PLE (`ple.h`): the n-gram history across rows; versions.
  - The round frame (`round.h` desc arrays, `version.h`), the decision and sampler exchanges (`samp_*` slot words),
    the commit.
  - The draft catch-up (`draft.h` `max_rows` 5): up to R committed tokens.
- **Runtime:** `lookup_propose` up to W proposals; step graphs captured for the wide row counts.

## Order of work

1. **Host routes for the joint path. Done (2026-11-24):** X-HMIX landed per lane, X-HC landed inside
   `hc_sub_joint_kernel`, joint X-A/X-F slots in host memory. Two clients: joint step 36.5 → ~22.6 ms, 154 → 229 tok/s
   in total, bitwise (`decode-headroom.md`, "Two requests at once").
2. **The sequence-coupled families at R rows** for one sequence, each with its tests extended (gd-test,
   rollback-test, qa tests, ple-test, state-test), bitwise against R consecutive 1-row rounds.
3. **The wide round's frame and runtime**: descriptor, versions, decision, commit, catch-up, lookup proposals,
   graphs. Greedy outputs must stay bitwise those of plain decoding (`edit_bench.py` md5s).
4. **Measure** with `edit_bench.py` and `bench.py`; then decide on tensor-core dense products.
