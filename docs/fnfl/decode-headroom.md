# Decode headroom on the development box: the card-to-card link, what was fixed, what is left

Measured 2026-11-23 on the development machine: i9-7900X on an EVGA X299 FTW K, two V100 32 GB (PG500-216) at PCIe 3.0
x16, on separate CPU root ports (`nvidia-smi topo`: NODE), Swift IQ3_S with the shared Q8_0 MTP draft. This page is the
record of the decode-speed investigation: the cause, the two fixes, where a step's time goes now, and the ideas left for
squeezing more decode later, ranked.

## The cause: peer writes collapse on this platform

- Kernel stores into the other card's memory run at full speed for bursts up to about 64 KB in flight, then collapse
  to 0.6–1 GB/s. The cost tracks how *fragmented* the writes are more than the byte count: many small pieces from many
  blocks are the worst case.
- Peer loads (pull) stay at about 9 GB/s, kernel stores to pinned host memory about 11 GB/s, and `cudaMemcpyPeer`
  about 10 GB/s. Only SM-issued peer stores collapse.
- FNFL's exchanges (`fn/xchg`) are all push-style SM stores, so a multi-token verify step paid heavily. The HC sublayer
  kernel at 5 rows took 23 µs with its sends redirected to the card's own memory, against 160 µs with real exchanges.
- Platform changes had no effect: driver P2P mode (`RMForceP2PType=1`), C-states and governor, uncore clock pinned at
  2.4 GHz. Moving both cards to one root port isn't possible here because of cooling.

## What was done: two server switches (2026-11-23)

| Server flag (fnfl-drive: environment) | What it does | Code |
|---|---|---|
| `--fnfl-hmix-route host` (`FNFL_HMIX_HOST=1`) | X-HMIX (the mixed row, 6,720 B a row) goes through pinned host memory. The sender writes the receiver's host slot; a small landing kernel on the receiver polls each 16-byte group once, copies it into the device slot the consumers already poll, and resets the host group. | `fn/xchg/xchg.{h,cu}` (`rings::host_hmix`, `land_hmix`, `set_hmix_host`), `hc_phase` in `fn/model/layer.cpp` |
| `--fnfl-hc-packed on` (`FNFL_HC_PACKED=1`) | X-HC (the moments) in the joint layout (`xchg.h hcj_run_at`): each down block gathers its moments of every row in shared memory and one warp sends them as one contiguous run; the reader polls each 16-byte group as two 8-byte pairs at remapped addresses. Fused, non-head `hc::sub` only. | `down_phase`/`up_phase` in `fn/kernels/hc/hc.cu`, `poll_mixed`/`hcp_at` in `fn/xchg/xchg.cuh`, `hc_packed` in `fn/kernels/hc/hc.h`, `set_hc_packed` in `fn/xchg` |

Both default off, so the engine behaves as upstream unless asked. The server sets them from its flags before loading
(`fn/glue/fnfl_glue.cpp`, which prints both at startup). `fnfl-drive` and the benches read the environment variables
instead.

Tests:
- `xchg-test` runs every round over both routes. The host route checks every X-HMIX word in every consumer block, as
  the peer route does, and checks that every host slot is back at its sentinels afterwards.
- `hc2-test` runs a `fused-packed` case beside each `fused` one, against the same reference bits, and adds packed
  rows to the batch-invariance check.

**Important: having the consumers read host memory directly is 2–3× slower than today.** Every consumer block reads the
whole message, so the PCIe reads multiply about 72 times, and L2 doesn't absorb them. The landing step is what makes
the host route work.

Coverage: the target's HC sublayers in decode, which are the exchanges that mattered. These still use the old route
and layout:
- the draft head's single Q8_0 HC sublayer a draft step (`fn/model/draft.cpp`): one small exchange, not worth a
  second path;
- prompt processing (`pass.cpp`);
- the joint multi-user kernel (it has its own packed layout already).

The host route needs the two cards on separate devices and streams, since the landing waits on the peer's send. Rings
made on one device never take the host slots, so single-card mode stays on the peer route.

### Results

`docs/iq3s-tools/bench.py`, 24 requests a setting (4 prompt lengths to 25k tokens, 3 tasks, 2 reps), decode tok/s:

| Proposals | Before | Host route | Host route + packed moments |
|---|---|---|---|
| 2 | 161.9 | 166.3 | 166.1 |
| 3 | 151.4 | 167.0 | **183.3** |
| 4 | 136.9 | 159.1 | 177.6 |

- The best setting moves from 2 proposals at 161.9 to **3 proposals with both switches at 183.3 (+13%)**. The gain
  holds at every prompt length, and prompt processing is unchanged.
- Correctness:
  - Token counts were identical in all 72 bench requests.
  - `fnfl-drive spec run` gave byte-identical tokens with sampling (T 1, fixed seed) and greedy, with the switches
    off, host only, and host + packed.
  - `xchg-test`, `hc2-test`, and `model-dry` in graph, eager and spec modes all pass with the switches on.
- Matched `fnfl-drive spec run` (one prompt, 3 runs, median): 4 proposals went 145 → 169 → 191 tok/s, and 3 proposals
  163 → 178 → 197 tok/s (before → host → host + packed).

## Where a decode step's time goes now

From the wait counters (`FNFL_WAIT_STATS`, below): the average wait per poll that waited, times the share of polls that
waited, times the exchanges per step. That's 96 for X-HC and X-HMIX (two HC sublayers a layer) and 48 each for X-A and
X-F. The diagnostic build's own overhead distorts step times, so use it for shares only.

| | 3 proposals, host only | 3 proposals, both | 4 proposals, host only | 4 proposals, both |
|---|---|---|---|---|
| X-HC | 28 µs, ~2.3 ms a step | 12 µs, ~1.0 ms | 39 µs, ~3.1 ms | 15 µs, ~1.2 ms |
| X-HMIX | ~0 | ~0 | ~0 | ~0 |
| X-A + X-F | ~0.3 ms | ~0.3 ms | ~0.4 ms | ~0.4 ms |
| Exchange stall, share of step | ~2.6 ms (15%) | **~1.3 ms (8%)** | ~3.5 ms (18%) | **~1.6 ms (9%)** |

The rest, about 90% of a step, is the work itself:
- the draft step, about 1.5 ms;
- reading the routed experts' and dense weights from HBM, and the layers' compute.

Upstream's 5-token verify on a box without the collapse takes 14.2 ms, against about 16 ms here; the remaining
exchange stall accounts for most of that difference. Draft acceptance is about 1.84 drafts a step at 3 proposals and
2.13 at 4.

## Ideas for later, ranked by expected payoff

1. **Shave X-HC further (about 1.0–1.2 ms a step left; maybe +3–5%).** Part of it can't be removed: each card waits for
   the other's down phase to finish, plus a couple of microseconds of PCIe latency. Options:
   - **Fewer, larger writes:** the fused kernel is cooperative and already has a grid barrier, so a few blocks could
     gather all 72 runs and send one or two large contiguous writes.
   - **Pull instead of push:** peer loads don't collapse. The receiver could read the sender's moments from the
     sender's memory after a single flag. That costs one extra latency hop; measure whether it beats the packed push.
   - **Overlap:** start the parts of the up phase that don't need the peer's moments before polling.
   - **Copy engine:** stage the moments locally and send them with `cudaMemcpyPeerAsync`. Launch latency probably
     makes this a loss at these sizes; worth a microbenchmark first.
2. **X-A and X-F (about 0.3–0.4 ms a step; maybe +1–2%).** Same treatment: host route with landing, or packed runs.
   Only worth it bundled with idea 1, since on their own they're small.
3. **X-DRAFT-IN (about 22 µs a wait, about half the polls wait).** The count per step is small; measure its share before
   touching it.
4. **The draft step (about 1.5 ms).** Its HC head still uses the old route and layout. Extending both switches to
   `fn/model/draft.cpp` would cover one small exchange a draft step; probably not measurable, so it was left out.
5. **Kernel time (about 90% of a step).** Profile the expert and dense products with Nsight Compute (`sudo -n
   /opt/cuda/nsight_compute/ncu` is allowed on this box). FNFL's kernels are already tuned, so the payoff is uncertain.
6. **Fewer sync points by design (large rewrite).** A layer split instead of a per-layer tensor split would replace
   about 200 card-to-card hand-offs a step with a few. Each hand-off costs at least a PCIe round trip, so that's a hard
   floor of about 0.3–0.5 ms a step today. Estimate the layer split's own costs (the pipeline bubble, the KV placement)
   before starting.
7. **Default depth.** With both switches, 3 proposals beats 2 and 4. Re-check after any of the above; cheaper deeper
   steps move the best depth up.

Ruled out:
- Fine-tuning the MTP head: Swift accepts drafts no worse than the base model (`docs/iq3s-swift.md`).
- Multi-user batching on this box: no throughput gain, and 3–4 users don't fit.
- Platform tweaks: listed above.

## Tools

- **Wait counters:** configure a build with `-DCMAKE_CUDA_FLAGS=-DFNFL_WAIT_STATS` (this box uses `build-wait`). Run
  the server with `FNFL_WAIT_STATS_DIR=DIR`; touching `DIR/dump-request` writes `DIR/wait-stats.txt`, with polls,
  polls that waited, cycles waited and the maximum for each message and card. Take a snapshot before and after a set
  of requests and subtract. Bit 14 (`X-HC-own`) counts the fused kernel's wait on its own card's moments separately.
- **`docs/iq3s-tools/p2p/`:**
  - `p2pall.cu.txt`: the platform regression set (peer write size sweep; stores and loads to local, peer and host).
  - `hcpat6.cu.txt`: the X-HC/X-HMIX stand-in, which compares peer, host-direct, host-with-landing and no X-HMIX, with
    the moments shipped or contiguous.
  - Both are saved as `.txt` because the include check rejects CUDA sources outside `fn/`. Copy one out and build it
    with `nvcc -O3 -arch=sm_70 FILE.cu`.
  - `hc_bench_xhc.patch`: an `xhc` mode for `fnfl-bench hc2-bench 0 1 xhc`, timing the fused sublayer at 1–5 rows with
    both messages crossing against X-HC only. Apply it with `git apply`, then build the `fnfl-bench` target. Its
    `xhc_none` point still sends to sinks on the *peer*, so it isn't a no-exchange bound. For that bound, point the
    sinks at the card's own memory.
  - `gpumon.sh FILE`: the GPU telemetry logger (temperature, HBM temperature, power, clocks, throttle reasons, every
    500 ms). Run it during every benchmark.
- **Telemetry:** GPU 0 runs 4–7 °C hotter than GPU 1 (its cooling), peaking at about 71 °C core and 73 °C HBM on the
  full bench. During long prompts it touches its 250 W software power cap (`0x4`, not thermal) for about 20–40 s, at
  1372 MHz instead of 1380. There has been no thermal throttling.
