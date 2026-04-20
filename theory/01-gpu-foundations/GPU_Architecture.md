## III. MEMORY HIERARCHY (The Speed vs. Size Trade-off)

Architecture is defined by how fast we can move data to the math units.

| Memory Type | Location | Size | Speed (cycles) | Visibility |
| :--- | :--- | :--- | :--- | :--- |
| **Registers** | On-Core (SM) | Tiny (~256 KB per SM) | ~1 cycle (instant) | One thread only |
| **L1 Cache / Shared Memory** | On-SM (same physical SRAM) | ~32–128 KB per SM | ~20–40 cycles | Block of threads (shared mem portion) |
| **L2 Cache** | On-chip, shared across all SMs | Several MB | ~100–200 cycles | Entire GPU (hardware managed) |
| **Global Memory (VRAM)** | Off-chip (HBM/GDDR chips) | 8–80+ GB | **400–800 cycles** | Entire Grid |

---

### REGISTERS — The GPU's Hands

Registers are the GPU's literal arithmetic hands — they live physically on the CUDA core itself. When a core executes `c += a * b`, both operands and the accumulator sit in registers. There is **zero fetch penalty**: the data is already where the math happens.

A single SM has tens of thousands of 32-bit registers, partitioned among all its resident threads. This creates an important hidden trap: if you launch too many threads per block, each thread gets fewer registers. When a thread runs out of registers, the compiler "spills" its variables into slower local memory (backed by VRAM), silently destroying performance without any error or warning.

---

### L1 CACHE & SHARED MEMORY — The Programmable Scratchpad

The L1 cache and shared memory occupy the **same physical SRAM bank** on the SM — the programmer controls the ratio between them in code (e.g., `cudaFuncSetAttribute`).

- **L1 cache**: Managed automatically by hardware. Holds recently-used local thread data. You don't control it directly.
- **Shared memory**: A **manually-controlled scratchpad**. You explicitly load data into it using CUDA code and share it among all threads in the same block. Access is roughly **10–20× faster than VRAM**.

Shared memory is your primary tool for escaping slow global memory. The entire tiling strategy (see below) exists to exploit it.

---

### L2 CACHE — The Chip-Wide Buffer

The L2 cache sits between all SMs and VRAM, shared across the entire GPU. You cannot program it directly — the hardware manages it automatically.

If the same memory addresses are requested repeatedly by different SMs, data stays warm in L2 and avoids the expensive VRAM round-trip. Modern GPUs (Ampere and later) ship with 40–80 MB of L2, large enough to hold entire weight layers during LLM inference — a key reason large L2 caches became critical for AI workloads.

---

### GLOBAL MEMORY (VRAM) — The Library

VRAM is the large memory chips on the GPU board — the "HBM" or "GDDR" stacks listed in spec sheets. Every matrix, weight tensor, and activation array lives here at runtime. It has enormous capacity (8–80+ GB) but is physically far from the SM cores.

Each access crosses a high-speed bus and a memory controller, costing **400–800 clock cycles**. At 1 GHz, that is 0.4–0.8 microseconds per fetch. Trivial for one thread — catastrophic when 16,384 threads are paying that cost on every instruction.

---

### THE PERFORMANCE PROBLEM: MEMORY-BOUND EXECUTION

To multiply two matrices `A × B`, every output element requires one multiply and one accumulate. The math takes **1–2 cycles per thread**. But before any math can happen, the values must be fetched from VRAM — costing **400–800 cycles**.

So for every 2 useful cycles of computation, the core sits idle for up to 800 cycles waiting for data. This is called being **memory bound**: adding more compute cores does nothing because every core is starved, waiting for the memory bus.

**Without tiling:**
Thread 0 fetches `A[0][0]` from VRAM → waits 600 cycles → does 1 multiply → fetches `B[0][0]` → waits 600 cycles → repeat for every element. Math units are **idle ~98% of the time**.

**With tiling:**
All 256 threads in a block cooperatively load a 16×16 tile into shared memory in one batched operation. Each thread then performs many multiplications from the fast scratchpad before any VRAM trip is needed again. Math units stay busy; VRAM trips are minimized.

> **Why the SM can't just "wait faster":** A VRAM fetch goes to off-chip memory chips via a physical bus. The latency is dominated by electrical signal travel time and memory row activation — physics, not clock speed. No amount of chip engineering shrinks it below a few hundred cycles at GPU frequencies.

---

### THE SOLUTION: TILING WITH SHARED MEMORY

Tiling (also called Blocking) is the technique of breaking a large matrix operation into smaller sub-problems ("tiles") that fit in fast shared memory. Instead of every thread independently hammering VRAM, the whole block collaborates to load one chunk at a time and works from it.

**The four phases of a tiled kernel:**

#### Phase 1 — Collaborative Load
All threads in the block simultaneously fetch one element each from the current VRAM tile into shared memory:

```cuda
s_A[threadIdx.y][threadIdx.x] = d_A[row * K + tile * TILE_SIZE + threadIdx.x];
s_B[threadIdx.y][threadIdx.x] = d_B[(tile * TILE_SIZE + threadIdx.y) * N + col];
```

Because all threads fire at once, the memory controller serves the entire warp in one or two 128-byte cache-line transactions — a **coalesced burst load** — rather than hundreds of individual round-trips.

#### Phase 2 — Barrier Sync (`__syncthreads()`)
All threads wait at a hardware barrier before any computation begins:

```cuda
__syncthreads();
```

This is essential and non-negotiable. Thread execution across different warps in a block is not guaranteed to be simultaneous. If a fast warp begins computing before a slow warp has written its element to shared memory, it reads **garbage data** — a silent, hard-to-debug correctness bug. `__syncthreads()` tells the SM's warp scheduler: *hold every warp here until all warps in this block have arrived.*

#### Phase 3 — Compute from Shared Memory
Each thread accumulates its partial dot-product sum using only the tile in fast shared memory — **zero VRAM accesses**:

```cuda
for (int k = 0; k < TILE_SIZE; k++) {
    acc += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
}
```

For a 32×32 tile, this is **1,024 multiply-accumulate operations per thread** without a single VRAM fetch. Since shared memory latency is ~20–40 cycles (vs. 400–800 for VRAM), the math units finally stay busy.

#### Phase 4 — Advance to the Next Tile
A second `__syncthreads()` is placed **before** the next load:

```cuda
__syncthreads();  // <-- critical: wait before overwriting shared memory
// tile index advances, loop repeats
```

This second barrier is required so that fast threads don't overwrite shared memory with the *next* tile's data while slow threads are still reading from the *current* tile's compute phase. The outer loop runs `K / TILE_SIZE` times, where `K` is the inner (shared) matrix dimension.

---

### ARITHMETIC INTENSITY — The Key Metric

**Arithmetic intensity** = FLOPs performed ÷ bytes read from VRAM.

Every GPU has a fixed peak ratio of compute throughput to memory bandwidth — called the **roofline**. If your kernel's arithmetic intensity falls below that ratio, you are memory bound: more cores won't help because they all stall waiting for the memory bus.

| Kernel | Arithmetic Intensity | Status |
| :--- | :--- | :--- |
| Naïve matrix multiply | ~0.25 FLOP/byte | Deeply memory bound |
| Tiled matrix multiply (tile=32) | ~8 FLOP/byte | Compute bound — cores stay busy |
| A100 GPU hardware roofline (FP16) | ~200 FLOP/byte | Peak achievable ratio |

Tiling alone moves a kernel from **0.25 to ~8 FLOP/byte** — a 32× improvement in compute utilisation — without changing the algorithm at all, just how data is staged. Further optimisations (double-buffering tiles, vectorised 128-bit loads, tensor core intrinsics) push arithmetic intensity closer to the hardware roofline.