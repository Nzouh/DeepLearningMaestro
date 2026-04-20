# SILICON-TO-SYNAPSE: GPU ARCHITECTURE FOUNDATIONS
> *"Understanding the hardware to write optimal software."*

---

## I. THE LOGICAL HIERARCHY (The Programmer's View)

Before writing CUDA, we must understand how we organize work.

**1. THE KERNEL**
A specialized program designed to run thousands of times simultaneously. Unlike a CPU function (run once), a kernel is launched across a massive Grid of threads.

**2. THE GRID**
The highest-level workspace for a single kernel launch. It represents the entire problem space (e.g., an entire image or matrix).

**3. THE BLOCK**
A Grid is divided into Blocks.
- **INDEPENDENCE:** Blocks cannot easily communicate with each other.
- **SCALABILITY:** Because they are independent, a GPU can run them in any order. A bigger GPU simply runs more blocks at the same time.

**4. THE THREAD**
The smallest unit of execution. A single thread represents one path of instructions (e.g., calculating one pixel).

---

## II. THE PHYSICAL HIERARCHY (The Silicon View)

This is how the logical work is mapped onto literal NVIDIA hardware.

**1. THE STREAMING MULTIPROCESSOR (SM)**
The "Control Center." A GPU is a collection of many SMs.
- **Role:** When you launch a Grid, the GPU distributes **Blocks** to available SMs.
- **Resources:** Each SM has its own schedulers, math cores, and memory banks.
- **Limit:** A Block must stay on its assigned SM until it finishes.

**2. THE WARP**
The "Unit of Action." Inside an SM, threads are bundled into groups of **32**.
- **SIMT (Single Instruction, Multiple Threads):** All 32 threads in a warp execute the exact same instruction at the exact same clock cycle.

---

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

- **L1 cache:** Managed automatically by hardware. Holds recently-used local thread data. You don't control it directly.
- **Shared memory:** A **manually-controlled scratchpad**. You explicitly load data into it using CUDA code and share it among all threads in the same block. Access is roughly **10–20× faster than VRAM**.

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

---

## IV. DATA MOVEMENT & OPTIMIZATION

### THE CACHE LINE & SECTORS

VRAM is like a library. A thread asking for a 4-byte float is like asking for one book. The memory controller is efficient: it never fetches one value in isolation — it always retrieves a full **128-byte cache line**. Whether you needed 4 bytes or 128, you pay the same latency cost either way.

### COALESCING — The Perfect Trip

If Thread 0 asks for element 0, Thread 1 asks for element 1, Thread 2 asks for element 2, and so on across the warp, all 32 requests map to the same cache line. The memory controller makes **one trip**. This is called a **coalesced access** — it is the ideal case and the target to aim for when writing kernels.

### UNCOALESCED ACCESS — The Bandwidth Killer

If threads in the same warp access scattered, non-contiguous addresses, the memory controller must fetch multiple cache lines to satisfy the warp. Most of the bytes in those cache lines are wasted — fetched but never used. This chokes memory bandwidth and is one of the most common performance killers in CUDA code.

---

## V. ARCHITECTURAL PERFORMANCE BOOSTERS

### LATENCY HIDING — The "Wait-Switch" Strategy

VRAM is slow (~400+ cycles of waiting). Instead of leaving math units idle during a fetch, the SM instantly context-switches to a **different warp** that is already ready to execute. This costs nothing — each warp has its own registers and state, so switching is instantaneous.

The GPU doesn't try to make one thread fast. It tries to keep the math units **100% busy** by juggling many warps. This is why GPUs launch far more threads than they have cores — the excess warps exist specifically to hide memory latency.

### WARP DIVERGENCE — The "Sequential" Trap

Since a warp is SIMT (all 32 threads execute the same instruction), an `if/else` branch is dangerous. If half the warp takes the `if` path and half takes the `else` path, the hardware must serialize them: run the `if` branch with the `else` threads masked off, then run the `else` branch with the `if` threads masked off. Throughput is effectively halved.

- **Solution:** Prefer branch-free math wherever possible. Example: `y = max(0.0f, x)` instead of `if (x < 0) y = 0; else y = x;`.

### TENSOR CORES — The Matrix Engines

Specialized hardware units designed specifically for deep learning workloads.

- **Normal CUDA cores:** 1 multiply-accumulate per cycle (`a * b + c`).
- **Tensor Cores:** Compute a full **4×4 matrix multiplication in a single cycle** — the equivalent of 64 multiply-accumulates simultaneously.

This is why FP16/BF16 precision is massively faster than FP32 for AI training and inference: Tensor Cores only activate on half-precision (and lower) data types.

---

## VI. NUMERIC PRECISION

| Format | Bytes | Notes |
| :--- | :--- | :--- |
| **FP32** (Single Precision) | 4 bytes | Standard for accuracy. Default in most scientific computing. |
| **FP16** (Half Precision) | 2 bytes | Half the memory, 2× throughput minimum. Unlocks Tensor Cores. Essential for LLMs. |
| **BF16** (Brain Float 16) | 2 bytes | Same exponent range as FP32 with reduced mantissa. Preferred for training stability over FP16. |