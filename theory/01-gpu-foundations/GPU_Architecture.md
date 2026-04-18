================================================================================
SILICON-TO-SYNAPSE: GPU ARCHITECTURE FOUNDATIONS
"Understanding the hardware to write optimal software."
================================================================================

I. THE LOGICAL HIERARCHY (The Programmer's View)
--------------------------------------------------------------------------------
Before writing CUDA, we must understand how we organize work.

1. THE KERNEL:
   A specialized program designed to run thousands of times simultaneously.
   Unlike a CPU function (run once), a Kernel is launched across a massive 
   Grid of threads.

2. THE GRID:
   The highest level workspace for a single kernel launch. It represents the 
   entire problem space (e.g., an entire image or matrix).

3. THE BLOCK:
   A Grid is divided into Blocks. 
   - INDEPENDENCE: Blocks cannot easily communicate with each other.
   - SCALABILITY: Because they are independent, a GPU can run them in any order. 
     A bigger GPU simply runs more blocks at the same time.

4. THE THREAD:
   The smallest unit of execution. A single thread represents one path of 
   instructions (e.g., calculating one pixel).


II. THE PHYSICAL HIERARCHY (The Silicon View)
--------------------------------------------------------------------------------
This is how the logical work is mapped onto literal NVIDIA hardware.

1. THE STREAMING MULTIPROCESSOR (SM):
   The "Control Center." A GPU is a collection of many SMs. 
   - Role: When you launch a Grid, the GPU distributes **Blocks** to available SMs.
   - Resources: Each SM has its own schedulers, math cores, and memory banks.
   - Limit: A Block must stay on its assigned SM until it finishes.

2. THE WARP:
   The "Unit of Action." Inside an SM, threads are bundled into groups of **32**.
   - SIMT (Single Instruction, Multiple Threads): All 32 threads in a warp 
     execute the exact same instruction at the exact same clock cycle.


III. MEMORY HIERARCHY (The Speed vs. Size Trade-off)
--------------------------------------------------------------------------------
Architecture is defined by how fast we can move data to the math units.

[ SLOWEST / LARGEST ]
1. VRAM (Global Memory):
   The big memory chips on the GPU. This is the "Library." Capacity is large (GBs), 
   but it is far away and slow to access.

2. L2 CACHE:
   A high-speed intermediate bank that handles frequently used data.

3. SHARED MEMORY (The Programmable Cache):
   The "Workbench." This is high-speed memory located physically INSIDE the SM.
   - Unlike the L2, the programmer manually controls what goes here.
   - It is used to share data between threads in the same block.
   - Access is ~100x faster than VRAM.

4. L1 CACHE:
   A tiny, automatic hardware cache for local thread data.

5. REGISTERS:
   The "Hands." This is where the actual math happens. Each thread has its 
   own registers to hold the variables it is currently working on.
[ FASTEST / SMALLEST ]


IV. DATA MOVEMENT & OPTIMIZATION (The Library Analogy)
--------------------------------------------------------------------------------
1. THE CACHE LINE & SECTORS:
   VRAM is like a library. A thread asking for a 4-byte float is like asking 
   for one book. The Librarian (Memory Controller) is efficient: they never 
   bring one book; they bring a "Crate" (128-byte Cache Line).

2. COALESCING (The Perfect Trip):
   If Thread 0 asks for Book 1, Thread 1 asks for Book 2, and so on, the 
   entire Warp's request fits in one "Crate." The Librarian makes one trip. 
   Warp is happy. MATH IS FAST.

3. UNCOALESCED ACCESS (The Bandwidth Killer):
   If threads ask for scattered books, the Librarian must bring multiple 
   crates just to satisfy one warp. Most of the bytes in the crates are 
   wasted. Bandwidth is choked.


V. ARCHITECTURAL PERFORMANCE BOOSTERS
--------------------------------------------------------------------------------
1. LATENCY HIDING (The "Wait-Switch" Strategy):
   VRAM is slow (~400+ cycles of waiting). Instead of sitting idle, the SM 
   instantly context-switches to a DIFFERENT Warp that is ready to work. 
   The GPU doesn't try to make one thread fast; it tries to keep the math 
   units 100% busy by juggling many warps.

2. WARP DIVERGENCE (The "Sequential" Trap):
   Since a Warp is SIMT, an `if/else` statement is dangerous. If half the 
   warp goes left and half goes right, the hardware must run the `if` path 
   while the `else` threads sit idle, then swap. 
   - SOLUTION: Use branch-free math (e.g., `y = max(0, x)` instead of `if`).

3. TENSOR CORES (The Matrix Engines):
   Specialized hardware designed for Deep Learning. 
   - Normal CUDA cores: 1 calculation per cycle (a * b + c).
   - Tensor Cores: Can multiply two 4x4 matrices in a SINGLE cycle.
   - This is why FP16/BF16 is massively faster for AI.


VI. NUMERIC PRECISION
--------------------------------------------------------------------------------
- FP32 (Single Precision): 4 bytes. The standard for accuracy.
- FP16 (Half Precision): 2 bytes. Half the space, 2x the speed (or more 
  via Tensor Cores). Essential for Large Language Models.
================================================================================
