# Silicon-to-Synapse: Deep Learning & CUDA Mastery

> **"From the logic gates to the neural networks."**

A rigorous, first-principles approach to mastering Deep Learning, PyTorch, and CUDA. This repository documents the journey from understanding matrix calculus and GPU hardware to implementing state-of-the-art architectures with minimal AI assistance.

---

## 🎯 The Mission

To bridge the gap between high-level deep learning research and low-level hardware optimization. The goal is to write code that isn't just mathematically correct, but **hardware-optimal**.

Modern AI development often abstracts away the "silicon" layer. This project is a commitment to peeling back those abstractions—understanding the synapse through the lens of the chip.

---

## 🏛️ Core Pillars

### 1. First-Principles Mathematics
No "black boxes." Every optimization algorithm, backpropagation step, and loss function is derived from scratch before being implemented.

### 2. Hardware-Aware Programming
Deep learning is bounded by compute and memory. We focus on:
- **CUDA Kernels:** Writing custom kernels for specialized operations.
- **Memory Hierarchy:** Optimizing for Shared Memory, L1/L2 Cache, and HBM bandwidth.
- **Arithmetic Intensity:** Understanding the balance between FLOPS and Memory Ops.

### 3. Implementation Rigor
Implementing SOTA architectures (Transformers, CNNs, GNNs) with minimal reliance on pre-baked libraries, ensuring a deep understanding of every tensor contraction.

---

## 📂 Repository Structure

The repository is divided into two primary tracks that feed into each other:

| Directory | Focus | Description |
| :--- | :--- | :--- |
| [`/theory`](./theory) | **The "Synapse"** | Mathematical derivations, logic diagrams, and hardware architecture conceptualization. |
| [`/practice`](./practice) | **The "Silicon"** | CUDA implementations, PyTorch extensions, and performance benchmarking. |

---

## 🛠️ Tech Stack

- **Languages:** Python, C++, CUDA C
- **Frameworks:** PyTorch (LibTorch for C++ integration)
- **Profiling:** NVIDIA Nsight Systems, Nsight Compute
- **Mathematics:** Matrix Calculus, Linear Algebra, Probability & Statistics

---

## 📜 Philosophy: "Minimal AI Assistance"

In an era of LLM-generated code, this repository serves as a sanctuary for manual mastery. 
- Code is written by hand to build intuition.
- AI is used for documentation or high-level conceptual brainstorming, not for solving the core engineering problems.
- If it isn't understood at the assembly/mathematical level, it isn't "done."

---

## 🛤️ Roadmap

- [x] **01 Foundation:** Matrix Calculus & GPU Hardware 101
- [ ] **02 Linear Layers:** Custom Linear Layer CUDA Kernels
- [ ] **03 Optimization:** SGD, Adam, and fused kernels
- [ ] **04 Sequence Models:** Implementing Attention from Scratch
- [ ] **05 Production:** TensorRT & Quantization (INT8/FP8)

---

*"What I cannot create, I do not understand." — Richard Feynman*
