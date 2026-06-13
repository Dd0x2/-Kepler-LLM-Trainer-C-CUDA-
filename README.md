# Kepler LLM Trainer (C++ / CUDA)

A custom C++ and CUDA 10.2 implementation designed to train Small Language Models (SLMs) on legacy **Nvidia Kepler Architecture (Tesla K20/K10)** using full FP32 precision.

## Features
- **Legacy Support**: Tailored for Compute Capability 3.5 (Kepler).
- **VRAM Optimizations**: Uses mini-batch data streaming to avoid Out-Of-Memory (OOM) errors on 5GB/6GB cards.
- **Pure FP32 Execution**: Custom kernels bypassed modern half-precision hardware dependencies.

## Prerequisites
- Nvidia Driver supporting CUDA 10.2 (Legacy)
- GCC/G++ Compiler compatible with CUDA 10.2
- Kepler GPU (Tesla K20, K10, GTX 680, etc.)

## How to Build & Run

1. Clone the repository:
   ```bash
   git clone https://github.com
   cd kepler-llm-trainer
   ```

2. Create a `dataset.txt` file in the root directory and populate it with chat interactions.

3. Build the project using Makefile:
   ```bash
   make
   ```

4. Run the trainer:
   ```bash
   ./kepler_train
   ```

## License
MIT License - feel free to use and modify for academic research.
