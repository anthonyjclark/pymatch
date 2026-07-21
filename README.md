# PyMatch (`pymatch` / `match`)

> Educational PyTorch-like Deep Learning Engine & Web Laboratory

**PyMatch** is an educational neural network library and interactive web platform designed for learning autograd, C-extensions, SIMD vectorization, and GPU compute shaders—running 100% client-side in the browser via WebAssembly (Pyodide) and WebGPU.

## Repository Structure

```
pymatch/
├── match/                # Core Python library (match.Tensor, autograd, nn, optim)
│   ├── __init__.py
│   ├── tensor.py         # Reverse topological autograd DAG engine
│   ├── nn.py             # Neural network layers (Linear, Sequential, ReLU, MSELoss, CrossEntropyLoss)
│   ├── optim.py          # Optimizers (SGD, Adam)
│   ├── backend.py        # Hardware execution dispatcher (Python, SIMD, WebGPU)
│   ├── utils.py          # Dataset generators (Spiral, Moons, XOR)
│   ├── c_extension.c     # Annotated C matrix multiplication & autograd routines
│   ├── simd_demo.c       # WASM 128-bit SIMD intrinsics (v128_f32x4)
│   └── cuda_demo.cu      # CUDA C++ kernel with shared memory SRAM tiling
│
└── docs/                 # Documentation website & interactive REPL notebook (Vite React)
    ├── src/
    │   ├── components/   # REPL Notebook, DAG Visualizer, Benchmarks, C/CUDA Lab
    │   └── engine/       # Pyodide WebWorker & WebGPU Compute Shader pipeline
    └── public/
```

## Quick Start

### 1. Python Library Installation
```bash
pip install -e .
```
```python
import match

# Create tensors with autograd tracking
x = match.tensor([3.0], requires_grad=True)
y = match.tensor([-2.0], requires_grad=True)

f = x * x * y + 4.0 * x
f.backward()

print("df/dx:", x.grad[0])  # Expected: -8.0
```

### 2. Website Development Server (`docs/`)
```bash
cd docs
npm install
npm run dev
```
Open `http://localhost:5173/` in your browser.
