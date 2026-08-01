# PyMatch (`match`)

> Educational PyTorch-like neural network library and guide.

**PyMatch** is an educational neural network library and interactive web platform designed for learning autograd, C-extensions, SIMD vectorization, and GPU compute shaders—running 100% client-side in the browser via WebAssembly (Pyodide) and WebGPU.

## Building the PyMatch Wheel

To build the PyMatch `.whl` wheel package and copy it to the site public folder for Pyodide web execution:

```bash
# Build the Python wheel distribution
uv build

# Copy the generated wheel into site/public
mkdir -p site/public
cp dist/match-0.1.0-py3-none-any.whl site/public/
```

TODO: automate the wheel copy step (either in vite or uv).

## Building the Docs Site

The site source code and interactive Pyodide playground live in the [`site/`](site/) directory, which builds the static website into the root [`docs/`](docs/) directory for GitHub Pages deployment.

```bash
# Navigate to site directory
cd site

# Install Node dependencies
npm install

# Start the Vite local development server
npm run dev

# Run Vitest Pyodide integration tests
npm test

# Autoformat TypeScript files using oxfmt
npm run format

# Build the production static site into root docs/ folder
npm run build
```

## Running Tests

Run the unit test suite using `uv`:

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_tensor.py
```

## Manual Testing in Pyodide

Running this and checking the UI experience is a good way to test the Pyodide build.

```python
import match
from time import time
start = time()
data = match.extras.load_mnist_dataset()
print(len(data[0]))
print(time() - start)
```

## Features

- [ ] Add full site search
- [ ] Add more interactive tutorials
- [ ] Add more PyTorch functionality
- [ ] Add copy-to-clipboard buttons for code and output blocks
- [ ] Find uses of `os` that should be replaced with `pathlib`
- [ ] Add a script to download the pyodide release
- [x] Move docs source files and use docs as the build output folder

