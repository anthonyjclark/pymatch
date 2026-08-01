import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { loadPyodide } from "pyodide";
import { describe, expect, it } from "vitest";

describe("Pyodide PyMatch Wheel Integration", () => {
  it("loads PyMatch wheel and performs autograd computation", async () => {
    const pyodide = await loadPyodide();

    // Absolute path to built wheel in public folder dynamically determined
    const wheelPath = resolve(process.cwd(), `public/${__MATCH_WHEEL_NAME__}`);

    // Load PyMatch wheel directly using pyodide.loadPackage
    await pyodide.loadPackage(wheelPath);

    // Run Python test using match, match.nn, and match.utils.data.DataLoader
    const result = pyodide.runPython(`
import match
import match.nn as nn
from match.utils.data import DataLoader, TensorDataset

x = match.tensor([3.0], requires_grad=True)
y = match.tensor([-2.0], requires_grad=True)

f = x * x * y + 4.0 * x
f.backward()

linear = nn.Linear(2, 1)

X = match.randn(10, 4)
targets = match.zeros(10)
loader = DataLoader(TensorDataset(X, targets), batch_size=4)
batch_shapes = [list(b[0].shape) for b in loader]

[f.item(), x.grad.item(), y.grad.item(), list(linear.weight.shape), batch_shapes]
`);

    const [fVal, xGrad, yGrad, linearShape, batchShapes] = result.toJs();
    expect(fVal).toBe(-6.0);
    expect(xGrad).toBe(-8.0);
    expect(yGrad).toBe(9.0);
    expect(linearShape).toEqual([2, 1]);
    expect(batchShapes).toEqual([[4, 4], [4, 4], [2, 4]]);

    // Test load_mnist_dataset inside Pyodide WebAssembly VFS
    const binBuffer = readFileSync(resolve(process.cwd(), "public/data/mnist.bin"));
    pyodide.FS.mkdir("/data");
    pyodide.FS.writeFile("/data/mnist.bin", binBuffer);

    const mnistResult = pyodide.runPython(`
from match.extras import load_mnist_dataset
tr, va = load_mnist_dataset('/data')
[len(tr), len(va)]
`);
    const [trainLen, validLen] = mnistResult.toJs();
    expect(trainLen).toBe(60000);
    expect(validLen).toBe(10000);
  }, 30000);
});
