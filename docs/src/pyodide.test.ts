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

    // Run Python test using match and match.nn
    const result = pyodide.runPython(`
import match
import match.nn as nn

x = match.tensor([3.0], requires_grad=True)
y = match.tensor([-2.0], requires_grad=True)

f = x * x * y + 4.0 * x
f.backward()

linear = nn.Linear(2, 1)
[f.item(), x.grad.item(), y.grad.item(), list(linear.weight.shape)]
`);

    const [fVal, xGrad, yGrad, linearShape] = result.toJs();
    expect(fVal).toBe(-6.0);
    expect(xGrad).toBe(-8.0);
    expect(yGrad).toBe(9.0);
    expect(linearShape).toEqual([2, 1]);
  }, 30000);
});
