import "./style.css";

window.MathJax = window.MathJax || {
  tex: {
    inlineMath: [
      ["$", "$"],
      ["\\(", "\\)"],
    ],
    displayMath: [
      ["$$", "$$"],
      ["\\[", "\\]"],
    ],
    processEscapes: true,
  },
};

import "mathjax/es5/tex-mml-chtml.js";

if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
  window.MathJax.typesetPromise();
}

const codeInput = document.getElementById("code-input") as HTMLTextAreaElement | null;
const runBtn = document.getElementById("run-btn") as HTMLButtonElement | null;
const outputText = document.getElementById("output-text") as HTMLPreElement | null;

if (!codeInput || !runBtn || !outputText) {
  throw new Error("Could not find code input, run button, or output text elements in the DOM.");
}

function init() {
  outputText!.textContent = "Initializing Pyodide Web Worker...";
  runBtn!.disabled = true;

  const worker = new Worker(new URL("./pyodide.worker.ts", import.meta.url), { type: "module" });

  let isWorkerReady = false;
  let isExecuting = false;

  worker.postMessage({
    type: "init",
    wheelName: __MATCH_WHEEL_NAME__,
    origin: window.location.origin,
  });

  worker.onmessage = (e: MessageEvent) => {
    const { type, text, output, error } = e.data;

    if (type === "status") {
      if (outputText && !isWorkerReady) {
        outputText.textContent = text;
      }
    } else if (type === "ready") {
      isWorkerReady = true;
      if (outputText) {
        outputText.textContent = `PyMatch v${__MATCH_VERSION__} ready! Type Python code using match and click Run.`;
      }
      if (runBtn) {
        runBtn.disabled = false;
      }
    } else if (type === "result") {
      isExecuting = false;
      if (runBtn) {
        runBtn.disabled = false;
        runBtn.textContent = "Run";
      }
      if (outputText) {
        if (output.includes("<div") || output.includes("<svg")) {
          outputText.innerHTML = output;
        } else {
          outputText.textContent = output;
        }
      }
    } else if (type === "error") {
      isExecuting = false;
      if (runBtn) {
        runBtn.disabled = false;
        runBtn.textContent = "Run";
      }
      if (outputText) {
        outputText.textContent = error;
      }
    }
  };

  const runExpression = () => {
    if (!codeInput || !outputText || !isWorkerReady || isExecuting) return;

    isExecuting = true;
    if (runBtn) {
      runBtn.disabled = true;
      runBtn.textContent = "Running...";
    }
    outputText.textContent = "Running Python code in background worker...";

    worker.postMessage({
      type: "run",
      code: codeInput.value,
    });
  };

  if (runBtn) {
    runBtn.addEventListener("click", runExpression);
  }

  if (codeInput) {
    codeInput.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        runExpression();
      }
    });
  }
}

init();
