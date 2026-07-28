import { loadPyodide } from "pyodide";

let pyodideInstance: any = null;
let isReady = false;

async function initWorker(wheelName: string, origin: string) {
  try {
    postMessage({ type: "status", text: "Loading Pyodide environment..." });
    pyodideInstance = await loadPyodide();

    postMessage({ type: "status", text: "Loading PyMatch package..." });
    const wheelUrl = new URL(`/${wheelName}`, origin).href;
    await pyodideInstance.loadPackage(wheelUrl);

    // Pre-fetch preprocessed MNIST dataset into Pyodide VFS
    try {
      const mnistUrl = new URL("/data/mnist.bin", origin).href;
      const res = await fetch(mnistUrl);
      if (res.ok) {
        const binBuf = new Uint8Array(await res.arrayBuffer());
        try {
          pyodideInstance.FS.mkdir("/data");
        } catch (_) {}
        pyodideInstance.FS.writeFile("/data/mnist.bin", binBuf);

        // Background pre-warm dataset cache in worker memory
        postMessage({ type: "status", text: "Pre-warming PyMatch dataset cache..." });
        pyodideInstance.runPython(
          "from match.extras import load_mnist_dataset\ntry:\n    load_mnist_dataset()\nexcept Exception:\n    pass"
        );
      }
    } catch (e) {
      console.warn("Worker MNIST pre-fetch warning:", e);
    }

    isReady = true;
    postMessage({ type: "ready" });
  } catch (err) {
    postMessage({ type: "error", error: String(err) });
  }
}

function runCode(code: string) {
  if (!pyodideInstance || !isReady) {
    postMessage({ type: "error", error: "Pyodide environment is still initializing." });
    return;
  }

  let outputLogs: string[] = [];
  pyodideInstance.setStdout({
    batched: (msg: string) => {
      outputLogs.push(msg);
    },
  });

  try {
    const result = pyodideInstance.runPython(code);
    let finalOutput = outputLogs.join("\n");
    if (result !== undefined && result !== null) {
      if (finalOutput.length > 0) {
        finalOutput += "\n" + String(result);
      } else {
        finalOutput = String(result);
      }
    }
    postMessage({ type: "result", output: finalOutput.length > 0 ? finalOutput : "(No output)" });
  } catch (err) {
    let errText = outputLogs.join("\n");
    if (errText.length > 0) {
      errText += "\n";
    }
    errText += String(err);
    postMessage({ type: "error", error: errText });
  }
}

self.onmessage = (e: MessageEvent) => {
  const { type, wheelName, origin, code } = e.data;
  if (type === "init") {
    initWorker(wheelName, origin);
  } else if (type === "run") {
    runCode(code);
  }
};
