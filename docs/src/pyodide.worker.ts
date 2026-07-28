import { loadPyodide } from "pyodide";

let pyodideInstance: any = null;
let isReady = false;

async function initWorker(wheelName: string, origin: string) {
  try {
    postMessage({ type: "status", text: "Loading Pyodide environment..." });
    pyodideInstance = await loadPyodide();

    postMessage({ type: "status", text: "Loading PyMatch and Matplotlib packages..." });
    const wheelUrl = new URL(`/${wheelName}`, origin).href;
    await pyodideInstance.loadPackage([wheelUrl, "matplotlib"]);

    // Configure Matplotlib dark theme defaults
    pyodideInstance.runPython(`
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('dark_background')
`);

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
    pyodideInstance.runPython(`
import matplotlib.pyplot as plt
plt.close('all')
`);

    const result = pyodideInstance.runPython(code);

    // Extract Matplotlib figures into base64 PNG images
    const plotHtml = pyodideInstance.runPython(`
import io, base64
import matplotlib.pyplot as plt
_html_out = ""
if plt.get_fignums():
    for _num in plt.get_fignums():
        _fig = plt.figure(_num)
        _buf = io.BytesIO()
        _fig.savefig(_buf, format="png", bbox_inches="tight", facecolor='#181825', edgecolor='none')
        _buf.seek(0)
        _b64 = base64.b64encode(_buf.read()).decode("ascii")
        _html_out += f'<div style="text-align:center; margin:12px 0;"><img src="data:image/png;base64,{_b64}" style="max-width:100%; height:auto; border-radius:6px; border:1px solid #313244;" /></div>'
    plt.close('all')
_html_out
`);

    let finalOutput = outputLogs.join("\n");
    if (result !== undefined && result !== null) {
      if (finalOutput.length > 0) {
        finalOutput += "\n" + String(result);
      } else {
        finalOutput = String(result);
      }
    }
    if (plotHtml && plotHtml.trim().length > 0) {
      finalOutput = (finalOutput ? finalOutput + "\n" : "") + plotHtml;
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
