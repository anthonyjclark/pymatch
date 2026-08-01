import { loadPyodide, type PyodideAPI } from "pyodide";

let pyodide: PyodideAPI;
let isReady = false;

//
// region: Message handler
//

self.onmessage = (e: MessageEvent) => {
  const { type, wheelName, origin, code } = e.data;
  if (type === "init") {
    initWorker(wheelName, origin);
  } else if (type === "run") {
    runCode(code);
  }
};

//
// region: Init
//

async function initWorker(wheelName: string, origin: string) {
  postMessage({ type: "status", text: "Loading Pyodide environment..." });
  try {
    pyodide = await loadPyodide({ indexURL: `${origin}/pyodide/` });
  } catch (e) {
    postMessage({ type: "error", error: "Failed to load Pyodide environment: " + String(e) });
    return;
  }

  postMessage({ type: "status", text: "Installing packages..." });
  let micropip;
  try {
    await pyodide.loadPackage("micropip");
    micropip = pyodide.pyimport("micropip");
  } catch (e) {
    postMessage({ type: "error", error: "Failed to load micropip: " + String(e) });
    return;
  }

  try {
    await micropip.install("matplotlib");
  } catch (e) {
    postMessage({ type: "error", error: "Failed to install Matplotlib: " + String(e) });
    return;
  }

  try {
    const wheelUrl = new URL(`/${wheelName}`, origin).href;
    await micropip.install(wheelUrl);
  } catch (e) {
    postMessage({ type: "error", error: "Failed to install PyMatch: " + String(e) });
    return;
  }

  //   // Configure Matplotlib dark theme defaults
  //   try {
  //     pyodide.runPython(`
  // import matplotlib
  // import matplotlib.pyplot as plt
  // plt.style.use('dark_background')
  // `);
  //   } catch (e) {
  //     console.warn("Matplotlib config warning:", e);
  //   }

  // // Pre-fetch preprocessed MNIST dataset into Pyodide VFS
  // try {
  //   const mnistUrl = new URL("/data/mnist.bin", origin).href;
  //   const res = await fetch(mnistUrl);
  //   if (res.ok) {
  //     const binBuf = new Uint8Array(await res.arrayBuffer());
  //     try {
  //       pyodide.FS.mkdir("/data");
  //     } catch (_) {}
  //     pyodide.FS.writeFile("/data/mnist.bin", binBuf);
  //     // Background pre-warm dataset cache in worker memory
  //     postMessage({ type: "status", text: "Pre-warming PyMatch dataset cache..." });
  //     pyodide.runPython(
  //       "from match.extras import load_mnist_dataset\ntry:\n    load_mnist_dataset()\nexcept Exception:\n    pass",
  //     );
  //   }
  // } catch (e) {
  //   console.warn("Worker MNIST pre-fetch warning:", e);
  // }

  isReady = true;
  postMessage({ type: "ready" });
}

//
// region: Run
//

async function runCode(code: string) {
  if (!pyodide || !isReady) {
    postMessage({ type: "error", error: "Pyodide environment is not initialized." });
    return;
  }

  let outputLogs: string[] = [];
  pyodide.setStdout({
    batched: (msg: string) => {
      outputLogs.push(msg);
    },
  });

  try {
    pyodide.runPython(`
import matplotlib.pyplot as plt
plt.close('all')
`);

    const result = await pyodide.runPythonAsync(code);

    // Extract Matplotlib figures into base64 PNG images
    const plotHtml = pyodide.runPython(`
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
      finalOutput = finalOutput.length > 0 ? `${finalOutput}\n${String(result)}` : String(result);
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
