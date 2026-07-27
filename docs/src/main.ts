import './style.css'
import { loadPyodide } from 'pyodide'

const codeInput = document.getElementById('code-input') as HTMLTextAreaElement | null
const runBtn = document.getElementById('run-btn') as HTMLButtonElement | null
const outputText = document.getElementById('output-text') as HTMLPreElement | null

async function init() {
  if (outputText) {
    outputText.textContent = 'Loading Pyodide environment...'
  }

  try {
    const pyodide = await loadPyodide()

    if (outputText) {
      outputText.textContent = 'Loading PyMatch package...'
    }

    // Load PyMatch wheel directly with pyodide.loadPackage (zero dependencies)
    const wheelUrl = new URL('/match-0.1.0-py3-none-any.whl', window.location.origin).href
    await pyodide.loadPackage(wheelUrl)

    if (outputText) {
      outputText.textContent = 'PyMatch loaded! Type Python code using match and click Run.'
    }

    const runExpression = () => {
      if (!codeInput || !outputText) return

      let outputLogs: string[] = []
      pyodide.setStdout({
        batched: (msg: string) => {
          outputLogs.push(msg)
        },
      })

      try {
        const result = pyodide.runPython(codeInput.value)
        let finalOutput = outputLogs.join('\n')
        if (result !== undefined && result !== null) {
          if (finalOutput.length > 0) {
            finalOutput += '\n' + String(result)
          } else {
            finalOutput = String(result)
          }
        }
        outputText.textContent = finalOutput.length > 0 ? finalOutput : '(No output)'
      } catch (err) {
        let errText = outputLogs.join('\n')
        if (errText.length > 0) {
          errText += '\n'
        }
        errText += String(err)
        outputText.textContent = errText
      }
    }

    if (runBtn) {
      runBtn.addEventListener('click', runExpression)
    }

    if (codeInput) {
      codeInput.addEventListener('keydown', (e: KeyboardEvent) => {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
          e.preventDefault()
          runExpression()
        }
      })
    }
  } catch (err) {
    if (outputText) {
      outputText.textContent = `Error loading Pyodide or PyMatch wheel: ${err}`
    }
  }
}

init()
