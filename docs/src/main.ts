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
      outputText.textContent = 'Pyodide ready! Type an expression and click Run.'
    }

    const runExpression = () => {
      if (!codeInput || !outputText) return
      try {
        const result = pyodide.runPython(codeInput.value)
        outputText.textContent = String(result ?? '')
      } catch (err) {
        outputText.textContent = String(err)
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
      outputText.textContent = `Error loading Pyodide: ${err}`
    }
  }
}

init()
