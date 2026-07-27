import './style.css'
import { loadPyodide } from 'pyodide';

const app = document.getElementById('app')
if (app) {
  app.innerHTML = '<h1>App</h1>'
}

let pyodide = await loadPyodide();
console.log(pyodide.runPython("4 + 2"));
