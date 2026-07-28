import { readdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig } from "vite";

// Dynamically read package version from pyproject.toml
const pyprojectPath = resolve(__dirname, "../pyproject.toml");
const pyprojectContent = readFileSync(pyprojectPath, "utf-8");
const versionMatch = pyprojectContent.match(/version\s*=\s*"([^"]+)"/);
const matchVersion = versionMatch ? versionMatch[1] : "0.1.0";
const matchWheelName = `match-${matchVersion}-py3-none-any.whl`;

// Collect all HTML pages in docs directory for multi-page build
const htmlFiles = readdirSync(__dirname)
  .filter((f) => f.endsWith(".html"))
  .reduce<Record<string, string>>((acc, file) => {
    const name = file.replace(/\.html$/, "");
    acc[name] = resolve(__dirname, file);
    return acc;
  }, {});

export default defineConfig({
  define: {
    __MATCH_VERSION__: JSON.stringify(matchVersion),
    __MATCH_WHEEL_NAME__: JSON.stringify(matchWheelName),
  },
  optimizeDeps: {
    exclude: ["pyodide"],
  },
  build: {
    rollupOptions: {
      input: htmlFiles,
    },
  },
});
