import { defineConfig } from "vite";
// import { viteStaticCopy } from "vite-plugin-static-copy";

import { readdirSync, readFileSync } from "node:fs";
import { resolve, join } from "node:path";

// const PYODIDE_EXCLUDE = ["!**/*.{md,html}", "!**/*.d.ts", "!**/*.map"];

// function viteStaticCopyPyodide() {
//   const pyodideDir = resolve(import.meta.dirname, "pyodide");
//   return viteStaticCopy({
//     targets: [
//       { src: [join(pyodideDir, "*").replace(/\\/g, "/")].concat(PYODIDE_EXCLUDE), dest: "pyodide" },
//     ],
//   });
// }

function getMatchPackageInfo() {
  const pyprojectPath = resolve(import.meta.dirname, "../pyproject.toml");
  const pyprojectContent = readFileSync(pyprojectPath, "utf-8");
  const versionMatch = pyprojectContent.match(/version\s*=\s*"([^"]+)"/);
  const matchVersion = versionMatch ? versionMatch[1] : "0.1.0";
  const matchWheelName = `match-${matchVersion}-py3-none-any.whl`;
  return { matchVersion, matchWheelName };
}

function getHtmlFiles() {
  return readdirSync(import.meta.dirname)
    .filter((f) => f.endsWith(".html"))
    .reduce<Record<string, string>>((acc, file) => {
      const name = file.replace(/\.html$/, "");
      acc[name] = resolve(import.meta.dirname, file);
      return acc;
    }, {});
}

const { matchVersion, matchWheelName } = getMatchPackageInfo();

export default defineConfig({
  define: {
    __MATCH_VERSION__: JSON.stringify(matchVersion),
    __MATCH_WHEEL_NAME__: JSON.stringify(matchWheelName),
  },
  optimizeDeps: { exclude: ["pyodide"] },
  // TODO: copy match wheel: viteStaticCopy({ targets: [ { src: resolve(import.meta.dirname, `../dist/${matchWheelName}`).replace(/\\/g, "/"), dest: ".", } ] })
  // plugins: [viteStaticCopyPyodide()],
  build: { rollupOptions: { input: getHtmlFiles() } },
});
