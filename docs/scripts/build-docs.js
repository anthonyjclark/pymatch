import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { marked } from "marked";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const rootDir = path.resolve(__dirname, "..");
const contentDir = path.join(rootDir, "content");

if (!fs.existsSync(contentDir)) {
  fs.mkdirSync(contentDir, { recursive: true });
}

const template = (title, contentHtml, filename) => `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="description" content="PyMatch - ${title}" />
    <title>PyMatch - ${title}</title>
  </head>

  <body>
    <header class="site-header">
      <div class="header-container">
        <h1><a href="/index.html" style="color: inherit; text-decoration: none;">PyMatch Documentation</a></h1>
        <nav class="site-nav">
          <a href="/index.html">Home</a>
          <a href="/01-introduction.html" class="${filename === "01-introduction.html" ? "active" : ""}">1. Introduction</a>
        </nav>
      </div>
    </header>

    <main class="content-container">
      <div class="math-defs" style="display: none">
        \\[ \\def\\i{{^{(i)}}} \\def\\vx{{\\mathbf{x}}} \\def\\vy{{\\mathbf{y}}} \\def\\vw{{\\mathbf{w}}}
        \\def\\vb{{\\mathbf{b}}} \\def\\vz{{\\mathbf{z}}} \\def\\va{{\\mathbf{a}}} \\def\\yhat{{\\hat y}}
        \\def\\vyhat{{\\mathbf{\\hat y}}} \\def\\mae{{||\\vyhat - \\vy||_1}} \\def\\vhmse{{\\frac{1}{2N} ||(\\vyhat
        - \\vy)^2||_1}} \\def\\vbce{{-\\frac{1}{N}\\sum_{i=1}^N (y\\i \\log{\\yhat\\i} + (1 - y\\i)\\log{(1-\\yhat\\i)})}} \\]
      </div>

      <article class="doc-article">
        ${contentHtml}
      </article>
    </main>

    <footer class="site-footer">
      <p>&copy; 2026 Anthony J. Clark. All rights reserved.</p>
    </footer>

    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
`;

const mdFiles = fs.readdirSync(contentDir).filter((file) => file.endsWith(".md"));

for (const file of mdFiles) {
  const filePath = path.join(contentDir, file);
  const mdContent = fs.readFileSync(filePath, "utf-8");

  // Extract title from first H1 heading if present
  const titleMatch = mdContent.match(/^#\s+(.+)$/m);
  const title = titleMatch ? titleMatch[1].trim() : path.basename(file, ".md");

  const htmlBody = marked.parse(mdContent);
  const outFileName = file.replace(/\.md$/, ".html");
  const outFilePath = path.join(rootDir, outFileName);

  const fullHtml = template(title, htmlBody, outFileName);
  fs.writeFileSync(outFilePath, fullHtml, "utf-8");
  console.log(`Generated ${outFileName} from ${file}`);
}
