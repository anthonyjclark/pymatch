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
          <a href="/index.html" class="${filename === "index.html" ? "active" : ""}">Home</a>
          <a href="/guide.html" class="${filename === "guide.html" ? "active" : ""}">Guide</a>
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

const scriptMtime = fs.statSync(__filename).mtimeMs;

// Discover all markdown files in content directory (excluding any non-chapter files if needed)
const mdFiles = fs
  .readdirSync(contentDir)
  .filter((file) => file.endsWith(".md"))
  .sort();

// Collect metadata for dynamic Table of Contents (guide page)
const chapters = [];
let maxContentMtime = 0;

for (const file of mdFiles) {
  const filePath = path.join(contentDir, file);
  const mdContent = fs.readFileSync(filePath, "utf-8");
  const stat = fs.statSync(filePath);
  if (stat.mtimeMs > maxContentMtime) {
    maxContentMtime = stat.mtimeMs;
  }

  // Extract title
  const titleMatch = mdContent.match(/^#\s+(.+)$/m);
  const title = titleMatch ? titleMatch[1].trim() : path.basename(file, ".md");

  // Extract first paragraph for description (skipping title and quotes)
  const paragraphMatch = mdContent
    .split("\n")
    .map((line) => line.trim())
    .find((line) => line.length > 0 && !line.startsWith("#") && !line.startsWith(">") && !line.startsWith("<") && !line.startsWith("-"));

  const description = paragraphMatch || "";

  const outFileName = file.replace(/\.md$/, ".html");
  chapters.push({
    file,
    outFileName,
    title,
    description,
    mtimeMs: stat.mtimeMs,
    filePath,
    mdContent,
  });
}

// 1. Build individual chapter pages
for (const ch of chapters) {
  const outFilePath = path.join(rootDir, ch.outFileName);
  const htmlExists = fs.existsSync(outFilePath);
  const htmlMtime = htmlExists ? fs.statSync(outFilePath).mtimeMs : 0;

  const needsRebuild = !htmlExists || ch.mtimeMs > htmlMtime || scriptMtime > htmlMtime;

  if (needsRebuild) {
    const htmlBody = marked.parse(ch.mdContent);
    const fullHtml = template(ch.title, htmlBody, ch.outFileName);
    fs.writeFileSync(outFilePath, fullHtml, "utf-8");
    console.log(`Generated ${ch.outFileName} from ${ch.file}`);
  } else {
    console.log(`Skipped ${ch.outFileName} (up to date)`);
  }
}

// 2. Dynamically generate the Guide (Table of Contents) page: guide.html
const guidePath = path.join(rootDir, "guide.html");
const guideExists = fs.existsSync(guidePath);
const guideMtime = guideExists ? fs.statSync(guidePath).mtimeMs : 0;
const needsGuideRebuild = !guideExists || maxContentMtime > guideMtime || scriptMtime > guideMtime;

if (needsGuideRebuild) {
  const guideContentHtml = `
    <h1>Educational Guide - Table of Contents</h1>
    <p class="lead-text">
      Welcome to the PyMatch educational neural network guide. Click on any chapter below to start reading:
    </p>

    <ul class="chapter-list">
      ${chapters
        .map(
          (ch) => `
        <li>
          <a href="/${ch.outFileName}"><strong>${ch.title}</strong></a>
          ${ch.description ? `<p>${ch.description}</p>` : ""}
        </li>
      `
        )
        .join("")}
    </ul>
  `;

  const fullGuideHtml = template("Guide - Table of Contents", guideContentHtml, "guide.html");
  fs.writeFileSync(guidePath, fullGuideHtml, "utf-8");
  console.log("Generated guide.html (Table of Contents)");
} else {
  console.log("Skipped guide.html (up to date)");
}
