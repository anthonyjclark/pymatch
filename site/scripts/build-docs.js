import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { marked } from "marked";
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";

hljs.registerLanguage("python", python);

marked.use({
  renderer: {
    code({ text, lang }) {
      const language = lang && hljs.getLanguage(lang) ? lang : "python";
      const highlighted = hljs.highlight(text, { language }).value;
      return `<pre class="hljs"><code class="language-${language}">${highlighted}</code></pre>`;
    },
  },
});


const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const rootDir = path.resolve(__dirname, "..");
const contentDir = path.join(rootDir, "content");

if (!fs.existsSync(contentDir)) {
  fs.mkdirSync(contentDir, { recursive: true });
}

// Protect math expressions from marked parser mangling while letting marked parse code blocks safely
function protectMath(mdText) {
  const mathBlocks = [];
  // Regex matches display math ($$...$$ or \[...\]) and inline math ($...$ or \(...\))
  const regex = /(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\(.*?\\\)|(?<!\\)\$[^\$\n]+?(?<!\\)\$)/g;

  const protectedText = mdText.replace(regex, (match) => {
    const placeholder = `MATHBLOCKTOKEN${mathBlocks.length}ENDTOKEN`;
    mathBlocks.push(match);
    return placeholder;
  });

  return { protectedText, mathBlocks };
}

function protectTextareas(mdText) {
  const textareas = [];
  const regex = /(<textarea[\s\S]+?<\/textarea>)/gi;
  const protectedText = mdText.replace(regex, (match) => {
    const placeholder = `TEXTAREABLOCKTOKEN${textareas.length}ENDTOKEN`;
    textareas.push(match);
    return placeholder;
  });
  return { protectedText, textareas };
}

function restoreMath(htmlText, mathBlocks) {
  let restored = htmlText;
  for (let i = 0; i < mathBlocks.length; i++) {
    const placeholder = `MATHBLOCKTOKEN${i}ENDTOKEN`;
    restored = restored.replace(placeholder, mathBlocks[i]);
  }
  return restored;
}

function restoreTextareas(htmlText, textareas) {
  let restored = htmlText;
  for (let i = 0; i < textareas.length; i++) {
    const placeholder = `TEXTAREABLOCKTOKEN${i}ENDTOKEN`;
    restored = restored.replace(placeholder, textareas[i]);
  }
  return restored;
}

const template = (title, contentHtml, filename) => `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="description" content="PyMatch - ${title}" />
    <title>PyMatch - ${title}</title>
    <script>
      window.MathJax = {
        tex: {
          inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
          displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
          processEscapes: true
        },
        options: {
          ignoreHtmlClass: 'tex2jax_ignore',
          processHtmlClass: 'tex2jax_process'
        }
      };
    </script>
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
        \\[
          \\def\\i{{^{(i)}}}
          \\def\\vx{{\\mathbf{x}}}
          \\def\\vy{{\\mathbf{y}}}
          \\def\\vw{{\\mathbf{w}}}
          \\def\\vb{{\\mathbf{b}}}
          \\def\\vz{{\\mathbf{z}}}
          \\def\\va{{\\mathbf{a}}}
          \\def\\yhat{{\\hat y}}
          \\def\\vyhat{{\\mathbf{\\hat y}}}
          \\def\\mae{{||\\vyhat - \\vy||_1}}
          \\def\\vhmse{{\\frac{1}{2N} ||(\\vyhat - \\vy)^2||_1}}
          \\def\\vbce{{-\\frac{1}{N}\\sum_{i=1}^N (y\\i \\log{\\yhat\\i} + (1 - y\\i)\\log{(1-\\yhat\\i)})}}
        \\]
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

// Discover all markdown files in content directory
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

  // Extract first paragraph for description
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

// 1. Build individual chapter pages with Prev/Next chapter navigation at bottom
for (let i = 0; i < chapters.length; i++) {
  const ch = chapters[i];
  const prevCh = i > 0 ? chapters[i - 1] : null;
  const nextCh = i < chapters.length - 1 ? chapters[i + 1] : null;

  const outFilePath = path.join(rootDir, ch.outFileName);
  const htmlExists = fs.existsSync(outFilePath);
  const htmlMtime = htmlExists ? fs.statSync(outFilePath).mtimeMs : 0;

  const needsRebuild = !htmlExists || ch.mtimeMs > htmlMtime || scriptMtime > htmlMtime;

  if (needsRebuild) {
    // Protect textareas and math expressions from marked parser mangling
    const { protectedText: p1, textareas } = protectTextareas(ch.mdContent);
    const { protectedText: p2, mathBlocks } = protectMath(p1);
    let parsedHtml = marked.parse(p2);
    let htmlBody = restoreMath(parsedHtml, mathBlocks);
    htmlBody = restoreTextareas(htmlBody, textareas);

    // Append Previous / Next chapter navigation
    const navHtml = `
      <nav class="chapter-nav">
        ${prevCh ? `<a href="/${prevCh.outFileName}" class="nav-prev">&larr; Previous: ${prevCh.title}</a>` : `<span></span>`}
        ${nextCh ? `<a href="/${nextCh.outFileName}" class="nav-next">Next: ${nextCh.title} &rarr;</a>` : `<span></span>`}
      </nav>
    `;
    htmlBody += navHtml;

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
