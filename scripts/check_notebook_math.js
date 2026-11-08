"use strict";
// Validate displayed and inline TeX in the five notebook Markdown sources.
const fs = require("fs");
const path = require("path");
const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { SVG } = require("mathjax-full/js/output/svg.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { AllPackages } = require("mathjax-full/js/input/tex/AllPackages.js");
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const doc = mathjax.document("", { InputJax: new TeX({ packages: AllPackages }), OutputJax: new SVG({ fontCache: "none" }) });
const names = ["01_advection_burgers", "02_euler_classical", "03_euler_riemann_solvers", "04_spatial_reconstruction", "05_time_integration"];
const regex = /(?<!\\)\$\$([\s\S]+?)(?<!\\)\$\$|(?<!\\)\$(?!\$)([\s\S]+?)(?<!\\)\$(?!\$)|(?<!\\)\\\[([\s\S]+?)(?<!\\)\\\]|(?<!\\)\\\(([\s\S]+?)(?<!\\)\\\)/g;
let total = 0;
const errors = [];
for (const name of names) {
  const nb = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "notebooks", name + ".ipynb"), "utf8"));
  let count = 0;
  nb.cells.forEach((cell) => {
    if (cell.cell_type !== "markdown") return;
    const source = Array.isArray(cell.source) ? cell.source.join("") : cell.source;
    const clean = source.replace(/^\s*```[^\n]*\n[\s\S]*?^\s*```[^\n]*$/gm, "").replace(/`[^`\n]*`/g, "");
    const matches = [...clean.matchAll(regex)];
    if (clean.replace(regex, "").includes("$")) errors.push({ name, cell: cell.id, reason: "Unmatched math delimiter" });
    for (const match of matches) {
      count += 1;
      const expression = match.slice(1).find(s => s !== undefined);
      try {
        const html = adaptor.outerHTML(doc.convert(expression, { display: match[1] !== undefined || match[3] !== undefined }));
        const bad = html.match(/data-mjx-error="([^"]+)"/);
        if (bad) errors.push({ name, cell: cell.id, reason: bad[1], expression: expression.slice(0, 170) });
      } catch (e) {
        errors.push({ name, cell: cell.id, reason: String(e), expression: expression.slice(0, 170) });
      }
    }
  });
  total += count;
  console.log(name + ": " + count + " formulas");
}
console.log("Total " + total + " formulas; errors " + errors.length);
if (total < 600 || errors.length) {
  for (const error of errors) console.error(JSON.stringify(error));
  process.exit(1);
}
