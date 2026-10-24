// Подробный отчёт (docs/full-report.md, обычный Markdown) → .docx для чтения (без требований ГОСТ к титулу).
const fs = require("fs");
const path = require("path");
const { build } = require("./gost");
const root = path.resolve(__dirname, "..");
const md = fs.readFileSync(path.resolve(root, "../docs/full-report.md"), "utf8").split("\n");
const out = [];
let inCode = false, inTable = false, docTitle = "";
for (const raw of md) {
  const line = raw.replace(/\s+$/, "");
  if (line.startsWith("```")) { out.push(inCode ? "@endcode" : "@code"); inCode = !inCode; continue; }
  if (inCode) { out.push(line); continue; }
  if (line.startsWith("|")) { if (!inTable) { out.push("@grid"); inTable = true; } out.push(line); continue; }
  inTable = false;
  if (/^# /.test(line)) { docTitle = line.slice(2); continue; }
  if (/^## /.test(line)) { out.push("# " + line.slice(3)); continue; }
  if (/^### /.test(line)) { out.push("## " + line.slice(4)); continue; }
  if (line === "---") continue;
  const num = line.match(/^(\s*)(\d+)\. (.*)$/);
  if (num) { out.push(num[1].length ? `  - ${num[3]}` : `${num[2]}) ${num[3]}`); continue; }
  out.push(line);
}
// ширины столбцов таблиц — по длине содержимого (иначе узкий столбец «№» занимает половину ширины)
for (let i = 0; i < out.length; i++) {
  if (out[i] !== "@grid") continue;
  const rows = [];
  for (let j = i + 1; j < out.length && out[j].startsWith("|"); j++) {
    const cells = out[j].replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
    if (!cells.every((c) => /^:?-+:?$/.test(c))) rows.push(cells);
  }
  const n = rows[0].length;
  const len = Array.from({ length: n }, (_, k) => Math.min(60, Math.max(4, ...rows.map((r) => (r[k] || "").length))));
  const total = len.reduce((a, b) => a + b, 0);
  out.splice(i, 0, "@widths " + len.map((x) => ((100 * x) / total).toFixed(1)).join(","));
  i++;
}
const tmp = path.join(root, "src", ".full-report.resolved.md");
fs.writeFileSync(tmp, out.join("\n"));
build({
  srcFile: tmp,
  outFile: path.join(root, "Полный отчёт по проекту (6 семестр – заморозка).docx"),
  sectionNewPage: true,
  title: {
    docTitle,
    lines: [
      { text: docTitle, bold: true, size: 32, after: 240 },
      { text: "Шелманова Ева Олеговна, группа Б23-605, НИЯУ МИФИ", size: 28, after: 120 },
      { text: "Состояние на 25.10.2026 (версия v1.0-freeze)", size: 28, after: 240 },
    ],
  },
  footerFromSecond: false,
}).then((c) => { console.log("ok", c.tableNo, "таблиц"); fs.unlinkSync(tmp); });
