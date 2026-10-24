// Отчёт по практике (ГОСТ 7.32-2017): ссылки [@ключ] нумеруются по первому упоминанию.
const fs = require("fs");
const path = require("path");
const os = require("os");
const { build } = require("./gost");
const title = require("./title");
const refs = require("./refs");
const root = path.resolve(__dirname, "..");

let src = fs.readFileSync(path.join(root, "src/practice-report.md"), "utf8");
const order = [];
src = src.replace(/\[(@[^\]]+)\]/g, (_, inner) => {
  const nums = inner.split(",").map((k) => k.trim().replace(/^@/, "")).map((k) => {
    if (!refs[k]) throw new Error(`нет источника ${k}`);
    if (!order.includes(k)) order.push(k);
    return order.indexOf(k) + 1;
  }).sort((a, b) => a - b);
  return `[${nums.join(", ")}]`;
});
src = src.replace("@references", order.map((k, i) => `${i + 1}) ${refs[k]}`).join("\n"));
const unused = Object.keys(refs).filter((k) => !order.includes(k));
if (unused.length) console.log("не процитированы:", unused.join(", "));
const tmp = path.join(root, "src", ".practice-report.resolved.md");
fs.writeFileSync(tmp, src);
build({
  srcFile: tmp,
  outFile: path.join(root, "Отчёт по производственной практике (ГОСТ 7.32).docx"),
  title: {
    docTitle: "Отчёт по производственной практике",
    lines: title({
      kind: "ОТЧЁТ ПО ЗАДАНИЮ НА ПРОИЗВОДСТВЕННУЮ ПРАКТИКУ",
      subtitle: "(7 семестр)",
      theme: "Интерактивный ассистент для поддержки принятия решений операторами АЭС на основе формирования и верификации гипотез о состоянии системы",
    }),
  },
}).then((c) => { console.log("ok", c.tableNo, "таблиц,", c.figNo, "рисунков,", order.length, "источников"); fs.unlinkSync(tmp); });
