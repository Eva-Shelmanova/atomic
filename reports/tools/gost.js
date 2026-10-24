// Конвертер простой разметки → .docx с оформлением по ГОСТ 7.32-2017.
// Разметка (одна строка = один абзац):
//   #* ЗАГОЛОВОК        структурный элемент (по центру, прописные, с новой страницы, в оглавлении)
//   #*- ЗАГОЛОВОК       то же, но без оглавления (РЕФЕРАТ, СОДЕРЖАНИЕ)
//   # 1 Раздел          раздел (с новой страницы), ## 1.1 подраздел, ### 1.1.1 пункт
//   - текст             элемент списка с тире; "  - " — второй уровень
//   1) текст            элемент нумерованного списка (номер как есть)
//   @table Название     подпись таблицы; следующие строки | a | b | — таблица (первая строка — шапка)
//   @widths 30,70       ширины столбцов следующей таблицы в процентах
//   @figure путь.png | Название | ширина_см
//   @code ... @endcode  моноширинный блок
//   @toc                оглавление (поле Word)
//   @pagebreak          разрыв страницы
//   @center текст       абзац по центру без отступа; @right текст — по правому краю; @left — без отступа
//   @raw                ничего (комментарий: строки, начинающиеся с //, пропускаются)
// Внутри текста: **жирный**, `код`.
const fs = require("fs");
const path = require("path");
const d = require("docx");

const MM = (mm) => Math.round((mm / 25.4) * 1440);
const CM = (cm) => MM(cm * 10);
const FONT = "Times New Roman";

function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new d.TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new d.TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new d.TextRun({ text: t.slice(1, -1), font: "Courier New", size: base.size ? base.size - 2 : 24, ...Object.fromEntries(Object.entries(base).filter(([k]) => k !== "font" && k !== "size")) }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new d.TextRun({ text: text.slice(last), ...base }));
  return out;
}

const body = (text, opts = {}) =>
  new d.Paragraph({
    children: runs(text, opts.run || {}),
    alignment: opts.align || d.AlignmentType.JUSTIFIED,
    indent: opts.indent || { firstLine: CM(1.25) },
    spacing: { line: 360, before: 0, after: 0, ...(opts.spacing || {}) },
    keepNext: opts.keepNext,
    pageBreakBefore: opts.pageBreakBefore,
  });

function tableFrom(rows, widthsPct, tableNo) {
  const total = 9638; // ширина текста: 210 − 30 − 15 мм ≈ 165 мм
  const ncol = rows[0].length;
  const w = (widthsPct || Array(ncol).fill(100 / ncol)).map((p) => Math.round((total * p) / 100));
  const diff = total - w.reduce((a, b) => a + b, 0);
  w[w.length - 1] += diff;
  const border = { style: d.BorderStyle.SINGLE, size: 4, color: "000000" };
  const borders = { top: border, bottom: border, left: border, right: border };
  return new d.Table({
    width: { size: total, type: d.WidthType.DXA },
    columnWidths: w,
    rows: rows.map((cells, ri) =>
      new d.TableRow({
        tableHeader: ri === 0,
        cantSplit: true,
        children: cells.map((c, ci) =>
          new d.TableCell({
            borders,
            width: { size: w[ci], type: d.WidthType.DXA },
            margins: { top: 40, bottom: 40, left: 80, right: 80 },
            children: c.split("<br>").map((part) =>
              new d.Paragraph({
                children: runs(part.trim(), { size: 24, bold: ri === 0 ? true : undefined }),
                alignment: ri === 0 ? d.AlignmentType.CENTER : d.AlignmentType.LEFT,
                spacing: { line: 240, before: 0, after: 0 },
              })),
          })),
      })),
  });
}

function parse(src, ctx) {
  // метки таблиц и рисунков: @table#id, @figure#id → {tab:id}, {fig:id} в тексте
  const ids = { tab: {}, fig: {} };
  let nt = 0, nf = 0;
  for (const l of src.split("\n")) {
    if (l.startsWith("@table")) { nt += 1; const m = l.match(/^@table#(\S+)/); if (m) ids.tab[m[1]] = nt; }
    if (l.startsWith("@figure")) { nf += 1; const m = l.match(/^@figure#(\S+)/); if (m) ids.fig[m[1]] = nf; }
  }
  src = src.replace(/\{(tab|fig):([^}]+)\}/g, (_, k, id) => {
    if (!(id in ids[k])) throw new Error(`нет метки ${k}:${id}`);
    return String(ids[k][id]);
  });
  const lines = src.split("\n");
  const out = [];
  let i = 0, pendingWidths = null;
  const capt = (text) => body(text, { align: d.AlignmentType.LEFT, indent: { firstLine: 0 }, keepNext: true, spacing: { before: 120 } });
  while (i < lines.length) {
    let line = lines[i];
    if (line.startsWith("//") || line.trim() === "") { i++; continue; }
    if (line.startsWith("#*")) {
      const noToc = line.startsWith("#*-");
      const text = line.replace(/^#\*-?\s*/, "");
      out.push(new d.Paragraph({
        heading: noToc ? undefined : d.HeadingLevel.HEADING_1,
        style: noToc ? "StructNoToc" : undefined,
        children: [new d.TextRun({ text: text.toUpperCase(), bold: true, size: 28, font: FONT })],
        alignment: d.AlignmentType.CENTER,
        pageBreakBefore: true,
        spacing: { line: 360, before: 0, after: 240 },
        indent: { firstLine: 0 },
      }));
    } else if (/^#{1,3} /.test(line)) {
      const level = line.match(/^#+/)[0].length;
      const text = line.replace(/^#+\s*/, "");
      out.push(new d.Paragraph({
        heading: [d.HeadingLevel.HEADING_1, d.HeadingLevel.HEADING_2, d.HeadingLevel.HEADING_3][level - 1],
        children: [new d.TextRun({ text, bold: true, size: 28, font: FONT })],
        alignment: d.AlignmentType.LEFT,
        indent: { firstLine: CM(1.25) },
        pageBreakBefore: level === 1 && ctx.sectionNewPage,
        keepNext: true,
        spacing: { line: 360, before: level === 1 ? 0 : 240, after: 120 },
      }));
    } else if (/^\s*- /.test(line)) {
      const lvl = line.match(/^\s*/)[0].length >= 2 ? 1 : 0;
      out.push(new d.Paragraph({
        numbering: { reference: "dash", level: lvl },
        children: runs(line.replace(/^\s*- /, "")),
        alignment: d.AlignmentType.JUSTIFIED,
        spacing: { line: 360, before: 0, after: 0 },
      }));
    } else if (/^\d+\) /.test(line)) {
      const m = line.match(/^(\d+\))\s+(.*)$/);
      out.push(new d.Paragraph({
        children: runs(`${m[1]} ${m[2]}`),
        alignment: d.AlignmentType.JUSTIFIED,
        indent: { firstLine: CM(1.25) },
        spacing: { line: 360, before: 0, after: 0 },
      }));
    } else if (line.startsWith("@widths")) {
      pendingWidths = line.replace("@widths", "").split(",").map((x) => parseFloat(x));
    } else if (line.startsWith("@grid")) {
      const rows = [];
      i++;
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        const cells = lines[i].trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
        if (!cells.every((c) => /^:?-+:?$/.test(c))) rows.push(cells);
        i++;
      }
      out.push(tableFrom(rows, pendingWidths, 0));
      out.push(new d.Paragraph({ children: [], spacing: { line: 240, after: 120 } }));
      pendingWidths = null;
      continue;
    } else if (line.startsWith("@table")) {
      ctx.tableNo += 1;
      out.push(capt(`Таблица ${ctx.tableNo} – ${line.replace(/^@table(#\S+)?/, "").trim()}`));
      const rows = [];
      i++;
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        const cells = lines[i].trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
        if (!cells.every((c) => /^:?-+:?$/.test(c))) rows.push(cells);
        i++;
      }
      out.push(tableFrom(rows, pendingWidths, ctx.tableNo));
      out.push(new d.Paragraph({ children: [], spacing: { line: 240, after: 120 } }));
      pendingWidths = null;
      continue;
    } else if (line.startsWith("@figure")) {
      const [p, caption, wcm] = line.replace(/^@figure(#\S+)?/, "").split("|").map((x) => x.trim());
      ctx.figNo += 1;
      const img = fs.readFileSync(path.resolve(ctx.dir, p));
      const [pw, ph] = pngSize(img);
      const wpx = (parseFloat(wcm || "16") / 2.54) * 96;
      out.push(new d.Paragraph({
        children: [new d.ImageRun({ type: "png", data: img, transformation: { width: Math.round(wpx), height: Math.round((wpx * ph) / pw) } })],
        alignment: d.AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 0 },
      }));
      out.push(body(`Рисунок ${ctx.figNo} – ${caption}`, { align: d.AlignmentType.CENTER, indent: { firstLine: 0 }, spacing: { after: 240 } }));
    } else if (line.startsWith("@code")) {
      i++;
      while (i < lines.length && !lines[i].startsWith("@endcode")) {
        out.push(new d.Paragraph({
          children: [new d.TextRun({ text: lines[i] || " ", font: "Courier New", size: 18 })],
          spacing: { line: 240, before: 0, after: 0 },
        }));
        i++;
      }
    } else if (line.startsWith("@toc")) {
      out.push(new d.TableOfContents("Содержание", { hyperlink: true, headingStyleRange: "1-3" }));
    } else if (line.startsWith("@pagebreak")) {
      out.push(new d.Paragraph({ children: [new d.PageBreak()] }));
    } else if (line.startsWith("@center")) {
      out.push(body(line.replace("@center", "").trim(), { align: d.AlignmentType.CENTER, indent: { firstLine: 0 } }));
    } else if (line.startsWith("@right")) {
      out.push(body(line.replace("@right", "").trim(), { align: d.AlignmentType.RIGHT, indent: { firstLine: 0 } }));
    } else if (line.startsWith("@left")) {
      out.push(body(line.replace("@left", "").trim(), { align: d.AlignmentType.LEFT, indent: { firstLine: 0 } }));
    } else {
      out.push(body(line));
    }
    i++;
  }
  return out;
}

function pngSize(buf) {
  return [buf.readUInt32BE(16), buf.readUInt32BE(20)];
}

function titleBlock(lines) {
  // строки титульного листа: {text, align, bold, size, before, after}
  return lines.map((l) => new d.Paragraph({
    children: (l.text || "").split("\\n").flatMap((t, k) => [
      ...(k ? [new d.TextRun({ break: 1 })] : []),
      new d.TextRun({ text: t, bold: l.bold, size: l.size || 28, font: FONT, allCaps: l.caps }),
    ]),
    alignment: { center: d.AlignmentType.CENTER, right: d.AlignmentType.RIGHT, left: d.AlignmentType.LEFT }[l.align || "center"],
    spacing: { line: l.line || 276, before: l.before || 0, after: l.after || 0 },
    indent: l.indentLeft ? { left: CM(l.indentLeft) } : undefined,
  }));
}

function build({ srcFile, outFile, title, sectionNewPage = true, footerFromSecond = true }) {
  const ctx = { dir: path.dirname(srcFile), tableNo: 0, figNo: 0, sectionNewPage };
  const content = parse(fs.readFileSync(srcFile, "utf8"), ctx);
  const doc = new d.Document({
    creator: "Шелманова Е. О.",
    title: title.docTitle,
    features: { updateFields: true }, // Word при открытии предложит обновить поля (оглавление)
    styles: {
      default: { document: { run: { font: FONT, size: 28 }, paragraph: { spacing: { line: 360 } } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: FONT, size: 28, bold: true, color: "000000" }, paragraph: { outlineLevel: 0, spacing: { line: 360 } } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: FONT, size: 28, bold: true, color: "000000" }, paragraph: { outlineLevel: 1, spacing: { line: 360 } } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { font: FONT, size: 28, bold: true, color: "000000" }, paragraph: { outlineLevel: 2, spacing: { line: 360 } } },
        { id: "StructNoToc", name: "Struct No TOC", basedOn: "Normal", run: { font: FONT, size: 28, bold: true } },
        { id: "TOC1", name: "toc 1", basedOn: "Normal", run: { font: FONT, size: 28 }, paragraph: { spacing: { line: 360 } } },
        { id: "TOC2", name: "toc 2", basedOn: "Normal", run: { font: FONT, size: 28 }, paragraph: { indent: { left: CM(0.75) }, spacing: { line: 360 } } },
        { id: "TOC3", name: "toc 3", basedOn: "Normal", run: { font: FONT, size: 28 }, paragraph: { indent: { left: CM(1.5) }, spacing: { line: 360 } } },
      ],
    },
    numbering: {
      config: [{
        reference: "dash",
        levels: [
          { level: 0, format: d.LevelFormat.BULLET, text: "–", alignment: d.AlignmentType.LEFT,
            style: { paragraph: { indent: { left: CM(1.25) + 360, hanging: 360 } }, run: { font: FONT } } },
          { level: 1, format: d.LevelFormat.BULLET, text: "•", alignment: d.AlignmentType.LEFT,
            style: { paragraph: { indent: { left: CM(2.25) + 360, hanging: 360 } }, run: { font: FONT } } },
        ],
      }],
    },
    sections: [{
      properties: {
        page: { size: { width: MM(210), height: MM(297) }, margin: { left: MM(30), right: MM(15), top: MM(20), bottom: MM(20) } },
        titlePage: footerFromSecond,
      },
      footers: {
        default: new d.Footer({ children: [new d.Paragraph({ alignment: d.AlignmentType.CENTER, children: [new d.TextRun({ children: [d.PageNumber.CURRENT], font: FONT, size: 28 })] })] }),
        first: new d.Footer({ children: [new d.Paragraph({ children: [] })] }),
      },
      children: [...titleBlock(title.lines), ...content],
    }],
  });
  return d.Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(outFile, buf); return ctx; });
}

module.exports = { build };

if (require.main === module) {
  const cfg = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
  build(cfg).then((ctx) => console.log(`ok: ${cfg.outFile}; таблиц ${ctx.tableNo}, рисунков ${ctx.figNo}`));
}
