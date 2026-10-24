const path = require("path");
const { build } = require("./gost");
const title = require("./title");
const root = path.resolve(__dirname, "..");
build({
  srcFile: path.join(root, "src/stage-A-checkpoint.md"),
  outFile: path.join(root, "Этап A — описание модели данных и методов.docx"),
  title: {
    docTitle: "Описание модели данных и методов (контрольная точка этапа A)",
    lines: title({
      kind: "ОПИСАНИЕ МОДЕЛИ ДАННЫХ И МЕТОДОВ",
      subtitle: "Контрольная точка этапа A производственной практики (7 семестр)",
      theme: "Интерактивный ассистент для поддержки принятия решений операторами АЭС на основе формирования и верификации гипотез о состоянии системы",
    }),
  },
}).then((c) => console.log("ok", c.tableNo, "таблиц,", c.figNo, "рисунков"));
