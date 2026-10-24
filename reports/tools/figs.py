import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "Times New Roman"
OUT = sys.argv[1]


def box(ax, x, y, w, h, text, fc="#ffffff", lw=1.2, fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="black", lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None, style="-|>", ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle=style, lw=1.1, color="black", linestyle=ls))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, ha="center", va="bottom", fontsize=9, style="italic")


# --- Рисунок 1: архитектура (два контура) ---
fig, ax = plt.subplots(figsize=(11, 6.6))
ax.set_xlim(0, 11); ax.set_ylim(0, 6.6); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.1, 3.45), 10.8, 3.05, boxstyle="round,pad=0.02", fc="#f2f2f2", ec="#777777", lw=0.8, ls="--"))
ax.text(0.25, 6.3, "Диагностический контур (новый, этапы A–E)", fontsize=11, weight="bold", va="center")
ax.add_patch(FancyBboxPatch((0.1, 0.1), 10.8, 3.15, boxstyle="round,pad=0.02", fc="#f9f9f9", ec="#777777", lw=0.8, ls="--"))
ax.text(0.25, 3.05, "Ассистент 6-го семестра (сохранён) + верификатор ответа", fontsize=11, weight="bold", va="center")
# верхний ряд
box(ax, 0.3, 4.9, 1.6, 0.9, "Телеметрия\n(NPPAD,\nтолько чтение)")
box(ax, 2.3, 4.9, 1.5, 0.9, "Информационный\nбарьер\n(6 каналов)")
box(ax, 4.2, 4.9, 1.5, 0.9, "Генератор\nгипотез\n(top-3)")
box(ax, 6.1, 4.9, 1.7, 0.9, "Верификатор\nгипотез\n(82 канала)")
box(ax, 8.2, 4.9, 1.5, 0.9, "Новизна и\nправило\nвыбора")
box(ax, 8.2, 3.7, 1.5, 0.7, "База событий\nETHZ (аналоги)", fc="#eeeeee")
box(ax, 0.3, 3.7, 1.6, 0.7, "Память о\nсобытии", fc="#eeeeee")
arrow(ax, 1.9, 5.35, 2.3, 5.35); arrow(ax, 3.8, 5.35, 4.2, 5.35); arrow(ax, 5.7, 5.35, 6.1, 5.35); arrow(ax, 7.8, 5.35, 8.2, 5.35)
arrow(ax, 1.1, 4.9, 1.1, 4.4)
# память о событии и все каналы окна → верификатор
ax.plot([1.9, 6.6], [4.05, 4.05], color="black", lw=1.1); arrow(ax, 6.6, 4.05, 6.6, 4.9)
ax.text(4.2, 4.1, "все наблюдаемые каналы + начало события", fontsize=9, style="italic", ha="center", va="bottom")
# аналоги ETHZ → верификатор
ax.plot([8.2, 7.4], [4.05, 4.05], color="black", lw=1.1); arrow(ax, 7.4, 4.05, 7.4, 4.9)
# нижний ряд
box(ax, 0.3, 1.6, 1.6, 0.9, "Запрос\nоператора")
box(ax, 2.3, 1.6, 1.5, 0.9, "RouterAgent")
box(ax, 4.2, 1.6, 1.8, 0.9, "Retriever |\nCommand |\nAnalyst")
box(ax, 6.4, 1.6, 1.9, 0.9, "Верификатор\nответа (правила,\nполитика)", fc="#e6e6e6", bold=True)
box(ax, 8.7, 1.6, 1.9, 0.9, "ALLOW /\nREQUIRE_HUMAN /\nBLOCK")
box(ax, 4.2, 0.35, 1.8, 0.75, "База знаний\n(pgvector / BM25)", fc="#eeeeee")
box(ax, 6.4, 0.35, 1.9, 0.75, "Langfuse\n(трассы)", fc="#eeeeee")
arrow(ax, 1.9, 2.05, 2.3, 2.05); arrow(ax, 3.8, 2.05, 4.2, 2.05); arrow(ax, 6.0, 2.05, 6.4, 2.05); arrow(ax, 8.3, 2.05, 8.7, 2.05)
arrow(ax, 5.1, 1.1, 5.1, 1.6); arrow(ax, 7.35, 1.6, 7.35, 1.1)
ax.plot([9.7, 10.5, 10.5, 5.6], [5.35, 5.35, 3.35, 3.35], color="black", lw=1.1); arrow(ax, 5.6, 3.35, 5.6, 2.5)
ax.text(8.1, 3.4, "DiagnosticReport → AnalystAgent.explain_diagnosis", fontsize=9, style="italic", ha="center", va="bottom")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_architecture.png", dpi=200); plt.close(fig)

# --- Рисунок 2: диагностический конвейер и исходы ---
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")
steps = [("Окно 120 с\n+ память\nо событии", 0.2), ("Проверка\nотклонения", 1.75), ("Генератор:\n3 гипотезы с\nуверенностью", 3.3),
         ("Проверка\nкаждой гипотезы\nпо всем каналам", 4.95), ("Проверка\nновизны", 6.75), ("Правило\nвыбора", 8.3)]
for t, x in steps:
    box(ax, x, 2.9, 1.4 if x != 4.95 else 1.6, 1.1, t, fs=10)
for (t1, x1), (t2, x2) in zip(steps, steps[1:]):
    arrow(ax, x1 + (1.4 if x1 != 4.95 else 1.6), 3.45, x2, 3.45)
outs = [("NO_ANOMALY\n«отклонений нет»", 0.3, "ALLOW"), ("NOVEL\n«незнакомая ситуация»", 2.45, "REQUIRE_HUMAN"),
        ("UNCONFIRMED\n«не подтверждено»", 4.6, "REQUIRE_HUMAN"), ("AMBIGUOUS\n«неоднозначно»", 6.75, "REQUIRE_HUMAN"), ("IDENTIFIED\n«вероятна ситуация X»", 8.9, "ALLOW")]
for t, x, dec in outs:
    box(ax, x, 0.95, 1.9, 0.9, t, fc="#eeeeee", fs=10)
    ax.text(x + 0.95, 0.6, dec, ha="center", fontsize=9, style="italic")
    ax.annotate("", xy=(x + 0.95, 1.85), xytext=(9.0, 2.9), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#444444"))
ax.text(5.5, 0.15, "Решение верификатора ответа по умолчанию для каждого исхода (пакет aes-core-1.1.0)", ha="center", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_pipeline.png", dpi=200); plt.close(fig)
print("ok")
