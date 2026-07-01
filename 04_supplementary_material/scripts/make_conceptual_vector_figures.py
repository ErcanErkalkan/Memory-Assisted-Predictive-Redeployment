"""Generate deterministic manuscript-facing vector figures for PRR-NSGA-II."""
from __future__ import annotations
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SUPPLEMENTARY_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_FIG_DIR = PACKAGE_ROOT / "02_latex_source" / "figures"
SUPP_FIG_DIR = SUPPLEMENTARY_ROOT / "figures"

BLUE = "#1f4e79"
TEAL = "#0f7c7c"
ORANGE = "#c45119"
GRAY = "#5c6670"
LIGHT_BLUE = "#eaf2fb"
LIGHT_TEAL = "#e8f6f6"
LIGHT_ORANGE = "#fff0e8"
LIGHT_GRAY = "#f4f6f8"
DARK = "#222222"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _box(ax, xy, w, h, title, fc="white", ec=BLUE, lw=1.25, title_color=None, fontsize=8.5):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        fc=fc, ec=ec, lw=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.055, title, ha="center", va="top",
            weight="bold", color=title_color or ec, fontsize=fontsize, linespacing=1.05)
    return patch


def _arrow(ax, p1, p2, color=DARK, lw=1.15):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="->", mutation_scale=10,
        lw=lw, color=color, shrinkA=3, shrinkB=3,
        connectionstyle="arc3,rad=0.0",
    ))


def _pareto_curve(x, shift=0.0):
    return 0.78 - 0.52 * np.sqrt(np.clip(x - shift, 0, 1))


def dynamic_setting(output_pdf: Path) -> None:
    fig = plt.figure(figsize=(7.2, 3.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.965, "Dynamic multi-objective setting and post-change recovery",
            ha="center", va="top", fontsize=12, weight="bold", color=DARK)

    _box(ax, (0.035, 0.13), 0.42, 0.75, "State-dependent objective space",
         fc=LIGHT_BLUE, ec=BLUE, fontsize=9)
    plot_ax = fig.add_axes([0.075, 0.30, 0.35, 0.46])
    plot_ax.set_xlim(0, 1)
    plot_ax.set_ylim(0, 1)
    plot_ax.set_xlabel(r"$f_1(x,t)$")
    plot_ax.set_ylabel(r"$f_2(x,t)$")
    plot_ax.spines[["top", "right"]].set_visible(False)
    plot_ax.tick_params(labelbottom=False, labelleft=False, length=0)
    x = np.linspace(0.12, 0.90, 150)
    y0 = _pareto_curve(x, 0.00)
    y1 = _pareto_curve(x, 0.05) + 0.06
    y2 = _pareto_curve(x, -0.03) - 0.05
    plot_ax.plot(x, y0, color=BLUE, lw=1.6, label=r"$PF(t-1)$")
    plot_ax.plot(x, y1, color=TEAL, lw=1.6, ls="--", label=r"$PF(t)$")
    plot_ax.plot(x, y2, color=ORANGE, lw=1.3, ls=":", label=r"later $PF$")
    rng = np.random.default_rng(42)
    pts = np.column_stack([rng.uniform(0.20, 0.82, 28), rng.uniform(0.22, 0.76, 28)])
    plot_ax.scatter(pts[:, 0], pts[:, 1], s=12, facecolors="white", edgecolors=GRAY, linewidths=0.8,
                    label="candidates")
    plot_ax.annotate("change", xy=(0.55, y1[80]), xytext=(0.37, 0.92),
                     arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0), color=GRAY, fontsize=8)
    plot_ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.text(0.245, 0.20,
            r"DMOP: minimize $F(x,t)=(f_1(x,t),\ldots,f_m(x,t))$, $x\in\Omega(t)$",
            ha="center", va="center", fontsize=8, color=DARK)

    _box(ax, (0.525, 0.13), 0.44, 0.75, "Post-change recovery logic",
         fc=LIGHT_GRAY, ec=GRAY, fontsize=9)
    yline = 0.72
    ax.plot([0.575, 0.915], [yline, yline], color=DARK, lw=1.1)
    ax.scatter([0.575], [yline], s=42, color=BLUE, zorder=3)
    ax.scatter([0.705], [yline], marker="*", s=110, color=ORANGE, edgecolor=DARK, linewidth=0.6, zorder=3)
    ax.plot([0.705, 0.705], [0.62, 0.83], color=GRAY, ls="--", lw=1)
    ax.scatter([0.915], [yline], s=42, color=TEAL, zorder=3)
    ax.text(0.575, 0.765, r"state $t-1$", ha="center", fontsize=8)
    ax.text(0.705, 0.84, "change event", ha="center", fontsize=8, color=ORANGE)
    ax.text(0.915, 0.765, r"state $t$", ha="center", fontsize=8)
    ax.text(0.81, 0.645, "recovery window", ha="center", color=TEAL, fontsize=8)
    ax.plot([0.725, 0.895], [0.60, 0.60], color=TEAL, lw=1.2)
    ax.plot([0.725, 0.725], [0.58, 0.62], color=TEAL, lw=1.2)
    ax.plot([0.895, 0.895], [0.58, 0.62], color=TEAL, lw=1.2)

    bx_y = 0.31
    comps = [
        ("bounded\nmemory", LIGHT_TEAL, TEAL),
        ("prediction\nproposals", "#eef8ff", BLUE),
        ("current-state\nselection", LIGHT_ORANGE, ORANGE),
    ]
    xs = [0.565, 0.695, 0.825]
    for x0, (label, fc, ec) in zip(xs, comps):
        _box(ax, (x0, bx_y), 0.105, 0.18, label, fc=fc, ec=ec, title_color=ec, fontsize=8.5)
    _arrow(ax, (0.670, bx_y + 0.09), (0.695, bx_y + 0.09), color=GRAY)
    _arrow(ax, (0.800, bx_y + 0.09), (0.825, bx_y + 0.09), color=GRAY)
    ax.text(0.745, 0.23,
            "Prediction and memory propose candidates;\nselection evaluates all candidates\nunder $F(\\cdot,t)$.",
            ha="center", va="center", fontsize=7.5, color=DARK)

    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def selection_filter_detail(output_pdf: Path) -> None:
    fig = plt.figure(figsize=(7.2, 3.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.965, "Selection-filtered predictive redeployment mechanism",
            ha="center", va="top", fontsize=12, weight="bold", color=DARK)
    ax.text(0.5, 0.855,
            r"$P_t^{PRR}=\mathcal{S}^{t}_{N}(P_t^{elite}\cup P_t^{recent}\cup P_t^{trans}\cup P_t^{pred}\cup P_t^{div})$",
            ha="center", va="center", fontsize=10, color=DARK)

    _box(ax, (0.035, 0.36), 0.17, 0.48, "Candidate sources", fc=LIGHT_GRAY, ec=GRAY,
         title_color=GRAY, fontsize=8.2)
    source_y = [0.74, 0.65, 0.56, 0.47]
    source_labels = ["elite survivors", "memory samples", "predicted candidates", "diversity immigrants"]
    source_colors = [BLUE, TEAL, ORANGE, GRAY]
    for y, lab, col in zip(source_y, source_labels, source_colors):
        ax.add_patch(Rectangle((0.058, y - 0.025), 0.024, 0.024, fc=col, ec=col, lw=0.8))
        ax.text(0.095, y - 0.012, lab, ha="left", va="center", fontsize=8, color=DARK)

    _box(ax, (0.285, 0.42), 0.16, 0.34, r"Merged pool $U_t$", fc="white", ec=BLUE,
         title_color=BLUE, fontsize=8.4)
    rng = np.random.default_rng(7)
    for col in source_colors:
        cx = 0.315 + 0.095 * rng.random(10)
        cy = 0.475 + 0.20 * rng.random(10)
        ax.scatter(cx, cy, s=14, color=col, alpha=0.9, edgecolor="white", linewidth=0.3)

    _box(ax, (0.530, 0.42), 0.17, 0.34, "NSGA-II\nenvironmental\nselection", fc=LIGHT_ORANGE,
         ec=ORANGE, title_color=ORANGE, fontsize=8.4)
    ax.text(0.615, 0.555, "rank\n+\ncrowding", ha="center", va="center", fontsize=9, color=DARK)

    _box(ax, (0.770, 0.42), 0.185, 0.34, "Redeployed\npopulation $P_t$", fc=LIGHT_BLUE,
         ec=BLUE, title_color=BLUE, fontsize=8.4)
    coords = [(0.810, 0.61, BLUE), (0.845, 0.64, TEAL), (0.875, 0.56, ORANGE),
              (0.915, 0.63, GRAY), (0.830, 0.51, BLUE), (0.895, 0.50, TEAL)]
    for xx, yy, col in coords:
        ax.scatter([xx], [yy], s=22, color=col, edgecolor="white", linewidth=0.5, zorder=4)

    for y in source_y:
        _arrow(ax, (0.205, y - 0.01), (0.285, 0.59), color=GRAY, lw=0.9)
    _arrow(ax, (0.445, 0.59), (0.530, 0.59), color=DARK)
    _arrow(ax, (0.700, 0.59), (0.770, 0.59), color=DARK)

    _box(ax, (0.055, 0.09), 0.40, 0.18, "Direct insertion risk", fc="#fff7f4", ec=ORANGE,
         title_color=ORANGE, fontsize=8.6)
    ax.text(0.255, 0.16, "fixed predicted slots can admit\nlow-priority predicted candidates",
            ha="center", va="center", fontsize=8, color=DARK)
    _box(ax, (0.545, 0.09), 0.40, 0.18, "Filtered PRR admission", fc=LIGHT_TEAL, ec=TEAL,
         title_color=TEAL, fontsize=8.6)
    ax.text(0.745, 0.16, "predicted candidates enter only if\nthey survive environmental selection",
            ha="center", va="center", fontsize=8, color=DARK)
    _arrow(ax, (0.455, 0.18), (0.545, 0.18), color=GRAY)

    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    MANUSCRIPT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    SUPP_FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        MANUSCRIPT_FIG_DIR / "prr_dynamic_setting.pdf",
        MANUSCRIPT_FIG_DIR / "prr_selection_filter_detail.pdf",
    ]
    dynamic_setting(outputs[0])
    selection_filter_detail(outputs[1])
    for pdf_path in outputs:
        for path in [pdf_path, pdf_path.with_suffix(".png")]:
            shutil.copy2(path, SUPP_FIG_DIR / path.name)


if __name__ == "__main__":
    main()
