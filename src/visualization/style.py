"""
Estilo visual unico do projeto -- paleta validada (CVD-safe) aplicada de
forma consistente em vez da mistura de hex ad-hoc (#16a085, #c0392b,
#008080...) e colormaps default (RdYlGn, "Blues", "Set2") que cada script
tinha antes. Ver src/visualization/eda_plots.py, explain.py e
business_questions.py.
"""

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# Ordem fixa -- nunca ciclada. Cores categoricas quando ha identidade
# (ex.: os 4 clusters da pergunta 3).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
BLUE = CATEGORICAL[0]

# Severidade ordenada (bom -> critico) -- usada quando as categorias sao
# niveis de risco/status, nao identidades arbitrarias (ex.: pergunta 4).
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

_SEQUENTIAL_BLUE_STEPS = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]
_DIVERGING_STEPS = ["#2a78d6", "#f0efec", "#e34948"]  # negativo -> neutro -> positivo

SEQUENTIAL_BLUES = LinearSegmentedColormap.from_list("brand_sequential_blues", _SEQUENTIAL_BLUE_STEPS)
DIVERGING_BLUE_RED = LinearSegmentedColormap.from_list("brand_diverging", _DIVERGING_STEPS)

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def apply_style() -> None:
    """Chama uma vez por processo, antes de gerar qualquer figura."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10.5,
        "figure.dpi": 100,
    })
