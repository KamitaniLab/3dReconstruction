"""Shared metadata used across analysis scripts."""

from __future__ import annotations

from matplotlib.colors import to_hex, to_rgb
import numpy as np


PUBLIC_SUBJECTS = ["S1", "S2", "S3", "S4", "S5"]
SUBJECT_COLORS = ["#D7DCE5", "#C3CBD8", "#AEB8C9", "#98A4B9", "#7F8CA3"]
WHOLE_VISUAL_ROI = "WholeVC"
SUBROIS = ["EarlyVC", "MTVC", "DorsalVC", "VentralVC"]
ALL_ROIS = [WHOLE_VISUAL_ROI, *SUBROIS]
CONTOUR_SLANT_ROIS = [*SUBROIS, WHOLE_VISUAL_ROI]
QUALITATIVE_EXAMPLE_SUBJECT = "S4"
ROI_BASE_COLORS = {
    "EarlyVC": "#75C1DF",
    "MTVC": "#907FBB",
    "DorsalVC": "#E49F8C",
    "VentralVC": "#A4D28A",
}
INSAMPLE_CATEGORIES = {
    "02691156",
    "02828884",
    "02933112",
    "02958343",
    "03001627",
    "03211117",
    "03636649",
    "03691459",
    "04090263",
    "04256520",
}
NATURAL_OBJECT_CATEGORY_INDEX = {
    "02691156": [1, 2, 3, 4],
    "02818832": [5, 6, 7, 8],
    "02828884": [9, 10, 11, 12],
    "02876657": [13, 14, 15, 16],
    "02933112": [17, 18, 19, 20],
    "02954340": [21, 22, 23, 24],
    "02958343": [25, 26, 27, 28],
    "03001627": [29, 30, 31, 32],
    "03211117": [33, 34, 35, 36],
    "03513137": [37, 38, 39, 40],
    "03624134": [41, 42, 43, 44],
    "03636649": [45, 46, 47, 48],
    "03691459": [49, 50, 51, 52],
    "03790512": [53, 54, 55, 56],
    "03928116": [57, 58, 59, 60],
    "03948459": [61, 62, 63, 64],
    "04090263": [65, 66, 67, 68],
    "04256520": [69, 70, 71, 72],
    "body": [73, 74, 75, 76],
    "human": [77, 78, 79, 80],
}

CONTOUR_SLANT_TRIALS_EXCLUDING_LARGEST_SLANT = {
    "horizontal_thin_bar": [
        {"stimulus_index": 2, "exp_true_deg": -34.72889379334044, "nominal_deg": -45},
        {"stimulus_index": 1, "exp_true_deg": -21.345388855923307, "nominal_deg": -30},
        {"stimulus_index": 0, "exp_true_deg": -10.17061305412802, "nominal_deg": -15},
        {"stimulus_index": 4, "exp_true_deg": 10.167495164765365, "nominal_deg": 15},
        {"stimulus_index": 5, "exp_true_deg": 21.332209251496867, "nominal_deg": 30},
        {"stimulus_index": 6, "exp_true_deg": 34.69700079757317, "nominal_deg": 45},
    ],
    "horizontal_thick_bar": [
        {"stimulus_index": 10, "exp_true_deg": -37.1224949004072, "nominal_deg": -45},
        {"stimulus_index": 9, "exp_true_deg": -22.75370073577296, "nominal_deg": -30},
        {"stimulus_index": 8, "exp_true_deg": -10.81565753323688, "nominal_deg": -15},
        {"stimulus_index": 12, "exp_true_deg": 10.812245483773168, "nominal_deg": 15},
        {"stimulus_index": 13, "exp_true_deg": 22.73933637083416, "nominal_deg": 30},
        {"stimulus_index": 14, "exp_true_deg": 37.08823256229996, "nominal_deg": 45},
    ],
    "horizontal_cylinder": [
        {"stimulus_index": 18, "exp_true_deg": -39.35715961370027, "nominal_deg": -45},
        {"stimulus_index": 17, "exp_true_deg": -24.693701932247155, "nominal_deg": -30},
        {"stimulus_index": 16, "exp_true_deg": -11.856237008943994, "nominal_deg": -15},
        {"stimulus_index": 20, "exp_true_deg": 11.852300524705662, "nominal_deg": 15},
        {"stimulus_index": 21, "exp_true_deg": 24.677696193412643, "nominal_deg": 30},
        {"stimulus_index": 22, "exp_true_deg": 39.320820102791686, "nominal_deg": 45},
    ],
    "vertical_thin_bar": [
        {"stimulus_index": 10, "exp_true_deg": -34.72886589726292, "nominal_deg": -45},
        {"stimulus_index": 9, "exp_true_deg": -21.34537459865315, "nominal_deg": -30},
        {"stimulus_index": 8, "exp_true_deg": -10.17061160023948, "nominal_deg": -15},
        {"stimulus_index": 12, "exp_true_deg": 10.16749807123461, "nominal_deg": 15},
        {"stimulus_index": 13, "exp_true_deg": 21.33222634590211, "nominal_deg": 30},
        {"stimulus_index": 14, "exp_true_deg": 34.69703264657337, "nominal_deg": 45},
    ],
}

CONTOUR_SLANT_TRIALS_INCLUDING_LARGEST_SLANT = {
    "horizontal_thin_bar": [
        {"stimulus_index": 3, "exp_true_deg": -51.74025661890929, "nominal_deg": -60},
        *CONTOUR_SLANT_TRIALS_EXCLUDING_LARGEST_SLANT["horizontal_thin_bar"],
        {"stimulus_index": 7, "exp_true_deg": 51.68127803289425, "nominal_deg": 60},
    ],
    "horizontal_thick_bar": [
        {"stimulus_index": 11, "exp_true_deg": -55.27544378576476, "nominal_deg": -60},
        *CONTOUR_SLANT_TRIALS_EXCLUDING_LARGEST_SLANT["horizontal_thick_bar"],
        {"stimulus_index": 15, "exp_true_deg": 55.21485452496986, "nominal_deg": 60},
    ],
    "horizontal_cylinder": [
        {"stimulus_index": 19, "exp_true_deg": -56.42495727760556, "nominal_deg": -60},
        *CONTOUR_SLANT_TRIALS_EXCLUDING_LARGEST_SLANT["horizontal_cylinder"],
        {"stimulus_index": 23, "exp_true_deg": 56.36214413200578, "nominal_deg": 60},
    ],
    "vertical_thin_bar": [
        {"stimulus_index": 11, "exp_true_deg": -51.74018856870874, "nominal_deg": -60},
        *CONTOUR_SLANT_TRIALS_EXCLUDING_LARGEST_SLANT["vertical_thin_bar"],
        {"stimulus_index": 15, "exp_true_deg": 51.68134179383535, "nominal_deg": 60},
    ],
}


def make_color_gradient(
    base_color: str,
    n_shades: int = 5,
    light_mix_max: float = 0.55,
    light_mix_min: float = 0.05,
) -> list[str]:
    """Return lighter-to-darker shades for one base color."""
    base = np.array(to_rgb(base_color))
    white = np.ones(3)
    mix_values = np.linspace(light_mix_max, light_mix_min, n_shades)
    shades = [(1 - mix) * base + mix * white for mix in mix_values]
    return [to_hex(color) for color in shades]


def roi_subject_colors(subject_colors: list[str] | None = None) -> dict[str, list[str]]:
    """Return per-ROI subject palettes."""
    subject_colors = subject_colors or SUBJECT_COLORS
    colors = {
        roi: make_color_gradient(color, n_shades=len(subject_colors))
        for roi, color in ROI_BASE_COLORS.items()
    }
    colors["WholeVC"] = subject_colors
    return colors


def natural_object_split_label(stimulus_index: int) -> str:
    """Return 'insample' or 'outsample' for a 0-based natural-object stimulus index."""
    for category, indices in NATURAL_OBJECT_CATEGORY_INDEX.items():
        if stimulus_index + 1 in indices:
            return "insample" if category in INSAMPLE_CATEGORIES else "outsample"
    return "unknown"


def public_stimulus_label(label: str) -> str:
    """Return the public stimulus label used across evaluation and visualization."""
    for suffix in (".npy", ".points.ply", ".ply"):
        label = label.removesuffix(suffix)
    return label.removeprefix("rds_")
