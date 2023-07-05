"""
Constant variables, paths, and functions shared across notebooks and scripts
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from itertools import combinations
from matplotlib import rcParams
from utilz import mapcat
from pymer4 import result_to_table, load_model, save_model, Lm, Lmer, Lm2
from gspread_pandas import Spread
import PIL

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Avenir"]
import matplotlib.pyplot as plt
import seaborn as sns
from pymer4.utils import upper
from toolz import curry

###################################################
# GLOBALS
###################################################
BASE_DIR = Path("../")
DATA_DIR = BASE_DIR / "data"
ANNOT_DIR = DATA_DIR / "annotations"
ANALYSIS_DIR = BASE_DIR / "analysis"
FIG_DIR = BASE_DIR / "figures"
OUTPUT_GSHEET = "Social_Memory_Behav_Tables"

SCAN_NAME = [
    "SID000120",
    "SID000791",
    "SID000134",
    "SID000804",
    "SID000435",
    "SID000473",
    "SID000813",
    "SID000138",
    "SID000815",
    "SID000819",
    "SID000052",
    "SID000820",
    "SID000499",
    "SID000584",
    "SID000824",
    "SID000826",
    "SID000829",
    "SID000576",
    "SID000836",
    "SID000839",
    "SID000841",
    "SID000857",
    "SID000860",
    "SID000703",
    "SID000861",
    "SID000862",
    "SID000863",
    "SID000864",
    "SID000868",
    "SID000872",
    "SID000875",
    "SID000278",
    "SID000894",
    "SID001018",
    "SID000973",
    "SID001141",
]
SUB_NAME = [
    "s01",
    "s02",
    "s03",
    "s04",
    "s05",
    "s06",
    "s07",
    "s08",
    "s09",
    "s10",
    "s11",
    "s12",
    "s13",
    "s14",
    "s15",
    "s16",
    "s17",
    "s18",
    "s19",
    "s20",
    "s21",
    "s22",
    "s23",
    "s24",
    "s25",
    "s26",
    "s27",
    "s28",
    "s29",
    "s30",
    "s31",
    "s32",
    "s33",
    "s34",
    "s35",
    "s36",
]

NAME_TO_SCAN = dict(zip(SUB_NAME, SCAN_NAME))
SCAN_TO_NAME = dict(zip(SCAN_NAME, SUB_NAME))
# 11 main characters
CHAR_LIST = [
    "BuddyGarrity",
    "CoachTaylor",
    "JasonStreet",
    "JulieTaylor",
    "LandryClarke",
    "LylaGarrity",
    "MattSaracen",
    "SmashWilliams",
    "TamiTaylor",
    "TimRiggins",
    "TyraCollette",
]
CHAR_NAMES = [
    "Buddy Garrity",
    "Coach Taylor",
    "Jason Street",
    "Julie Taylor",
    "Landry Clarke",
    "Lyla Garrity",
    "Matt Saracen",
    "Smash Williams",
    "Tami Taylor",
    "Tim Riggins",
    "Tyra Collette",
]
CHAR_NAMES_LB = [
    "Buddy\nGarrity",
    "Coach\nTaylor",
    "Jason\nStreet",
    "Julie\nTaylor",
    "Landry\nClarke",
    "Lyla\nGarrity",
    "Matt\nSaracen",
    "Smash\nWilliams",
    "Tami\nTaylor",
    "Tim\nRiggins",
    "Tyra\nCollette",
]
CUE_ORDER = [
    "CasuallyHangout",
    "FightArgue",
    "Flirt",
    "Gossip",
    "Romance",
    "SupportAdviseComfort",
    "Attractive",
    "Ethical",
    "Hardworking",
    "Influential",
    "Outgoing",
    "Diner",
    "FootballStadium",
    "Home",
    "Hospital",
    "School",
    "BuddyGarrity",
    "CoachTaylor",
    "JasonStreet",
    "JulieTaylor",
    "LandryClarke",
    "LylaGarrity",
    "MattSaracen",
    "SmashWilliams",
    "TamiTaylor",
    "TimRiggins",
    "TyraCollette",
]
CATEGORY_ORDER = ["Action", "Trait", "Place", "Character"]
CATEGORY_LABELS = mapcat(
    None, [[label] * count for label, count in zip(CATEGORY_ORDER, [6, 5, 5, 11])]
)
CUE_MAP = dict(zip(CUE_ORDER, CATEGORY_LABELS))

# 55 pairs of character combinations
CHAR_PAIRS = list(combinations(CHAR_LIST, 2))

# Lengths of each episode in TRs (2s)
EP_LENS = [1364, 1317, 1294, 1287]

# SID000973 feel asleep while watching
# SID000820 fell asleep during episode 2
# SID000875 was high while watching all episodes
ALL_EXCLUSIONS = ["SID000973", "SID000820", "SID000875"]

# SID000836 technical error recording all cues
CUE_EXCLUSIONS = ALL_EXCLUSIONS + ["SID000836"]

# Could not transcribe reliably due to technical errors
CONTENT_EXCLUSIONS = ALL_EXCLUSIONS + ["SID000120", "SID000813", "SID000864"]

POS_MAP = {
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary",
    "CCONJ": "coordinating_conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper_noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating_conjunction",
    "SPACE": "space",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other",
}

###################################################
# FUNCTIONS
###################################################
def chars2initials(characters=None):
    """Convert list of characters to lowercase initials"""
    characters = CHAR_LIST if characters is None else characters
    get_initials = lambda s: "".join([c for c in s if c.isupper()]).lower()
    return [get_initials(c) for c in CHAR_LIST]


def thresh_mat(mat, thresh="default", as_array=False):
    """
    Given a matrix, threshold it at some value returning a sparser version of mat
    as well a binarized adjacency matrix.
    Used by:
        - plot_character_graph
    """
    if thresh == "default":
        thresh = 1 / (mat.shape[0] - 1)
    # Convert to pd dataframe cause masking is easier
    mat = pd.DataFrame(mat)
    adj = np.abs(mat) >= thresh
    mat_thresh = mat[adj]
    adj = adj.astype(int)
    mat_thresh = mat_thresh.fillna(0)
    if as_array:
        mat_thresh = mat_thresh.values
        adj = adj.values
    return mat_thresh, adj


def fill_diagonal(mat, val):
    """Non in-place fill-diagonal that supports dataframes"""
    as_dataframe = False
    if isinstance(mat, pd.DataFrame):
        index = list(mat.index)
        cols = list(mat.columns)
        mat = mat.to_numpy()
        as_dataframe = True
    out = np.copy(mat)
    np.fill_diagonal(out, val)
    if as_dataframe:
        return pd.DataFrame(out, index=index, columns=cols)
    else:
        return out


def comm_to_df(comm):
    """Covert output of an nx.communcability_exp call to a dataframe"""

    df_comm = pd.DataFrame()
    for k, v in comm.items():
        df_comm = df_comm.append(pd.DataFrame(v, index=[k]))
    return df_comm


def communicability_weighted(adjacency):
    """
    Computes the communicability of pairs of nodes in `adjacency`

    BSD 3-Clause License

    Copyright (c) 2018, Network Neuroscience Lab
    All rights reserved.

    Parameters
    ----------
    adjacency : (N, N) array_like
        Weighted, direct/undirected connection weight/length array

    Returns
    -------
    cmc : (N, N) numpy.ndarray
        Symmetric array representing communicability of nodes {i, j}

    References
    ----------
    Crofts, J. J., & Higham, D. J. (2009). A weighted communicability measure
    applied to complex brain networks. Journal of the Royal Society Interface,
    6(33), 411-414.

    """

    from scipy.linalg import expm

    # negative square root of nodal degrees
    row_sum = adjacency.sum(1)
    neg_sqrt = np.power(row_sum, -0.5)
    square_sqrt = np.diag(neg_sqrt)

    # normalize input matrix
    for_expm = square_sqrt @ adjacency @ square_sqrt

    # calculate matrix exponential of normalized matrix
    cmc = expm(for_expm)
    cmc[np.diag_indices_from(cmc)] = 0

    return cmc


def distmat_to_df(mat, val_name="Dist"):
    dists = upper(mat.to_numpy())
    return pd.DataFrame({"Pair": CHAR_PAIRS, val_name: dists})


@curry
def format_and_save2sheet(
    df: pd.DataFrame,
    sheet: str = "",
    format: bool = True,
    iv_name: str = "Predictor",
    comparison_name: str = "b",
    ci_name: str = "ci",
    fetch_name_col: str = "index",
) -> pd.DataFrame:

    if format:
        df = result_to_table(
            df,
            iv_name=iv_name,
            comparison_name=comparison_name,
            fetch_name_col=fetch_name_col,
            ci_name=ci_name,
        )
    try:
        Spread(OUTPUT_GSHEET).df_to_sheet(df, sheet=sheet, replace=True, index=False)
        print("Successfully wrote to g-sheet")
    except Exception as e:
        print(e)
    return df


@curry
def fit_lm(data: pd.DataFrame, formula: str, model_file: Path, **kwargs):
    """Fit a basic lm model or load results from disk"""

    if model_file.exists():
        print("Loading existing model from file")
        model = load_model(str(model_file))
    else:
        model = Lm(formula, data=data)
        model.fit(summarize=False, **kwargs)
        save_model(model, str(model_file))

    return model


@curry
def fit_lmer(data: pd.DataFrame, formula: str, model_file: Path, factors: None | dict):
    """Fit simple lmer model or load from disk. No contrasts computed"""

    if model_file.exists():
        print("Loading existing model from file")
        model = load_model(str(model_file))
    else:
        model = Lmer(formula, data=data)
        model.fit(summarize=False, factors=factors)
        save_model(model, str(model_file))

    return model


@curry
def fit_lm2(data: pd.DataFrame, formula: str, group: str, model_file: Path, **kwargs):
    """Fit simple lm2 model or load from disk. No contrasts computed"""

    if model_file.exists():
        print("Loading existing model from file")
        model = load_model(str(model_file))
    else:
        model = Lm2(formula, group=group, data=data)
        model.fit(summarize=False, **kwargs)
        save_model(model, str(model_file))

    return model


###################################################
# PLOTTING
###################################################
@curry
def annotate_axis(
    ax,
    xstart,
    y,
    texts,
    thickness=1.5,
    xend=None,
    color="k",
    fontsize=18,
    offset=0.01,
    xycoords="data",
    despine=True,
):
    """
    Draw comparison lines and text/stars on a given matplotlib axis.

    Args:
        ax (matplotlib.axes): axes to annotate
        xstart (list): list of starting x-coords
        xend (list): list of ending x-coords
        y (list): list of y-coords that determine comparison bar height
        texts (list): list of texts to add at yannot
        width (int): thickness of all comparison bars
        color (str): color of all comparison bars and text/stars
        fontsize (int): size of text/stars
        offset (float): how much higher than y, text/stars should appear

    Returns:
        ax: annotated matplotlib axis
    """

    if not isinstance(xstart, list):
        xstart = [xstart]
    if xend is not None and not isinstance(xend, list):
        xend = [xend]
    if not isinstance(y, list):
        y = [y]
    if not isinstance(texts, list):
        texts = [texts]

    if xend is not None:
        assert (
            len(xstart) == len(xend) == len(y) == len(texts)
        ), "All coordinates and annotations need to have the same number of elements"

        for x1, x2, y, t in zip(xstart, xend, y, texts):
            _ = ax.annotate(
                "",
                xy=(x1, y),
                xycoords=xycoords,
                xytext=(x2, y),
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="-",
                    ec=color,
                    connectionstyle="arc3,rad=0",
                    linewidth=thickness,
                ),
            )
            if np.abs(x1) == np.abs(x2):
                midpoint = 0
            else:
                midpoint = np.mean([x1, x2])
            _ = ax.text(
                midpoint,
                y + offset,
                t,
                fontsize=fontsize,
                horizontalalignment="center",
                verticalalignment="center",
            )
    else:
        assert (
            len(xstart) == len(y) == len(texts)
        ), "All coordinates and annotations need to have the same number of elements"

        for x, y, t in zip(xstart, y, texts):

            _ = ax.text(
                x,
                y + offset,
                t,
                fontsize=fontsize,
                horizontalalignment="center",
                verticalalignment="center",
            )

    if despine:
        sns.despine()
    return ax


def plot_mat(
    mat,
    fig_size=(8, 6),
    annot=False,
    ax=None,
    title=None,
    xlabel=None,
    ylabel=None,
    xticklabels=None,
    yticklabels=None,
    cbar=True,
    title_fontsize=16,
    is_corr=True,
    standard_range=True,
    ignore_diagonal=False,
    save=False,
    **kwargs,
):
    """Plot a matrix nicely. If standard_range is true, vmin and vmax will be set to -1, 1 for correlation matrices and 0, 1 otherwise. Otherwise, will be set to the range of the data."""

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=fig_size)

    if standard_range:
        if is_corr:
            vmin = kwargs.pop("vmin", -1)
            vmax = kwargs.pop("vmax", 1)
            cmap = kwargs.pop("cmap", "RdBu_r")
            center = kwargs.pop("center", 0)
        else:
            vmin = kwargs.pop("vmin", 0)
            vmax = kwargs.pop("vmax", 1)
            cmap = kwargs.pop("cmap", "Reds")
            center = kwargs.pop("center", None)
    else:
        if isinstance(mat, np.ndarray):
            m = mat
        else:
            m = mat.values
        if ignore_diagonal:
            matmin = m[~np.eye(m.shape[0], dtype=bool)].min()
            matmax = m[~np.eye(m.shape[0], dtype=bool)].max()
        else:
            matmin = m.min()
            matmax = m.max()
        vmin = kwargs.pop("vmin", matmin)
        vmax = kwargs.pop("vmax", matmax)
        if is_corr:
            cmap = kwargs.pop("cmap", "RdBu_r")
            center = kwargs.pop("center", 0)
        else:
            cmap = kwargs.pop("cmap", "Reds")
            center = kwargs.pop("center", None)

    fmt = kwargs.pop("fmt", ".2f")
    ax = sns.heatmap(
        mat,
        ax=ax,
        annot=annot,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        cbar=cbar,
        center=center,
        fmt=fmt,
    )
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    if xlabel is not None:
        ax.set(xlabel=xlabel)
    if ylabel is not None:
        ax.set(ylabel=ylabel)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation=90)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels, rotation=0)

    if save:
        save_fig(save)
    return ax


def plot_graph(
    data,
    threshold=None,
    layout="circular",
    color_edges=False,
    alpha_edges=False,
    weight_edges=False,
    alpha_enhance=0.0,
    plot_diagonal=False,
    figsize=(12, 8),
    labels=None,
    ax=None,
    title=None,
    cbar=False,
    standard_range=True,
    graph_type="undirected",
    arrows=True,
    node_border_color="gray",
    connectionstyle="arc3",
    node_border_width=1.0,
    return_graph=False,
    edge_start_gap=0,
    edge_end_gap=0,
    random_seed=None,
    k=None,
    use_char_images=False,
    img_scale=0.025,
    pos=None,
    title_fontsize=18,
    fixed_img_size=False,
    plot_labels=False,
    **kwargs,
):
    """
    General function to plot a graph provided a 2d numpy array, pandas dataframe or networkX object. If standard_range=True, then colors will span -1, 1 if the data are signed or 0, 1 if not. Otherwise will use the min and max of the data.
    Most aesthetic settings are adjustable via keyword calls, e.g. "node_size" = 500. Properties include:
    vmin, vmax, node_color, node_size, arrow_size, font_size, edge_colors, edge_alphas, edge_weights, edge_weight_scale, arrow_style

    Args:
        data (nx, np.array, pd.DataFrame): networkx graph object or 2d numpy array or pandas dataframe
        threshold (bool): visualize only edges with absolute values greater than this threshold; default None, optionally pass 'default' to threshold by number of edges - 1
        layout (str) : type of networkx layout (e.g. circular, spring)
        color_edges (bool): whether to color edges based on their values; default True; see default value of edge_colors if False
        alpha_edges (bool): whether to adjust edge transparency based on values; default False; see default value of edge_alphas if False
        weight_edges (bool): whether to adjust edge thickness based on values; default False; see default value of edge_weights if False
        vmin (int): minumum value for edge colors (like sns.heatmap); defaults to -1 for signed data, and 0 for unsigned data
        vmax (int): maximum value for edge colors; default to 1 for both signed and unsigned data
        node_color (str): node color; default red
        node_size (int): size of nodes; default 500
        arrow_size (int): size of arrows between nodes; default 15
        font_size (int): node label font size; default 14
        edge_colors (str): edge colors; default 'black'
        edge_alphas (int): edge alphas; default 1.
        edge_weights (int): edge thickness; default 2.
        edge_weight_scale (int): multiplier if weight_edges is True; larger values increase thickness disparity between edges
        alpha_enhance (float): value between 0 and 1 to darken edges if alpha_edges =
        True
        plot_diagonal (bool): where to plot the diagonal of the matrix as self-loops
        Default False
        figsize (tup): figure size
        labels (dict): dictionary of node:label pairs; default character list
        ax (mpl axis): matplotlib axis; optional
        title (str): title; optional
        cbar (bool): whether to display color bar or not
        graph_type (str): 'directed' or 'undirected'; default 'directed'
        label_x_adjust (float): value to move node label right or left
        label_y_adjust (float): value to move node label up or down

    """

    import matplotlib as mpl
    import matplotlib.cbook
    from collections import Iterable
    import warnings

    warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

    # First sort out data type
    if isinstance(data, pd.DataFrame) or (data.__class__.__bases__[0] == pd.DataFrame):
        # Pandas df/design matrix
        mat = data.to_numpy()
        labels = dict(zip(range(len(data.index.values)), data.index.values))
    elif isinstance(data, nx.DiGraph) or isinstance(data, nx.Graph):
        # Networkx object
        mat = nx.to_numpy_array(data)
    else:
        # numpy array
        mat = data

    if not plot_diagonal:
        mat = fill_diagonal(mat, 0)
    if threshold is not None:
        print(f"Edges thresholded at +/- {threshold}")
        mat, _ = thresh_mat(mat, threshold, as_array=True)

    if graph_type == "directed":
        func = nx.DiGraph
    elif graph_type == "undirected":
        func = nx.Graph
    else:
        raise ValueError("graph_type must be 'directed' or 'undirected'")

    # Create graph object
    G = nx.from_numpy_array(mat, create_using=func())

    # Get range of values, e.g. all positive or positive and negative
    if (mat < 0).any():
        cmap = kwargs.pop("cmap", plt.cm.RdBu_r)
        edge_vmin = kwargs.pop("vmin", -1)
        edge_vmax = kwargs.pop("vmax", 1)
    else:
        cmap = kwargs.pop("cmap", plt.cm.Reds)
        edge_vmin = kwargs.pop("vmin", 0)
        edge_vmax = kwargs.pop("vmax", 1)

    if not standard_range:
        edge_vmin = kwargs.pop("vmin", mat[~np.eye(mat.shape[0], dtype=bool)].min())
        edge_vmax = kwargs.pop("vmax", mat[~np.eye(mat.shape[0], dtype=bool)].max())

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Set other attributes
    # Edge colors based on values
    if color_edges:
        edge_colors = []
        for i, edge_loc in enumerate(list(G.edges())):
            val = mat[edge_loc]
            edge_colors.append(val)
    else:
        edge_colors = kwargs.pop("edge_colors", "black")

    # Edge thickness based on values
    edge_weight_scale = kwargs.pop("edge_weight_scale", 15)
    if weight_edges:
        edge_weights = []
        for i, edge_loc in enumerate(list(G.edges())):
            val = np.abs(mat[edge_loc])
            edge_weights.append(2.0 + (val * edge_weight_scale))
    else:
        edge_weights = kwargs.pop("edge_weights", 2.0)

    # Edge transparency based on values
    if alpha_edges:
        edge_alphas = []
        _mat = (mat - mat.min()) / mat.ptp()
        for i, edge_loc in enumerate(list(G.edges())):
            # val = np.abs(mat[edge_loc])
            val = _mat[edge_loc]
            edge_alphas.append(max(0, min(val + alpha_enhance, 1)))
    else:
        edge_alphas = kwargs.pop("edge_alphas", 1.0)

    # Node labels
    if labels is None:
        labels = dict(zip(G.nodes, CHAR_LIST))
    G = nx.relabel_nodes(G, labels)

    # Get other settable properties
    node_color = kwargs.pop("node_color", "red")
    node_size = kwargs.pop("node_size", 500)
    arrow_size = kwargs.pop("arrow_size", 15)
    label_font_size = kwargs.pop("label_font_size", 14)
    label_y_adjust = kwargs.pop("label_y_adjust", 0)
    label_x_adjust = kwargs.pop("label_x_adjust", 0)
    default_arrow_style = "-|>" if graph_type == "directed" else "<|-|>"
    arrow_style = kwargs.pop("arrow_style", default_arrow_style)
    tight_layout = kwargs.pop("tight_layout", False)
    title_offset = kwargs.pop("title_offset", None)

    # Setup plot
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=figsize)

    if pos is None:
        func = getattr(nx.layout, layout + "_layout")
        if layout == "spring":
            pos = func(G, seed=random_seed, k=k)
        else:
            pos = func(G)

    # Draw edges
    edges = nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle=arrow_style,
        arrowsize=arrow_size,
        edge_color=edge_colors,
        edge_cmap=cmap,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        width=edge_weights,
        alpha=edge_alphas,
        node_size=node_size,
        min_source_margin=edge_start_gap,
        min_target_margin=edge_end_gap,
        arrows=arrows,
        connectionstyle=connectionstyle,
        ax=ax,
    )

    # Draw nodes
    if not use_char_images:
        _ = nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_color,
            node_size=node_size,
            edgecolors=node_border_color,
            linewidths=node_border_width,
            ax=ax,
        )
    else:
        # Load images
        images = dict(zip(CHAR_LIST, sorted((DATA_DIR / "char_images").glob("*.png"))))
        for char, imgfile in images.items():
            G.add_node(char, image=PIL.Image.open(imgfile))

        # Transform from data coordinates (scaled between xlim and ylim) to display
        # coordinates
        ax.set(ylim=(-1.1, 1.1), xlim=(-1.1, 1.1))
        tr_figure = ax.transData.transform
        f = ax.get_figure()
        # Transform from display to figure coordinates
        tr_axes = f.transFigure.inverted().transform
        # Select the size of the image (relative to the X axis)
        if fixed_img_size:
            icon_size = fixed_img_size * img_scale
        else:
            icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * img_scale
        icon_center = icon_size / 2.0
        # Add the respective image to each node
        for n in G.nodes:
            xf, yf = tr_figure(pos[n])
            xa, ya = tr_axes((xf, yf))
            # get overlapped axes and plot icon
            a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
            a.imshow(G.nodes[n]["image"])
            a.axis("off")

    if plot_labels:
        # Label nodes but first adjust position
        label_pos = {}
        for k, v in pos.items():
            x, y = v
            if layout == "circular":
                if x < 0:
                    new_x = x - label_x_adjust
                else:
                    new_x = x + label_x_adjust
                if y < 0:
                    new_y = y - label_y_adjust
                else:
                    new_y = y + label_y_adjust
                label_pos[k] = np.array([new_x, new_y])
            else:
                label_pos[k] = v + np.array([label_x_adjust, label_y_adjust])

        _ = nx.draw_networkx_labels(
            G,
            label_pos,
            dict(
                zip(CHAR_LIST, list(map(lambda name: name.split(" ")[0], CHAR_NAMES)))
            ),
            font_size=label_font_size,
            ax=ax,
        )

    # Make a color bar
    # if color_edges and cbar:
    #     pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    #     pc.set_array(np.arange(edge_vmin, edge_vmax + 0.01, 0.01))
    #     plt.colorbar(pc)

    # Set the title
    if title:
        ax.set_title(label=title, fontsize=title_fontsize, y=title_offset)
    ax.set_facecolor("white")
    ax.set_axis_off()
    if tight_layout:
        plt.tight_layout()

    if return_graph:
        return ax, G, pos
    else:
        return ax


def plot_free_recall_lm2(model, plot_order, ax):
    plot_data = (
        model.fixef.reset_index()
        .rename(columns={"index": "SID"})
        .melt(id_vars="SID", var_name="model", value_name="correlation")
        .assign(
            category=lambda x: x.model.map(
                {
                    "communicability": "annot",
                    "cooccurence": "annot",
                    "salience": "annot",
                    "family": "graph",
                    "friendship": "graph",
                    "football": "graph",
                    "trait": "q",
                    "preference": "q",
                    "action": "q",
                    "moral": "q",
                    "location": "q",
                }
            )
        )
    )
    ax = sns.barplot(
        x="model",
        y="correlation",
        hue="category",
        units="SID",
        order=plot_order,
        data=plot_data,
        ax=ax,
    )
    ax = sns.stripplot(
        x="model",
        y="correlation",
        color="grey",
        hue="category",
        dodge=True,
        order=plot_order,
        data=plot_data,
        ax=ax,
    )
    ax.set(
        ylim=(-0.4, 0.4),
        xlabel="",
        ylabel="Model similarity with memory\n(rank correlation)",
    )
    ax.get_legend().remove()
    sns.despine()
    plt.xticks(rotation=45)
    return ax
