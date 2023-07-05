# #NOTE: To rerun the algorithm requires annotations csv files which are not included with the repo as they're currently in use for other unpublished lab projects. However, we do include the pre-computed motifs.joblib file which can be used to reproduce the results in the paper.
# %% 1. Imports and paths
from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
from nltools.data import Adjacency, Design_Matrix
from joblib import load, dump
from scipy.spatial.distance import squareform, pdist

# lib functions and globals
from utils import DATA_DIR, CHAR_LIST, ANALYSIS_DIR, plot_graph, FIG_DIR
from utilz import savefig, tweak

sns.set_context("talk", font_scale=1.2)
sns.set_style("whitegrid")
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Avenir"]
FIG_DIR = FIG_DIR / "motif_detection"

# %% 2. Functions
def vec2graph(v, return_square=False):
    """
    Given an n dimensional binary vector encoding the appearance of n nodes, convert
    it into an undirect adjacnecy matrix where edges indicate if two nodes are both "on"
    together

    Args:
        v: binary vector
        return_square (bool): whether to return the upper-triangle of the adjacency matrix or the full matrix
    """

    from itertools import combinations
    from scipy.spatial.distance import squareform

    edges = np.array(
        [
            1 if p[0] == 1.0 and p[1] == 1.0 and p[0] == p[1] else 0
            for p in combinations(v, 2)
        ]
    )
    if return_square:
        edges = squareform(edges)
    return edges


def load_ep_annot(num):
    return pd.read_csv(DATA_DIR / "annotations" / f"char_mat_ep{num}.csv")


def detect_motifs(
    annots: pd.DataFrame, look_ahead: int = 5, save: str = "motifs.joblib"
) -> dict:
    """
    Contiguous motif detection algorithm:

    Given a time x character annotation matrix:

    1) Convert it to a flattened adjacency matrix where edges indicated two or more characters co-occured. This ignores solo character appearances by treating those time-point as an empty adjacency matrix, i.e. graph with no edges

    2) Given a look_ahead criteria, find all unique graphs (i.e. edge combinations) that occur for at least look_ahead contiguous time-points. Only considers graphs with at least 1 edge.


    Returns:
        dict: dictionary of motifs where each value is sub dict with the motif's
        adjacency, the number of time-points it was present for, and the exact time-pointsit appeared
        edge_list: time x flattened adjacency representation of annotations
    """

    out_path = ANALYSIS_DIR / save

    if out_path.exists():
        print("Loading pre-computed motifs...")
        motif_dict = load(out_path)
        print(f"Number of unique motifs: {len(list(motif_dict.keys())) - 1}")
        return motif_dict

    # Convert TR x char annotation matrix to TR x flattened adjacency matrix
    all_edges = np.vstack([vec2graph(row.values) for i, row in annots.iterrows()])

    # Initialize motif cache
    unique_motifs = []

    for i in range(all_edges.shape[0]):
        # Start with nans and an empty cache since we're always comparing backwards
        if i == 0:
            pass
        else:
            # From TR 2 onwards
            current_motif = all_edges[i, :]

            # If there are at least 2 people on the screen
            if current_motif.sum() >= 1:

                # See if the next look_ahead TRs are the same motif
                are_contiguous = all(
                    [
                        np.allclose(current_motif, all_edges[i + l, :])
                        for l in range(1, look_ahead)
                    ]
                )

                if are_contiguous:
                    if not unique_motifs:
                        # If our motif cache is empty, add this motif
                        unique_motifs.append(current_motif)

                    else:
                        # Otherwise see if it's unique and if not add it to the store
                        have_it = False
                        for ie, e in enumerate(unique_motifs):
                            if np.allclose(e, current_motif):
                                have_it = True
                                break
                        # If we've hit this point we've looped through all our unique motifs and still dont have it so append it
                        if not have_it:
                            unique_motifs.append(current_motif)

    unique_motifs = np.array(unique_motifs, dtype=int)

    # Count motif occurences and identify time-points they occur
    # Clunky code with extra looping but it allows to address the following counting
    # issue:
    # It's tricky to both identify unique motifs that appear for at least N TRs
    # while also counting how often they appear because if a motif appears for *exactly*
    # N TRs it'll only get counted once, but if it appears for longer it'll get counted more
    # hence conflating the count with the time on screen.
    # So just do a second pass over all edges with our identified unique motif to count
    # them properly
    unique_motifs_counter = np.zeros((len(unique_motifs)))
    unique_motifs_idx = []
    for i, m in enumerate(unique_motifs):
        for TR, e in enumerate(all_edges):
            if np.allclose(m, e):
                unique_motifs_counter[i] += 1
                if unique_motifs_counter[i] == 1:
                    unique_motifs_idx.append([TR])
                else:
                    unique_motifs_idx[i].append(TR)

    unique_motifs_counter = np.array(unique_motifs_counter)
    unique_motifs_idx = np.array(unique_motifs_idx)

    # Make sure the TR locations matches the number of times we've observed the motif as a sanity check
    assert np.allclose(list(map(len, unique_motifs_idx)), unique_motifs_counter)

    print(f"Number of unique motifs: {len(unique_motifs_counter)}")

    # Return
    motif_dict = {}
    motif_dict["all_edges"] = all_edges
    for i in range(unique_motifs_counter.shape[0]):
        d = {}
        d["motif"] = unique_motifs[i]
        d["count"] = unique_motifs_counter[i]
        d["idx"] = np.array(unique_motifs_idx[i])
        motif_dict[i] = d

    dump(motif_dict, out_path)

    return motif_dict


def to_charmat(flat_adj):
    out = squareform(flat_adj)
    return pd.DataFrame(out, index=CHAR_LIST, columns=CHAR_LIST)


# %% 3. Load annotations and detect motifs
# Load and concat annotations
annots = pd.concat([load_ep_annot(i) for i in range(1, 5)])[CHAR_LIST]
# Plot annotations
ax = sns.heatmap(annots, cmap="gray", cbar=False)
ax = tweak(ax, yticklabels=[], ylabel="Time")
savefig(ax.get_figure(), path=FIG_DIR, name="character_onsets")

# Detect motifs
unsorted_motif_dict = detect_motifs(annots)

# Expand dict to variables for convenience
all_edges = unsorted_motif_dict.pop("all_edges")
unique_motifs_counter = np.array([v["count"] for k, v in unsorted_motif_dict.items()])
unique_motifs = np.array([v["motif"] for k, v in unsorted_motif_dict.items()])
unique_motifs_idx = np.array([v["idx"] for k, v in unsorted_motif_dict.items()])

# Sort motifs by the total time they occur
sort_idx = np.argsort(unique_motifs_counter)[::-1]
unique_motifs = unique_motifs[sort_idx]
unique_motifs_counter = unique_motifs_counter[sort_idx]
unique_motifs_idx = unique_motifs_idx[sort_idx]

# Rebuild the dict with keys corresponding to the sorted order
motif_dict = dict()
for i, idx in enumerate(sort_idx):
    motif_dict[i] = unsorted_motif_dict[idx]

dump(motif_dict, ANALYSIS_DIR / "motifs_sorted.joblib")
# %% 4a. Plot how often we see each motif
ax = sns.barplot(unique_motifs_counter * 2, orient="v")
ax = sns.stripplot(unique_motifs_counter * 2, orient="v", color="black", ax=ax)
_ = ax.set(ylabel="Number of Seconds", xlabel="", xticklabels=[])
sns.despine()


# %% 4b. Explore motifs
def plot_motif(idx, motif_dict=None, **kwargs):
    """Nicely plots a motif"""

    if motif_dict is not None:
        motif_data = motif_dict[idx]

        total_sec = motif_data["count"] * 2
        num_min = int(total_sec // 60)
        num_min = "00" if num_min == 0 else num_min
        num_sec = int(total_sec % 60)
        num_sec = "00" if num_sec == 0 else num_sec
        title = kwargs.pop("title", f"{num_min}m {num_sec}s")
        graph = squareform(motif_data["motif"])
    else:
        graph = kwargs.pop("graph", None)
        title = kwargs.pop("title", "")
        if graph is None:
            raise ValueError("must provide motif_dict or a 2d numpy array")
    edge_gap = kwargs.pop("edge_gap", 0)
    return plot_graph(
        graph,
        title=title,
        edge_start_gap=edge_gap,
        edge_end_gap=edge_gap,
        **kwargs,
    )


# %% Plot all motifs style 1 - just nodes
f, axs = plt.subplots(5, 5, figsize=(12, 12))
for i, ax in enumerate(axs.flat):
    if i in [4, 8, 12, 14, 15, 19, 20, 21, 24]:
        connectionstyle = "arc3, rad = -0.1"
    else:
        connectionstyle = "arc3, rad = 0.1"
    _ = plot_motif(
        i,
        motif_dict=motif_dict,
        ax=ax,
        node_size=150,
        edge_gap=9,
        edge_weights=3,
        arrow_size=1,
        show_arrows=True,
        figsize=(4, 4),
        node_color="orange",
        node_border_color="gray",
        node_border_width=2,
        connectionstyle=connectionstyle,
        title=f"Motif {i+1}",
        title_offset=1.05,
        title_fontsize=22,
    )
plt.tight_layout()
savefig(f, path=FIG_DIR, name="all_motifs")

# %% Plot all motifs style 2 - char images
f, axs = plt.subplots(5, 5, figsize=(16, 16))
for i, ax in enumerate(axs.flat):
    if i in [4, 8, 12, 14, 15, 19, 20, 21, 24]:
        connectionstyle = "arc3, rad = -0.1"
    else:
        connectionstyle = "arc3, rad = 0.1"
    _ = plot_motif(
        i,
        motif_dict=motif_dict,
        ax=ax,
        use_char_images=True,
        node_size=50,
        img_scale=0.01,
        edge_gap=15,
        weight_edges=False,
        edge_weights=1.85,
        show_arrows=True,  # just for edge spacing
        arrow_size=8,
        connectionstyle=connectionstyle,
        title_offset=1.03,
    )
savefig(f, path=FIG_DIR, name="all_motifs_char_imgs")

# %% Plot all motifs separately
sort_idx = np.argsort(unique_motifs_counter)[::-1]
for ii, i in enumerate(sort_idx):
    ax = plot_motif(
        i,
        motif_dict,
        use_char_images=True,
        img_scale=0.055,
        figsize=(6, 6),
        node_size=50,
        show_arrows=True,  # just for edge spacing
        arrow_size=15,
        edge_gap=30,
        connectionstyle="arc3, rad = 0.05",
        title="",
    )
    savefig(ax.get_figure(), path=FIG_DIR, name=f"motif_{ii+1}")
    plt.close("all")
# %% Fix up a couple where edges are hidden
motif = 25
ii = motif - 1
ax = plot_motif(
    ii,
    motif_dict=motif_dict,
    use_char_images=True,
    img_scale=0.055,
    figsize=(6, 6),
    node_size=50,
    show_arrows=True,  # just for edge spacing
    arrow_size=15,
    edge_gap=30,
    connectionstyle="arc3, rad = -0.1",
    title_offset=1.05,
)
# savefig(ax.get_figure(), path=FIG_DIR, name=f"motif_{motif}")

# %% Basic circular graph for motif animation
f, ax = plt.subplots(1, 1, figsize=(6, 6))
ax = plot_motif(
    23,
    # graph=squareform(all_edges[1540]),
    motif_dict=motif_dict,
    node_size=850,
    use_char_images=False,
    edge_gap=20,
    weight_edges=False,
    edge_weights=4,
    arrow_size=22,
    show_arrows=True,
    figsize=(4, 4),
    node_color="orange",
    node_border_color="gray",
    node_border_width=2.5,
    connectionstyle="arc3, rad = 0.05",
    plot_labels=True,
    label_font_size=18,
    label_x_adjust=0,
    label_y_adjust=0.2,
    ax=ax,
)
tweak(ax, ylim=(-1.25, 1.25), xlim=(-1.15, 1.2))
# _ = ax.text(0.4, 1.05, f"02:23", transform=ax.transAxes, fontsize=28)

# %% Generate talk figures for summing multiple motifs a char belongs to


def find_char_motifs(char, unique_motifs_dfs):
    """Return a list of all motifs to which a character belongs as well as a sum of those
    graphs with unweighted edges"""

    all_char_edges = pd.Series(np.zeros((11,)), index=CHAR_LIST)
    out = []
    for motif in unique_motifs_dfs:
        if motif[char].sum() > 0:
            out.append(motif)
            all_char_edges += motif[char]
    all_char_edges = (all_char_edges >= 1).astype(int)
    return out, all_char_edges


def plot_combined_char_motifs(character, unique_motifs_dfs, title=""):
    """Plot the sum of all motifs a character belongs to"""

    # Find all motifs the char belogns to
    char_motifs, char_edges = find_char_motifs(character, unique_motifs_dfs)
    # Initialize an adjacency
    char_graph = pd.DataFrame(np.zeros((11, 11)), index=CHAR_LIST, columns=CHAR_LIST)
    # Add in the edges
    for edge in char_edges[char_edges == 1].index:
        char_graph.loc[character, edge] = 1
        # We don't need symmetric edges for plotting
        # char_graph.loc[edge, character] = 1

    char_graph = char_graph.to_numpy().astype(int)

    return char_graph, plot_motif(
        np.nan,
        motif_dict=None,
        graph=char_graph,
        use_char_images=True,
        img_scale=0.055,
        figsize=(6, 6),
        node_size=50,
        show_arrows=True,  # just for edge spacing
        arrow_size=15,
        graph_type="directed",
        edge_gap=30,
        connectionstyle="arc3, rad = -0.1",
        title=title,
    )


# Turn flattened adjacencies into squareform with char names
unique_motifs_dfs = list(
    map(
        lambda adj: pd.DataFrame(squareform(adj), index=CHAR_LIST, columns=CHAR_LIST),
        unique_motifs,
    )
)


# %% Fig for example subject memory
landry_motifs, all_landry_edges = find_char_motifs("LandryClarke", unique_motifs_dfs)
example_memory = landry_motifs[0].copy()
example_memory.loc["LandryClarke", "MattSaracen"] = 1
example_memory.loc["LandryClarke", "CoachTaylor"] = 1
example_memory.loc["LandryClarke", "JulieTaylor"] = 1
example_memory.loc["MattSaracen", "LandryClarke"] = 0
plot_motif(
    np.nan,
    motif_dict=None,
    graph=example_memory.to_numpy(),
    use_char_images=True,
    img_scale=0.055,
    figsize=(6, 6),
    node_size=50,
    show_arrows=True,  # just for edge spacing
    arrow_size=15,
    graph_type="directed",
    edge_gap=30,
    connectionstyle="arc3, rad = -0.1",
    # title=title,
)


# %% Animate motifs
# NOTE: This cell creates a video for a range of TRs based on our annotations
from celluloid import Camera


def tr_to_mmss(TR):
    sec = TR * 2
    num_min = int(sec // 60)
    num_sec = int(sec % 60)
    num_min = f"0{num_min}" if num_min <= 9 else str(num_min)
    num_sec = f"0{num_sec}" if num_sec <= 9 else str(num_sec)
    return f"{num_min}:{num_sec}"


f, ax = plt.subplots(1, 1, figsize=(6, 6))
camera = Camera(f)

for i in range(480, 1000):
    title = tr_to_mmss(i)
    ax = plot_motif(
        np.nan,
        graph=squareform(all_edges[i]),
        motif_dict=None,
        node_size=850,
        use_char_images=False,
        edge_gap=20,
        weight_edges=False,
        edge_weights=4,
        arrow_size=22,
        show_arrows=True,
        figsize=(4, 4),
        node_color="orange",
        node_border_color="gray",
        node_border_width=2.5,
        connectionstyle="arc3, rad = 0.05",
        plot_labels=True,
        label_font_size=18,
        label_x_adjust=0,
        label_y_adjust=0.2,
        ax=ax,
    )
    _ = tweak(ax, ylim=(-1.25, 1.25), xlim=(-1.15, 1.2))
    _ = ax.text(0.4, 1.05, f"{title}", transform=ax.transAxes, fontsize=24)
    _ = camera.snap()

animation = camera.animate()

animation.save(str(ANALYSIS_DIR / "motif.mp4"))

# %% Friendship and Family graphs

family = squareform(unique_motifs[8] + unique_motifs[22])
friendship = squareform(
    unique_motifs[1]
    + unique_motifs[2]
    + unique_motifs[4]
    + unique_motifs[5]
    + unique_motifs[13]
    + unique_motifs[16]
    + unique_motifs[17]
    + unique_motifs[23]
)
plot_motif(
    np.nan,
    motif_dict=None,
    graph=friendship,
    use_char_images=True,
    img_scale=0.055,
    figsize=(6, 6),
    node_size=50,
    show_arrows=True,  # just for edge spacing
    arrow_size=15,
    graph_type="undirected",
    edge_gap=30,
    connectionstyle="arc3, rad = -0.1",
    # title=title,
)
