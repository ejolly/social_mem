# SUMMARY
#  Another approach to analyzing memory data is by performing model comparison to
#  predict people's memories. Here we take "behavioral RSA" approach where we model the
#  absolute value distance between the *order* characters were recalled in a
#  "free-recall". We do this by fitting a ranked linear regression to the a vector of
#  pairwise distances between characters using several distance models as predictors.
#  Then we perform inference *over participants* by using permutation testing and
#  bootstrapping confidence intervals. This allows us to test in a data-driven way
#  (rather than through experimental manipulation) what model(s) best fit to the
#  structure of participant's memory.
#  We also compare the empirical transition probability between characters to
#  transition probabilities made by participants.

# %% Imports and paths
from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib import rcParams
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from nltools.stats import matrix_permutation

from joblib import load, dump

# FP data-analysis tools
from utilz import (
    savefig,
    pipe,
    tweak,
    newax,
)
import utilz.dfverbs as _
import utilz.dfverbs as do

# lib functions and globals
from utils import (
    DATA_DIR,
    CHAR_LIST,
    FIG_DIR,
    CHAR_PAIRS,
    ANALYSIS_DIR,
    ALL_EXCLUSIONS,
    distmat_to_df,
    fit_lm2,
    annotate_axis,
)

sns.set_context("talk", font_scale=1.2)
sns.set_style("whitegrid")
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Avenir"]

OUTPUT_DIR = ANALYSIS_DIR / "n03"
FIG_DIR = FIG_DIR / "n03"
OUTPUT_DIR.exists() or OUTPUT_DIR.mkdir()
FIG_DIR.exists() or FIG_DIR.mkdir()

# %% Load and munge data
recall_data = pipe(
    DATA_DIR / "recalls.csv",
    _.read_csv(),
    _.query(lambda df: ~df.SID.isin(ALL_EXCLUSIONS)),
    _.query("Cue == 'FreeRecall'"),
    _.mutate(
        Recall=lambda Recall: Recall.astype(int),
        # Create a new column that simply indicates whether a character was or wasn't recalled
        Was_Recalled=lambda Recall: Recall.apply(lambda x: 1 if x > 0 else x),
    ),
    ...,
    lambda df: print(
        f"""
        Num participants: {df.SID.nunique()}
        Num characters per participant: {df.groupby('SID').size().unique()}
        """
    ),
)

# %% Calculate memory distances and load distance models
# NOTE: For this analysis we first compute the absolute value *distance* between
#  characters within each participant's free-recall. This provides a symmetric character
#  x character matrix that captures how certain characters "cluster" together in
#  memory.
#  Our goal is to test several competing models to predict this matrix that are also
#  comprised of *distances* between characters as a function of different features
#  e.g. family ties, participant's impression ratings, etc.
#  The model used is a two-level regression approach where each participant's own
#  recall data is first fit separately. Then parameter estimates for each predictor
#  (i.e. "betas" or semi-partial correlations) are passed to another regression to
#  perform inference over participants using permutation testing (sign-flipping across
#  participants).


def calc_recall_distances(SID, CHAR_PAIRS):
    """
    For a given subject return a character x character distance matrix
    containing the absolute value distance between each character based upon their
    relative recall positions. If either character was not recalled set the recall
    distance to 0 so it can be excluded in future analyses.
    """
    df = recall_data.query("SID == @SID").reset_index(drop=True)
    out = pd.DataFrame(np.zeros(shape=(11, 11)), index=CHAR_LIST, columns=CHAR_LIST)
    get_char = lambda df, p: df.loc[df.Character == p, "Recall"].values[0]
    for p in CHAR_PAIRS:
        p1, char1 = get_char(df, p[0]), p[0]
        p2, char2 = get_char(df, p[1]), p[1]
        if p1 == 0 or p2 == 0:
            out.loc[char1, char2] = 0
            out.loc[char2, char1] = 0
        else:
            diff = np.abs(p1 - p2)
            out.loc[char1, char2] = diff
            out.loc[char2, char1] = diff
    return out


def calc_or_load_tidy_dists(MODEL_PATH, rescale_to_dists=True):
    """Calculates the distance between participants recalls, loads distance models, and
    puts them all together in a tidy dataframe for analysis"""

    if MODEL_PATH.exists():
        print("Loading previously calculated dataframe...")
        model_df = load(ANALYSIS_DIR / "models" / "free_recall_model_df.sav")

    else:
        # Load in pre-computed distance models
        salience = load(ANALYSIS_DIR / "models" / "character_salience_distances.sav")
        graph_comm = load(
            ANALYSIS_DIR / "models" / "crafted_model_communicabilities.sav"
        )
        coccur = load(ANALYSIS_DIR / "models" / "character_simple_cooccurence.sav")
        traits = load(ANALYSIS_DIR / "models" / "character_trait_distances.sav")
        # Build dataframe for modeling with rows as character recall pairs and columns as
        # predictors, with vstacked subjects
        # Use non-degenerate graphs, others are strict sub-graphs
        graph_models = [
            "family",
            "friendship",
            "football",
        ]
        model_df = []
        recall_counts = []

        for SID in recall_data.SID.unique():

            # Calculate recall distances
            distances = calc_recall_distances(SID, CHAR_PAIRS)
            df = distmat_to_df(distances, "recalls")
            df["SID"] = SID

            # Filter out non-recalls
            df = df.query("recalls != 0").reset_index(drop=True)
            recall_counts.append((SID, df.shape[0]))

            # Add in salience model
            salience_long = distmat_to_df(salience, "salience")
            df = df.merge(salience_long, on="Pair")

            # Add in co-occurence model
            coccur_long = distmat_to_df(coccur, "cooccurence")
            df = df.merge(coccur_long, on="Pair")

            # Add in graph models
            for graph in graph_models:
                graph_dists = distmat_to_df(graph_comm[graph], graph)
                df = df.merge(graph_dists, on="Pair")

            # Add in rating models
            rating_dists = distmat_to_df(
                traits.query("SID == @SID").iloc[:, :11], "trait"
            )
            df = df.merge(rating_dists, on="Pair")

            model_df.append(df)

        model_df = pd.concat(model_df, ignore_index=True)
        recall_counts = pd.DataFrame(recall_counts, columns=["SID", "Num_pairs"])

        print(
            f"Mean pairs: {recall_counts.Num_pairs.mean()} +/- {recall_counts.Num_pairs.std()}"
        )

        print("Saving...")
        dump(model_df, ANALYSIS_DIR / "models" / "free_recall_model_df.sav")

    # Flip the sign of measures to make everything in terms of "distance", i.e.  #
    # larger numbers == further away
    if rescale_to_dists:
        model_df.family *= -1
        model_df.friendship *= -1
        model_df.football *= -1
        model_df.cooccurence = 1 - model_df.cooccurence  # correlation distance

    return model_df


model_df = calc_or_load_tidy_dists(ANALYSIS_DIR / "models" / "free_recall_model_df.sav")

# %% Fit Lm2 predicting recall_dist ~ trait_dist + football_graph_comm +
# family_graph_comm + friendship_graph_comm + co-occur + salience
# NOTE: Traits were self-reported, co-occur refers to pairwise correlation between
# characters from our annotations, football/family graph communicability is computing
# the comm between character pairs given a crafted graph of their relationships. We
# don't include romance and gender because they're strictly sub-graphs of friendship and football.

dist_model = pipe(
    model_df,
    fit_lm2(
        formula="recalls ~ trait + football + family + friendship + cooccurence + salience",
        group="SID",
        model_file=OUTPUT_DIR / "recall_dist_model.h5",
        rank=True,
        to_corrs="semi",
        ztrans_corrs=False,
        conf_int="boot",
        n_boot=1000,
        permute=5000,
    ),
)
dist_model.summary()

# %% Plot effects

# Munge subject-level effects for plotting
sub_effects = dist_model.fixef.reset_index()
sub_effects = pipe(
    sub_effects,
    _.rename({"index": "SID"}),
    _.pivot_longer(id_vars="SID", into=("model", "correlation")),
    _.mutate(
        category=lambda model: model.map(
            {
                "cooccurence": "annot",
                "salience": "annot",
                "family": "graph",
                "friendship": "graph",
                "football": "q",
                "trait": "q",
            }
        )
    ),
    _.mutate(
        model=lambda model: model.map(
            {
                "cooccurence": "Co-occurr",
                "salience": "Salience",
                "family": "Family",
                "friendship": "Friendship",
                "football": "Social Identity\n(football)",
                "trait": "Trait\nImpressions",
            }
        )
    ),
)

# Plot full
f, ax = plt.subplots(figsize=(14, 4.8))

model_plot_order = [
    "Trait\nImpressions",
    "Social Identity\n(football)",
    "Family",
    "Friendship",
    "Co-occurr",
    "Salience",
]
colors = sns.color_palette("Set2", n_colors=4)
colors = [colors[1], colors[-1], colors[2]]

pipe(
    sub_effects,
    _.stripbarplot(
        x="model",
        y="correlation",
        hue="category",
        order=model_plot_order,
        dodge=False,
        legend=False,
        pointcolor="black",
        alpha=0.2,
        ax=ax,
        palette=colors,
    ),
    tweak(
        xlabel="",
        ylabel="Model fit to memory\n(semi-partial correlation)",
        despine=True,
        ylim=(-0.5, 0.5),
        xticklabel_fontsize=16,
        xticklabels=[
            "Trait\nImpressions",
            "Social Identity\n(football)",
            "Family\nCommunicability",
            "Friendship\nCommunicability",
            "Temporal\nCo-occurence",
            "Visual\nSalience",
        ],
    ),
    annotate_axis(
        xstart=[2, 3, 5],
        y=[0.4] * 3,
        texts=["***", "**", "***"],
        fontsize=20,
    ),
    savefig(path=FIG_DIR, name="free_recall_lm2"),
)

# %% Calcluate empirical and recall transition probabilities


def load_ep_annot(num):
    return pd.read_csv(DATA_DIR / "annotations" / f"char_mat_ep{num}.csv")


def compute_empirical_trans(mat, look_ahead=1, fill_diagonal=None):
    """
    Given a dataframe of TR X character annotation convert it to a
    multi-state transition probability matrix of character x character.
    """

    out_file = OUTPUT_DIR / "empirical_trans_prob.csv"

    if out_file.exists():
        print("loading pre-computed empirical transition probabilities")
        trans_prob = pd.read_csv(out_file)
        trans_prob.index = CHAR_LIST
    else:
        # Get character names
        char_list = mat.columns

        # Initialize empty matrix
        trans_prob = np.zeros((len(char_list), len(char_list)))

        for ci, char in enumerate(char_list):
            for i, row in mat.iterrows():
                if i < mat.shape[0] - look_ahead:
                    # If a character appears at the current TR
                    if mat.iloc[i, ci]:
                        # Update the count of all characters at next TR
                        trans_prob[ci, :] += mat.iloc[i + look_ahead, :]
            # Normalize
            trans_prob[ci, :] = trans_prob[ci, :] / trans_prob[ci, :].sum()

        if fill_diagonal is not None:
            np.fill_diagonal(trans_prob, fill_diagonal)

        # Convert to pandas
        trans_prob = pd.DataFrame(
            trans_prob, index=char_list, columns=char_list
        ).fillna(0)
        trans_prob.to_csv(out_file, index=False)

    return trans_prob


def compute_sub_trans(recall_data):

    out_file = OUTPUT_DIR / "sub_trans_prob.csv"
    if out_file.exists():
        print("loading pre-computed subject transition probabilities")
        sub_trans = pd.read_csv(out_file)
        sub_trans.index = CHAR_LIST
    else:
        sub_trans = pd.DataFrame(
            np.zeros(shape=(len(CHAR_LIST), len(CHAR_LIST))),
            index=CHAR_LIST,
            columns=CHAR_LIST,
        )

        for s in recall_data["SID"].unique():
            sub = recall_data.query("SID == @s")

            for c in CHAR_LIST:
                # Get the position they recalled this character
                this_char_idx = sub.query("Character == @c")["Recall"].values[0]
                # Ignore if this is the last character they recalled or if they didn't recall
                # them at all
                if this_char_idx == sub["Recall"].max() or this_char_idx == 0:
                    continue
                else:
                    next_recall = this_char_idx + 1
                    next_char = sub.query("Recall == @next_recall")["Character"].values[
                        0
                    ]
                    sub_trans.loc[c, next_char] += 1

        # Normalize by the number of subjects so the transitions are interpretable as the
        # proportion of participants who made that transition
        sub_trans = sub_trans / recall_data.SID.nunique()
        sub_trans.to_csv(out_file, index=False)
    return sub_trans


def zero_diago(mat):
    m = mat.to_numpy() if isinstance(mat, pd.DataFrame) else mat
    out = m.copy()
    np.fill_diagonal(out, 0)
    if isinstance(mat, pd.DataFrame):
        return pd.DataFrame(out, index=mat.index, columns=mat.columns)
    else:
        return out


annots = pd.concat([load_ep_annot(i) for i in range(1, 5)])[CHAR_LIST]
empirical_trans = compute_empirical_trans(annots)
empirical_trans = zero_diago(empirical_trans)
sub_trans = compute_sub_trans(recall_data)

# To see numeric annotations on the heatmaps
# with sns.plotting_context("talk", font_scale=0.4):
#     ax = sns.heatmap(empirical_trans, vmin=0, cmap="Blues", annot=True, ax=newax())
#     tweak(ax, ylabel="From", xlabel="To", title="Empirical Transition Probabilities")
#     ax = sns.heatmap(sub_trans, vmin=0, cmap="Blues", annot=True, ax=newax())
#     tweak(ax, ylabel="From", xlabel="To", title="Participant Transition
#     Probabilities")
pipe(
    empirical_trans,
    do.heatmap(vmin=0, cmap="Blues"),
    tweak(ylabel="From", xlabel="To", title="Empirical Transition Probabilities"),
    savefig(path=FIG_DIR, name="empirical_trans_probs"),
)
pipe(
    sub_trans,
    do.heatmap(vmin=0, cmap="Blues"),
    tweak(ylabel="From", xlabel="To", title="Proportion of Participant Recalls"),
    savefig(path=FIG_DIR, name="recalled_trans_probs"),
)


# %% Compare empirical and subject transitions

results = matrix_permutation(
    sub_trans, empirical_trans, random_state=991, return_perms=True, how="full"
)

# %% Plot similarity

f, ax = plt.subplots(1, 1)
ax.hist(results["perm_dist"])
ax.vlines(x=results["correlation"], ymin=0, ymax=1500, color="k", ls="--")
pipe(
    ax,
    tweak(
        xlim=(-1, 1),
        title=f"Permuted Correlation:\n{np.round(results['correlation'],3)}, p={np.round(results['p'],4)}",
        despine=True,
    ),
    savefig(path=FIG_DIR, name="trans_prob_perm_dist"),
)

ax = sns.regplot(
    x=rankdata(offdiag(empirical_trans)), y=rankdata(offdiag(sub_trans)), ax=newax()
)
pipe(
    ax,
    tweak(
        xlabel="Observed Transitions (ranks)",
        ylabel="Recalled Transitions (ranks)",
        despine=True,
    ),
    savefig(path=FIG_DIR, name="trans_prob_rank_corr"),
)

# %% Nicer weighted edge-graphs of empirical and participant transitions
from utils import plot_graph

threshold = 0.05
recall_threshold = 0
graph_type = "directed"
layout = "spring"
node_color = "orange"
node_size = 1000
node_border_color = "grey"
node_border_width = 3
edge_colors = "black"
connectionstyle = "arc3, rad = 0.1"
edge_start_gap = 40
edge_end_gap = 40
weight_edges = True
alpha_edges = True
arrow_size = 25
alpha_enhance = -0.10
alpha_enhance_recall = -0.1
label_y_adjust = 0.1
edge_weight_scale = 6
random_seed = 10
k = 1
use_char_images = True
img_scale = 0.05

ax, G, pos = plot_graph(
    empirical_trans,
    threshold=threshold,
    graph_type=graph_type,
    layout=layout,
    node_color=node_color,
    node_size=node_size,
    node_border_color=node_border_color,
    node_border_width=node_border_width,
    edge_colors=edge_colors,
    connectionstyle=connectionstyle,
    edge_start_gap=edge_start_gap,
    edge_end_gap=edge_end_gap,
    weight_edges=weight_edges,
    alpha_edges=alpha_edges,
    arrow_size=arrow_size,
    alpha_enhance=alpha_enhance,
    label_y_adjust=label_y_adjust,
    edge_weight_scale=edge_weight_scale,
    random_seed=random_seed,
    k=k,
    use_char_images=use_char_images,
    img_scale=img_scale,
    return_graph=True,
    title="Empirical Transitions",
    title_fontsize=24,
)
savefig(ax, path=FIG_DIR, name="empirical_trans_probs_graph")

ax = plot_graph(
    sub_trans,
    threshold=recall_threshold,
    graph_type=graph_type,
    layout=layout,
    node_color=node_color,
    node_size=node_size,
    node_border_color=node_border_color,
    node_border_width=node_border_width,
    edge_colors=edge_colors,
    connectionstyle=connectionstyle,
    edge_start_gap=edge_start_gap,
    edge_end_gap=edge_end_gap,
    weight_edges=weight_edges,
    alpha_edges=alpha_edges,
    arrow_size=arrow_size,
    alpha_enhance=alpha_enhance_recall,
    label_y_adjust=label_y_adjust,
    edge_weight_scale=edge_weight_scale,
    random_seed=random_seed,
    k=k,
    use_char_images=use_char_images,
    img_scale=img_scale,
    pos=pos,
    title="Participant Recall Transitions",
    title_fontsize=24,
)

savefig(ax, path=FIG_DIR, name="recall_trans_probs_graph")
