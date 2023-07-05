# SUMMARY
#  One approach to analyzing memory data is by testing specific *hypotheses* about the
#  dimensions that *could* organize person memories. To do this we probed participants'
#  memories from cues that belonged to 4 categories we theorized could be these
#  dimensions:
#  1. Actions (e.g. fighting, gossiping, romance)
#  2. Places (e.g. football field, diner)
#  3. Traits (e.g. vain, attractive, intelligent)
#  4. People (i.e. *other* characters in the show)

# %% Imports and paths
from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import itertools as it
from pymer4 import Lmer, save_model, load_model
from pathlib import Path
from joblib import load
from tqdm import tqdm

# FP data-analysis tools
from utilz import (
    savefig,
    map,
    mapcat,
    pipe,
    curry,
    alongwith,
    unpack,
    tweak,
)

import utilz.dfverbs as _

# lib functions and globals
from utils import (
    DATA_DIR,
    CHAR_LIST,
    ANALYSIS_DIR,
    FIG_DIR,
    CUE_EXCLUSIONS,
    CUE_ORDER,
    CUE_MAP,
    annotate_axis,
    fit_lmer,
)

sns.set_context("talk", font_scale=1.2)
sns.set_style("whitegrid")
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Avenir"]

OUTPUT_DIR = ANALYSIS_DIR / "n01"
ANALYSIS_DIR = ANALYSIS_DIR / "n01"
FIG_DIR = FIG_DIR / "n01"
OUTPUT_DIR.exists() or OUTPUT_DIR.mkdir()
FIG_DIR.exists() or FIG_DIR.mkdir()

# %% Load and clean data

recall_data = pipe(
    DATA_DIR / "recalls.csv",
    _.read_csv(),
    _.fillna(0),
    _.replace(" ", 0),
    _.query(lambda df: ~df.SID.isin(CUE_EXCLUSIONS)),
    _.query("Cue != 'FreeRecall'"),
    _.mutate(
        Recall=lambda df: df.Recall.astype(int),
        Was_Recalled=lambda df: df.Recall.apply(lambda x: 1 if x > 0 else x),
    ),
    ...,
    lambda df: print(
        f"""
           Num participants: {df.SID.nunique()}
            Num cues: {df.groupby('Category').Cue.nunique()}
            Num chars per participant: {df.groupby('SID').size().unique()}
            """
    ),
)


# %% What category of cue elicits the most *similar* recalls across individuals?
# NOTE: Since this isn't a constrained learning paradigm (e.g. list-learning task) there
#  is no notion of memory *accuracy* because there is no "ground-truth" to compare
#  participants' recalls to. We can instead borrow from naturalistic fMRI style analyses
#  and instead compare the similarity of all participants memories for each category.
#  Given the fact that each participant viewed the same input stimulus we can reason
#  that the mean intersubject-similarity of their recalls acts as a kind of
#  *reliability* score for any given category. Then we can compare across participants
#  in a fashion similar to how we might compare accuracy in more conventional paradigms.


def calc_pair_sim(sub_pair: tuple, recall_bin_mat: pd.DataFrame) -> pd.Series:
    """Compute the jaccard similarity between a pair of subs for each cue"""

    def jaccard_sim(a, b):
        """Jaccard similarity metric"""
        from scipy.spatial.distance import jaccard

        return 1 - jaccard(a, b)

    s1, s2 = sub_pair  # type: ignore
    s1_data = recall_bin_mat.query("SID == @s1").iloc[:, 2:].reset_index(drop=True)
    s2_data = recall_bin_mat.query("SID == @s2").iloc[:, 2:].reset_index(drop=True)
    sims = [jaccard_sim(s1_data[col], s2_data[col]) for col in s1_data]
    return pd.Series(sims, index=s1_data.columns)


recall_sims, char_by_cue_stacked = pipe(
    recall_data,
    # Reshape so dataframe is vstack of char x cue binary matrix per sub with 0s if
    # character wasn't mentioned for that cue, and 1 if they were
    _.groupby("SID"),
    _.apply(
        lambda s: s.pivot(values="Was_Recalled", index="Character", columns="Cue"),
        reset_index="reset",
    ),
    # -> piv: Dataframe of sub stacked char x cue binary mats
    # Generate a list of subject pairs
    alongwith(lambda df: list(it.combinations(df.SID.unique(), 2))),
    # -> ( piv: DataFrame,  pairs: list[tuple])
    # Loop over subject pairs and compute similarity
    alongwith(
        lambda recall_data, pairs: mapcat(
            calc_pair_sim,
            pairs,
            recall_bin_mat=recall_data,
            concat_axis=1,
        ),
    ),
    # -> ( piv: DataFrame, pairs: list[tuple], sims: DataFrame of cue x pairs )
    # melt into longform
    unpack(
        lambda piv, pairs, sims: pipe(
            sims.T[CUE_ORDER],
            _.mutate(S_pair=map(lambda p: "_".join(p), pairs)),
            _.pivot_longer(id_vars="S_pair", into=("Cue", "Similarity")),
            _.mutate(Category=lambda df: df.Cue.map(CUE_MAP)),
            alongwith(piv),
            show=False,
        )
    ),
)

# %% Run Lmer using Chen et al method to compute ISC with correct dof
# NOTE: Use Chen et al, method to compute corrected dof when performing ISC comparison


@curry
def fit_sim_lmer(
    data: pd.DataFrame, formula: str, model_file: Path, factors: list | None = None
) -> Lmer:
    """Loads or fits an lmer model with Chen et al ISC rfx structure"""

    def create_isc_rfx(df, col="S_pair"):
        """Takes a long form dataset with a col indicating the subject pair and splits on
        '_' and appends a full copy of the dataset with separate id columns for each
        subject. This is design to be used with Lmer"""
        p = df.copy()
        p = p.assign(
            sub1=p.S_pair.apply(lambda x: x.split("_")[0]),
            sub2=p.S_pair.apply(lambda x: x.split("_")[1]),
        )
        p2 = p.copy().rename(columns={"sub1": "sub2", "sub2": "sub1"})
        pp = pd.concat([p, p2], axis=0)
        return pp

    if model_file.exists():
        print("Loading existing model from file")
        model = load_model(str(model_file))
    else:
        sim_data = create_isc_rfx(data)
        model = Lmer(formula, data=sim_data)
        model.fit(factors=factors, ordered=False, summarize=False)
        model.anova(force_orthogonal=True)
        save_model(model, str(model_file))

    return model


@curry
def run_sim_posthoc(model: Lmer, marginal_vars: str) -> pd.DataFrame:
    """Performs dof corrected pairwise comparisons"""

    def correct_isc_dof_p(df, correct="holm"):
        from scipy.stats import t as tdist
        from pymer4.utils import _sig_stars
        from pymer4.stats import correct_pvals

        dfs = df["DF"].to_numpy() / 2
        ts = df["T-stat"].to_numpy()
        ps = tdist.sf(np.abs(ts), dfs) * 2
        out = df.copy()
        out["DF"] = dfs
        out["T-stat"] = ts
        if correct is not None:
            out["P-val"] = correct_pvals(ps, method=correct)
        else:
            out["P-val"] = ps
        out["Sig"] = out["P-val"].apply(_sig_stars)
        return out

    model.post_hoc(marginal_vars=marginal_vars, p_adjust="none", summarize=False)
    cons = correct_isc_dof_p(model.marginal_contrasts, correct="holm")
    print(f"p-values corrected for multiple comparisons by holm-bonferroni procedure")
    return cons.round(3)


model, comparisons = pipe(
    recall_sims,
    # Fit and save model or load existing; more complicated rfx fails to converge
    fit_sim_lmer(
        model_file=OUTPUT_DIR / "cuesim_lmer.h5",
        formula="Similarity ~ Category + (Category | sub1) + (Category | sub2)",
        factors={"Category": ["Character", "Action", "Place", "Trait"]},
    ),
    # Run pairwise comparisons
    alongwith(
        run_sim_posthoc(marginal_vars="Category"),
    ),
    # Save to gsheet
    # format_and_save2sheet(
    #     iv_name="Cued Category",
    #     comparison_name="Mean Difference",
    #     fetch_name_col="Contrast",
    #     sheet="cuesim",
    # ),
    show=True,
)

# %% Plot ISC
pipe(
    recall_sims,
    _.mutate(
        Category=lambda Category: Category.map(
            {
                "Action": "Actions",
                "Trait": "Traits",
                "Place": "Locations",
                "Character": "People",
            }
        )
    ),
    # plot it
    _.stripbarplot(
        x="Category",
        y="Similarity",
        hue="Category",
        units="S_pair",
        n_boot=100,
        dodge=False,
        ncol=4,
        loc="upper center",
        legend=False,
        palette=sns.color_palette("Set2"),
        pointcolor="black",
        alpha=0.005,
        xlabel="",
    ),
    tweak(title="", ylabel="Mnemonic Convergence\n(mean jaccard similarity)"),
    # # add significance labels
    annotate_axis(
        xstart=[0, 1, 0, 1, 2],
        xend=[1, 2, 3, 3, 3],
        y=[0.57, 0.63, 0.7, 0.8, 0.9],
        texts=["***"] * 5,
        thickness=2,
        fontsize=20,
        despine=True,
    ),
    savefig(path=FIG_DIR, name="cue_recall_similarity"),
)

# %% Compare num recalled characters across categories

num_chars_by_cue = pipe(
    char_by_cue_stacked,
    _.groupby("SID"),
    _.sum(),
    _.reset_index(),
    _.pivot_longer(id_vars="SID", into=("Cue", "Num_Chars")),
    _.mutate(Category=lambda Cue: Cue.map(CUE_MAP)),
)

num_chars_evoc_model = pipe(
    num_chars_by_cue,
    fit_lmer(
        formula="Num_Chars ~ Category + (1 | SID)",
        model_file=OUTPUT_DIR / "num_chars_evoc.h5",
        factors={"Category": ["Character", "Action", "Trait", "Place"]},
    ),
    show=True,
)
num_chars_evoc_model.summary()

# Agg stats
pipe(
    num_chars_by_cue,
    _.groupby("Category"),
    _.summarize(cat_mean="Num_Chars.mean()", cat_std="Num_Chars.std()"),
)


# Num chars by category
pipe(
    num_chars_by_cue,
    _.mutate(
        Category=lambda Category: Category.map(
            {
                "Action": "Actions",
                "Trait": "Traits",
                "Place": "Locations",
                "Character": "People",
            }
        )
    ),
    _.stripbarplot(
        x="Category",
        y="Num_Chars",
        units="SID",
        n_boot=100,
        dodge=True,
        order=["Actions", "Traits", "Locations", "People"],
        alpha=0.05,
        pointcolor="black",
        palette=sns.color_palette("Set2"),
    ),
    tweak(
        xlabel="",
        ylabel="Number of Characters\n Recalled (Average)",
        ylim=(0, 11),
        despine=True,
    ),
    annotate_axis(
        xstart=[0, 1, 2],
        xend=[3, 3, 3],
        y=[9.8, 9, 7],
        texts=["***"] * 3,
        thickness=2,
        fontsize=20,
        despine=True,
    ),
    savefig(path=FIG_DIR, name="num_chars_by_category"),
)


# %% How are accurate are characted-cued recalls if motifs are ground truth?
from scipy.spatial.distance import squareform
from pymer4.utils import upper


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


def score_memory(
    SID, Character, char_only_cues, unique_motifs_dfs, sigma=0.000001, shuffle=False
):
    "Score a participants recall by comparing the edges they mentioned to the ones present in any motifs containing this character. Score using TP, FP, TN, and FN to generate a balanced accuracy score and other SDT metrics"

    # Get sub data
    s = char_only_cues.query("SID == @SID").set_index("Character").drop(columns="SID")

    # See who they said in response to character
    mem = s[Character]
    if shuffle:
        mem = pd.Series(
            np.random.choice(mem.values, len(mem), replace=False), index=mem.index
        )

    # See who that char was connected to based on motifs they particpated in
    char_motifs, char_graph = find_char_motifs(Character, unique_motifs_dfs)

    # Score
    score_card = pd.concat([char_graph, mem], axis=1).astype(int)
    score_card.columns = ["motif", "memory"]

    tp = score_card.query("motif == 1 and memory == 1").shape[0]
    fp = score_card.query("motif == 0 and memory == 1").shape[0]
    tn = score_card.query("motif == 0 and memory == 0").shape[0]
    fn = score_card.query("motif == 1 and memory == 0").shape[0]

    out = dict(
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        sensitivity=tp / (tp + fn + sigma),
        specificity=tn / (tn + fp + sigma),  # recall
        precision=tp / (tp + fp + sigma),
        acc=(tp + tn) / (tp + tn + fp + fn + sigma),
    )
    out["bal_acc"] = np.mean([out["sensitivity"], out["specificity"]])

    out["f1"] = (out["precision"] * out["sensitivity"]) / (
        out["precision"] + out["sensitivity"] + sigma
    )
    if not shuffle:
        out["score_card"] = score_card
    return out


# %% Score memories

char_only_cues = char_by_cue_stacked[["SID", "Character"] + CHAR_LIST]
motif_dict = load(ANALYSIS_DIR / "motifs.joblib")

# Expand dict to variables for convenience
all_edges = motif_dict.pop("all_edges")
unique_motifs_counter = np.array([v["count"] for k, v in motif_dict.items()])
unique_motifs = np.array([v["motif"] for k, v in motif_dict.items()])
unique_motifs_idx = np.array([v["idx"] for k, v in motif_dict.items()])

# Turn flattened adjacencies into squareform with char names
unique_motifs_dfs = map(
    lambda adj: pd.DataFrame(squareform(adj), index=CHAR_LIST, columns=CHAR_LIST),
    unique_motifs,
)

sub_motif_score_cards = []
sub_motif_scores = []
for sid in char_only_cues.SID.unique():
    for char in CHAR_LIST:
        score_dict = score_memory(sid, char, char_only_cues, unique_motifs_dfs)
        score_card = score_dict.pop("score_card")
        score_card["SID"] = sid
        score_card["Character"] = char
        sub_motif_score_cards.append(score_card)

        score_dict["SID"] = sid
        score_dict["Character"] = char
        sub_motif_scores.append(score_dict)

sub_motif_scores = pd.DataFrame(sub_motif_scores)
sub_motif_score_cards = pd.concat(sub_motif_score_cards, ignore_index=True)

# Melt for plotting
metrics_of_interest = [
    "sensitivity",
    "specificity",
    "precision",
    "acc",
    "bal_acc",
    "f1",
]

sub_motif_scores = sub_motif_scores[metrics_of_interest + ["SID", "Character"]]
sub_motif_scores = pipe(
    sub_motif_scores,
    _.pivot_longer(columns=metrics_of_interest, into=("metric", "score")),
)

# %% Run permuted version for inference
# Run permuted version
out_path = ANALYSIS_DIR / "sub_motif_scores_perms.csv"
if out_path.exists():

    print("Loading pre-computed permuted memories...")
    sub_motif_scores_perms = pd.read_csv(out_path)
else:
    print("Generating and scoring permuted memories...")
    metrics_of_interest = [
        "sensitivity",
        "specificity",
        "precision",
        "acc",
        "bal_acc",
        "f1",
    ]

    # sub_motif_scores_perms = []

    for i in tqdm(range(1000)):
        scores = []
        # Score everyone
        for sid in char_only_cues.SID.unique():
            for char in CHAR_LIST:
                score_dict = score_memory(
                    sid, char, char_only_cues, unique_motifs_dfs, shuffle=True
                )
                score_dict["SID"] = sid
                score_dict["Character"] = char
                scores.append(score_dict)

        scores = pd.DataFrame(scores)
        scores = scores[metrics_of_interest + ["SID", "Character"]]

        # Aggregate memories within subject by metric
        perm_means = pipe(
            scores,
            _.pivot_longer(columns=metrics_of_interest, into=("metric", "score")),
            _.groupby("SID", "metric"),
            _.summarize(mean="score.mean()"),
            _.assign(perm=i),
        )
        # perm_means = perm_means.set_index("metric")
        # perm_means = perm_means.T
        # sub_motif_scores_perms.append(perm_means)
        perm_means.to_csv(out_path, index=False, mode="a", header=i == 0)

    # sub_motif_scores_perms = pd.concat(sub_motif_scores_perms, ignore_index=True)
perm_means = pd.read_csv(out_path)


# %% Plot performance
# Sensitivty = true positive rate, i.e. character has edge and they were recalled
# Specificity = true negative rate, i.e. character has no edge and none was recalled
# Precision = true positive rate penalized just mentioning a lot of characters
# Acc = true
xticklabels = [
    "Sensitivity",
    "Specificity",
    "Precision",
    "Accuracy",
    "Balanced\nAccuracy",
    "F1",
]
xorder = ["sensitivity", "specificity", "precision", "acc", "bal_acc", "f1"]

# Get summary stats
agg_sub_motif_scores = pipe(
    sub_motif_scores,
    _.groupby("SID", "metric"),
    _.summarize(mean="score.mean()", std="score.std()"),
    show=True,
)
agg_sub_motif_scores = agg_sub_motif_scores.set_index("metric")

ax = pipe(
    sub_motif_scores,
    _.stripbarplot(
        x="metric", y="score", units="SID", alpha=0.01, pointcolor="black", order=xorder
    ),
    tweak(
        xlabel="",
        ylabel="Score",
        despine=True,
        tight_layout=True,
        xtick_rotation=45,
        xticklabels=xticklabels,
    ),
    annotate_axis(
        xstart=[0, 1, 2, 3, 4, 5],
        y=[1.02] * 6,
        texts=["***"] * 6,
        thickness=2,
        fontsize=20,
        despine=True,
    ),
    ...,
    lambda ax: ax.hlines(
        sub_motif_scores_perms.groupby("metric")["mean"].mean()[xorder].tolist(),
        [-0.4, 0.6, 1.6, 2.6, 3.6, 4.6],
        [0.4, 1.4, 2.4, 3.4, 4.4, 5.4],
        color="black",
        ls="dotted",
    ),
)
savefig(ax.get_figure(), path=FIG_DIR, name="motif_performance")

# %% Nicer plot just accuracy
sub_acc = pipe(
    agg_sub_motif_scores,
    _.query("metric == 'acc'"),
)
sub_perm_acc = pipe(
    perm_means,
    _.query("metric == 'acc'"),
    _.groupby("perm"),
    _.summarize(mean="mean.mean()"),
)
f, ax = plt.subplots(figsize=(4, 6))
ax = sns.boxplot(y="mean", data=sub_acc, width=0.2, color="lightsteelblue", ax=ax)
ax = sns.stripplot(
    y="mean",
    data=sub_acc,
    color="dimgray",
    size=8,
    edgecolor="gray",
    linewidth=2,
    alpha=0.25,
)
ax = sns.stripplot(
    y="mean",
    data=sub_perm_acc,
    color="lightcoral",
    size=8,
    edgecolor="gray",
    linewidth=2,
    alpha=0.25,
)
line = ax.hlines(
    sub_perm_acc["mean"].mean(), -0.1, 0.1, color="black", ls="solid", linewidth=3
)
line.set_zorder(20)

tweak(
    ax, ylim=(0.2, 1.01), xlim=(-0.5, 0.5), despine=True, xlabel="", ylabel="Accuracy"
)
annotate_axis(
    ax,
    xstart=[0],
    y=[1.01],
    texts=["***"],
    thickness=2,
    fontsize=20,
)
custom_lines = [
    Line2D([0], [0], color="lightsteelblue", lw=6, alpha=0.75),
    Line2D([0], [0], color="lightcoral", lw=6, alpha=0.75),
]
plt.legend(
    custom_lines,
    ["Cued Recalls", "Permuted Recalls"],
    fontsize=16,
    loc="center",
    ncol=2,
    bbox_to_anchor=(0.5, 1.1),
    frameon=False,
)
savefig(ax.get_figure(), path=FIG_DIR, name="motif_performance_acc")

# %% Nicer plot just sensitivity and specificity
sub_scores = pipe(
    agg_sub_motif_scores,
    _.query("metric == 'sensitivity' or metric == 'specificity'"),
)
sub_perm_scores = pipe(
    perm_means,
    _.query("metric == 'sensitivity' or metric == 'specificity'"),
    _.groupby("metric", "perm"),
    _.summarize(mean="mean.mean()"),
)
f, ax = plt.subplots(figsize=(4, 6))
ax = sns.boxplot(
    x="metric", y="mean", data=sub_scores, width=0.2, color="lightsteelblue", ax=ax
)
ax = sns.stripplot(
    x="metric",
    y="mean",
    data=sub_scores,
    color="dimgray",
    size=8,
    edgecolor="gray",
    linewidth=2,
    alpha=0.25,
)
ax = sns.stripplot(
    x="metric",
    y="mean",
    data=sub_perm_scores,
    color="lightcoral",
    size=8,
    edgecolor="gray",
    linewidth=2,
    alpha=0.25,
)
tweak(
    ax,
    ylim=(0.0, 1.01),
    despine=True,
    xlabel="",
    ylabel="",
    xticklabels=["Sensitivity", "Specificity"],
)
# %% Nicer plot all 3
sub_scores = pipe(
    agg_sub_motif_scores,
    _.query("metric == 'sensitivity' or metric == 'specificity' or metric == 'acc'"),
)
sub_perm_scores = pipe(
    perm_means,
    _.query("metric == 'sensitivity' or metric == 'specificity' or metric == 'acc'"),
    _.groupby("metric", "perm"),
    _.summarize(mean="mean.mean()"),
)
f, ax = plt.subplots(figsize=(8, 6))
ax = sns.boxplot(
    x="metric", y="mean", data=sub_scores, width=0.2, color="lightsteelblue", ax=ax
)
ax = sns.stripplot(
    x="metric",
    y="mean",
    data=sub_scores,
    color="dimgray",
    size=8,
    edgecolor="gray",
    linewidth=2,
    alpha=0.25,
)
ax = sns.stripplot(
    x="metric",
    y="mean",
    data=sub_perm_scores,
    color="lightcoral",
    size=8,
    edgecolor="gray",
    linewidth=2,
    alpha=0.25,
)

lines = (
    ax.hlines(
        sub_perm_scores.groupby("metric")["mean"].mean().to_list(),
        [-0.13, 0.88, 1.88],
        [0.13, 1.13, 2.13],
        color="black",
        ls="solid",
        linewidth=3,
    ),
)
for l in lines:
    l.set_zorder(20)


ax = tweak(
    ax,
    ylim=(0.2, 1.03),
    despine=True,
    xlabel="",
    ylabel="",
    xticklabels=["Accuracy", "Sensitivity", "Specificity"],
)
annotate_axis(
    ax,
    xstart=[0, 1, 2],
    y=[1.01] * 3,
    texts=["***"] * 3,
    thickness=2,
    fontsize=20,
),
savefig(ax.get_figure(), path=FIG_DIR, name="motif_performance_nicer")
