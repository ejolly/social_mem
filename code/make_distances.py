# #NOTE: To rerun some of the distance calculations requires annotations csv files which are not included with the repo as they're currently in use for other unpublished lab projects. However, we do include pre-computed distances (as with motifs) which can be used to reproduce the results in the paper.
# %%
# IMPORTS
import numpy as np
import pandas as pd
import networkx as nx
from utils import (
    ANALYSIS_DIR,
    CHAR_LIST,
    DATA_DIR,
)
from joblib import dump
from scipy.spatial.distance import pdist, squareform

# %% Create hand-crafted and motif-based character graphs
def make_social_graph(graph_type):
    """
    Make one of the possible model character graphs and compute communicability between
    characters on that graph. Return the graph and the communicability.

    Args:
        graph_type (str): one of 'family', 'romance', 'frienship', 'football', 'gender',
        'age'

    Returns:
        tuple: (pd.DataFrame, pd.DataFrame)
    """

    # Family and friendships are from motifs!
    graph_types = {
        "football": np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        ),
        "friendship": [
            ("BuddyGarrity", "CoachTaylor"),
            ("CoachTaylor", "JasonStreet"),
            ("CoachTaylor", "TamiTaylor"),
            ("JulieTaylor", "MattSaracen"),
            ("TimRiggins", "TyraCollette"),
            ("TimRiggins", "JasonStreet"),
            ("JasonStreet", "LylaGarrity"),
            ("JasonStreet", "TyraCollette"),
            ("JasonStreet", "MattSaracen"),
            ("MattSaracen", "LandryClarke"),
            ("TyraCollette", "SmashWilliams"),
        ],
        "family": [
            ("CoachTaylor", "TamiTaylor"),
            ("CoachTaylor", "JulieTaylor"),
            ("TamiTaylor", "JulieTaylor"),
            ("BuddyGarrity", "LylaGarrity"),
        ],
    }
    mat = graph_types[graph_type]
    if isinstance(mat, np.ndarray):
        graph = pd.DataFrame(mat, index=CHAR_LIST, columns=CHAR_LIST)
        graph = nx.from_pandas_adjacency(graph)
    else:
        graph = nx.Graph()
        graph.add_nodes_from(CHAR_LIST)
        graph.add_edges_from(mat)

    # Get pairwise communicabiility
    communicability = nx.communicability(graph)

    # Convert it to a dataframe
    df_comm = comm_to_df(communicability)
    df_comm = df_comm.fillna(0)

    df_adj = nx.to_pandas_adjacency(graph)

    return df_adj, df_comm


# %%
# Graph communicability of these relationships
model_list = ["family", "friendship", "football"]

comms = {}
for m in model_list:
    g, c = make_social_graph(m)
    comms[m] = c

# Save em
dump(comms, ANALYSIS_DIR / "models" / "crafted_model_communicabilities.sav")

# %% Character salience similarity: euclidean distance between vectors of length 4
def load_episode(num):
    return pd.read_csv(
        DATA_DIR / "annotations" / f"char_mat_ep{num}.csv", usecols=CHAR_LIST
    )


# %%
# Similarity of frequency of appearance across all 4 episodes
char_salience = pd.DataFrame(
    squareform(
        pdist(
            pd.concat(
                [load_episode(e).sum(axis=0) for e in range(1, 5)],
                axis=1,
            )
        )
    ),
    index=CHAR_LIST,
    columns=CHAR_LIST,
)
dump(char_salience, ANALYSIS_DIR / "models" / "character_salience.sav")

# %% Simple temporal co-occurence
# Condense design matrices into only 11 columns by summing with interactions and then
episodes = pd.concat([load_episode(num) for num in range(1, 5)])
char_cooccur = episodes.corr()
char_cooccur.index = CHAR_LIST
char_cooccur.columns = CHAR_LIST
dump(char_cooccur, ANALYSIS_DIR / "models" / "character_simple_cooccurence.sav")

# %% Trait distance
from utilz import pipe
import utilz.dfverbs as _


def compute_character_distance(sdf):
    """Compute the distances between characters for a given subject using the questions
    within a specific category as features"""
    distances = pdist(sdf.loc[:, "annoy":].to_numpy(), metric="euclidean")
    assert len(distances) == 11 * 10 / 2
    out = pd.DataFrame(squareform(distances), index=CHAR_LIST, columns=CHAR_LIST)
    out["SID"] = sdf.SID.to_numpy()
    return out


# Ratings csv already excluded subjects!
dists = pipe(
    DATA_DIR / "ratings.csv",
    _.read_csv,
    _.pivot_wider("dimension", "rating", "character"),
    _.groupby("SID"),
    _.apply(compute_character_distance),
)
dump(dists, ANALYSIS_DIR / "character_trait_distances.sav")
