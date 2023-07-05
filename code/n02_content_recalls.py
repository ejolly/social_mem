# SUMMARY
#  How often to people use langauge from our hypothesized dimensions in their longer more contentful memories?

# %% Imports and paths
from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from pathlib import Path
from pymer4.models import Lmer
from pymer4.io import load_model, save_model
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import warnings

# lib functions and globals
from utils import (
    DATA_DIR,
    ANALYSIS_DIR,
    FIG_DIR,
    CONTENT_EXCLUSIONS,
    NAME_TO_SCAN,
    POS_MAP,
    CHAR_NAMES,
    CHAR_LIST,
    annotate_axis,
)

from utilz import (
    map,
    mapcat,
    mapwith,
    filter,
    pipe,
    alongwith,
    curry,
    savefig,
    pairs,
    tweak,
    newax,
)
import utilz.dfverbs as _
import spacy

sns.set_context("talk", font_scale=1.2)
sns.set_style("whitegrid")
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Avenir"]

nlp = spacy.load("en_core_web_sm")

OUTPUT_DIR = ANALYSIS_DIR / "n02"
FIG_DIR = FIG_DIR / "n02"

OUTPUT_DIR.mkdir() if not OUTPUT_DIR.exists() else None
FIG_DIR.mkdir() if not FIG_DIR.exists() else None

lemmatize = lambda word: nlp(word.lower())[0].lemma_

# Load data and filter subs
transcripts = pipe(
    DATA_DIR / "transcripts.csv",
    pd.read_csv,
    _.rename({"Subject": "SID"}),
    _.query(lambda df: ~df.SID.isin(CONTENT_EXCLUSIONS)),
)

print(f"Num unique subjects: {transcripts.SID.nunique()}")

# Funcs to estimate, visualize, and save LSA
# %% POS tagging and filtering
@curry
def tag_pos(transcripts: pd.DataFrame) -> pd.DataFrame:

    output_file = OUTPUT_DIR / "transcripts_pos_tagged.csv"
    if output_file.exists():
        print("Loading existing POS tags...")
        return pd.read_csv(output_file)

    name_fixes = {
        "Layla": "Lyla",
        "Connie": "Tami",
        "Britton": "Taylor",
        "Sorenson": "Saracen",
        "Tanner": "Taylor",
        "Riggs": "Riggins",
        "Rigs": "Riggins",
        "Saracin": "Saracen",
        "Landy": "Landry",
        "Tara": "Tyra",
        "Laura": "Lyla",
        "Lyra": "Lyla",
        "Streeter": "Street",
        "Streets": "Street",
        "Jordan": "Jason",
        "Collettte": "Collette",
        "Jim": "Tim",
        "Jimmy": "Tim",
        "Lya": "Lyla",
        "Mike": "Matt",
        "Michael": "Matt",
        "Kyra": "Tyra",
        "Tammy": "Tami",
        "Lila": "Lyla",
        "Jasonas": "Jason",
        "CoachTaylor": "Coach",
    }

    tagged = []

    print("Tagging recalls POS with spacy...")
    for _, row in transcripts.iterrows():
        doc = nlp(row["Text"])

        # keep track of previous word's POS so we can ignore counts for character first
        # names and last names as separate tokens
        prev_pos = None
        for _, word in enumerate(doc):
            if word.is_alpha:
                word_type = "alpha"
            elif word.is_digit:
                word_type = "number"
            elif word.is_bracket:
                word_type = "bracket"
            elif word.is_punct:
                word_type = "punctuation"
            else:
                word_type = "unknown"

            # Fix improper names
            word_text = name_fixes.get(word.text, word.text)

            # Check for character names
            if len(
                filter(
                    word_text.capitalize(),
                    CHAR_NAMES + CHAR_LIST,
                    assert_notempty=False,
                )
            ):
                # Don't double count first and last names
                if prev_pos == "character":
                    word_pos = "duplicate_character"
                elif prev_pos is None:
                    word_pos = "character"
                    prev_pos = "character"
            else:
                word_pos = POS_MAP[word.pos_]
                prev_pos = None

            tagged.append(
                {
                    "SID": row["SID"],
                    "Character": row["Character"],
                    "Text": word_text,
                    "Pos": word_pos,
                    "Tag": word.tag_,
                    "Explanation": spacy.explain(word.tag_),
                    "Sentiment": word.sentiment,
                    "Idx": word.i,
                    "Entity": word.ent_type_,
                    "Type": word_type,
                }
            )
    tagged = pd.DataFrame(tagged)
    tagged.to_csv(output_file, index=False)
    return tagged


@curry
def get_unique_tokens(df: pd.DataFrame) -> tuple[list, list, list]:

    if (OUTPUT_DIR / "nouns.npy").exists():
        print("loading existing unique POS...")
        out = map(
            lambda pos: np.load(OUTPUT_DIR / f"{pos}", allow_pickle=True),
            ["verbs.npy", "nouns.npy", "adjectives.npy"],
        )
        return tuple(out)

    # Setup filters for things we don't want to count
    filters = {
        "Type": ["unknown", "punctuation", "number", "bracket"],
        "Pos": [
            "duplicate_character",
            "auxiliary",
            "adverb",
            "determiner",
            "adposition",
            "pronoun",
            "interjection",
            "subordinating_conjunction",
            "coordinating_conjunction",
            "space",
            "symbol",
            "particle",
            "numeral",
        ],
        "Entity": ["ORDINAL", "QUANTITY", "TIME", "PERCENT"],
        "Tag": ["FW", "MD", "TO", "XX"],
    }
    temp = df.copy()
    for col, ignore_words in filters.items():
        temp = temp.query(f"{col} not in @ignore_words")

    # Combine noun + proper_noun tags since they were vetted above for character names
    temp["Pos"] = temp["Pos"].replace({"proper_noun": "noun"})
    print(f"Removed {df.shape[0] - temp.shape[0]} rows")

    # Get unique POS counts
    unique_pos = temp.groupby("Pos")["Text"].unique().reset_index().explode("Text")
    verbs = unique_pos.query("Pos == 'verb'").Text.unique()
    nouns = unique_pos.query("Pos == 'noun'").Text.unique()
    adjectives = unique_pos.query("Pos == 'adjective'").Text.unique()

    # Save em
    mapwith(
        lambda arr, pos: np.save(OUTPUT_DIR / f"{pos}", arr),
        ["verbs", "nouns", "adjectives"],
        [verbs, nouns, adjectives],
    )
    return verbs, nouns, adjectives


tagged, verbs, nouns, adjectives = pipe(
    transcripts,
    tag_pos,
    alongwith(get_unique_tokens),
    flatten=True,
)


# %% Build vocabularies for each category, lemmatize, and create 1 hot represention
#  of all subject recalls
# NOTE: To project each transcribed content recall into an location/trait embedding
#  space, we first need to learn the vocabulary (i.e. *features*) that make up this
#  space. To do so we first get all the unique *nouns* across all subjects' recalls. We
#  then *manually curated* these based upon location words or known locations in the
#  show. Finally we lemmatize them to collapse over tense, plurals, etc. Then we can
#  look to see which of our location-word lemmas are present in each subjects' recall to
#  represetn the transcript at a vector of length N where N = number of
#  location-related noun lemmas.
#  We do the same using *adjectives* for traits and cross-reference them against several
#  papers in the literature e.g:
#  - https://www.nature.com/articles/s41467-019-10309-7
#  - https://link.springer.com/article/10.1007/s11682-019-00254-w
#  - Mega-analysis of traits: https://academic.oup.com/cercor/article/32/6/1131/6352380#336800867

# NOTE: To project each transcribed content recall into an action-embedding space, we
#  first need to learn the vocabulary (i.e. *features*) that make up this space. To do
#  so we first get all the unique *verbs* across all subjects' recalls and lemmatize
#  them to collapse over tense, plurals, etc. Then we use [Thornton et
#  al](https://osf.io/abzpg/)'s reduced set of verbs which include 1875 unique verb
#  lemmas that have already aggregated/filtered words by semantic similarity and
#  frequency norms. Then we look up to see which of our verb lemmas are included in
#  Thornton's set and throw away the ones that aren't. This should filter down from our
#  full list of verbs to action-related verb lemmas. Then we can represent transcript as
#  a vector of length N where N = number of action-related verb lemmas.


def build_location_vocab(nouns):
    """Get the unique lemmatize location words recall across all Ps"""

    print(f"Unique (unfiltered) NOUN words recalled: {len(nouns)}")

    out_file = OUTPUT_DIR / "location_vocab.csv"
    if not out_file.exists():
        # Manually filtered so we dont actually need nouns
        nouns_filtered = pd.read_csv(OUTPUT_DIR / "location_nouns_curated.csv")
        assert not all(nouns_filtered.isnull().any())
        # lemmatize
        location_vocab = sorted(
            list(set(lemmatize(word) for word in nouns_filtered.iloc[:, 0].tolist()))
        )
        pd.DataFrame(location_vocab).to_csv(out_file, index=False)
    else:
        print("loading existing location vocab...")
        location_vocab = pd.read_csv(out_file).iloc[:, 0].to_list()

    print(f"Final LOCATION vocab size: {len(location_vocab)}")
    return location_vocab


def build_action_vocab(verbs):
    """Get the unique lemmatized action words recall across all Ps"""
    from ast import literal_eval

    print(f"Unique (unfiltered) VERB words recalled: {len(verbs)}")

    out_file = OUTPUT_DIR / "action_vocab.csv"
    if not out_file.exists():
        # Get the unique set of verb lemmas for our words
        # We call spacy in a loop because for some reason passing the entire string of text
        # doesn't always produce accurate lemmatization
        verb_lemmas = sorted(set(map(lemmatize, verbs)))
        # print(f"Unique verb LEMMAs recalled: {len(verb_lemmas)}")

        # Get lemmas from Thornton et al
        verbset = pd.read_csv(DATA_DIR / "Thorntonetal_verbset.csv")
        verb_lookup = verbset.lemma.tolist()
        # Add list of other verb forms in case we can't lemmatize some words
        verb_lookup += map(None, verbset.verbforms.apply(literal_eval).tolist())
        # Uniquify
        verb_lookup = list(set(verb_lookup))
        # print(f"Full action-word set lookup table: {len(verb_lookup)}")

        action_vocab = sorted(filter(verb_lookup, verb_lemmas))
        pd.DataFrame(action_vocab).to_csv(out_file, index=False)
    else:
        print("loading existing action vocab...")
        action_vocab = pd.read_csv(out_file).iloc[:, 0].to_list()
    print(f"Final ACTION vocab size: {len(action_vocab)}")
    return action_vocab


def build_trait_vocab(adjectives):
    """Get the unique lemmatized trait words recall across all Ps"""

    print(f"Unique (unfiltered) ADJECTIVE words recalled: {len(adjectives)}")

    out_file = OUTPUT_DIR / "trait_vocab.csv"
    if not out_file.exists():

        # Manually filtered and load it back up
        adj_filtered = pd.read_csv(OUTPUT_DIR / "trait_adjectives_curated.csv")
        assert not all(adj_filtered.isnull().any())

        # lemmatize
        trait_vocab = sorted(
            list(set(lemmatize(word) for word in adj_filtered.iloc[:, 0].tolist()))
        )
        pd.DataFrame(trait_vocab).to_csv(out_file, index=False)
    else:
        print("loading existing trait vocab...")
        trait_vocab = pd.read_csv(out_file).iloc[:, 0].to_list()
    print(f"Final TRAIT vocab size: {len(location_vocab)}")
    return trait_vocab


def load_character_onehot():
    """Load up annotated recalls and reshape"""

    out_file = OUTPUT_DIR / "character_onehot.csv"
    if out_file.exists():
        print("Exist csv found loading...")
        character_onehot = pd.read_csv(out_file)
    else:
        character_onehot = pipe(
            DATA_DIR / "annotated_recalls.csv",
            pd.read_csv,
            _.pivot_wider("Character", using="Mention"),
            _.rename({"Content": "Character", "Subject": "SID"}),
            _.mutate(SID=lambda SID: SID.map(NAME_TO_SCAN)),
            _.query(lambda df: ~df.SID.isin(CONTENT_EXCLUSIONS)),
            _.to_csv(out_file),
        )

    print(f"Final CHARACTER vocab size: {len(CHAR_LIST)}")
    return character_onehot


@curry
def text2onehot(recall_text, vocab: list | None = None):
    """Create a dict of token counts for each word in vocab in recall_text"""
    words = nlp(recall_text)
    embedding = np.zeros_like(vocab, dtype=int)
    for word in words:
        try:
            idx = vocab.index(word.lemma_)
            embedding[idx] += 1
        except ValueError as e:
            continue
    return dict(zip(vocab, embedding))


@curry
def onehot2frame(onehot: list, fname: str) -> pd.DataFrame:
    """Convert list of dicts to dataframe"""
    out = pd.DataFrame(onehot)
    out.insert(0, "SID", transcripts["SID"])
    out.insert(1, "Character", transcripts["Character"])
    out.to_csv(OUTPUT_DIR / f"{fname}.csv", index=False)
    return out


# Build up lemmatize vocabs to compare recalls against
location_vocab = build_location_vocab(nouns)
action_vocab = build_action_vocab(verbs)
trait_vocab = build_trait_vocab(adjectives)
# These were manually annotated so we just loaded them up
character_onehot = load_character_onehot()

vocabs = dict(
    Place=location_vocab,
    Action=action_vocab,
    Trait=trait_vocab,
    Character=character_onehot,
)

# %% LSA helpers


@curry
def fit_lsa(transcripts, vocab=None, n_components=0.95):

    # For actions, traits, and locations
    if isinstance(vocab, list):
        corpus = transcripts["Text"].squeeze()
        vectorizer = TfidfVectorizer(vocabulary=vocab)
        tfidf_mat = vectorizer.fit_transform(corpus)
        transformer = vectorizer

    elif isinstance(vocab, pd.DataFrame):
        # For characters
        transformer = TfidfTransformer()
        tfidf_mat = transformer.fit_transform(vocab.iloc[:, 2:]).toarray()

    else:
        raise TypeError(
            f"vocab must be either a list or pd.DataFrame. Received {type(vocab)}"
        )

    if isinstance(n_components, float):
        # Perform LSA but first solving for n_components where explained variance is >= 95%
        svd = TruncatedSVD(n_components=np.min(tfidf_mat.shape) - 1, random_state=0)
        fit = svd.fit(tfidf_mat)
        var_explained = np.cumsum(svd.explained_variance_ratio_)
        n_components = np.argwhere(var_explained >= n_components).min()
        print(n_components)

        # Refit with optimal n_components
        svd = TruncatedSVD(n_components=n_components, random_state=0)
        svd = svd.fit(tfidf_mat)

    elif isinstance(n_components, int):
        svd = TruncatedSVD(n_components=n_components)
        svd = svd.fit(tfidf_mat)
    else:
        raise TypeError(
            f"n_components must be either a float or int. Received {type(n_components)}"
        )

    return svd, transformer


@curry
def make_cv_embeddings(vocab):

    # First estimate the number of components that retain 95% variance across everyone
    # We need to do this so that each subject's embeddings have the same dimensionality
    svd, transformer = fit_lsa(transcripts, vocab, n_components=0.95)
    n_components = svd.components_.shape[0]

    # Now do cv with that many components
    subjects = transcripts.SID.unique()
    out = []

    # filter user warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        for subject in subjects:

            train_transcripts = transcripts.query("SID != @subject")
            test_transcripts = transcripts.query("SID == @subject")

            # For characters the vocab is already one-hot encoded
            if isinstance(vocab, pd.DataFrame):
                train_vocab = vocab.query("SID != @subject")
                test_vocab = vocab.query("SID == @subject")
            else:
                train_vocab = test_vocab = vocab

            # Compute embeddings for everyone excluding left out sub
            svd, transformer = fit_lsa(
                train_transcripts, train_vocab, n_components=n_components
            )

            # Project this sub's memories into the learned space and save as character x
            # component dataframe
            if isinstance(vocab, list):
                leftout_tfidf = transformer.transform(
                    test_transcripts["Text"].squeeze()
                )
            # characters are already annotated this way
            else:
                leftout_tfidf = transformer.transform(test_vocab.iloc[:, 2:].to_numpy())

            leftout_sub_embeddings = svd.transform(leftout_tfidf)
            leftout_sub_embeddings = pd.DataFrame(
                leftout_sub_embeddings,
                columns=[f"C{i+1}" for i in range(leftout_sub_embeddings.shape[1])],
            )

            leftout_sub_embeddings.insert(0, "SID", test_transcripts.SID.to_list())
            leftout_sub_embeddings.insert(
                1, "Character", test_transcripts.Character.to_list()
            )
            out.append(leftout_sub_embeddings)

    return pd.concat(out, axis=0, ignore_index=True)


@curry
def plot_transformed(model_and_dtm, **kwargs):
    lda, dtm = model_and_dtm
    ax = sns.heatmap(
        lda.components_,
        vmin=0,
        cmap="Blues",
        ax=newax(),
        yticklabels=[f"topic_{i}" for i in range(lda.components_.shape[0])],
    )
    return tweak(ax, **kwargs)


@curry
def plot_component_loadings(model_and_dtm, num=None, col_wrap=5, title="", **kwargs):
    lda, dtm = model_and_dtm
    topic_term = lda.components_
    topic_term = topic_term[:, :num] if num is not None else topic_term
    comp_num = topic_term.shape[0]

    num_rows = comp_num // col_wrap
    num_rows = num_rows + 1 if comp_num % col_wrap else num_rows

    f, axs = plt.subplots(
        num_rows,
        col_wrap,
        figsize=(col_wrap * 4, num_rows * col_wrap),
        sharex=True,
        sharey=False,
    )

    for i in range(comp_num):
        idx = np.argsort(topic_term[i])[::-1]
        sorted_loadings = topic_term[i][idx]
        ax = sns.barplot(x=sorted_loadings, y=dtm.columns[idx], ax=axs.flat[i])
        ax = tweak(ax, title=f"Topic {i}", **kwargs)

    plt.suptitle(f"{title.capitalize()} {comp_num}d\n(top {num})")
    plt.tight_layout()
    return f


@curry
def calc_pair_sim(
    df_pair: tuple,
    category=None,
    method="pearson",
) -> pd.DataFrame:
    """Compute the similarity between 2 character x dimension dataframes either by
    character or by column (LSA dimension)"""

    df1, df2 = df_pair
    s1, s2 = df1.SID.unique()[0], df2.SID.unique()[0]

    # This returns a series with the index as characters or components
    sims = (
        df1.drop("SID", axis=1)
        .set_index("Character")
        .corrwith(
            df2.drop("SID", axis=1).set_index("Character"),
            method=method,
            axis=1,
        )
    )
    sims = (
        sims.to_frame()
        .rename(columns={0: "Similarity"})
        .T.assign(S_pair=f"{s1}_{s2}", Category=category)
        .pivot_longer(id_vars=["S_pair", "Category"], into=("Character", "Similarity"))
    )

    # Some recalls have no mentions of other characters so we get back nans for the
    # similarities and it screws up things like error bar computation
    return sims.fillna(0.0)


@curry
def calc_pair_sim_fix_d(
    df_pair: tuple, category=None, method="pearson", ndim=9
) -> pd.DataFrame:
    """Compute the similarity between 2 character x dimension dataframes either by
    character or by column (LSA dimension) but fixing the LSA dimension to ndim"""

    df1, df2 = df_pair
    s1, s2 = df1.SID.unique()[0], df2.SID.unique()[0]

    # This returns a series with the index as characters or components
    sims = (
        df1.drop("SID", axis=1)
        .set_index("Character")
        .iloc[:, :ndim]
        .corrwith(
            df2.drop("SID", axis=1).set_index("Character"),
            method=method,
            axis=1,
        )
    )
    sims = (
        sims.to_frame()
        .rename(columns={0: "Similarity"})
        .T.assign(S_pair=f"{s1}_{s2}", Category=category)
        .pivot_longer(id_vars=["S_pair", "Category"], into=("Character", "Similarity"))
    )

    # Some recalls have no mentions of other characters so we get back nans for the
    # similarities and it screws up things like error bar computation
    return sims.fillna(0.0)


# %% Compute CV-embeddings and the pairwise similarity between them in each embedding space

out_file = OUTPUT_DIR / "cv_embedding_sims.csv"
if out_file.exists():
    print("Loading existing embeddings and similarities")
    sims = pd.read_csv(out_file)
    # embeddings = load(OUTPUT_DIR / "cv_embeddings.h5")
else:
    embeddings = {}
    sims = []
    for name, vocab in tqdm(vocabs.items()):

        embeddings[name] = make_cv_embeddings(vocab)

        embedding_sims = pipe(
            embeddings[name],
            _.groupby("SID"),
            _.split_groups(),
            pairs(),
            mapcat(calc_pair_sim(category=name)),
        )
        sims.append(embedding_sims)

    sims = pd.concat(sims, ignore_index=True)
    sims.to_csv(out_file, index=False)
    dump(embeddings, OUTPUT_DIR / "cv_embeddings.h5")
# %% Compute CV-embeddings with fixed dimensionality

out_file = OUTPUT_DIR / "cv_embedding_sims_fix_d.csv"
if out_file.exists():
    print("Loading existing embeddings and similarities")
    sims_fix_d = pd.read_csv(out_file)
    # embeddings_fix_d = load(OUTPUT_DIR / "cv_embeddings_fix_d.h5")
else:
    embeddings_fix_d = {}
    sims_fix_d = []
    for name, vocab in tqdm(vocabs.items()):

        embeddings_fix_d[name] = make_cv_embeddings(vocab)

        embedding_sims_fix_d = pipe(
            embeddings_fix_d[name],
            _.groupby("SID"),
            _.split_groups(),
            pairs(),
            mapcat(calc_pair_sim_fix_d(category=name)),
        )
        sims_fix_d.append(embedding_sims_fix_d)

    sims_fix_d = pd.concat(sims_fix_d, ignore_index=True)
    sims_fix_d.to_csv(out_file, index=False)
    dump(embeddings_fix_d, OUTPUT_DIR / "cv_embeddings_fix_d.h5")

# %% Plot original

pipe(
    sims,
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
        order=["Actions", "Traits", "Locations", "People"],
        hue_order=["Actions", "Traits", "Locations", "People"],
        legend=False,
        n_boot=100,
        dodge=False,
        pointcolor="black",
        alpha=0.005,
        xlabel="",
        palette=sns.color_palette("Set2"),
    ),
    tweak(
        title="",
        ylabel="Mnemonic Convergence\n(mean CV correlation)",
        despine=True,
        ylim=(0, 1),
    ),
    # add significance labels
    annotate_axis(
        xstart=[0, 0, 1, 1, 2],
        xend=[2, 3, 2, 3, 3],
        y=[0.4, 0.7, 0.33, 0.8, 0.87],
        texts=["***"] * 5,
        thickness=2,
        fontsize=20,
        despine=True,
    ),
    savefig(path=FIG_DIR, name="content_recall_similarity_cv"),
)

# %% Plot fix-d
pipe(
    sims_fix_d,
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
        order=["Actions", "Traits", "Locations", "People"],
        hue_order=["Actions", "Traits", "Locations", "People"],
        legend=False,
        n_boot=100,
        dodge=False,
        pointcolor="black",
        alpha=0.005,
        xlabel="",
        palette=sns.color_palette("Set2"),
    ),
    tweak(
        title="",
        ylabel="Mnemonic Convergence\n(mean CV correlation)",
        despine=True,
        ylim=(0, 1),
    ),
    # add significance labels
    annotate_axis(
        xstart=[0, 0, 0, 1, 1, 2],
        xend=[1, 2, 3, 2, 3, 3],
        y=[0.6, 0.52, 0.7, 0.42, 0.8, 0.87],
        texts=["***"] * 6,
        thickness=2,
        fontsize=20,
        despine=True,
    ),
    savefig(path=FIG_DIR, name="content_recall_similarity_cv_fix_d"),
)
# %% Run lmer original


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


sim_model, sim_comparisons = pipe(
    sims,
    # Fit and save model or load exisint
    fit_sim_lmer(
        model_file=OUTPUT_DIR / "cv_contentsim_lmer.h5",
        formula="Similarity ~ Category + (Category | sub1) + (Category | sub2)",
        factors={"Category": ["Character", "Action", "Place", "Trait"]},
    ),
    # Run pairwise comparisons
    alongwith(run_sim_posthoc(marginal_vars="Category")),
    show=True,
)


# %% Run lmer fix-d
sim_model_fix_d, sim_comparisons_fix_d = pipe(
    sims_fix_d,
    # Fit and save model or load exisint
    fit_sim_lmer(
        model_file=OUTPUT_DIR / "cv_contentsim_lmer_fix_d.h5",
        formula="Similarity ~ Category + (Category | sub1) + (Category | sub2)",
        factors={"Category": ["Character", "Action", "Place", "Trait"]},
    ),
    # Run pairwise comparisons
    alongwith(run_sim_posthoc(marginal_vars="Category")),
    show=True,
)

# %% How does mnemonic convergence change with increasing dimensionality?

out_file = OUTPUT_DIR / "embedding_dim_explore.csv"
if out_file.exists():
    print("Loading precomputed embedding explorations")
    sims_d_explore = pd.read_csv(out_file)
else:
    sims_d_explore = []
    for name, embeddings in tqdm(embeddings_fix_d.items()):
        for i in tqdm(range(5, embeddings.shape[1], 10)):
            embedding_sims_d_explore = pipe(
                embeddings,
                _.groupby("SID"),
                _.split_groups(),
                pairs(),
                mapcat(calc_pair_sim_fix_d(category=name, ndim=i)),
            )
            embedding_sims_d_explore["dimensionality"] = i
            sims_d_explore.append(embedding_sims_d_explore)
    sims_d_explore = pd.concat(sims_d_explore, ignore_index=True)
    sims_d_explore.to_csv(out_file)


# %% Plot it
pipe(
    sims_d_explore,
    _.assign(
        Category=lambda df: df.Category.map(
            {
                "Place": "Locations",
                "Action": "Actions",
                "Trait": "Traits",
                "Character": "People",
            }
        )
    ),
    _.lineplot(
        x="dimensionality",
        y="Similarity",
        hue="Category",
        hue_order=["Actions", "Traits", "Locations", "People"],
        palette=sns.color_palette("Set2"),
    ),
    tweak(
        title="",
        ylabel="Mnemonic Convergence\n(mean CV correlation)",
        xlabel="Embedding Dimensionality",
        despine=True,
        ylim=(0, 1),
    ),
    savefig(path=FIG_DIR, name="embedding_dim_explore"),
)
