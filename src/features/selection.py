"""Feature selection and extraction (PCA) for the 1D direction model.

Every function here is meant to be fit on the train split only. Using val or
test to decide which features to keep would leak information about the
evaluation data into the model, the same way fitting a scaler on test would.

Selection order (each step only sees the survivors of the previous one):
  1. near-constant attributes
  2. low-variance numerics
  3. clear noise                (fails both ANOVA and mutual information)
  4. highly correlated pairs    (keep the member with higher MI)
Near-unique attributes only apply to nominal columns like IDs,
which this feature set does not contain.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MAX_VALUE_SHARE = 0.95
MIN_STD = 1e-8
ANOVA_ALPHA = 0.05
CORR_THRESHOLD = 0.85
N_PERMUTATIONS = 20
NOISE_QUANTILE = 0.95
PCA_VARIANCE = 0.90
MANIFEST_PATH = Path(__file__).with_name("selected_features.json")


def near_constant(X: pd.DataFrame, max_share: float = MAX_VALUE_SHARE) -> list[str]:
    """Columns where a single value covers more than `max_share` of the rows."""
    return [c for c in X.columns if X[c].value_counts(normalize=True).iloc[0] > max_share]


def low_variance(X: pd.DataFrame, min_std: float = MIN_STD) -> list[str]:
    """Numeric columns whose standard deviation is effectively zero.

    The threshold is deliberately tiny: features here live on very different scales
    (a daily return has std ~0.03, RSI ~10), so a larger raw-std cutoff would remove
    small-scale features for their units, not for a lack of information.
    """
    return [c for c in X.columns if X[c].std() < min_std]


def _discrete_mask(X: pd.DataFrame) -> np.ndarray:
    """True for integer-valued columns (e.g. is_weekend) so MI uses the discrete estimator."""
    return np.array([pd.api.types.is_integer_dtype(X[c]) for c in X.columns])


def relevance(X: pd.DataFrame, y: pd.Series, n_permutations: int = N_PERMUTATIONS,
              random_state: int = 0) -> pd.DataFrame:
    """Per-feature relevance to the label: ANOVA F/p-value and mutual information.

    ANOVA is the standard test for a continuous feature vs a categorical target.
    MI also catches non-linear relationships ANOVA misses. MI is never exactly 0 on
    finite samples, so it is compared with `mi_noise`: the NOISE_QUANTILE of the MI
    each feature gets against randomly shuffled labels, i.e. what pure chance scores.
    """
    discrete = _discrete_mask(X)
    f_stat, p_value = f_classif(X, y)
    mi = mutual_info_classif(X, y, discrete_features=discrete, random_state=random_state)

    rng = np.random.default_rng(random_state)
    null_mi = np.array([
        mutual_info_classif(X, rng.permutation(y.to_numpy()), discrete_features=discrete,
                            random_state=random_state)
        for _ in range(n_permutations)
    ])
    out = pd.DataFrame({
        "anova_f": f_stat,
        "anova_p": p_value,
        "mi": mi,
        "mi_noise": np.quantile(null_mi, NOISE_QUANTILE, axis=0),
    }, index=X.columns)
    out["is_noise"] = (out["anova_p"] > ANOVA_ALPHA) & (out["mi"] <= out["mi_noise"])
    return out.sort_values("mi", ascending=False)


def correlated_pairs(corr: pd.DataFrame, threshold: float = CORR_THRESHOLD) -> pd.DataFrame:
    """All feature pairs with |correlation| above `threshold`, strongest first."""
    cols = corr.columns
    rows = [
        {"a": cols[i], "b": cols[j], "corr": corr.iloc[i, j]}
        for i in range(len(cols)) for j in range(i + 1, len(cols))
        if abs(corr.iloc[i, j]) > threshold
    ]
    pairs = pd.DataFrame(rows, columns=["a", "b", "corr"])
    return pairs.reindex(pairs["corr"].abs().sort_values(ascending=False).index)


def drop_correlated(corr: pd.DataFrame, mi: pd.Series,
                    threshold: float = CORR_THRESHOLD) -> dict[str, str]:
    """Greedy pass over correlated pairs, strongest first: drop the member with lower MI.

    A pair is skipped once either member is already dropped, so a cluster of three
    mutually correlated features keeps exactly its highest-MI member.
    Returns {dropped_feature: reason}.
    """
    dropped: dict[str, str] = {}
    for a, b, r in correlated_pairs(corr, threshold).itertuples(index=False):
        if a in dropped or b in dropped:
            continue
        keep, drop = (a, b) if mi[a] >= mi[b] else (b, a)
        dropped[drop] = f"correlated with {keep} (rho={r:.2f}), lower MI"
    return dropped


def select_features(X: pd.DataFrame, y: pd.Series, corr_threshold: float = CORR_THRESHOLD,
                    random_state: int = 0) -> dict:
    """Run the full selection pass on train data and return a JSON-ready manifest.

    The manifest lists the kept features (in their original column order) and,
    for every dropped feature, the step and reason that removed it.
    """
    dropped: dict[str, str] = {}
    for col in near_constant(X):
        dropped[col] = f"near-constant (one value > {MAX_VALUE_SHARE:.0%} of rows)"
    for col in low_variance(X.drop(columns=list(dropped))):
        dropped[col] = "low variance (std ~ 0)"

    remaining = X.drop(columns=list(dropped))
    scores = relevance(remaining, y, random_state=random_state)
    for col in scores.index[scores["is_noise"]]:
        dropped[col] = (f"noise: ANOVA p={scores.loc[col, 'anova_p']:.2f} and "
                        f"MI={scores.loc[col, 'mi']:.4f} <= shuffled-label baseline")

    remaining = X.drop(columns=list(dropped))
    corr = remaining.corr(method="spearman")
    dropped.update(drop_correlated(corr, scores["mi"], corr_threshold))

    return {
        "selected": [c for c in X.columns if c not in dropped],
        "dropped": dropped,
        "params": {
            "corr_method": "spearman",
            "corr_threshold": corr_threshold,
            "anova_alpha": ANOVA_ALPHA,
            "noise_quantile": NOISE_QUANTILE,
            "n_permutations": N_PERMUTATIONS,
            "max_value_share": MAX_VALUE_SHARE,
            "fit_on": "train",
            "train_rows": len(X),
        },
    }


def fit_pca(X: pd.DataFrame, variance: float = PCA_VARIANCE) -> Pipeline:
    """Standardize, then keep the fewest principal components explaining `variance`.

    Standardizing first is the same as running PCA on the correlation matrix
    instead of the covariance matrix: otherwise RSI (std ~10) would
    dominate every component purely because of its units.
    """
    return Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=variance))]).fit(X)


def pca_transform(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Project X onto the fitted components, columns named pc1, pc2, ..."""
    values = pipeline.transform(X)
    return pd.DataFrame(values, index=X.index, columns=[f"pc{i + 1}" for i in range(values.shape[1])])


def save_manifest(manifest: dict, path: Path = MANIFEST_PATH) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_selected_features(path: Path = MANIFEST_PATH) -> list[str]:
    """The feature columns chosen by the last selection run (written by notebook 04)."""
    return json.loads(path.read_text(encoding="utf-8"))["selected"]
