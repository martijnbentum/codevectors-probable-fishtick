import numpy as np

from ci_analysis import sort_w2v2_model_names
import locations

CB_MATRIX_DIR = locations.data / 'cb_matrices'


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_cb_matrix(model_name):
    """Load the codebook matrix (640, 128) for a model."""
    return np.load(CB_MATRIX_DIR / f'{model_name}.npy')


# ---------------------------------------------------------------------------
# Per-codevector distances between two models
# ---------------------------------------------------------------------------

def codevector_l2(model_a, model_b):
    """Per-codevector L2 distance between two models. Returns array (640,)."""
    a, b = load_cb_matrix(model_a), load_cb_matrix(model_b)
    return np.linalg.norm(a - b, axis=1)


def codevector_cosine_distance(model_a, model_b):
    """Per-codevector cosine distance (1 − similarity). Returns array (640,) in [0, 2]."""
    a, b = load_cb_matrix(model_a), load_cb_matrix(model_b)
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return 1.0 - (a_n * b_n).sum(axis=1)


def _drift_fn(metric):
    return codevector_l2 if metric == 'l2' else codevector_cosine_distance


# ---------------------------------------------------------------------------
# Intra-codebook distance (reference scale for interpreting L2 drift)
# ---------------------------------------------------------------------------

def mean_intra_distance(model_name):
    """Mean pairwise L2 distance between all codevectors within the codebook.

    Use as a reference scale: drift / mean_intra_distance gives a value in [0, 1]
    where 1 means a codevector has moved by a full inter-codevector step.
    """
    a = load_cb_matrix(model_name)
    norms_sq = (a ** 2).sum(axis=1)
    gram = a @ a.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * gram
    np.fill_diagonal(dist_sq, 0.0)
    dist = np.sqrt(np.maximum(dist_sq, 0.0))
    n = len(a)
    return float(dist.sum() / (n * (n - 1)))


# ---------------------------------------------------------------------------
# Mean drift scalar
# ---------------------------------------------------------------------------

def mean_drift(model_a, model_b, metric='l2'):
    """Mean per-codevector drift between two models. metric: 'l2' or 'cosine'."""
    return float(_drift_fn(metric)(model_a, model_b).mean())


# ---------------------------------------------------------------------------
# Drift trajectory over training checkpoints
# ---------------------------------------------------------------------------

def drift_trajectory(models, reference='first', metric='l2'):
    """Mean codevector drift vs a reference checkpoint at each training step.

    reference: 'first', 'last', or a model name
    Returns (values, models) — parallel arrays.
    """
    if reference == 'first':
        ref = models[0]
    elif reference == 'last':
        ref = models[-1]
    else:
        ref = reference
    values = [mean_drift(ref, m, metric=metric) for m in models]
    return np.array(values), list(models)


# ---------------------------------------------------------------------------
# Consecutive-step drift matrix  (per-codevector × checkpoint-transition)
# ---------------------------------------------------------------------------

def consecutive_drift_matrix(models, metric='l2'):
    """Per-codevector drift between each pair of consecutive checkpoints.

    Returns (matrix, models) where matrix shape is (640, len(models)-1).
    Column j is the per-codevector drift from models[j] to models[j+1].
    """
    fn = _drift_fn(metric)
    cols = [fn(a, b) for a, b in zip(models[:-1], models[1:])]
    return np.column_stack(cols), list(models)


# ---------------------------------------------------------------------------
# Usage-weighted drift  (requires CI store)
# Assumes store index layout matches the matrix rows: 0-319 = first codebook,
# 320-639 = second codebook.
# ---------------------------------------------------------------------------

def _weighted_drift_from_counts(drifts, weights):
    total = weights.sum()
    if total == 0:
        return float(drifts.mean())
    return float((drifts * weights / total).sum())


def weighted_drift(store, model_a, model_b, metric='l2'):
    """Usage-weighted mean codevector drift between two models.

    Weights each codevector's drift by its total usage count (summed over all
    phones) in model_a.
    """
    drifts = _drift_fn(metric)(model_a, model_b)
    weights = store.get(model=model_a).sum(axis=0).astype(float)
    return _weighted_drift_from_counts(drifts, weights)


def weighted_drift_trajectory(store, models=None, reference='first', metric='l2'):
    """Usage-weighted drift vs reference checkpoint at each training step.

    Returns (values, models) — parallel arrays.
    """
    if models is None:
        models = sort_w2v2_model_names(store.models)
    if reference == 'first':
        ref = models[0]
    elif reference == 'last':
        ref = models[-1]
    else:
        ref = reference
    values = [weighted_drift(store, ref, m, metric=metric) for m in models]
    return np.array(values), list(models)


# ---------------------------------------------------------------------------
# Phone-specific weighted drift
# ---------------------------------------------------------------------------

def phone_weighted_drift(store, model_a, model_b, phone, metric='l2'):
    """Drift weighted by usage of a specific phone in model_a."""
    drifts = _drift_fn(metric)(model_a, model_b)
    weights = store.get(model=model_a, phone=phone).astype(float)
    return _weighted_drift_from_counts(drifts, weights)


def phone_weighted_drift_trajectory(store, models=None, reference='first',
                                     phone=None, metric='l2'):
    """Phone-weighted drift vs reference at each training step.

    Returns (values, models) — parallel arrays.
    """
    if models is None:
        models = sort_w2v2_model_names(store.models)
    if reference == 'first':
        ref = models[0]
    elif reference == 'last':
        ref = models[-1]
    else:
        ref = reference
    values = [phone_weighted_drift(store, ref, m, phone=phone, metric=metric)
              for m in models]
    return np.array(values), list(models)


def phones_weighted_drift(store, model_a, model_b, phones=None, metric='l2'):
    """Phone-weighted drift for each phone in phones.

    phones: list of phone labels, or None to use all phones in the store.
    Returns dict {phone: drift_value}.
    """
    if phones is None:
        phones = store.phones
    drifts = _drift_fn(metric)(model_a, model_b)
    return {
        p: _weighted_drift_from_counts(drifts,
                                       store.get(model=model_a, phone=p).astype(float))
        for p in phones
    }


def phones_weighted_drift_trajectory(store, models=None, reference='first',
                                      phones=None, metric='l2'):
    """Phone-weighted drift trajectory for each phone.

    Returns dict {phone: (values, models)}.
    """
    if models is None:
        models = sort_w2v2_model_names(store.models)
    if phones is None:
        phones = store.phones
    return {
        p: phone_weighted_drift_trajectory(store, models=models, reference=reference,
                                           phone=p, metric=metric)
        for p in phones
    }
