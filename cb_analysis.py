import numpy as np

import ci_store
from ci_analysis import sort_w2v2_model_names

CB_SLICES = {None: slice(None), 1: slice(0, 320), 2: slice(320, 640)}


# ---------------------------------------------------------------------------
# Per-codevector distances between two models
# ---------------------------------------------------------------------------

def load_cb_matrix(model_name):
    return ci_store.load_codebook_matrix(model_name)

def codevector_l2(model_a, model_b):
    """Per-codevector L2 distance between two models. Returns array (640,)."""
    a, b = load_cb_matrix(model_a), load_cb_matrix(model_b)
    return np.linalg.norm(a - b, axis=1)


def codevector_cosine_distance(model_a, model_b):
    """Per-codevector cosine distance (1 − similarity). 
    Returns array (640,) in [0, 2]."""
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

def mean_drift(model_a, model_b, metric='l2', codebook=None):
    """Mean per-codevector drift between two models. metric: 'l2' or 'cosine'."""
    s = CB_SLICES[codebook]
    return float(_drift_fn(metric)(model_a, model_b)[s].mean())


# ---------------------------------------------------------------------------
# Drift trajectory over training checkpoints
# ---------------------------------------------------------------------------

def drift_trajectory(models, reference='first', metric='l2', codebook=None):
    """Mean codevector drift vs a reference checkpoint at each training step.

    reference: 'first', 'last', 'previous', or a model name
    codebook:  None (all 640 codes), 1 (codes 0–319), or 2 (codes 320–639)
    Returns (values, models) — parallel arrays.
    """
    if reference == 'previous':
        values = [0.0] + [mean_drift(a, b, metric=metric, codebook=codebook)
                          for a, b in zip(models[:-1], models[1:])]
        return np.array(values), list(models)
    if reference == 'first':
        ref = models[0]
    elif reference == 'last':
        ref = models[-1]
    else:
        ref = reference
    values = [mean_drift(ref, m, metric=metric, codebook=codebook) for m in models]
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

def phone_weighted_drift(store, model_a, model_b, phone, metric='l2', codebook=None):
    """Drift weighted by usage of a specific phone in model_a.

    codebook: None (all 640 codes), 1 (codes 0–319), or 2 (codes 320–639).
    """
    s = CB_SLICES[codebook]
    drifts = _drift_fn(metric)(model_a, model_b)[s]
    weights = store.get(model=model_a, phone=phone).astype(float)[s]
    return _weighted_drift_from_counts(drifts, weights)


def phone_weighted_drift_trajectory(store, models=None, reference='first',
                                     phone=None, metric='l2', codebook=None):
    """Phone-weighted drift vs reference at each training step.

    reference: 'first', 'last', 'previous', or a model name
    codebook:  None (all 640 codes), 1 (codes 0–319), or 2 (codes 320–639)
    Returns (values, models) — parallel arrays.
    """
    if models is None:
        models = sort_w2v2_model_names(store.models)
    if reference == 'previous':
        values = [0.0] + [phone_weighted_drift(store, a, b, phone=phone,
                                               metric=metric, codebook=codebook)
                          for a, b in zip(models[:-1], models[1:])]
        return np.array(values), list(models)
    if reference == 'first':
        ref = models[0]
    elif reference == 'last':
        ref = models[-1]
    else:
        ref = reference
    values = [phone_weighted_drift(store, ref, m, phone=phone,
                                   metric=metric, codebook=codebook)
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


# ---------------------------------------------------------------------------
# Drift × usage correlation
# ---------------------------------------------------------------------------

def _pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def drift_usage_vectors(store, model_a, model_b, metric='l2', phone=None):
    """Per-codevector drift and usage count arrays.

    phone: restrict usage to one phone; None uses all phones combined.
    Returns (drift, usage) — parallel float arrays of length n_codes.
    """
    drift = _drift_fn(metric)(model_a, model_b)
    if phone is not None:
        usage = store.get(model=model_a, phone=phone).astype(float)
    else:
        usage = store.get(model=model_a).sum(axis=0).astype(float)
    return drift, usage


def drift_usage_correlation(store, model_a, model_b, metric='l2', phone=None):
    """Pearson and Spearman correlation between per-codevector drift and usage.

    Usage is measured in model_a (the source checkpoint).
    phone: restrict usage to one phone; None uses all phones combined.
    Returns dict with keys 'drift', 'usage', 'pearson', 'spearman'.
    """
    drift, usage = drift_usage_vectors(store, model_a, model_b,
                                       metric=metric, phone=phone)
    return {
        'drift': drift,
        'usage': usage,
        'pearson': _pearson(drift, usage),
        'spearman': _spearman(drift, usage),
    }


def drift_usage_correlation_trajectory(store, models=None, reference='first',
                                        metric='l2', phone=None):
    """Pearson and Spearman r between drift and usage at each training step.

    reference: 'first', 'last', 'previous', or a model name.
    phone:     restrict usage to one phone; None uses all phones combined.
    Returns dict with keys 'pearson', 'spearman', 'models' — parallel arrays.
    """
    if models is None:
        models = sort_w2v2_model_names(store.models)
    if not models:
        raise ValueError("models list is empty")

    if reference == 'previous':
        pairs = list(zip(models[:-1], models[1:]))
        pearson = [float('nan')] + [
            drift_usage_correlation(store, a, b, metric=metric, phone=phone)['pearson']
            for a, b in pairs
        ]
        spearman = [float('nan')] + [
            drift_usage_correlation(store, a, b, metric=metric, phone=phone)['spearman']
            for a, b in pairs
        ]
    else:
        if reference == 'first':
            ref = models[0]
        elif reference == 'last':
            ref = models[-1]
        else:
            ref = reference
            if ref not in models:
                raise ValueError(
                    f"reference {ref!r} is not in models; available: {list(models)}"
                )
        results = [drift_usage_correlation(store, ref, m, metric=metric, phone=phone)
                   for m in models]
        pearson = [r['pearson'] for r in results]
        spearman = [r['spearman'] for r in results]

    return {
        'pearson': np.array(pearson),
        'spearman': np.array(spearman),
        'models': list(models),
    }


# ---------------------------------------------------------------------------
# Cross-codebook geometry  (cb1 vs cb2 distance structure)
# ---------------------------------------------------------------------------

def cross_codebook_distances(model_name):
    """Pairwise L2 distance matrix between cb1 and cb2 codevectors.

    Returns an array of shape (320, 320) where entry [i, j] is the L2 distance
    between cb1 codevector i and cb2 codevector j.
    """
    a = load_cb_matrix(model_name)
    cb1, cb2 = a[CB_SLICES[1]], a[CB_SLICES[2]]
    diff = cb1[:, None, :] - cb2[None, :, :]
    return np.linalg.norm(diff, axis=2)


def cross_codebook_geometry_trajectory(models, normalize=False):
    """Mean and nearest-neighbour cross-codebook distance at each training step.

    normalize: if True, divide by mean_intra_distance of the first model so
               values are in units of the average within-codebook codevector spacing.
    Returns dict with keys 'mean_cross', 'mean_nn', 'models'.
    """
    scale = mean_intra_distance(models[0]) if normalize else 1.0
    mean_cross, mean_nn = [], []
    for m in models:
        d = cross_codebook_distances(m)
        mean_cross.append(float(d.mean()) / scale)
        mean_nn.append(float(d.min(axis=1).mean()) / scale)
    return {
        'mean_cross': np.array(mean_cross),
        'mean_nn': np.array(mean_nn),
        'models': list(models),
    }


# ---------------------------------------------------------------------------
# Drift difference  (cb1 drift minus cb2 drift)
# ---------------------------------------------------------------------------

def drift_difference_trajectory(store=None, models=None, reference='first',
                                 phone=None, metric='l2'):
    """cb1 drift minus cb2 drift at each training step.

    Positive values mean cb1 drifted more than cb2 from the reference.
    If phone is given, uses phone-weighted drift; otherwise uses raw mean drift.
    If store is None, phone must also be None (raw drift only).

    Returns (diff_values, models) — parallel arrays.
    """
    if models is None:
        if store is not None:
            models = sort_w2v2_model_names(store.models)
        else:
            raise ValueError("models must be provided when store is None")
    if not models:
        raise ValueError("models list is empty")

    if phone is not None:
        v1, _ = phone_weighted_drift_trajectory(store, models=models,
                                                reference=reference, phone=phone,
                                                metric=metric, codebook=1)
        v2, _ = phone_weighted_drift_trajectory(store, models=models,
                                                reference=reference, phone=phone,
                                                metric=metric, codebook=2)
    else:
        v1, _ = drift_trajectory(models, reference=reference, metric=metric, codebook=1)
        v2, _ = drift_trajectory(models, reference=reference, metric=metric, codebook=2)
    return v1 - v2, list(models)


# ---------------------------------------------------------------------------
# Phone-pair geometry
# ---------------------------------------------------------------------------

def phone_pair_distance(store, model_name, phone1, phone2, codebook=None):
    """Expected L2 distance between codevectors drawn from phone1 and phone2 distributions.

    Computes E[||cv_i - cv_j||] where i ~ P(code | phone1) and j ~ P(code | phone2).
    Geometry-aware: two phones can diverge in code selection but still sit close
    in vector space if their preferred codes are nearby.
    codebook: None (all 640 codes), 1 (codes 0–319), or 2 (codes 320–639).
    Returns a scalar.
    """
    s = CB_SLICES[codebook]
    cb = load_cb_matrix(model_name)[s]
    p1 = store.get(model=model_name, phone=phone1).astype(float)[s]
    p2 = store.get(model=model_name, phone=phone2).astype(float)[s]
    t1, t2 = p1.sum(), p2.sum()
    if t1 == 0:
        raise ValueError(f"no counts for model={model_name!r}, phone={phone1!r}")
    if t2 == 0:
        raise ValueError(f"no counts for model={model_name!r}, phone={phone2!r}")
    p1, p2 = p1 / t1, p2 / t2
    norms_sq = (cb ** 2).sum(axis=1)
    gram = cb @ cb.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * gram
    dist = np.sqrt(np.maximum(dist_sq, 0.0))
    return float((p1[:, None] * p2[None, :] * dist).sum())


def phone_pair_distance_trajectory(store, phone1, phone2, models=None, codebook=None):
    """Expected inter-phone codevector distance at each training step.

    codebook: None (all 640 codes), 1 (codes 0–319), or 2 (codes 320–639).
    Returns (values, models) — parallel arrays.
    """
    if models is None:
        models = sort_w2v2_model_names(store.models)
    if not models:
        raise ValueError("models list is empty")
    values = [phone_pair_distance(store, m, phone1, phone2, codebook=codebook)
              for m in models]
    return np.array(values), list(models)
