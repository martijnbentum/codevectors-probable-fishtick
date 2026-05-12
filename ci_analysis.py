import numpy as np

def sort_w2v2_model_names(model_names):
    '''Sort model names by training checkpoint.'''
    return sorted(model_names, key=lambda n: int(n.split('-')[-1]))

def entropy(counts, base=2):
    '''Compute entropy from counts.
    counts:             iterable of non-negative counts
    base:               logarithm base (default bits)
    '''
    total = counts.sum()
    if total == 0: return 0.0
    p = counts[counts > 0] / total
    return -(p * np.log(p) / np.log(base)).sum()


# ---------------------------------------------------------------------------
# Probability distributions
# ---------------------------------------------------------------------------

def ci_pdf_for_model_phone(store, model, phone, smoothing=1e-10):
    """P(code | model, phone) as a normalized array of length n_codes."""
    counts = store.get(model=model, phone=phone).astype(float) + smoothing
    return counts / counts.sum()


def ci_pdf_for_model(store, model, smoothing=1e-10):
    """P(code | model) collapsed over all phones."""
    counts = store.get(model=model).sum(axis=0).astype(float) + smoothing
    return counts / counts.sum()


def ci_pdfs_for_all_phones_model(store, model, smoothing=1e-10):
    """Return dict {phone: distribution array} for every phone in the store."""
    return {p: ci_pdf_for_model_phone(store, model, p, smoothing) for p in store.phones}


# ---------------------------------------------------------------------------
# Divergence primitives
# ---------------------------------------------------------------------------

def kl_divergence(p, q):
    """KL(P || Q) in bits. Assumes no zero entries in Q where P > 0."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits (symmetric, range [0, 1])."""
    m = 0.5 * (np.asarray(p) + np.asarray(q))
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


# ---------------------------------------------------------------------------
# Model-level divergence
# ---------------------------------------------------------------------------

def model_kl(store, model_a, model_b, phone=None, smoothing=1e-10):
    """KL(model_a || model_b). Pass phone to restrict to one phone."""
    if phone is not None:
        p = ci_pdf_for_model_phone(store, model_a, phone, smoothing)
        q = ci_pdf_for_model_phone(store, model_b, phone, smoothing)
    else:
        p = ci_pdf_for_model(store, model_a, smoothing)
        q = ci_pdf_for_model(store, model_b, smoothing)
    return kl_divergence(p, q)


def model_js(store, model_a, model_b, phone=None, smoothing=1e-10):
    """JS divergence between two models (or model/phone pairs)."""
    if phone is not None:
        p = ci_pdf_for_model_phone(store, model_a, phone, smoothing)
        q = ci_pdf_for_model_phone(store, model_b, phone, smoothing)
    else:
        p = ci_pdf_for_model(store, model_a, smoothing)
        q = ci_pdf_for_model(store, model_b, smoothing)
    return js_divergence(p, q)


# ---------------------------------------------------------------------------
# Pairwise divergence matrix
# ---------------------------------------------------------------------------

def pairwise_divergence_matrix(store, models=None, phone=None,
                               divergence='js', smoothing=1e-10):
    """
    Compute an N×N pairwise divergence matrix for a list of models.

    divergence:     'js' (symmetric, default) or 'kl' (KL(i||j))
    Returns (matrix, models) where models is the ordered list used.
    """
    if models is None:
        models = store.models
    n = len(models)

    if phone is not None:
        dists = {m: ci_pdf_for_model_phone(store, m, phone, smoothing) for m in models}
    else:
        dists = {m: ci_pdf_for_model(store, m, smoothing) for m in models}

    fn = js_divergence if divergence == 'js' else kl_divergence
    mat = np.zeros((n, n))
    for i, mi in enumerate(models):
        for j, mj in enumerate(models):
            if i != j:
                mat[i, j] = fn(dists[mi], dists[mj])
    return mat, list(models)


# ---------------------------------------------------------------------------
# Training trajectory
# ---------------------------------------------------------------------------

def training_trajectory(store, models=None, reference='first', phone=None,
                        divergence='js', smoothing=1e-10):
    """
    Divergence from a reference checkpoint at each training step.

    models:         ordered list of model names (checkpoints); defaults to
                    store.models sorted by sort_w2v2_model_names
    reference:      'first', 'last', or a model name
    divergence:     'js' or 'kl'
    Returns (divergences, models) — parallel arrays.
    """
    if models is None:
        models = sort_w2v2_model_names(store.models)

    if reference == 'first':
        ref = models[0]
    elif reference == 'last':
        ref = models[-1]
    else:
        ref = reference

    fn = model_js if divergence == 'js' else model_kl
    values = [fn(store, ref, m, phone=phone, smoothing=smoothing) for m in models]
    return np.array(values), list(models)


# ---------------------------------------------------------------------------
# Per-phone divergence
# ---------------------------------------------------------------------------

def per_phone_divergence(store, model_a, model_b, divergence='js', smoothing=1e-10):
    """
    Divergence between two models broken down per phone.

    Returns dict {phone: divergence_value}.
    """
    fn = model_js if divergence == 'js' else model_kl
    return {p: fn(store, model_a, model_b, phone=p, smoothing=smoothing)
            for p in store.phones}


# ---------------------------------------------------------------------------
# Phone-vs-phone divergence
# ---------------------------------------------------------------------------

def phone_divergence(store, model, phone1, phone2, divergence='js', smoothing=1e-10):
    """
    Divergence between two phones for a given model.

    Compares P(code | model, phone1) against P(code | model, phone2).
    divergence: 'js' (symmetric) or 'kl' (KL(phone1 || phone2))
    """
    fn = js_divergence if divergence == 'js' else kl_divergence
    p = ci_pdf_for_model_phone(store, model, phone1, smoothing)
    q = ci_pdf_for_model_phone(store, model, phone2, smoothing)
    return fn(p, q)


def phone_vs_all_phones(store, model, phone, divergence='js', smoothing=1e-10):
    """
    Divergence between one phone and every other phone for a given model.

    Returns dict {other_phone: divergence_value}.
    """
    return {
        q_phone: phone_divergence(store, model, phone, q_phone, divergence, smoothing)
        for q_phone in store.phones
        if q_phone != phone
    }


def phone_vs_rest(store, model, phone, divergence='js', smoothing=1e-10):
    """
    Divergence between one phone and the aggregate of all other phones.

    Compares P(code | model, phone) against the distribution derived from
    summing CI counts over all other phones.
    divergence: 'js' (symmetric) or 'kl' (KL(phone || rest))
    """
    fn = js_divergence if divergence == 'js' else kl_divergence
    p = ci_pdf_for_model_phone(store, model, phone, smoothing)
    other_counts = sum(
        store.get(model=model, phone=q).astype(float)
        for q in store.phones
        if q != phone
    )
    other_counts += smoothing
    q = other_counts / other_counts.sum()
    return fn(p, q)


# ---------------------------------------------------------------------------
# Codebook utilization
# ---------------------------------------------------------------------------

def codebook_utilization(store, model, phone=None, min_count=1):
    """
    Fraction of codebook entries used at least min_count times.

    Pass phone to restrict to one phone; omit to collapse over all phones.
    """
    if phone is not None:
        counts = store.get(model=model, phone=phone)
    else:
        counts = store.get(model=model).sum(axis=0)
    return float((counts >= min_count).sum()) / store.n_codes


def utilization_trajectory(store, models=None, phone=None, min_count=1):
    """Codebook utilization at each training step. Returns (values, models)."""
    if models is None:
        models = sort_w2v2_model_names(store.models)
    values = [codebook_utilization(store, m, phone=phone, min_count=min_count)
              for m in models]
    return np.array(values), list(models)


# ---------------------------------------------------------------------------
# Top-k codes
# ---------------------------------------------------------------------------

def top_codes(store, model, phone=None, k=10):
    """
    Indices of the k most-used codebook entries (descending order).

    Pass phone to restrict to one phone; omit to collapse over all phones.
    """
    if phone is not None:
        counts = store.get(model=model, phone=phone)
    else:
        counts = store.get(model=model).sum(axis=0)
    indices = np.argsort(counts)[::-1][:k]
    return indices, counts[indices]


def top_k_jaccard(store, model_a, model_b, phone=None, k=10):
    """
    Jaccard similarity between the top-k code sets of two models.

    Pass phone to restrict to one phone; omit to collapse over all phones.
    Returns a value in [0, 1] where 1 means identical top-k sets.
    """
    set_a = set(top_codes(store, model_a, phone=phone, k=k)[0].tolist())
    set_b = set(top_codes(store, model_b, phone=phone, k=k)[0].tolist())
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def top_k_stability_trajectory(store, models=None, phone=None,
                                reference='first', k=10):
    """
    Jaccard similarity of the top-k code set vs a reference checkpoint
    at each training step.

    reference:  'first', 'last', or a model name
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
    values = [top_k_jaccard(store, ref, m, phone=phone, k=k) for m in models]
    return np.array(values), list(models)


def top_k_rank_matrix(store, phone, models=None, k=10, reference='last'):
    """
    Rank matrix for the top-k codes over training checkpoints.

    Rows are the union of top-k code indices seen across all checkpoints,
    sorted by rank in the reference checkpoint (rank 1 first; codes absent
    from the reference sort to the bottom by mean rank).
    Columns are ordered checkpoints.
    Values are 1-based ranks; entries outside the top-k are NaN.

    reference:  'first', 'last', or a model name
    Returns (matrix, code_indices, models).
    """
    if models is None:
        models = sort_w2v2_model_names(store.models)
    if reference == 'first':
        ref = models[0]
    elif reference == 'last':
        ref = models[-1]
    else:
        ref = reference
    model_ranks = {
        m: {int(idx): rank + 1
            for rank, idx in enumerate(top_codes(store, m, phone=phone, k=k)[0])}
        for m in models
    }
    all_indices = sorted({idx for ranks in model_ranks.values() for idx in ranks})
    matrix = np.full((len(all_indices), len(models)), np.nan)
    for j, m in enumerate(models):
        for i, idx in enumerate(all_indices):
            if idx in model_ranks[m]:
                matrix[i, j] = model_ranks[m][idx]
    ref_col = models.index(ref)
    ref_ranks = matrix[:, ref_col]
    fallback = np.nanmean(matrix, axis=1)
    sort_key = np.where(np.isnan(ref_ranks), fallback + k, ref_ranks)
    order = np.argsort(sort_key)
    return matrix[order], [all_indices[i] for i in order], list(models)
