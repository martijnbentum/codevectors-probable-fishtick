import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ci_analysis import (entropy, sort_w2v2_model_names,
                          model_js, model_kl,
                          codebook_utilization, per_phone_divergence,
                          phone_vs_rest,
                          top_k_stability_trajectory, top_k_rank_matrix,
                          CodebookSliceStore,
                          cb_usage_divergence_trajectory,
                          per_phone_cb_usage_divergence,
                          phone_pair_divergence_trajectory)
from cb_analysis import (drift_trajectory, phones_weighted_drift,
                          phone_weighted_drift_trajectory, mean_intra_distance,
                          drift_usage_correlation, drift_usage_correlation_trajectory,
                          cross_codebook_geometry_trajectory,
                          drift_difference_trajectory,
                          phone_pair_distance_trajectory)


def _checkpoint_xticks(models):
    try:
        return [int(m.split('-')[-1]) for m in models]
    except (ValueError, IndexError):
        return list(models)


def _sorted_models(store, models):
    return sort_w2v2_model_names(store.models) if models is None else models


def _codebook_store(store, codebook):
    return CodebookSliceStore(store, codebook) if codebook in (1, 2) else store


def _codebook_cases(store, codebook):
    """Return [(store, linestyle, label_suffix)] for the given codebook setting."""
    if codebook == 'both':
        return [(CodebookSliceStore(store, 1), ':', ' (cb1)'),
                (CodebookSliceStore(store, 2), '-', ' (cb2)')]
    s = _codebook_store(store, codebook)
    suffix = f' (cb{codebook})' if codebook in (1, 2) else ''
    return [(s, ':' if codebook == 1 else '-', suffix)]


def _phone_color_map(phones):
    """Assign a consistent color from the default pyplot color cycle to each phone."""
    return {p: f'C{i}' for i, p in enumerate(phones)}


def _plot_phone_lines(ax, x, phones, values_dict, color_map,
                      linestyle=':', label_suffix='', with_label=True):
    for phone in phones:
        lbl = f'{phone}{label_suffix}' if with_label else None
        ax.plot(x, values_dict[phone], marker='o', markersize=3,
                color=color_map[phone], linestyle=linestyle, label=lbl)


def _apply_legend(ax, phones, color_map, codebook):
    """Legend with one colour entry per phone and one linestyle entry per codebook.

    Only builds the custom split legend when codebook='both' and phones are present;
    otherwise falls back to the standard matplotlib legend.
    """
    if codebook == 'both' and phones:
        phone_handles = [
            Line2D([0], [0], color=color_map[p], marker='o', markersize=1, label=p)
            for p in phones
        ]
        cb_handles = [
            Line2D([0], [0], color='black', linestyle=':', label='cb1'),
            Line2D([0], [0], color='black', linestyle='-', label='cb2'),
        ]
        ax.legend(handles=phone_handles + cb_handles, fontsize=8)
    else:
        ax.legend(fontsize=8)


def plot_over_checkpoints(models, values_dict, ylabel='', title='',
                          show_legend=True, xlim=(1000, 50000), ax=None):
    """
    Base function: one line per entry in values_dict over ordered checkpoints.

    models:       ordered list of checkpoint names (x-axis)
    values_dict:  {label: sequence_of_values} — one entry per line
    xlim:         x-axis limits; set to None to use matplotlib defaults
    Returns the Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    x = _checkpoint_xticks(models)
    for label, values in values_dict.items():
        ax.plot(x, values, marker='o', markersize=1, linestyle=':', label=label)

    ax.set_xlabel('training step')
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if title:
        ax.set_title(title)
    if show_legend and values_dict:
        ax.legend(fontsize=8)
    ax.figure.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def plot_entropy(store, phones=None, models=None, codebook=None,
                 xlim=(1000, 50000), ax=None):
    """
    Plot codebook entropy (bits) over model checkpoints.

    phones:   None      → single line: all phones combined
              'all'     → one line per phone in the store
              list      → one line per phone in the list
    codebook: None → combined; 1 or 2 → single codebook; 'both' → cb1 (solid)
              and cb2 (dashed) per phone, coloured by phone
    models:   ordered checkpoint list; defaults to all store models sorted
    """
    models = _sorted_models(store, models)
    x = _checkpoint_xticks(models)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    if phones is None:
        for s, ls, suffix in _codebook_cases(store, codebook):
            label = suffix.strip().strip('()') or 'all phones combined'
            values = [entropy(s.get(model=m).sum(axis=0)) for m in models]
            ax.plot(x, values, marker='o', markersize=1, linestyle=ls, label=label)
        title = 'Codebook entropy over training'
    else:
        if phones == 'all':
            phones = store.phones
        elif isinstance(phones, str):
            phones = [phones]
        color_map = _phone_color_map(phones)
        for s, ls, suffix in _codebook_cases(store, codebook):
            values_dict = {p: [entropy(s.get(model=m, phone=p)) for m in models]
                           for p in phones}
            _plot_phone_lines(ax, x, phones, values_dict, color_map, ls,
                              label_suffix=suffix, with_label=codebook != 'both')
        title = (f'Codebook entropy over training — phone: {phones[0]}'
                 if len(phones) == 1 else 'Codebook entropy over training per phone')

    ax.set_xlabel('training step')
    ax.set_ylabel('entropy (bits)')
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_title(title)
    _apply_legend(ax, phones or [], color_map if phones else None, codebook)
    ax.grid(alpha=0.3)
    ax.figure.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Divergence trajectory
# ---------------------------------------------------------------------------

def _divergence_values(store, models, ref_model, phone, fn):
    """Compute divergence from ref_model to each model in models."""
    return [fn(store, ref_model, m, phone=phone) for m in models]


def _divergence_values_previous(store, models, phone, fn):
    """Compute divergence between each consecutive pair of checkpoints."""
    values = [0.0]  # first checkpoint has no predecessor
    for prev, curr in zip(models[:-1], models[1:]):
        values.append(fn(store, prev, curr, phone=phone))
    return values


def plot_divergence_trajectory(store, phones=None, models=None,
                                reference='first', divergence='js',
                                codebook=None, xlim=(1000, 50000), ax=None):
    """
    Plot JS or KL divergence over training.

    phones:     None        → model-level divergence (phones summed)
                'all'       → one line per phone
                list/str    → one line per phone in the list
    reference:  'first'     → divergence from the first checkpoint
                'last'      → divergence from the last checkpoint
                'previous'  → divergence from the preceding checkpoint
                model name  → divergence from that specific checkpoint
    divergence: 'js' (default, symmetric) or 'kl'
    codebook:   None → combined; 1 or 2 → single codebook; 'both' → cb1 (solid)
                and cb2 (dashed) per phone, coloured by phone
    """
    models = _sorted_models(store, models)
    fn = model_js if divergence == 'js' else model_kl
    div_label = 'JS divergence' if divergence == 'js' else 'KL divergence'
    x = _checkpoint_xticks(models)

    if reference == 'previous':
        ref_desc = 'previous checkpoint'
        def _make_compute(s):
            return lambda phone: _divergence_values_previous(s, models, phone, fn)
    else:
        ref = models[0] if reference == 'first' else (
              models[-1] if reference == 'last' else reference)
        try:
            ref_desc = f'checkpoint {int(ref.split("-")[-1])}'
        except (ValueError, IndexError):
            ref_desc = ref
        def _make_compute(s):
            return lambda phone: _divergence_values(s, models, ref, phone, fn)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    if phones is None:
        for s, ls, suffix in _codebook_cases(store, codebook):
            label = suffix.strip().strip('()') or 'all phones combined'
            ax.plot(x, _make_compute(s)(None), marker='o', markersize=1,
                    linestyle=ls, label=label)
        title = f'Codebook {div_label} from {ref_desc}'
    else:
        if phones == 'all':
            phones = store.phones
        elif isinstance(phones, str):
            phones = [phones]
        color_map = _phone_color_map(phones)
        for s, ls, suffix in _codebook_cases(store, codebook):
            compute = _make_compute(s)
            values_dict = {p: compute(p) for p in phones}
            _plot_phone_lines(ax, x, phones, values_dict, color_map, ls,
                              label_suffix=suffix, with_label=codebook != 'both')
        title = (f'Codebook {div_label} from {ref_desc} — phone: {phones[0]}'
                 if len(phones) == 1
                 else f'Codebook {div_label} from {ref_desc} per phone')

    ax.set_xlabel('training step')
    ax.set_ylabel(div_label)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_title(title)
    _apply_legend(ax, phones or [], color_map if phones else None, codebook)
    ax.grid(alpha=0.3)
    ax.figure.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Codebook utilization
# ---------------------------------------------------------------------------

def plot_utilization(store, phones=None, models=None, min_count=1,
                     codebook=None, xlim=(1000, 50000), ax=None):
    """
    Plot fraction of codebook entries used (>= min_count) over training.

    phones:   None      → model-level utilization (phones summed)
              'all'     → one line per phone
              list/str  → one line per phone in the list
    codebook: None → combined; 1 or 2 → single codebook; 'both' → cb1 (solid)
              and cb2 (dashed) per phone, coloured by phone
    """
    models = _sorted_models(store, models)
    x = _checkpoint_xticks(models)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    if phones is None:
        for s, ls, suffix in _codebook_cases(store, codebook):
            label = suffix.strip().strip('()') or 'all phones combined'
            values = [codebook_utilization(s, m, min_count=min_count) for m in models]
            ax.plot(x, values, marker='o', markersize=1, linestyle=ls, label=label)
        title = 'Codebook utilization over training'
    else:
        if phones == 'all':
            phones = store.phones
        elif isinstance(phones, str):
            phones = [phones]
        color_map = _phone_color_map(phones)
        for s, ls, suffix in _codebook_cases(store, codebook):
            values_dict = {p: [codebook_utilization(s, m, phone=p, min_count=min_count)
                               for m in models]
                           for p in phones}
            _plot_phone_lines(ax, x, phones, values_dict, color_map, ls,
                              label_suffix=suffix, with_label=codebook != 'both')
        title = (f'Codebook utilization over training — phone: {phones[0]}'
                 if len(phones) == 1 else 'Codebook utilization over training per phone')

    ax.set_xlabel('training step')
    ax.set_ylabel('utilization')
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_title(title)
    _apply_legend(ax, phones or [], color_map if phones else None, codebook)
    ax.grid(alpha=0.3)
    ax.figure.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Per-phone divergence (bar chart between two checkpoints)
# ---------------------------------------------------------------------------

def plot_per_phone_divergence(store, model_a, model_b,
                               divergence='js', phones=None, ax=None):
    """
    Bar chart of divergence per phone between two checkpoints.

    phones:   None / 'all'  → all phones in the store
              list           → subset of phones
    """
    if phones is None or phones == 'all':
        phones = store.phones
    elif isinstance(phones, str):
        phones = [phones]

    div_values = per_phone_divergence(store, model_a, model_b, divergence=divergence)
    values = [div_values[p] for p in phones]

    div_label = 'JS divergence' if divergence == 'js' else 'KL divergence'

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, len(phones) * 0.4), 4))

    ax.bar(range(len(phones)), values)
    ax.set_xticks(range(len(phones)))
    ax.set_xticklabels(phones, rotation=90, fontsize=7)
    ax.set_ylabel(div_label)
    step_a = int(model_a.split('-')[-1])
    step_b = int(model_b.split('-')[-1])
    ax.set_title(f'Per-phone codebook {div_label}: step {step_a} vs {step_b}')
    ax.figure.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Phone-vs-rest divergence trajectory
# ---------------------------------------------------------------------------

def plot_phone_vs_rest(store, phones=None, models=None, divergence='js',
                       codebook=None, xlim=(1000, 50000), ax=None):
    """
    Plot JS or KL divergence between each phone and the aggregate of all
    other phones, over training checkpoints.

    phones:     None / 'all'  → one line per phone in the store
                list / str    → one line per phone in the list
    divergence: 'js' (default) or 'kl'
    codebook:   None → combined; 1 or 2 → single codebook; 'both' → cb1 (solid)
                and cb2 (dashed) per phone, coloured by phone
    """
    models = _sorted_models(store, models)
    x = _checkpoint_xticks(models)

    if phones is None or phones == 'all':
        phones = store.phones
    elif isinstance(phones, str):
        phones = [phones]

    div_label = 'JS divergence' if divergence == 'js' else 'KL divergence'

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    color_map = _phone_color_map(phones)
    for s, ls, suffix in _codebook_cases(store, codebook):
        values_dict = {p: [phone_vs_rest(s, m, p, divergence=divergence) for m in models]
                       for p in phones}
        _plot_phone_lines(ax, x, phones, values_dict, color_map, ls,
                          label_suffix=suffix, with_label=codebook != 'both')

    title = (f'Phone vs rest codebook {div_label} over training'
             if len(phones) > 1
             else f'Phone vs rest codebook {div_label} — phone: {phones[0]}')

    ax.set_xlabel('training step')
    ax.set_ylabel(div_label)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_title(title)
    _apply_legend(ax, phones, color_map, codebook)
    ax.grid(alpha=0.3)
    ax.figure.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Top-k code stability trajectory
# ---------------------------------------------------------------------------

def plot_top_k_stability(store, phones=None, models=None, reference='first',
                          k=10, xlim=(1000, 50000), ax=None):
    """
    Plot Jaccard similarity of top-k codes vs a reference checkpoint over training.

    phones:     None        → model-level (phones summed)
                'all'       → one line per phone
                list/str    → one line per phone in the list
    reference:  'first', 'last', or a model name
    k:          number of top codes to compare
    """
    models = _sorted_models(store, models)

    if reference == 'first':
        ref_desc = f'step {_checkpoint_xticks(models)[0]}'
    elif reference == 'last':
        ref_desc = f'step {_checkpoint_xticks(models)[-1]}'
    else:
        try:
            ref_desc = f'step {int(reference.split("-")[-1])}'
        except (ValueError, IndexError):
            ref_desc = reference

    if phones is None:
        values, _ = top_k_stability_trajectory(store, models=models, phone=None,
                                                reference=reference, k=k)
        values_dict = {'all phones combined': values}
        title = f'Top-{k} code stability from {ref_desc}'
    else:
        if phones == 'all':
            phones = store.phones
        elif isinstance(phones, str):
            phones = [phones]
        values_dict = {
            phone: top_k_stability_trajectory(store, models=models, phone=phone,
                                               reference=reference, k=k)[0]
            for phone in phones
        }
        title = (f'Top-{k} code stability from {ref_desc} — phone: {phones[0]}'
                 if len(phones) == 1
                 else f'Top-{k} code stability from {ref_desc} per phone')

    return plot_over_checkpoints(
        models, values_dict, ylabel='Jaccard similarity', title=title,
        xlim=xlim, ax=ax
    )


# ---------------------------------------------------------------------------
# Top-k code rank heatmap
# ---------------------------------------------------------------------------

def plot_top_k_rank_heatmap(store, phone, models=None, k=10,
                             reference='last', ax=None):
    """
    Heatmap of top-k code ranks over training checkpoints for a single phone.

    Rows are code indices (union of top-k across all checkpoints), sorted by
    rank in the reference checkpoint. Columns are checkpoints. Color encodes
    rank (1=top); grey cells were outside the top-k at that checkpoint.

    reference:  'first', 'last', or a model name
    """
    models = _sorted_models(store, models)
    matrix, code_indices, models = top_k_rank_matrix(store, phone, models=models,
                                                      k=k, reference=reference)

    n_codes, n_models = matrix.shape
    if ax is None:
        _, ax = plt.subplots(figsize=(max(8, n_models * 0.35),
                                      max(4, n_codes * 0.4)))

    masked = np.ma.array(matrix, mask=np.isnan(matrix))
    cmap = plt.get_cmap('viridis_r').copy()
    cmap.set_bad(color='lightgrey')

    im = ax.imshow(masked, aspect='auto', cmap=cmap, vmin=1, vmax=k,
                   interpolation='none')
    plt.colorbar(im, ax=ax, label=f'rank within top-{k}')

    x_labels = _checkpoint_xticks(models)
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(x_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n_codes))
    ax.set_yticklabels(code_indices, fontsize=7)
    ax.set_xlabel('training step')
    ax.set_ylabel('code index')
    if reference in ('first', 'last'):
        ref_desc = reference
    else:
        try:
            ref_desc = f'step {int(reference.split("-")[-1])}'
        except (ValueError, IndexError):
            ref_desc = reference
    ax.set_title(f'Top-{k} code ranks over training — phone: {phone} (sorted by {ref_desc})')
    ax.figure.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Codevector drift trajectory
# ---------------------------------------------------------------------------

def _ref_model(models, reference):
    if reference == 'first':
        return models[0]
    if reference == 'last':
        return models[-1]
    if reference == 'previous':
        return models[0]  # use first checkpoint as normalization scale
    return reference


def _ref_desc(models, reference):
    if reference == 'previous':
        return 'previous checkpoint'
    ref = _ref_model(models, reference)
    try:
        return f'step {int(ref.split("-")[-1])}'
    except (ValueError, IndexError):
        return ref


def plot_drift_trajectory(models, reference='first', metric='l2',
                           normalize=False, xlim=(1000, 50000), ax=None):
    """
    Plot mean codevector drift vs a reference checkpoint over training.

    Purely geometric — no store needed.
    models:     ordered list of checkpoint names
    reference:  'first', 'last', or a model name
    metric:     'l2' (default) or 'cosine'
    normalize:  divide by mean intra-codebook distance of the reference model,
                so 1.0 = a full inter-codevector step
    """
    values, _ = drift_trajectory(models, reference=reference, metric=metric)

    if normalize and metric == 'l2':
        scale = mean_intra_distance(_ref_model(models, reference))
        values = values / scale

    ref_desc = _ref_desc(models, reference)
    ylabel = ('drift / mean intra-codebook distance' if normalize and metric == 'l2'
              else f'mean {metric} drift')
    title = f'Codevector drift from {ref_desc}'
    return plot_over_checkpoints(
        models, {'mean drift': values}, ylabel=ylabel, title=title,
        show_legend=False, xlim=xlim, ax=ax
    )


def plot_phone_drift_trajectory(store, phones=None, models=None,
                                  reference='first', metric='l2',
                                  normalize=False, codebook=None,
                                  xlim=(1000, 50000), ax=None):
    """
    Plot phone-weighted codevector drift vs a reference checkpoint over training.

    phones:     None / 'all'  → one line per phone in the store
                list / str    → one line per phone in the list
    reference:  'first', 'last', or a model name
    metric:     'l2' (default) or 'cosine'
    normalize:  divide by mean intra-codebook distance of the reference model
    codebook:   None → all 640 codes; 1 or 2 → single codebook; 'both' → cb1
                (solid) and cb2 (dashed) per phone, coloured by phone
    """
    models = _sorted_models(store, models)

    if phones is None or phones == 'all':
        phones = store.phones
    elif isinstance(phones, str):
        phones = [phones]

    scale = mean_intra_distance(_ref_model(models, reference)) if normalize and metric == 'l2' else 1.0
    x = _checkpoint_xticks(models)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    cb_cases = ([(1, ':', ' (cb1)'), (2, '-', ' (cb2)')] if codebook == 'both'
                else [(codebook, ':' if codebook == 1 else '-', f' (cb{codebook})' if codebook in (1, 2) else '')])
    color_map = _phone_color_map(phones)

    for cb, ls, suffix in cb_cases:
        for p in phones:
            values, _ = phone_weighted_drift_trajectory(
                store, models=models, reference=reference,
                phone=p, metric=metric, codebook=cb)
            lbl = None if codebook == 'both' else f'{p}{suffix}'
            ax.plot(x, values / scale, marker='o', markersize=3,
                    color=color_map[p], linestyle=ls, label=lbl)

    ref_desc = _ref_desc(models, reference)
    ylabel = ('drift / mean intra-codebook distance' if normalize and metric == 'l2'
              else f'mean {metric} drift (phone-weighted)')
    title = (f'Phone-weighted codevector drift from {ref_desc} — phone: {phones[0]}'
             if len(phones) == 1
             else f'Phone-weighted codevector drift from {ref_desc} per phone')

    ax.set_xlabel('training step')
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_title(title)
    _apply_legend(ax, phones, color_map, codebook)
    ax.grid(alpha=0.3)
    ax.figure.tight_layout()
    return ax


def plot_drift_per_phone(store, model_a, model_b, phones=None,
                          metric='l2', normalize=False, ax=None):
    """
    Bar chart of phone-weighted codevector drift per phone between two checkpoints.

    phones:    None / 'all'  → all phones in the store
               list / str    → subset of phones
    metric:    'l2' (default) or 'cosine'
    normalize: divide by mean intra-codebook distance of model_a
    """
    if phones is None or phones == 'all':
        phones = store.phones
    elif isinstance(phones, str):
        phones = [phones]

    drift_values = phones_weighted_drift(store, model_a, model_b,
                                          phones=phones, metric=metric)
    scale = mean_intra_distance(model_a) if normalize and metric == 'l2' else 1.0
    values = [drift_values[p] / scale for p in phones]

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, len(phones) * 0.4), 4))

    ax.bar(range(len(phones)), values)
    ax.set_xticks(range(len(phones)))
    ax.set_xticklabels(phones, rotation=90, fontsize=7)
    ylabel = ('drift / mean intra-codebook distance' if normalize and metric == 'l2'
              else f'mean {metric} drift (phone-weighted)')
    ax.set_ylabel(ylabel)
    step_a = int(model_a.split('-')[-1])
    step_b = int(model_b.split('-')[-1])
    ax.set_title(f'Per-phone codevector {metric} drift: step {step_a} vs {step_b}')
    ax.figure.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Overview panel
# ---------------------------------------------------------------------------

def plot_overview(store, models=None, codebook=None,
                  phones_subset=None):
    """
    4-row × 2-column panel summary.

    Row 1  entropy          — subset phones / all phones
    Row 2  JS divergence    — subset phones (ref=previous) / all phones (ref=last)
    Row 3  JS divergence    — subset phones, wide panel (ref=last)
    Row 4  utilization      — subset phones / all phones

    codebook:      None → combined (default); 1 or 2 → single codebook;
                   'both' → cb1 (solid) and cb2 (dashed) per phone
    phones_subset: phones shown in the left column; defaults to ['r', 's', 'm', 'd', 'z']
    """
    if phones_subset is None:
        phones_subset = ['r', 's', 'm', 'd', 'z']

    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.3)

    ax_r1c1 = fig.add_subplot(gs[0, 0])
    ax_r1c2 = fig.add_subplot(gs[0, 1])
    ax_r2c1 = fig.add_subplot(gs[1, 0])
    ax_r2c2 = fig.add_subplot(gs[1, 1])
    ax_r3c1   = fig.add_subplot(gs[2, 0])
    ax_r3c2   = fig.add_subplot(gs[2, 1])
    ax_r4c1 = fig.add_subplot(gs[3, 0])
    ax_r4c2 = fig.add_subplot(gs[3, 1])

    plot_entropy(store, phones=phones_subset, models=models,
                 codebook=codebook, xlim=(0, 30_000), ax=ax_r1c1)
    plot_entropy(store, phones=None, models=models,
                 codebook=codebook, xlim=(0, 100_000), ax=ax_r1c2)

    plot_divergence_trajectory(store, phones=phones_subset, models=models,
        reference='last', codebook=codebook, xlim=(0, 100_000), ax=ax_r2c1)
    plot_divergence_trajectory(store, phones=None, models=models,
        reference='last', codebook=codebook, xlim=(0, 100_000), ax=ax_r2c2)

    plot_divergence_trajectory(store, phones=phones_subset, models=models,
        reference='previous', codebook=codebook, xlim=(0, 30_000), ax=ax_r3c1)
    plot_phone_vs_rest(store, phones=phones_subset, models=models,
        codebook=codebook, xlim=(0, 30_000), ax=ax_r3c2)

    plot_utilization(store, phones=phones_subset, models=models,
        codebook=codebook, xlim=(0, 30_000), ax=ax_r4c1, min_count=100)
    plot_utilization(store, phones=None, models=models,
        codebook=codebook, xlim=(0, 100_000), ax=ax_r4c2, min_count=100)

    axs = [ax_r1c1, ax_r1c2, ax_r2c1, ax_r2c2, ax_r3c1, ax_r3c2, ax_r4c1, ax_r4c2]
    for ax in axs:
        ax.grid(alpha=0.3)

    return fig


# ---------------------------------------------------------------------------
# Drift × usage correlation
# ---------------------------------------------------------------------------

def plot_drift_usage_scatter(store, model_a, model_b, metric='l2', phone=None,
                              log_usage=True, ax=None):
    """
    Scatter plot of per-codevector drift vs usage frequency.

    Each point is one codevector. X-axis: usage count in model_a (source).
    Y-axis: drift from model_a to model_b. Annotated with Pearson and Spearman r.

    log_usage:  plot usage on a log(1+x) scale (default True; counts are skewed)
    phone:      restrict usage to one phone; None uses all phones combined
    metric:     'l2' (default) or 'cosine'
    """
    result = drift_usage_correlation(store, model_a, model_b,
                                     metric=metric, phone=phone)
    drift, usage = result['drift'], result['usage']
    pearson, spearman = result['pearson'], result['spearman']

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    x = np.log1p(usage) if log_usage else usage
    ax.scatter(x, drift, s=8, alpha=0.5, linewidths=0)

    ax.set_xlabel('log(1 + usage count)' if log_usage else 'usage count')
    ax.set_ylabel(f'{metric} drift')

    step_a = model_a.split('-')[-1]
    step_b = model_b.split('-')[-1]
    phone_desc = f' — phone: {phone}' if phone else ''
    ax.set_title(f'Drift vs usage: step {step_a} → {step_b}{phone_desc}')

    ax.text(0.97, 0.97,
            f'Pearson r = {pearson:.3f}\nSpearman r = {spearman:.3f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.grid(alpha=0.3)
    ax.figure.tight_layout()
    return ax


def plot_drift_usage_correlation_trajectory(store, models=None, reference='first',
                                             metric='l2', phone=None,
                                             xlim=(1000, 50000), ax=None):
    """
    Plot Pearson and Spearman r between per-codevector drift and usage over training.

    reference:  'first', 'last', 'previous', or a model name
    phone:      restrict usage to one phone; None uses all phones combined
    metric:     'l2' (default) or 'cosine'
    """
    models = _sorted_models(store, models)
    result = drift_usage_correlation_trajectory(store, models=models,
                                                reference=reference,
                                                metric=metric, phone=phone)
    x = _checkpoint_xticks(models)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(x, result['pearson'],  marker='o', markersize=1, linestyle=':', label='Pearson r')
    ax.plot(x, result['spearman'], marker='o', markersize=1, linestyle=':', label='Spearman r')
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')

    ax.set_xlabel('training step')
    ax.set_ylabel('correlation (r)')
    if xlim is not None:
        ax.set_xlim(xlim)

    phone_desc = f' — phone: {phone}' if phone else ''
    ref_desc = reference if reference in ('first', 'last', 'previous') else reference.split('-')[-1]
    ax.set_title(f'Drift–usage correlation over training '
                 f'(ref: {ref_desc}, metric: {metric}){phone_desc}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.figure.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# cb1 vs cb2 — approach 1: cross-codebook geometry
# ---------------------------------------------------------------------------

def plot_cross_codebook_geometry(models, normalize=False,
                                  xlim=(1000, 50000), ax=None):
    """
    Plot mean and nearest-neighbour cross-codebook L2 distance over training.

    mean_cross: average L2 distance from every cb1 codevector to every cb2 codevector.
    mean_nn:    average distance from each cb1 codevector to its nearest cb2 neighbour.
    Purely geometric — no store needed.

    normalize:  if True, divide by mean_intra_distance of the first model so values
                are in units of the average within-codebook codevector spacing.
    """
    models = sort_w2v2_model_names(models)
    result = cross_codebook_geometry_trajectory(models, normalize=normalize)
    x = _checkpoint_xticks(models)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(x, result['mean_cross'], marker='o', markersize=1, linestyle='-',
            label='mean cross-codebook distance')
    ax.plot(x, result['mean_nn'],    marker='o', markersize=1, linestyle='-',
            label='mean nearest-neighbour distance')

    ax.set_xlabel('training step')
    ylabel = ('distance / mean intra-codebook distance' if normalize
              else 'L2 distance')
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_title('cb1 vs cb2 cross-codebook geometry over training')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.figure.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# cb1 vs cb2 — approach 2: usage distribution divergence
# ---------------------------------------------------------------------------

def plot_cb_usage_divergence(store, phones=None, models=None, divergence='js',
                              xlim=(1000, 50000), ax=None):
    """
    Plot cb1 vs cb2 usage distribution divergence over training.

    phones:    None      → model-level (all phones combined), one line
               'all'    → one line per phone in the store
               list/str → one line per phone in the list
    divergence: 'js' (default, symmetric) or 'kl'
    """
    models = _sorted_models(store, models)
    x = _checkpoint_xticks(models)
    div_label = 'JS divergence' if divergence == 'js' else 'KL divergence'

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    if phones is None:
        values, _ = cb_usage_divergence_trajectory(store, models=models,
                                                    divergence=divergence)
        ax.plot(x, values, marker='o', markersize=1, linestyle='-', label='all phones')
        title = f'cb1 vs cb2 usage {div_label} over training'
    else:
        if phones == 'all':
            phones = store.phones
        elif isinstance(phones, str):
            phones = [phones]
        color_map = _phone_color_map(phones)
        for p in phones:
            values, _ = cb_usage_divergence_trajectory(store, models=models,
                                                        phone=p, divergence=divergence)
            ax.plot(x, values, marker='o', markersize=1, linestyle='-',
                    color=color_map[p], label=p)
        title = (f'cb1 vs cb2 usage {div_label} over training — phone: {phones[0]}'
                 if len(phones) == 1
                 else f'cb1 vs cb2 usage {div_label} over training per phone')

    ax.set_xlabel('training step')
    ax.set_ylabel(div_label)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.figure.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# cb1 vs cb2 — approach 3: drift difference trajectory
# ---------------------------------------------------------------------------

def plot_drift_difference(store=None, models=None, phones=None,
                           reference='first', metric='l2',
                           normalize=False, xlim=(1000, 50000), ax=None):
    """
    Plot cb1 drift minus cb2 drift over training (positive = cb1 drifted more).

    store:      CI store for phone-weighted drift; omit for raw geometric drift.
    phones:     None              → one line (raw or all-phones-weighted drift)
                'all' / list/str → one line per phone (requires store)
    reference:  'first', 'last', 'previous', or a model name
    metric:     'l2' (default) or 'cosine'
    normalize:  divide by mean intra-codebook distance of the reference model
    """
    if models is None:
        if store is not None:
            models = _sorted_models(store, models)
        else:
            raise ValueError("models must be provided when store is None")

    x = _checkpoint_xticks(models)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    scale = (mean_intra_distance(_ref_model(models, reference))
             if normalize and metric == 'l2' else 1.0)

    if phones is None or store is None:
        diff, _ = drift_difference_trajectory(store=store, models=models,
                                               reference=reference, metric=metric)
        ax.plot(x, diff / scale, marker='o', markersize=1, linestyle='-',
                label='cb1 − cb2 drift')
        title = f'cb1 vs cb2 drift difference from {_ref_desc(models, reference)}'
    else:
        if phones == 'all':
            phones = store.phones
        elif isinstance(phones, str):
            phones = [phones]
        color_map = _phone_color_map(phones)
        for p in phones:
            diff, _ = drift_difference_trajectory(store=store, models=models,
                                                   reference=reference, phone=p,
                                                   metric=metric)
            ax.plot(x, diff / scale, marker='o', markersize=1, linestyle='-',
                    color=color_map[p], label=p)
        title = (f'cb1 vs cb2 drift difference from {_ref_desc(models, reference)}'
                 f' — phone: {phones[0]}'
                 if len(phones) == 1
                 else f'cb1 vs cb2 drift difference from {_ref_desc(models, reference)} per phone')

    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_xlabel('training step')
    ylabel = ('(cb1 − cb2 drift) / mean intra-codebook distance'
              if normalize and metric == 'l2'
              else f'cb1 − cb2 mean {metric} drift')
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.figure.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Phone-pair comparison
# ---------------------------------------------------------------------------

def _pair_cb_cases(store, codebook):
    """Return [(store_or_None, cb_int, linestyle, label_suffix)] for codebook splitting.

    For divergence (store-based) and distance (codebook-int-based) in one structure.
    """
    if codebook == 'both':
        return [(CodebookSliceStore(store, 1), 1, '-', ' (cb1)'),
                (CodebookSliceStore(store, 2), 2, ':', ' (cb2)')]
    s = CodebookSliceStore(store, codebook) if codebook in (1, 2) else store
    cb_int = codebook if codebook in (1, 2) else None
    ls = ':' if codebook == 2 else '-'
    suffix = f' (cb{codebook})' if codebook in (1, 2) else ''
    return [(s, cb_int, ls, suffix)]


def _pair_legend(ax, pairs, color_map, codebook):
    if codebook == 'both':
        pair_handles = [
            Line2D([0], [0], color=color_map[pair], marker='o', markersize=1,
                   label=f'{pair[0]} vs {pair[1]}')
            for pair in pairs
        ]
        cb_handles = [
            Line2D([0], [0], color='black', linestyle='-', label='cb1'),
            Line2D([0], [0], color='black', linestyle=':', label='cb2'),
        ]
        ax.legend(handles=pair_handles + cb_handles, fontsize=8)
    else:
        ax.legend(fontsize=8)


def plot_phone_pair_divergence(store, pairs, models=None, divergence='js',
                                codebook=None, xlim=(1000, 50000), ax=None):
    """
    Usage distribution divergence for one or more phone pairs over training.

    pairs:      list of (phone1, phone2) tuples, e.g. [('s', 'z'), ('r', 'l')]
    divergence: 'js' (default, symmetric) or 'kl'
    codebook:   None → all 640 codes; 1 or 2 → single codebook;
                'both' → cb1 (solid) and cb2 (dotted) per pair, coloured by pair
    """
    models = _sorted_models(store, models)
    x = _checkpoint_xticks(models)
    div_label = 'JS divergence' if divergence == 'js' else 'KL divergence'
    color_map = {pair: f'C{i}' for i, pair in enumerate(pairs)}

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    for s, _cb, ls, suffix in _pair_cb_cases(store, codebook):
        for pair in pairs:
            p1, p2 = pair
            values, _ = phone_pair_divergence_trajectory(s, p1, p2, models=models,
                                                          divergence=divergence)
            lbl = None if codebook == 'both' else f'{p1} vs {p2}{suffix}'
            ax.plot(x, values, marker='o', markersize=1, linestyle=ls,
                    color=color_map[pair], label=lbl)

    ax.set_xlabel('training step')
    ax.set_ylabel(div_label)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_title(f'Phone-pair {div_label} over training')
    _pair_legend(ax, pairs, color_map, codebook)
    ax.grid(alpha=0.3)
    ax.figure.tight_layout()
    return ax


def plot_phone_pair_distance(store, pairs, models=None, codebook=None,
                              xlim=(1000, 50000), ax=None):
    """
    Expected inter-phone codevector L2 distance for one or more phone pairs over training.

    pairs:    list of (phone1, phone2) tuples, e.g. [('s', 'z'), ('r', 'l')]
    codebook: None → all 640 codes; 1 or 2 → single codebook;
              'both' → cb1 (solid) and cb2 (dotted) per pair, coloured by pair
    """
    models = _sorted_models(store, models)
    x = _checkpoint_xticks(models)
    color_map = {pair: f'C{i}' for i, pair in enumerate(pairs)}

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    for _s, cb_int, ls, suffix in _pair_cb_cases(store, codebook):
        for pair in pairs:
            p1, p2 = pair
            values, _ = phone_pair_distance_trajectory(store, p1, p2, models=models,
                                                        codebook=cb_int)
            lbl = None if codebook == 'both' else f'{p1} vs {p2}{suffix}'
            ax.plot(x, values, marker='o', markersize=1, linestyle=ls,
                    color=color_map[pair], label=lbl)

    ax.set_xlabel('training step')
    ax.set_ylabel('expected codevector L2 distance')
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_title('Expected inter-phone codevector distance over training')
    _pair_legend(ax, pairs, color_map, codebook)
    ax.grid(alpha=0.3)
    ax.figure.tight_layout()
    return ax
