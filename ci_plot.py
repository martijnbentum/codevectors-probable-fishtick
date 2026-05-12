import matplotlib.pyplot as plt
from ci_analysis import (entropy, sort_w2v2_model_names,
                          model_js, model_kl,
                          codebook_utilization, per_phone_divergence)


def _checkpoint_xticks(models):
    try:
        return [int(m.split('-')[-1]) for m in models]
    except (ValueError, IndexError):
        return list(models)


def _sorted_models(store, models):
    return sort_w2v2_model_names(store.models) if models is None else models


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
        ax.plot(x, values, marker='o', markersize=3, label=label)

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

def plot_entropy(store, phones=None, models=None, xlim=(1000, 50000), ax=None):
    """
    Plot codebook entropy (bits) over model checkpoints.

    phones:   None      → single line: all phones combined
              'all'     → one line per phone in the store
              list      → one line per phone in the list
    models:   ordered checkpoint list; defaults to all store models sorted
    """
    models = _sorted_models(store, models)

    if phones is None:
        values = [entropy(store.get(model=m).sum(axis=0)) for m in models]
        values_dict = {'all phones combined': values}
        title = 'Codebook entropy over training'
    else:
        if phones == 'all':
            phones = store.phones
        elif isinstance(phones, str):
            phones = [phones]
        values_dict = {
            phone: [entropy(store.get(model=m, phone=phone)) for m in models]
            for phone in phones
        }
        if len(phones) == 1:
            title = f'Codebook entropy over training — phone: {phones[0]}'
        else:
            title = 'Codebook entropy over training per phone'

    return plot_over_checkpoints(
        models, values_dict, ylabel='entropy (bits)', title=title, xlim=xlim, ax=ax
    )


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
                                xlim=(1000, 50000), ax=None):
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
    """
    models = _sorted_models(store, models)
    fn = model_js if divergence == 'js' else model_kl
    div_label = 'JS divergence' if divergence == 'js' else 'KL divergence'

    if reference == 'previous':
        ref_desc = 'previous checkpoint'
        compute = lambda phone: _divergence_values_previous(store, models, phone, fn)
    else:
        if reference == 'first':
            ref = models[0]
        elif reference == 'last':
            ref = models[-1]
        else:
            ref = reference
        ref_desc = f'checkpoint {int(ref.split("-")[-1])}'
        compute = lambda phone: _divergence_values(store, models, ref, phone, fn)

    if phones is None:
        values_dict = {'all phones combined': compute(None)}
        title = f'Codebook {div_label} from {ref_desc}'
    else:
        if phones == 'all':
            phones = store.phones
        elif isinstance(phones, str):
            phones = [phones]
        values_dict = {phone: compute(phone) for phone in phones}
        if len(phones) == 1:
            title = f'Codebook {div_label} from {ref_desc} — phone: {phones[0]}'
        else:
            title = f'Codebook {div_label} from {ref_desc} per phone'

    return plot_over_checkpoints(
        models, values_dict, ylabel=div_label, title=title, xlim=xlim, ax=ax
    )


# ---------------------------------------------------------------------------
# Codebook utilization
# ---------------------------------------------------------------------------

def plot_utilization(store, phones=None, models=None, min_count=1,
                     xlim=(1000, 50000), ax=None):
    """
    Plot fraction of codebook entries used (>= min_count) over training.

    phones:   None      → model-level utilization (phones summed)
              'all'     → one line per phone
              list/str  → one line per phone in the list
    """
    models = _sorted_models(store, models)

    if phones is None:
        values = [codebook_utilization(store, m, min_count=min_count) for m in models]
        values_dict = {'all phones combined': values}
        title = 'Codebook utilization over training'
    else:
        if phones == 'all':
            phones = store.phones
        elif isinstance(phones, str):
            phones = [phones]
        values_dict = {
            phone: [codebook_utilization(store, m, phone=phone, min_count=min_count)
                    for m in models]
            for phone in phones
        }
        if len(phones) == 1:
            title = f'Codebook utilization over training — phone: {phones[0]}'
        else:
            title = 'Codebook utilization over training per phone'

    return plot_over_checkpoints(
        models, values_dict, ylabel='utilization', title=title, xlim=xlim, ax=ax
    )


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
# Overview panel
# ---------------------------------------------------------------------------

def plot_overview(store, models=None):
    """
    4-row × 2-column panel summary.

    Row 1  entropy          — subset phones / all phones
    Row 2  JS divergence    — subset phones (ref=previous) / all phones (ref=last)
    Row 3  JS divergence    — subset phones, wide panel (ref=last)
    Row 4  utilization      — subset phones / all phones
    """
    phones_subset = ['r', 's', 'm', 'd', 'z']

    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.3)

    ax_r1c1 = fig.add_subplot(gs[0, 0])
    ax_r1c2 = fig.add_subplot(gs[0, 1])
    ax_r2c1 = fig.add_subplot(gs[1, 0])
    ax_r2c2 = fig.add_subplot(gs[1, 1])
    ax_r3   = fig.add_subplot(gs[2, :])
    ax_r4c1 = fig.add_subplot(gs[3, 0])
    ax_r4c2 = fig.add_subplot(gs[3, 1])

    plot_entropy(store, phones=phones_subset, models=models,
                 xlim=(0, 30_000), ax=ax_r1c1)
    plot_entropy(store, phones=None, models=models,
                 xlim=(0, 100_000), ax=ax_r1c2)

    plot_divergence_trajectory(store, phones=phones_subset, models=models,
                               reference='last', xlim=(0, 100_000), ax=ax_r2c1)
    plot_divergence_trajectory(store, phones=None, models=models,
                               reference='last', xlim=(0, 100_000), ax=ax_r2c2)

    plot_divergence_trajectory(store, phones=phones_subset, models=models,
                               reference='previous', xlim=(0, 16_000), ax=ax_r3)

    plot_utilization(store, phones=phones_subset, models=models,
                     xlim=(0, 30_000), ax=ax_r4c1)
    plot_utilization(store, phones=None, models=models,
                     xlim=(0, 100_000), ax=ax_r4c2)

    for ax in [ax_r1c1, ax_r1c2, ax_r2c1, ax_r2c2, ax_r3, ax_r4c1, ax_r4c2]:
        ax.grid(alpha=0.3)

    return fig
