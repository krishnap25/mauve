# Author: Krishna Pillutla
# License: GPLv3

import math
import numpy as np
import time
from types import SimpleNamespace

import faiss
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import auc as compute_area_under_curve

try:
    import torch
    FOUND_TORCH = True
except (ImportError, ModuleNotFoundError):
    FOUND_TORCH = False

try:
    import transformers
    FOUND_TRANSFORMERS = True
except (ImportError, ModuleNotFoundError):
    FOUND_TRANSFORMERS = False

if FOUND_TORCH and FOUND_TRANSFORMERS:
    # only needed for tokenizing
    from .utils import get_tokenizer, get_model, featurize_tokens_from_model, get_device_from_arg


MODEL, TOKENIZER, MODEL_NAME = None, None, None

def compute_mauve(
        p_features=None, q_features=None,
        p_tokens=None, q_tokens=None,
        p_text=None, q_text=None,
        num_buckets='auto', pca_max_data=-1, kmeans_explained_var=0.9,
        kmeans_num_redo=5, kmeans_max_iter=500,
        featurize_model_name='gpt2-large', device_id=-1, max_text_length=1024,
        divergence_curve_discretization_size=25, mauve_scaling_factor=5,
        verbose=False, seed=25, batch_size=1, use_float64=False,
):

    """
    Compute the MAUVE score between two text generations P and Q.

    P is either specified as ``p_features``, ``p_tokens``, or ``p_text``. Same with Q.

    :param ``p_features``: ``numpy.ndarray`` of shape (n, d), where n is the number of generations.
    :param ``q_features``: ``numpy.ndarray`` of shape (n, d), where n is the number of generations.
    :param ``p_tokens``: list of length n, each entry is torch.LongTensor of shape (1, length).
    :param ``q_tokens``: list of length n, each entry is torch.LongTensor of shape (1, length).
    :param ``p_text``: list of length n, each entry is a string.
    :param ``q_text``: list of length n, each entry is a string.
    :param ``num_buckets``: the size of the histogram to quantize P and Q. Options: ``'auto'`` (default, which is n/10) or an integer.
    :param ``pca_max_data``: the number data points to use for PCA. If `-1`, use all the data. Default -1.
    :param ``kmeans_explained_var``: amount of variance of the data to keep in dimensionality reduction by PCA. Default 0.9.
    :param ``kmeans_num_redo``: number of times to redo k-means clustering (the best objective is kept). Default 5. 
        Try reducing this to 1 in order to reduce running time.
    :param ``kmeans_max_iter``: maximum number of k-means iterations. Default 500.
        Try reducing this to 100 in order to reduce running time.
    :param ``featurize_model_name``: name of the model from which features are obtained. Default 'gpt2-large'.
        We support all models which can be loaded from ``transformers.AutoModel.from_pretrained(featurize_model_name)``.
    :param ``device_id``: Device for featurization. Supply gpu_id (e.g. 0 or 3) to use GPU or -1 to use CPU.
    :param ``max_text_length``: maximum number of tokens to consider. Default 1024.
    :param ``divergence_curve_discretization_size``: Number of points to consider on the divergence curve. Default 25.
        Larger values do not offer much of a difference. 
    :param ``mauve_scaling_factor``: The constant``c`` from the paper. Default 5.
        See `Best Practices <index.html#best-practices-for-mauve>`_ for details.
    :param ``verbose``: If True, print running time updates.
    :param ``seed``: random seed to initialize k-means cluster assignments.
    :param ``batch_size``: Batch size for feature extraction.
        A larger batch size speeds up computation.
        You might have to experiment to find the largest batch size that fits in your GPU memory.
        See `here <https://github.com/krishnap25/mauve/issues/8#issuecomment-1082075240>`_ for details.

    :return: an object with fields mauve, frontier_integral, mauve_star, frontier_integral_star, p_hist, q_hist, divergence_curve, num_buckets.

    * ``out.mauve`` is a number between 0 and 1, the MAUVE score. Higher values means P is closer to Q.
    * ``out.frontier_integral``, a number between 0 and 1. Lower values mean that P is closer to Q. 
    * ``out.mauve_star`` and ``out.frontier_integral_star``  are the corresponding variants with smoothing (see Pillutla et al., JMLR 2023).
    * ``out.p_hist`` is the obtained histogram for P. Same for ``out.q_hist``.
    * ``out.divergence_curve`` contains the points in the divergence curve. It is of shape (m, 2), where m is ``divergence_curve_discretization_size``.
    * ``out.num_buckets`` is the quantization size and the shape of ``out.p_hist`` and ``out.q_hist``.

    """

    if p_features is None and p_tokens is None and p_text is None:
        raise ValueError('Supply at least one of p_features, p_tokens, p_text')
    if q_features is None and q_tokens is None and q_text is None:
        raise ValueError('Supply at least one of q_features, q_tokens, q_text')
    p_features = get_features_from_input(
        p_features, p_tokens, p_text, featurize_model_name, max_text_length,
        device_id, name="p", verbose=verbose, batch_size=batch_size, use_float64=use_float64,
    )
    q_features = get_features_from_input(
        q_features, q_tokens, q_text, featurize_model_name, max_text_length,
        device_id, name="q", verbose=verbose, batch_size=batch_size, use_float64=use_float64,
    )
    if num_buckets == 'auto':
        # Heuristic: use num_clusters = num_generations / 10.
        num_buckets = max(2, int(round(min(p_features.shape[0], q_features.shape[0]) / 10)))
    elif not isinstance(num_buckets, int):
        raise ValueError('num_buckets is expected to be an integer or "auto"')

    # Bin the features into histograms.
    t1 = time.time()
    p, q, p_smoothed, q_smoothed = cluster_feats(
        p_features, q_features, num_clusters=num_buckets,
        norm='l2', whiten=False, pca_max_data=pca_max_data,
        explained_variance=kmeans_explained_var, num_redo=kmeans_num_redo,
        max_iter=kmeans_max_iter, seed=seed, verbose=verbose
    )
    t2 = time.time()
    if verbose:
        print('total discretization time:', round(t2-t1, 2), 'seconds')

    # Divergence curve and MAUVE (no smoothing).
    mixture_weights = np.linspace(1e-6, 1-1e-6, divergence_curve_discretization_size)
    divergence_curve = get_divergence_curve_for_multinomials(p, q, mixture_weights, mauve_scaling_factor)
    x, y = divergence_curve.T
    idxs1 = np.argsort(x)
    idxs2 = np.argsort(y)
    mauve_score = 0.5 * (
        compute_area_under_curve(x[idxs1], y[idxs1]) +
        compute_area_under_curve(y[idxs2], x[idxs2])
    )
    fi_score = get_fronter_integral(p, q)

    # Divergence curve and MAUVE (with smoothing).
    x_s, y_s = get_divergence_curve_for_multinomials(p_smoothed, q_smoothed, mixture_weights, mauve_scaling_factor).T
    idxs1, idxs2 = np.argsort(x_s), np.argsort(y_s)
    mauve_star = 0.5 * (
        compute_area_under_curve(x_s[idxs1], y_s[idxs1]) +
        compute_area_under_curve(y_s[idxs2], x_s[idxs2])
    )
    fi_star = get_fronter_integral(p_smoothed, q_smoothed)
    to_return = SimpleNamespace(
        p_hist=p, q_hist=q, divergence_curve=divergence_curve, 
        mauve=mauve_score, frontier_integral=fi_score,
        mauve_star=mauve_star, frontier_integral_star=fi_star,
        num_buckets=num_buckets,
    )
    return to_return

def get_features_from_input(features, tokenized_texts, texts,
                            featurize_model_name, max_len, device_id, name, batch_size,
                            verbose=False, use_float64=False):
    global MODEL, TOKENIZER, MODEL_NAME
    if features is None:
        # Featurizing is necessary. Make sure the required packages are available.
        if not FOUND_TORCH:
            raise ModuleNotFoundError(
                """PyTorch not found. Please install PyTorch if you would like to use the featurization.
                    For details, see `https://github.com/krishnap25/mauve` 
                    and `https://pytorch.org/get-started/locally/`.
                """)
        if not FOUND_TRANSFORMERS:
            raise ModuleNotFoundError(
                """Transformers not found. Please install Transformers if you would like to use the featurization.
                    For details, see `https://github.com/krishnap25/mauve` 
                    and `https://huggingface.co/transformers/installation.html`.
                """)

        if tokenized_texts is None:
            # Tokenize texts.
            texts = [sen for sen in texts if len(sen) > 0]  # Remove empty strings.
            if len(texts) == 0:
                raise ValueError(f'Variable `{name}_text` is empty. Please provide non-empty strings.')
            if TOKENIZER is None or MODEL_NAME != featurize_model_name:
                if verbose: print('Loading tokenizer')
                TOKENIZER = get_tokenizer(featurize_model_name)
            if verbose: print('Tokenizing text...')
            tokenized_texts = [
                TOKENIZER.encode(sen, return_tensors='pt', truncation=True, max_length=max_len)
                for sen in texts
            ]
        else:
            # Make sure tokenized texts are not empty.
            if len(tokenized_texts) == 0:
                raise ValueError(f'Variable `{name}_tokens` is empty. Please provide non-empty tokenized texts.')
    
        # Use tokenized_texts to featurize.
        if TOKENIZER is None or MODEL_NAME != featurize_model_name:
            if verbose: print('Loading tokenizer')
            TOKENIZER = get_tokenizer(featurize_model_name)
        if MODEL is None or MODEL_NAME != featurize_model_name:
            if verbose: print('Loading model')
            MODEL = get_model(featurize_model_name, TOKENIZER, device_id)
            MODEL_NAME = featurize_model_name
        else:
            MODEL = MODEL.to(get_device_from_arg(device_id))
        if use_float64:
            MODEL = MODEL.double()
        if verbose: print('Featurizing tokens')
        features = featurize_tokens_from_model(MODEL, tokenized_texts, batch_size, name).detach().cpu().numpy()
    else:
        features = np.asarray(features)
        if features.shape[0] == 0:
            raise ValueError(f'Variable `{name}_features` is an empty array. Please provide non-empty features.')
    return features

def cluster_feats(p, q, num_clusters,
                  norm='none', whiten=True,
                  pca_max_data=-1,
                  explained_variance=0.9,
                  num_redo=5, max_iter=500,
                  seed=0, verbose=False):
    assert 0 < explained_variance < 1
    def _normalize(array):
        # Normalize sum of array to 1.
        # We assume non-negative entries with non-zero sum.
        return array / array.sum()
    if verbose:
        print(f'seed = {seed}')
    assert norm in ['none', 'l2', 'l1', None]
    data1 = np.vstack([q, p])
    if norm in ['l2', 'l1']:
        data1 = normalize(data1, norm=norm, axis=1)
    pca = PCA(n_components=None, whiten=whiten, random_state=seed+1)
    if pca_max_data < 0 or pca_max_data >= data1.shape[0]:
        pca.fit(data1)
    elif 0 < pca_max_data < data1.shape[0]:
        rng = np.random.RandomState(seed+5)
        idxs = rng.choice(data1.shape[0], size=pca_max_data, replace=False)
        pca.fit(data1[idxs])
    else:
        raise ValueError(f'Invalid argument pca_max_data={pca_max_data} with {data1.shape[0]} datapoints')
    s = np.cumsum(pca.explained_variance_ratio_)
    idx = np.argmax(s >= explained_variance)  # last index to consider
    if verbose:
        print(f'performing clustering in lower dimension = {idx}')
    data1 = pca.transform(data1)[:, :idx+1]
    # Cluster features and obtain the labels for each data point.
    data1 = data1.astype(np.float32)  # Faiss requires float32.
    t1 = time.time()
    kmeans = faiss.Kmeans(data1.shape[1], num_clusters, niter=max_iter,
                          verbose=verbose, nredo=num_redo, update_index=True,
                          seed=seed+2)
    kmeans.train(data1)
    _, labels = kmeans.index.search(data1, 1)
    labels = labels.reshape(-1)
    t2 = time.time()
    if verbose:
        print('kmeans time:', round(t2-t1, 2), 's')

    q_labels = labels[:len(q)]
    p_labels = labels[len(q):]

    # Convert cluster labels to histograms.
    q_bin_counts = np.histogram(
        q_labels, bins=num_clusters,
        range=[0, num_clusters], density=False
    )[0]
    p_bin_counts = np.histogram(
        p_labels, bins=num_clusters,
        range=[0, num_clusters], density=False
    )[0]
    # Histograms without smoothing (used for the original MAUVE).
    p_hist = _normalize(p_bin_counts)
    q_hist = _normalize(q_bin_counts)
    # Histograms with Krichevsky-Trofimov smoothing.
    # Used for MAUVE* suggested by by Pillutla et al. (JMLR 2023).
    p_hist_smoothed = _normalize(p_bin_counts + 0.5)
    q_hist_smoothed = _normalize(q_bin_counts + 0.5)
    return p_hist, q_hist, p_hist_smoothed, q_hist_smoothed


def kl_multinomial(p, q):
    assert p.shape == q.shape
    if np.logical_and(p != 0, q == 0).any():
        return np.inf
    else:
        idxs = np.logical_and(p != 0, q != 0)
        return np.sum(p[idxs] * np.log(p[idxs] / q[idxs]))


def get_divergence_curve_for_multinomials(p, q, mixture_weights, scaling_factor):
    divergence_curve = [[0, np.inf]] # extreme point
    for w in np.sort(mixture_weights):
        r = w * p + (1 - w) * q
        divergence_curve.append([kl_multinomial(q, r), kl_multinomial(p, r)])
    divergence_curve.append([np.inf, 0]) # other extreme point
    return np.exp(-scaling_factor * np.asarray(divergence_curve))

def get_fronter_integral(p, q, scaling_factor=2):
    total = 0.0
    for p1, q1 in zip(p, q):
        if p1 == 0 and q1 == 0:
            pass
        elif p1 == 0:
            total += q1 / 4
        elif q1 == 0:
            total += p1 / 4
        elif abs(p1 - q1) > 1e-8:
            t1 = p1 + q1
            t2 = p1 * q1 * (math.log(p1) - math.log(q1)) / (p1 - q1)
            total += 0.25 * t1 - 0.5 * t2
        # else: contribution is 0 
    return total * scaling_factor
