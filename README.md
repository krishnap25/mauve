# MAUVE

This package provides an implementation of MAUVE, an evaluation metric for open-ended text generation.
It was introduced [in this paper](https://arxiv.org/pdf/2102.01454.pdf).

MAUVE directly  compares  the distribution of machine-generated text to 
that of human language as the area under the divergence curve for the two distributions. 
MAUVE summarizes the  trade-off  between two types of errors: 
those arising from parts of the human distribution that the model distribution 
approximates  well, and those it does not. 

**Features**:
- MAUVE with quantization using *k*-means. 
- Adaptive selection of *k*-means hyperparameters. 
- Compute MAUVE using pre-computed GPT-2 features (i.e., terminal hidden state), 
    or featurize raw text using HuggingFace transformers + PyTorch.

For scripts to reproduce the experiments in the paper, please see 
[this repo](https://github.com/krishnap25/mauve-experiments).

## Installation

For a direct install, run this command from your terminal:
```
pip install mauve
``` 
If you wish to edit or contribute to MAUVE, you should install from source
```
git clone git@github.com:krishnap25/mauve.git
cd mauve
pip install -e .
``` 
Some functionality requires more packages. Please see the requirements below.

## Requirements
The installation command above installs the main requirements, which are:
- `numpy>=1.18.1`
- `scikit-learn>=0.22.1`
- `faiss-gpu>=1.7.0`
- `tqdm>=4.40.0`

In addition, if you wish to use featurization within MAUVE, you need to manually install:
- `torch>=1.1.0`: [Instructions](https://pytorch.org/get-started/locally/)
- `transformers>=3.2.0`:  Simply run `pip install transformers` after PyTorch has been installed 
    ([Detailed Instructions](https://huggingface.co/transformers/installation.html))



## Quick Start
Let `p_text` and `q_text` each be a list of strings, where each string is a complete generation (including context). 
For best practice, MAUVE needs at least a few thousand generations each for `p_text` and `q_text`
(the paper uses 5000 each).
For our demo, we use 100 generations each for fast running time.

To demonstrate the functionalities of this package on some real data, 
this repository provides some functionalities to
download and use sample data in the `./examples` folder
(these are not a part of the MAUVE package, you need to clone the repository for these).

Let use download some Amazon product reviews as well as
machine generations, provided by the 
[GPT-2 output dataset repo](https://github.com/openai/gpt-2-output-dataset)
 by running this command in our shell (downloads ~17M in size):
```bash
python examples/download_gpt2_dataset.py

```
The data is downloaded into the `./data` folder. 
We can load the data (100 samples out of the available 5000) in Python as 
```python
from examples import load_gpt2_dataset
p_text = load_gpt2_dataset('data/amazon.valid.jsonl', num_examples=100) # human
q_text = load_gpt2_dataset('data/amazon-xl-1542M.valid.jsonl', num_examples=100) # machine
```

We can now compute MAUVE as follows
(note that this requires installation of [PyTorch](https://pytorch.org) 
and HF [Transformers](https://huggingface.co/transformers)). 
```python
import mauve 

# call mauve.compute_mauve using raw text on GPU 0; each generation is truncated to 256 tokens
out = mauve.compute_mauve(p_text=p_text, q_text=q_text, device_id=0, max_text_length=256, verbose=False)
print(out.mauve) # prints 0.9917
```
This first downloads GPT-2 large tokenizer and pre-trained model (if you do not have them downloaded already). 
Even if you have the model offline, it takes it up to 30 seconds to load the model the first time. 
`out` now contains the fields:
- `out.mauve`: MAUVE score, a number between 0 and 1
- `out.divergence_curve`: a `numpy.ndarray` of shape (m, 2); plot it with matplotlib to view the divergence curve
- `out.p_hist`: a discrete distribution, which is a quantized version of the text distribution `p_text`
- `out.q_hist`: same as above, but with `q_text`  

You can plot the divergence curve using
```python
# Make sure matplotlib is installed in your environment
import matplotlib.pyplot as plt  
plt.plot(out.divergence_curve[:, 1], out.divergence_curve[:, 0])
```

## Other Ways of Using MAUVE 
For each text (in both `p_text` and `q_text`), 
MAUVE internally uses the terimal hidden state from GPT-2 large as a feature representation.
This featurization process can be rather slow 
(~10 mins for 5000 generations at a max length of 1024; 
but the implementation can be made more efficient, see [Contributing](#contributing)).
Alternatively, this package allows you to use cached hidden states directly
(this does not require PyTorch and HF Transformers to be installed): 
```python
# call mauve.compute_mauve using features obtained directly
# p_feats and q_feats are `np.ndarray`s of shape (n, dim)
# we use a synthetic example here
import numpy as np
p_feats = np.random.randn(100, 1024)  # feature dimension = 1024
q_feats = np.random.randn(100, 1024)
out = mauve.compute_mauve(p_features=p_feats, q_features=q_feats)
```

You can also compute MAUVE using the tokenized (BPE) representation using the GPT-2 vocabulary 
(e.g., obtained from using an explicit call to `transformers.GPT2Tokenizer`).
```python
# call mauve.compute_mauve using tokens on GPU 1
# p_toks, q_toks are each a list of LongTensors of shape [1, length]
# we use synthetic examples here
import torch
p_toks = [torch.LongTensor(np.random.choice(50257, size=(1, 32), replace=True)) for _ in range(100)]
q_toks = [torch.LongTensor(np.random.choice(50257, size=(1, 32), replace=True)) for _ in range(100)]
out = mauve.compute_mauve(p_tokens=p_toks, q_tokens=q_toks, device_id=1, max_text_length=1024)
```
To view the progress messages, pass in the argument `verbose=True` to `mauve.compute_mauve`.
You can also use different forms as inputs for `p` and `q`, e.g., 
`p` via `p_text` and `q` via `q_features`. 

## Available Options
`mauve.compute_mauve` takes the following arguments
- `p_features`: `numpy.ndarray` of shape (n, d), where n is the number of generations
- `q_features`: `numpy.ndarray` of shape (n, d), where n is the number of generations
- `p_tokens`: list of length n, each entry is torch.LongTensor of shape (1, length); length can vary between generations
- `q_tokens`: list of length n, each entry is torch.LongTensor of shape (1, length); length can vary between generations
- `p_text`: list of length n, each entry is a string
- `q_text`: list of length n, each entry is a string
- `num_buckets`: the size of the histogram to quantize P and Q. Options: 'auto' (default) or an integer
- `pca_max_data`: the number data points to use for PCA dimensionality reduction prior to clustering. If `-1`, use all the data. Default -1
- `kmeans_explained_var`: amount of variance of the data to keep in dimensionality reduction by PCA. Default 0.9
- `kmeans_num_redo`: number of times to redo k-means clustering (the best objective is kept). Default 5
- `kmeans_max_iter`: maximum number of k-means iterations. Default 500
- `featurize_model_name`: name of the model from which features are obtained. Default `'gpt2-large'`
    Use one of `['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']`.
- `device_id`: Device for featurization. Supply a GPU id (e.g. 0 or 3) to use GPU. If no GPU with this id is found, use CPU
- `max_text_length`: maximum number of tokens to consider. Default 1024
- `divergence_curve_discretization_size`: Number of points to consider on the divergence curve. Default 25
- `mauve_scaling_factor`: "c" from the paper. Default 5.
- `verbose`: If True (default), print running time updates
- `seed`: random seed to initialize *k*-means cluster assignments.

Note: `p` and `q` can be of different lengths, but it is
recommended that they are the same length.


## Contributing
If you find any bugs, please raise an issue on GitHub. 
If you would like to contribute, please submit a pull request.
We encourage and highly value community contributions.

Some features which would be good to have are:
- batched implementation featurization (current implementation sequentially featurizes generations); 
    this requires appropriate padding/masking
- featurization in HuggingFace Transformers with a  TensorFlow backend.
    
## Best Practices for MAUVE
MAUVE is quite different from most metrics in common use, so here are a few guidelines on proper usage of MAUVE:
1. *Relative comparisons*: 
    - We find that MAUVE is best suited for relative comparisons while 
    the absolute MAUVE score is less meaningful. 
    - For instance if we wish to find which of `model1` and `model2` are better at generating 
    the human distribution, we can compare `MAUVE(text_model1, text_human)` and `MAUVE(text_model2, text_human)`.
    - The absolute number  `MAUVE(text_model1, text_human)` can vary based on the hyperparameters selected below, 
        but the relative trends remain the same.
    - One must ensure that the hyperparameters are exactly the same for 
        the MAUVE scores under comparison.
    - Some hyperparameters are described below. 
2. *Number of generations*: 
    - MAUVE computes the similarity between two *distributions*. 
    - Therefore, each distribution must contain at least
    a few thousand samples (we use 5000 each). MAUVE with a smaller number of samples is biased towards optimism
    (that is, MAUVE typically goes down as the number of samples increase) 
    and exhibits a larger standard deviation between runs.
3. *Number of clusters (discretization size)*: 
    - We take `num_buckets` to be 0.1 * the number of samples. 
    - The performance of MAUVE is quite robust to this, provided the number of generations is not too small. 
4. *MAUVE is too large or too small*:
    - The parameter `mauve_scaling_parameter` controls the absolute value of the MAUVE score,
        without changing the relative ordering between various methods. 
        The main purpose of this parameter is to help with interpretability.  
    - If you find that all your methods get a very high MAUVE score (e.g., 0.995, 0.994),
        try increasing the value of `mauve_scaling_factor`.
        (note: this also increases the per-run standard deviation of MAUVE). 
    - If you find that all your methods get a very low MAUVE score (e.g. < 0.4), then 
        try decreasing the value of `mauve_scaling_factor`.
5. *MAUVE takes too long to run*: 
    - You can also try reducing the number of clusters using the argument `num_buckets`. The
        clustering algorithm's run time scales as the square of the number of clusters. 
        Once the number of clusters exceeds 500, the clustering really starts to slow down. 
        In this case, it could be helpful to set the number of clusters to 500
        by overriding the default (which is `num_data_points / 10`, so use this when the number of 
        samples for each of p and q is over 5000).
    - In this case, try reducing the clustering hyperparameters: 
        set `kmeans_num_redo` to `1`, and if this does not work, `kmeans_max_iter` to `100`.
        This enables the clustering to run faster at the cost of returning a worse clustering. 
    
## Citation
If you find this package useful, or you use it in your research, please cite:
```
@article{pillutla-etal:mauve:preprint2021,
title = {{MAUVE: Human-Machine Divergence Curves for Evaluating Open-Ended Text Generation}},
author = {Krishna Pillutla and Swabha Swayamdipta and Rowan Zellers and John Thickstun and Yejin Choi and Zaid Harchaoui}
journal={arXiv preprint},
year = {2021},
}
```
    
## Acknowledgements
This work was supported by NSF CCF-2019844,the DARPA MCS program through NIWC Pacific(N66001-19-2-4031),
the CIFAR program "Learning in Machines and Brains", 
a Qualcomm Innovation Fellowship, and faculty research awards. 


