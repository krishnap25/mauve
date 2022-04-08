# MAUVE

MAUVE is a library built on PyTorch and HuggingFace Transformers to measure the gap between neural text and human text 
with the eponymous MAUVE measure, 
introduced [in this paper](https://arxiv.org/pdf/2102.01454.pdf) (NeurIPS 2021 Ooutstanding Paper).


### [Documentation Link](krishnap25.github.io/mauve/)

### _New: MAUVE is available via [HuggingFace Datasets](https://huggingface.co/docs/datasets/how_to_metrics.html)!_


**Features**:
- MAUVE with quantization using *k*-means. 
- Adaptive selection of *k*-means hyperparameters. 
- Compute MAUVE using pre-computed GPT-2 features (i.e., terminal hidden state), 
    or featurize raw text using HuggingFace transformers + PyTorch.
- New: minibatching for efficient implementation. 

## Installation

For a direct install, run this command from your terminal:
```
pip install mauve-text
``` 
    
## Citation
If you find this package useful, or you use it in your research, please cite:
```
@inproceedings{pillutla-etal:mauve:neurips2021,
  title={MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers},
  author={Pillutla, Krishna and Swayamdipta, Swabha and Zellers, Rowan and Thickstun, John and Welleck, Sean and Choi, Yejin and Harchaoui, Zaid},
  booktitle = {NeurIPS},
  year      = {2021}
}

```
Further, the Frontier Integral was introduced in this paper:
```
@inproceedings{liu-etal:divergence:neurips2021,
  title={{Divergence Frontiers for Generative Models: Sample Complexity, Quantization Effects, and Frontier Integrals}},
  author={Liu, Lang and Pillutla, Krishna and  Welleck, Sean and Oh, Sewoong and Choi, Yejin and Harchaoui, Zaid},
  booktitle = {NeurIPS},
  year      = {2021}
}
```
    
## Acknowledgements
This work was supported by NSF DMS-2134012, NSF CCF-2019844, NSF DMS-2023166, the DARPA MCS program through NIWC Pacific (N66001-19-2-4031), the CIFAR "Learning in Machines & Brains" program, a Qualcomm Innovation Fellowship, and faculty research awards.


