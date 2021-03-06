# Stochastic Shared Embeddings: Data-driven Regularization of Embedding Layers
[![Hex.pm](https://img.shields.io/hexpm/l/plug.svg)](LICENSE)
This paper has been accepted to this year's (2019) NeurIPS.
Please cite the paper:
```
@article{wu2019stochastic,
  title={Stochastic Shared Embeddings: Data-driven Regularization of Embedding Layers},
  author={Wu, Liwei and Li, Shuqing and Hsieh, Cho-Jui and Sharpnack, James},
  journal={arXiv preprint arXiv:1905.10630},
  year={2019}
}
```

## Description: 
This repo consists of 3 folders:
1. SSE-MF for Explicit Feedback
2. SSE-BPR for Implicit Feedback
3. SSE-PT for Sequential Recommendation

Note that:
- SSE stands for Stochastic Shared Embeddings
- MF stands for Matrix Factorization
- BPR stands for Bayseian Personalized Ranking
- PT stands for Personalized Transformer

## Instructions on how to run the code:
1. For explicit feedback setting, `cd SSE-MF` and follow README file there
2. For implicit feedback setting, `cd SSE-BPR` and follow README file there
3. For sequential recommendation setting, `cd SSE-PT` and follow README file there


## System Requirements:
- We assume everyone uses a linux machine/server. We don't consider the Windows/Mac usage case.
- For SSE-MF, Julia 0.6 is needed. Julia 0.7 may be okay but the codes won't work in Julia 1.0 without proper modifications.
- For SSE-BPR, gcc 5.0+, CMake 2.8+ and glog, gflags and lapack libraries are needed for training. Julia 0.6 is needed for evaluation.
- For SSE-PT, tensorflow 1.11.0+, Python 2.7/3.5 and Nvdia GPUs are needed for training and evaluation at a reasonable amount of time.  

