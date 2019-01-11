# Stochastic Shared Embeddings (SSE) for Better Recommendations


## Description: 
This repo consists of 3 folders:
1. SSE-MF for Explicit Feedback
2. SSE-BPR for Implicit Feedback
3. SSE-PT for Sequential Recommendation


## Instructions on how to run the code:
1. For Explicit Feedback, go to folder `./SSE-MF` and follow README file there
2. For Implicit Feedback, go to folder `./SSE-BPR` and follow README file there
3. For Sequentail Recommendation, go to folder `./SSE-PT` and follow README file there


## System Requirements:
- We assume everyone uses a linux machine/server. We don't consider the Windows/Mac usage case.
- For SSE-MF, Julia 0.6 is needed. Julia 0.7 may be okay but the codes won't work in Julia 1.0 for sure without proper modifications.
- For SSE-BPR, gcc 5.0+, CMake 2.8+ and glog, gflags and lapack libraries are needed for training. Julia 0.6 is needed to evaluation.
- For SSE-PT, tensorflow 1.11.0+, Python 2.7/3.5 and Nvdia GPUs are needed for training and evaluation at a reasonable amount of time.  

