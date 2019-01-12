# Codes for MF and SSE-MF for the paper submission
- Codes should work under Julia 0.6.0, we include movielens1m data as reference.


## Desription
- ```data``` includes the test and train data for movielens1m in the form of "user id, movie id, rating" .
- ```code``` includes Julia code which implements the MF, SSE-MF algorithm.
- ```mf.jl``` is Julia code implementation of the Matrix Factorization and its evaluation.
- ```sse_mf.jl``` is Julia code implementation of the Stocastic Shared Emeddings and its evaluation.
- ```see_dropout_sgf_mf``` is Julia code implementation of the l_2 + Dropout + SSE and its evaluation.
 

## Instructions on how to run the code
1. Prepare the dataset of the form (user, item, ratings) in csv files. (Example: data/ml-1m_train_ratings.csv)

2. Go to the folder containing ```data``` and ```code``` and use command line to start Julia:
```$ julia```

3.
- To run MF, do
```
include("code/mf.jl")
main("data/ml1m_train_ratings.csv", "data/ml1m_test_ratings.csv", 100, 0.1)
```
Note: The third paramter ```100``` is the rank we choose for the movielens1m data. The fourth parameter ```0.1``` is the l_2 regularization lambda.
 
- To run SSE-MF, do
```
include("code/sse_mf.jl")
main("data/ml1m_train_ratings.csv", "data/ml1m_test_ratings.csv", 100, 0.1, 0.995)
```
Note: The fifth parameter is the user and item threshold. (eg. ```0.995``` means the replacement probability of both user side and item side are ```1-0.995 = 0.005```)

- To run the l_2 + Dropout + SSE, do
```
include("sse_dropout_sgd_mf")
main("data/ml1m_train_ratings.csv", "data/ml1m_test_ratings.csv", 100, 0.1, 0.995, 0.9)
```
Note: The sixth parameter is the dropout threshould. (eg. ```0.9``` means the dropout rate is ```1-0.9 = 0.1```)  
