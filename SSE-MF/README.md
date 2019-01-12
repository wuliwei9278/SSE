# Codes for MF and SSE-MF


Note that we only tested our codes under Julia 0.6, and we only include movielens1m data for reference.


## Desription
- data folder: training data “ml1m_train_ratings.csv” and test data "ml1m_test_ratings.csv" can be found there, the data is of the form: “user id, movie id, rating”.
- code folder: Julia codes which implement MF and SSE-MF algorithms described in the paper is put into this folder (instructions on how to run the codes are given below).

## Instructions on how to run the code

### To run MF

Type below in Julia command line
```
julia>  include("code/mf.jl")
julia>  main("data/ml1m_train_ratings.csv", "data/ml1m_test_ratings.csv", 100, 0.1)
```

Note: The first two arguments are the paths to training data and test data. The third argument `100` is the rank we choose for MF. The fourth argument `0.1` is the l_2 regularization lambda. 
 
### To run SSE-MF

Same as before, start julia and then
```
julia>  include("code/sse_mf.jl")
julia>  main("data/ml1m_train_ratings.csv", "data/ml1m_test_ratings.csv", 100, 0.1, 0.995, 0.995)
```

Note: The first 4 arguments are the same as MF. The last 2 arguments are thresholds of replacing user index and item item index. (eg. `0.995` means the replacement probability of both user side and item side are `1 - 0.995 = 0.005`)

### To run best method (l2 + SSE + dropout)

Same as before, start julia and then
```
julia>  include("code/sse_dropout_sgd_mf")
julia>  main("data/ml1m_train_ratings.csv", "data/ml1m_test_ratings.csv", 100, 0.1, 0.995, 0.9)
```

Note: For simplicity reason, here we combine the two arguments in SSE-MF into one argument `0.995` and the last argument `0.9` represents `1 - dropout_probability`, i.e. here we use dropout probability 0.1 for both user and item embeddings.
