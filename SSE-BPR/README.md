# Codes for BPR and SSE-BPR 

## Building the codes

To install libraries dependencies:
```
sudo apt-get install libgoogle-glog-dev libgflags-dev liblapack-dev
```

To build the binaries:
```
cmake .
make
```

## Running the codes
We include movielens1m data in `ml1m-50-oc` as reference.

To run the codes for BPR:
```
./rep_bpr.sh
```

To run the codes for SSE-BPR:
1. `vim rep_bpr.sh` and change p1 and p2 to `p1=0.9` and `p2=0.99`, because for SSE-BPR, the option `--user_threshold` and `--item_threshold` become 2 additional parameters requires tuning 
2. type
```
./rep_bpr.sh
```


Options for BPR in the shell script:
* `--nepochs` (default 10): number of iterations of SGD
* `--nfactors` (default 30): dimensionality of the learned user and item factors
* `--use_biases` (default false): whether to use additive item biases
* `--user_lambda`: regularization coefficient on user factors
* `--item_lambda`: regularization coefficient on item factors
* `--bias_lambda`: regularization coefficient on biases
* `--init_learning_rate`: initial learning rate
* `--num_negative_samples` (default 3): number of random negatives sampled for each positive item
* `--user_threshold` (default 1.0): with probability 1 - user_threshold to replace user index
* `--item_threshold` (default 1.0): with probability 1 - user_threshold to replace item index

For more details, refer to the original README in `qmf` folder or the original github repo.
