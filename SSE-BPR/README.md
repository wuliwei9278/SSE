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

### Running the codes
We include movielens1m data in `ml1m-50-oc` as reference.

To run the codes:
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
