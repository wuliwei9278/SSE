import random
import numpy as np
from multiprocessing import Process, Queue


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, item_graph, usernum, itemnum, 
                    batch_size, maxlen, rho, 
                    threshold_user, threshold_item,
                    result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])

        #candidates = range(1, itemnum + 1)

        for i in reversed(user_train[user][:-1]):
            seq[idx] = i

            if random.random() > threshold_item:
                if nxt not in item_graph:
                    nxt = np.random.randint(1, itemnum + 1)
                else:
                    ratio = float(itemnum) / (len(item_graph[nxt]) * (rho - 1) + itemnum)   
                    if random.random() > ratio:
                        nxt = random.choice(item_graph[nxt])
                    else:
                        nxt = np.random.randint(1, itemnum + 1)
            pos[idx] = nxt

            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break
        
        if random.random() > threshold_user:
            user = np.random.randint(1, usernum + 1)
             
        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, item_graph, usernum, itemnum, batch_size=64, maxlen=10, 
                 rho=2.0, threshold_user=1.0, threshold_item=1.0, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      item_graph, 
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      rho,
                                                      threshold_user,
                                                      threshold_item,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
