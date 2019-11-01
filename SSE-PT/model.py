from modules import *
import time
import random
import numpy as np


class SSE():
    """ Create a SSE class assuming we know indices for each class"""
    def __init__(self, idx_img_class, idx_txt_class, sse_prob_img, sse_prob_txt,
                rho_img, rho_txt, num_img, num_txt):
        self.idx_img_class = [list(x) for x in idx_img_class]
        self.idx_txt_class = [list(x) for x in idx_txt_class]
        self.sse_prob_img = sse_prob_img
        self.sse_prob_txt = sse_prob_txt
        self.rho_img = rho_img
        self.rho_txt = rho_txt
        self.num_img = num_img
        self.num_txt = num_txt

    def apply_sse_se(self, batch_idx):
        sse_batch_idx = []
        for i in range(len(batch_idx)):
            batch = batch_idx[i]
            idx_img = batch[0]
            idx_txt = batch[1]
            if random.random() < self.sse_prob_img:
                idx_img = np.random.randint(0, self.num_img)
            if random.random() < self.sse_prob_txt:
                idx_txt = np.random.randint(0, self.num_txt)
            sse_batch_idx.append((idx_img, idx_txt))			
        return sse_batch_idx 

    def apply_sse_graph(self, batch_idx, class_labels):
        # tuple: first being image, second being text
        #batch_idx = [(2, 7), (4, 5), (1, 3)]
        #class_labels = [(0, 0), (1, 2), (2, 2)]
        sse_batch_idx = []
        new_labels = []
        for i in range(len(batch_idx)):
            batch = batch_idx[i]
            idx_img = batch[0]
            idx_txt = batch[1]
            # class labels
            labels = class_labels[i]
            label_img = labels[0]
            label_txt = labels[1]
            if random.random() < self.sse_prob_img:
                if random.random() < 1 / self.rho_img:
                    idx_img = np.random.randint(0, self.num_img)
                else:
                    rand_idx = np.random.randint(0, len(self.idx_img_class[label_img]))
                    idx_img = self.idx_img_class[label_img][rand_idx]
                    #idx_img = np.random.choice(self.idx_img_class[label_img])
            if random.random() < self.sse_prob_txt:
                if random.random() < 1 / self.rho_txt:
                    idx_txt = np.random.randint(0, self.num_txt)
                else:
                    rand_idx = np.random.randint(0, len(self.idx_txt_class[label_txt]))
                    idx_txt = self.idx_txt_class[label_txt][rand_idx]
                    #idx_txt = np.random.choice(self.idx_txt_class[label_txt])
            sse_batch_idx.append((idx_img, idx_txt))
            if random.random() < 0.5:
                new_label = (labels[0], labels[0])
            else:
                new_label = (labels[1], labels[1])
            new_labels.append(new_label)
        return sse_batch_idx, new_labels
'''
start_time = time.time()
idx_img_class = [set([0, 2, 7, 8]), set([4, 5]), set([1, 3, 6])]
idx_txt_class = [set([0, 2, 7, 8]), set([4, 6]), set([1, 3, 5])]
batch_idx = [(2, 7), (4, 5), (1, 3)]
class_labels = [(0, 0), (1, 2), (2, 2)]
sse_prob = 0.1
rho = 100
n = 9
sse = SSE(idx_img_class, idx_txt_class, sse_prob, sse_prob, rho, rho, n, n)
for i in range(1000):
    sse_batch_idx = sse.apply_sse_se(batch_idx)
    sse_batch_idx, new_labels = sse.apply_sse_graph(batch_idx, class_labels)
    print(sse_batch_idx, new_labels)

elapsed_time = time.time() - start_time
print(elapsed_time)
'''

class Model():
    def __init__(self, usernum, itemnum, args, reuse=tf.AUTO_REUSE):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        with tf.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.item_hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )

            # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.item_hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )
            self.seq += t

            # User Encoding
            u0_latent, user_emb_table = embedding(self.u[0],
                                                 vocab_size=usernum + 1,
                                                 num_units=args.user_hidden_units,
                                                 zero_pad=False,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="user_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )
            # Has dim: B by C
            u_latent = embedding(self.u,
                                 vocab_size=usernum + 1,
                                 num_units=args.user_hidden_units,
                                 zero_pad=False,
                                 scale=True,
                                 l2_reg=args.l2_emb,
                                 scope="user_embeddings",
                                 with_t=False,
                                 reuse=reuse
                                 )
            # Change dim to B by T by C
            self.u_latent = tf.tile(tf.expand_dims(u_latent, 1), [1, tf.shape(self.input_seq)[1], 1])

            # Concat item embedding with user embedding
            self.hidden_units = args.item_hidden_units + args.user_hidden_units
            self.seq = tf.reshape(tf.concat([self.seq, self.u_latent], 2),
                                  [tf.shape(self.input_seq)[0], -1, self.hidden_units])
            
            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            # Build blocks
            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):

                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=self.hidden_units,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_units, self.hidden_units],
                                           dropout_rate=args.dropout_rate, is_training=self.is_training)
                    self.seq *= mask

            self.seq = normalize(self.seq)
        
        user_emb = tf.reshape(self.u_latent, [tf.shape(self.input_seq)[0] * args.maxlen, 
                                              args.user_hidden_units])

        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)

        pos_emb = tf.reshape(tf.concat([pos_emb, user_emb], 1), [-1, self.hidden_units])
        neg_emb = tf.reshape(tf.concat([neg_emb, user_emb], 1), [-1, self.hidden_units])

        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, self.hidden_units])

        self.test_item = tf.placeholder(tf.int32, shape=(101))
        test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        
        test_user_emb = tf.tile(tf.expand_dims(u0_latent, 0), [101, 1])
        # combine item and user emb
        test_item_emb = tf.reshape(tf.concat([test_item_emb, test_user_emb], 1), [-1, self.hidden_units])

        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, 101])
        self.test_logits = self.test_logits[:, -1, :]

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        self.loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        tf.summary.scalar('auc', self.auc)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq, item_idx):
        return sess.run(self.test_logits,
                        {self.u: u, self.input_seq: seq, self.test_item: item_idx, self.is_training: False})
