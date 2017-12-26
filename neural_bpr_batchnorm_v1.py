"""
Neural Bayesian Personalized Ranking
embedding layer + one hidden layer with relu activation function
output layer is BPR loss function
add batch normalization module
"""
import numpy as np
import tensorflow as tf
import os
import sys
import random
from collections import defaultdict
import heapq
import math
import argparse


def load_data():
    '''
    As for experiment, all ratings are removed.
    '''
    data_path = "movie/processed_ratings.dat"
    user_ratings = defaultdict(set)
    max_u_id = -1
    max_i_id = -1
    with open(data_path, 'r') as f:
        for line in f:
            linetuple = line.strip().split("::")
            u = int(linetuple[0])
            i = int(linetuple[1])
            user_ratings[u].add(i)
            max_u_id = max(u, max_u_id)
            max_i_id = max(i, max_i_id)

    return max_u_id, max_i_id, user_ratings


def generate_test(user_ratings):
    '''
    for each user, random select one of his(her) rating into test set
    leave one out
    '''
    user_test = dict()
    for u, i_set in user_ratings.items():
        user_test[u] = random.sample(i_set, 1)[0]
    return user_test


def generate_train_batch(user_ratings, user_ratings_test, item_count, batch_size):
    '''
    uniform sampling (user, item_rated, item_not_rated)
    '''
    t = []
    for b in xrange(batch_size):
        u = random.sample(user_ratings.keys(), 1)[0]
        i = random.sample(user_ratings[u], 1)[0]
        while i == user_ratings_test[u]:
            i = random.sample(user_ratings[u], 1)[0]

        j = random.randint(1, item_count)
        while j in user_ratings[u]:
            j = random.randint(1, item_count)
        t.append([u, i, j])

    return np.array(t)


def generate_test_batch(user_ratings, user_ratings_test, item_count):
    '''
    for an user u and an item i rated by u,
    generate pairs (u,i,j) for all subsampled item j which u has not rated
    it's convinent for computing AUC score for u
    '''
    for u in user_ratings.keys():
        t = []
        negative_item_list = []
        i = user_ratings_test[u]
        cnt = 0
        while cnt < 100:
            j = random.choice(xrange(1, item_count + 1))
            if j not in negative_item_list and j not in user_ratings[u]:
                t.append([u, i, j])
                negative_item_list.append(j)
                cnt += 1

        yield np.array(t), [u, i, negative_item_list]


"""
compute HR@10 and NDCG@10 for one user
"""
def eval_one_rating(user_mat, item_mat, item_bias, u, i, j_list):

    map_item_score = {}
    map_item_score[i] = np.dot(user_mat[u], item_mat[i]) + item_bias[i]
    for j in j_list:
        map_item_score[j] = np.dot(user_mat[u], item_mat[j]) + item_bias[j]

    rank_list = heapq.nlargest(10, map_item_score, key = map_item_score.get)
    hr = getHitRate(rank_list, i)
    ndcg = getNDCG(rank_list, i)
    return [hr, ndcg]


def getHitRate(rank_list, purchased_item):
    if purchased_item in rank_list:
        return 1
    return 0


# normalized discounted cumulative gain (NDCG)
def getNDCG(rank_list, purchased_item):
    if purchased_item in rank_list:
        ranked_idx = rank_list.index(purchased_item)
        return float(1) / (math.log(ranked_idx + 2, 2))
    return 0


# nbpr: neural Bayesian personalized ranking

def nbpr(user_count, item_count, L, M, regulation_rate, learning_rate):
    """
    L: embedding layer dimension
    M: hidden layer dimension
    """
    u = tf.placeholder(tf.int32, [None])
    i = tf.placeholder(tf.int32, [None])
    j = tf.placeholder(tf.int32, [None])
    # the behavior of batch normalization is different between training and test phase
    phase = tf.placeholder(tf.bool, name = 'phase')

    # embedding layer
    user_emb_w = tf.Variable(tf.random_normal([user_count+1, L], stddev = 0.01),
                             name = "user_emb_w")
    item_emb_w = tf.Variable(tf.random_normal([item_count+1, L], stddev = 0.01),
                             name = "item_emb_w")
    item_b = tf.Variable(tf.zeros([item_count+1]), name = "item_b")

    # W11: weight parameter for first hidden layer for users
    # W12: weight parameter for first hidden layer for items
    W11 = tf.Variable(tf.random_normal([L, M], stddev = 0.01))
    B11 = tf.Variable(tf.ones([M]) / 10)
    W12 = tf.Variable(tf.random_normal([L, M], stddev = 0.01))
    B12 = tf.Variable(tf.ones([M]) / 10)

    u_emb = tf.nn.embedding_lookup(user_emb_w, u)
    i_emb = tf.nn.embedding_lookup(item_emb_w, i)
    j_emb = tf.nn.embedding_lookup(item_emb_w, j)
    Y11 = tf.nn.relu(tf.matmul(u_emb, W11) + B11)
    Y11 = tf.contrib.layers.batch_norm(Y11, center = True, scale = True,
                                       is_training = phase)
    Y12 = tf.nn.relu(tf.matmul(i_emb, W12) + B12)
    Y12 = tf.contrib.layers.batch_norm(Y12, center = True, scale = True,
                                       is_training = phase)
    Y13 = tf.nn.relu(tf.matmul(j_emb, W12) + B12)
    Y13 = tf.contrib.layers.batch_norm(Y13, center = True, scale = True,
                                       is_training = phase)
    i_b = tf.nn.embedding_lookup(item_b, i)
    j_b = tf.nn.embedding_lookup(item_b, j)

    # BPR loss in output layer
    Y = i_b - j_b + tf.reduce_sum(tf.multiply(Y11, (Y12 - Y13)), 1)

    # AUC for one user:
    # reasonable iff all (u,i,j) pairs are from the same user
    # average AUC = mean( auc for each user in test set)
    # mf_auc = tf.reduce_mean(tf.to_float(x > 0))
    embedding_score = i_b - j_b + tf.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1)
    correct_pred = tf.greater(embedding_score, 0)
    mf_auc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    
    l2_norm = tf.add_n([
            tf.reduce_sum(tf.multiply(u_emb, u_emb)),
            tf.reduce_sum(tf.multiply(i_emb, i_emb)),
            tf.reduce_sum(tf.multiply(j_emb, j_emb)),
            tf.reduce_sum(tf.multiply(i_b, i_b)),
            tf.reduce_sum(tf.multiply(j_b, j_b)),
            tf.reduce_sum(tf.multiply(W11, W11)),
            tf.reduce_sum(tf.multiply(W12, W12)),
            tf.reduce_sum(tf.multiply(B11, B11)),
            tf.reduce_sum(tf.multiply(B12, B12))
        ])

    #regulation_rate = 0.0001
    bprloss = regulation_rate * l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(Y)))
    train_op = tf.train.AdagradOptimizer(learning_rate).minimize(bprloss)

    return u, i, j, phase, mf_auc, bprloss, train_op, user_emb_w, item_emb_w, item_b


def run_nbpr(user_count, item_count, user_ratings, user_ratings_test,
             L, M, regulation_rate, learning_rate, batch_size):

    with tf.Session() as session:
        u, i, j, phase, mf_auc, bprloss, train_op, user_emb_w, item_emb_w, item_b = \
                nbpr(user_count, item_count, L, M, regulation_rate, learning_rate)
        session.run(tf.global_variables_initializer())
        for epoch in xrange(1, 11):
            _batch_bprloss = 0
            for k in xrange(1, 5000): # uniform samples from training set
                uij = generate_train_batch(user_ratings, user_ratings_test, item_count, batch_size)

                _bprloss, train_opt = session.run([bprloss, train_op],
                                          feed_dict={u:uij[:,0], i:uij[:,1], j:uij[:,2], phase: 1})

                _batch_bprloss += _bprloss

            print "epoch: ", epoch
            print "bpr_loss: ", _batch_bprloss / k

            _auc_sum = 0.0
            _hr_sum = 0.0
            _ndcg_sum = 0.0
            # each batch will return only one user's auc, hit rate, and ndcg
            user_mat, item_mat, item_bias = session.run([user_emb_w, item_emb_w, item_b])
            for t_uij, uij_list in generate_test_batch(user_ratings, user_ratings_test, item_count):

                _auc = session.run(mf_auc, feed_dict={u:t_uij[:,0], i:t_uij[:,1], j:t_uij[:,2], phase:0})
                _auc_sum += _auc

                _hr, _ndcg = eval_one_rating(user_mat, item_mat, item_bias, uij_list[0], uij_list[1], uij_list[2])
                _hr_sum += _hr
                _ndcg_sum += _ndcg

            print  "test_auc: ", _auc_sum/user_count
            print "test_hr: ", _hr_sum / user_count
            print "test_ndcg: ", _ndcg_sum / user_count
            print


def parse_args():
    """
    parse the embedding model arguments
    """
    parser_arg = argparse.ArgumentParser(description =
                                        "Bayesian Personalized Ranking for top-N recommendation system")
    parser_arg.add_argument('L', type = int, default = 20,
                            help = 'number of embedding in BPR')
    parser_arg.add_argument('M', type = int, default = 10,
                            help = 'number of neurons in first hidden layer')
    parser_arg.add_argument('regulation_rate', type = float, default = 0.0001,
                            help = 'matrix regularization parameter')
    parser_arg.add_argument('learning_rate', type = float, default = 0.01,
                            help = 'learning rate during min-batch gradient descent')
    parser_arg.add_argument('batch_size', type = int, default = 50,
                            help = 'min-batch size')
    return parser_arg.parse_args()


def main(args):

    user_count, item_count, user_ratings = load_data()
    user_ratings_test = generate_test(user_ratings)
    run_nbpr(user_count, item_count, user_ratings, user_ratings_test,
            args.L, args.M, args.regulation_rate, args.learning_rate, args.batch_size)


if __name__ == '__main__':

    args = parse_args()
    main(args)
