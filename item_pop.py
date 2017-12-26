"""
top-N recommendation purely based on item popularity
most basic baseline
"""
import heapq
from collections import defaultdict
import random
import math


def load_data():

    data_path = "movie/processed_ratings.dat"
    user_ratings = defaultdict(set)
    item_user_dict = defaultdict(set)
    max_u_id = -1
    max_i_id = -1
    with open(data_path, 'r') as f:
        for line in f:
            linetuple = line.strip().split("::")
            u = int(linetuple[0])
            i = int(linetuple[1])
            user_ratings[u].add(i)
            item_user_dict[i].add(u)
            max_u_id = max(u, max_u_id)
            max_i_id = max(i, max_i_id)
            
    item_pop_dict = {}
    for k, v in item_user_dict.items():
        item_pop_dict[k] = len(v)
    return max_u_id, max_i_id, user_ratings, item_pop_dict


def generate_test(user_ratings):
    '''
    for each user, random select one of his(her) rating into test set
    leave one out
    '''
    user_test = dict()
    for u, i_set in user_ratings.items():
        user_test[u] = random.sample(i_set, 1)[0]
    return user_test


def generate_test_batch(user_ratings, user_ratings_test, item_count):
    '''
    for an user u and an item i rated by u,
    generate pairs (u,i,j) for all subsampled item j which u has not rated
    it's convinent for computing AUC score for u
    '''
    for u in user_ratings.keys():
        negative_item_list = []
        i = user_ratings_test[u]
        cnt = 0
        while cnt < 100:
            j = random.choice(xrange(1, item_count + 1))
            if j not in negative_item_list and j not in user_ratings[u]:
                negative_item_list.append(j)
                cnt += 1

        yield u, i, negative_item_list


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


"""
compute HR@10 and NDCG@10 for one user
item score is purely based its popularity, which is measured by number of interactions with users
"""
def eval_one_rating(u, i, j_list, item_pop_dict):
    map_item_score = {}
    map_item_score[i] = item_pop_dict[i]
    for j in j_list:
        map_item_score[j] = item_pop_dict[j]
    rank_list = heapq.nlargest(10, map_item_score, key = map_item_score.get)
    hr = getHitRate(rank_list, i)
    ndcg = getNDCG(rank_list, i)
    return [hr, ndcg]


def eval_all_users(user_ratings, user_ratings_test, user_count,
                   item_count, item_pop_dict):
    _hr_sum = 0.0
    _ndcg_sum = 0.0
    for u, i, j_list in generate_test_batch(user_ratings, user_ratings_test, item_count):
        _hr, _ndcg = eval_one_rating(u, i, j_list, item_pop_dict)
        _hr_sum += _hr
        _ndcg_sum += _ndcg
    return _hr_sum / user_count, _ndcg_sum / user_count


if __name__ == '__main__':

    user_count, item_count, user_ratings, item_pop_dict = load_data()
    user_test = generate_test(user_ratings)
    hr, ndcg = eval_all_users(user_ratings, user_test, user_count, item_count, item_pop_dict)

    print 'test_hr: ', hr
    print 'test_ndcg: ', ndcg
