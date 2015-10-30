from __future__ import division
import numpy as np
import os
from numpy.random import random
from random import shuffle
import cPickle

class SVD_C:

    def __init__(self, X, k=20):
        self.X = np.array(X)
        self.k = k
        self.ave = np.mean(self.X[:, 2])
        print "the input data size is ", self.X.shape
        self.bi = {}
        self.bu = {}
        self.qi = {}
        self.pu = {}
        self.movie_user = {}
        self.user_movie = {}
        for i in range(self.X.shape[0]):
            uid = self.X[i][0]
            mid = self.X[i][1]
            rat = self.X[i][2]
            self.movie_user.setdefault(mid, {})
            self.user_movie.setdefault(uid, {})
            self.movie_user[mid][uid] = rat
            self.user_movie[uid][mid] = rat
            self.bi.setdefault(mid, 0)
            self.bu.setdefault(uid, 0)
            self.qi.setdefault(mid, random((self.k, 1)) /
                               10 * (np.sqrt(self.k)))
            self.pu.setdefault(uid, random((self.k, 1)) /
                               10 * (np.sqrt(self.k)))

    def pred(self, uid, mid):
        self.bi.setdefault(mid, 0)
        self.bu.setdefault(uid, 0)
        self.qi.setdefault(mid, np.zeros((self.k, 1)))
        self.pu.setdefault(uid, np.zeros((self.k, 1)))
        if (self.qi[mid] == None):
            self.qi[mid] = np.zeros((self.k, 1))
        if (self.pu[uid] == None):
            self.pu[uid] = np.zeros((self.k, 1))
        ans = self.ave + self.bi[mid] + self.bu[uid] + \
            np.sum(self.qi[mid] * self.pu[uid])
        if ans > 5:
            return 5
        elif ans < 1:
            return 1
        return ans

    def train(self, steps=1, gamma=0.04, Lambda=0.15):
        for step in range(steps):
            print 'the ', step, '-th  step is running'
            rmse_sum = 0.0
            kk = np.random.permutation(self.X.shape[0])
            for j in range(self.X.shape[0]):
                i = kk[j]
                uid = self.X[i][0]
                mid = self.X[i][1]
                rat = self.X[i][2]
                eui = rat - self.pred(uid, mid)
                rmse_sum += eui**2
                self.bu[uid] += gamma * (eui - Lambda * self.bu[uid])
                self.bi[mid] += gamma * (eui - Lambda * self.bi[mid])
                temp = self.qi[mid]
                self.qi[mid] += gamma * \
                    (eui * self.pu[uid] - Lambda * self.qi[mid])
                self.pu[uid] += gamma * (eui * temp - Lambda * self.pu[uid])
            gamma = gamma * 0.93
            print "the rmse of this step on train data is ", np.sqrt(rmse_sum / self.X.shape[0])
            # self.test(test_data)

    def test(self, test_X):
        output = []
        sums = 0
        test_X = np.array(test_X)
        # print "the test data size is ",test_X.shape
        for i in range(test_X.shape[0]):
            pre = self.pred(test_X[i][0], test_X[i][1])
            print "Predict  :", pre
            output.append(pre)
            # print pre,test_X[i][2]
            sums += (pre - test_X[i][2])**2
        rmse = np.sqrt(sums / test_X.shape[0])
        print "the rmse on test data is ", rmse
        return output

    def rank_user(self, uid):
        output = []
        movie_name = self.movie_name
        # print "the test data size is ",test_X.shape
        for i in movie_name.keys():
            pre = self.pred(uid, i)
            output.append([pre, i])
        output = sorted(output, key=lambda x: x[0], reverse=True)
        return output

    def get_user(self, uid):
        seened_movie = []
        if uid in self.seened:
            for a in sorted(self.seened[uid], key=lambda x: x[1], reverse=True)[:10]:
                if a[0] not in self.movie_name:
                    continue
                name = self.movie_name[a[0]]
                try:
                    name = name.encode("utf-8")
                except Exception, e:
                    print e
                name, genre = name.split("\t")
                seened_movie.append({"rate": a[1], "name": name, "genre" : genre})
            if len(self.seened[uid]) >= 10:
                for a in sorted(self.seened[uid], key=lambda x: x[1], reverse=True)[-10:]:
                    if a[0] not in self.movie_name:
                        continue
                    name = self.movie_name[a[0]]
                    try:
                        name = name.encode("utf-8")
                    except Exception, e:
                        print e
                    name, genre = name.split("\t")
                    seened_movie.append({"rate": a[1], "name": name, "genre" : genre})

        output = self.rank_user(uid)
        res = []
        for i in range(10):
            name = self.movie_name[output[i][1]]
            try:
                name = name.encode("utf-8")
            except Exception, e:
                print e
            name, genre = name.split("\t")
            res.append({"name" : name, "genre" : genre, "rate" : output[i][0]})
        return seened_movie, res,

    def rank_all_user(self):
        self.seened_all = {}
        self.res_all = {}
        for uid in self.user_movie.keys():
            seened_movie, res = self.get_user(uid)
            self.seened_all[uid] = seened_movie
            self.res_all[uid] = res

def add_rating(adding):

    X = load_rate()[:100000] + adding
    # shuffle(X)
    brk = int(len(X) * 1)
    train_x = np.array(X[:brk])

    cf = SVD_C(train_x, 20)
    cf.movie_name = load_data()
    cf.seened = {}
    for a in train_x:
        if a[0] not in cf.seened:
            cf.seened[a[0]] = []
        cf.seened[a[0]].append([a[1], a[2]])


    # with open("/Users/shawn/Code/PyCharmWorkspace/MovieRec/model.pkl", "r") as f:
    #     cf = cPickle.load(f)


    cf.train(20)

    cf.rank_all_user()
    with open("/Users/shawn/Code/PyCharmWorkspace/MovieRec/cf.pkl", "w") as f:
        cPickle.dump([cf.seened_all, cf.res_all], f)



    # with open("load_datamodel.pkl", "w") as f:
    #     cPickle.dump(cf, f)


data_path = os.path.join("extract", "ml-latest-small")


def load_rate():
    rate_path = os.path.join(data_path, "/Users/shawn/Code/PyCharmWorkspace/MovieRec/extract/ml-latest-small/ratings.csv")
    X = []
    with open(rate_path, "r") as f:
        f.readline()
        for line in f.readlines():
            (userId, movieId, rating, timestamp) = line.split(",")
            X.append([eval(userId), eval(movieId), eval(rating)])
    return X


def load_data():
    rate_path = os.path.join(data_path, "/Users/shawn/Code/PyCharmWorkspace/MovieRec/extract/ml-latest-small/movies.csv")
    movie_name = {}
    with open(rate_path, "r") as f:
        f.readline()
        for line in f.readlines():
            if line.find("\"") == -1:
                (mid, name, genre) = line.split(',')
            else:
                mid = line.split(",")[0]
                name = line.split("\"")[1]
                genre = line.split("\",")[1]
            s = (name + "\t" + genre)
            try:
                s = s.encode("utf-8")
            except Exception, e:
                # print e
                continue
            movie_name[eval(mid)] = s
    return movie_name


def similarity(x, y, item):
    if x[0, item] == 0:
        return 0
    dic = {}
    for i in x:
        dic[i] = 1
    idx = filter(lambda i: x[0, i] != 0, y.nonzero()[1])
    if idx == []:
        return 0
    res = 0
    a = []
    b = []
    for i in idx:
        a.append(x[0, i])
        b.append(y[0, i])
    return 1


def predict(X, x, item):
    # sim = np.array([similarity(X[i, :], x, item) for i in range(X.shape[0])])
    # print sim
    return 1
    # return np.dot(sim, X.toarray()[:, item]) / np.sum(sim)


def MF(X):
    # model = model = ProjectedGradientNMF(
    #     n_components=5, init='random', random_state=0)
    # A = model.fit_transform(X_full.T)
    # B = model.components_

    # A, Y = nmf(np.array(X), 100)

    # C = (A.dot(B)).T

    # print similarity(X_sparse[1, :], X_sparse[2, :])

    pass


def init():
    X = load_rate()[:100000]
    shuffle(X)
    brk = int(len(X) * 1)
    train_x = np.array(X[:brk])
    test_x = X[brk:]

    # a = zip(*train_x)
    # X_full = csc_matrix(
    #     (np.array(a[2]), (np.array(a[0]), np.array(a[1])))).toarray()
    # appear_time = [np.sum(X_full[:, i]) for i in range(X_full.shape[1])]
    # idx = filter(lambda i: appear_time[i] > 300, range(X_full.shape[1]))
    # ind = {}
    # for i in range(len(idx)):
    #     ind[idx[i]] = i
    # print len(idx)
    # X_full = X_full[:, idx]
    # print X_full.shape


    # print appear_time

    # X_sparse = csc_matrix(
    #     (np.array(a[2]), (np.array(a[0]), np.array(a[1]))))
    # X_sparse = csc_matrix(X_full)

    # for p in test_x:
    #     u, m, r = p
    #     if m not in ind:
    #         continue
    #     # print r, C[u, m]
    #     print r, predict(X_sparse, X_sparse[u, :], ind[m])

    # print X_full
    # print len(a[0])

    # print test_x
    # train_x = [[0, 0, 1], [0, 1, 5], [1, 0, 5], [
    #     1, 1, 1], [2, 0, 5], [3, 0, 5], [3, 1, 1]]
    # test_x = [[3, 1, 1]]

    # test_x = filter(lambda x: x[2] < X_full.shape[1], test_x)
    cf = SVD_C(train_x, 20)
    cf.movie_name = load_data()
    cf.seened = {}
    for a in train_x:
        if a[0] not in cf.seened:
            cf.seened[a[0]] = []
        cf.seened[a[0]].append([a[1], a[2]])

    cf.train(20)

    cf.rank_all_user()

    with open("/Users/shawn/Code/PyCharmWorkspace/MovieRec/cf.pkl", "w") as f:
        cPickle.dump([cf.seened_all, cf.res_all], f)

    return cf

if __name__ == "__main__":
    cf = init()
    # cf.train()
    # cf.test(test_x)
