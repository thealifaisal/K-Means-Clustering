# ******************************************
#
#   AUTHOR: ALI FAISAL
#   Github: https://github.com/thealifaisal/
#
# ******************************************

import math
import operator
from sklearn.cluster import KMeans
import numpy as np


class MachineLearning:

    @staticmethod
    def createTrainVectors(vocabulary, train_set):

        # train_vectors = {"id-1": [tf-1, tf-2, tf-3, ...], "id-2": [tf-1, tf-2, ...], ...}
        train_vectors = {}
        trainset_len = len(train_set)
        vocabulary_len = len(vocabulary)
        idf_vector = [0] * vocabulary_len

        # iterates for every vocabulary
        i = 0
        for voc in vocabulary:
            # wb_sheet.cell(i + 2, 1).value = voc

            df = 0
            for doc in range(0, trainset_len):

                doc_id = train_set[doc]["id"]
                label = train_set[doc]["label"]
                key = doc_id + "_" + label

                try:
                    if key not in train_vectors.keys():
                        train_vectors[key] = [0] * vocabulary_len

                    # assigns tf of term in doc
                    train_vectors[key][i] = train_set[doc]["features"][voc]
                    df += 1

                except KeyError:
                    # when, term from vocab not in doc
                    train_vectors[key][i] = 0

            idf_vector[i] = (float(format(math.log10(trainset_len / df), ".5f")))

            for doc in range(0, trainset_len):

                doc_id = train_set[doc]["id"]
                label = train_set[doc]["label"]
                key = doc_id + "_" + label

                tf = float(format(train_vectors[key][i], ".5f"))
                train_vectors[key][i] = float(format(idf_vector[i] * tf, ".5f"))

            i += 1

        return train_vectors, idf_vector

    @staticmethod
    def createTestVectors(vocabulary, idf_vector, test_set):
        # train_vectors = {"id-1": [tf-1, tf-2, tf-3, ...], "id-2": [tf-1, tf-2, ...], ...}
        test_vectors = {}
        test_set_len = len(test_set)
        vocabulary_len = len(vocabulary)

        # iterates for every vocabulary
        for i, voc in enumerate(vocabulary):

            for doc in range(0, test_set_len):

                doc_id = test_set[doc]["id"]
                label = test_set[doc]["label"]
                key = doc_id + "_" + label

                try:
                    if key not in test_vectors.keys():
                        test_vectors[key] = [0] * vocabulary_len

                    # assigns tf-idf of term in test-doc
                    tf = test_set[doc]["features"][voc]
                    test_vectors[key][i] = float(format(idf_vector[i] * tf, ".5f"))

                except KeyError:
                    # when, term from vocab not in doc
                    test_vectors[key][i] = 0

        return test_vectors

    @staticmethod
    def prepareSheet(wb_sheet, train_set):
        # train_set = [{id, label, features: {term: tf}}, {}, ...]
        trainset_len = len(train_set)
        wb_sheet.cell(1, 1).value = "bag"
        for _i in range(2, trainset_len + 1):
            # e.g: doc-id = 037a, 110t, ...
            wb_sheet.cell(1, _i).value = train_set[_i - 2]["id"] + train_set[_i - 2]["label"][0]
        wb_sheet.cell(1, trainset_len + 2).value = "idf"
        return

    # splits the data-set into train and test sets
    # returns trains and test sets
    @staticmethod
    def dataSplit(jsonlist):
        # jsonlist = [{id: id1, label: lb1, features: {term: tf}}, {}, {}, ...]

        docs_len = len(jsonlist)  # 737
        train_len = round(70 / 100 * docs_len)  # 516
        test_len = docs_len - train_len  # 221
        trainset = []
        testset = []

        for f in range(0, train_len):
            trainset.append(jsonlist[f].copy())

        for f in range(train_len, docs_len):
            testset.append(jsonlist[f].copy())

        jsonlist.clear()

        return trainset, testset

    @staticmethod
    def createVocabulary(trainset):
        vocab = []
        for file in trainset:
            for feature in file["features"].keys():
                if feature not in vocab:
                    vocab.append(feature)
        return vocab

    @staticmethod
    def cosineSimilarity(train_vectors, test_vectors, test_key, vocabulary_len):
        result_set = {}
        # test_vector => list
        test_vector = test_vectors[test_key]
        np_test = np.array(test_vector)

        for train_key in train_vectors.keys():

            # train_vector => list
            train_vector = train_vectors[train_key]
            np_train = np.array(train_vector)

            dot_product = np.sum(np.multiply(np_train, np_test))

            train_modulus = np.sqrt(np.sum(np.square(np_train)))
            test_modulus = np.sqrt(np.sum(np.square(np_test)))

            try:
                cosine = dot_product / (train_modulus * test_modulus)
            except ZeroDivisionError:
                cosine = -1

            if cosine != -1:
                result_set[train_key] = float(format(cosine, ".5f"))

        result_set = dict(sorted(result_set.items(), key=operator.itemgetter(1), reverse=True))

        # cosines_file = open("../out/cosines.txt", "w")
        # cosines_file.write(str(result_set))
        # cosines_file.close()

        return result_set

    @staticmethod
    def classifyKNN(result_set, k, test_composite_key):
        # KNN:
        #   key: doc-class
        #   value: number of docs that belong to that class from top K result_set
        # e.g: KNN = {"athletics": 2, "tennis": 1}
        # e.g: KNN = {"athletics": 1`, "tennis": 1, "rugby": 1}
        KNN = {}
        labels = {"athletics": "athletics", "cricket": "cricket", "football": "football",
                  "rugby": "rugby", "tennis": "tennis"}

        # i=0, 1, 2, .. | item=("067_cricket", 0.12458), ...
        # {'063_rugby': 0.13047, '141_football': 0.12382, '012_rugby': 0.10805
        for i, item in enumerate(result_set.items()):

            if i >= k:
                # after selecting top k, break
                break

            # item = ("003_athletics", 0.12312)
            # item[0]="003_athletics", item[1]=0.12312
            lab = item[0].split("_")[1]

            # if a class of doc from top K result is not in KNN
            if lab not in KNN.keys():
                # insert class with its initial doc occurrence = 1
                KNN[lab] = 1
            else:
                # else increment doc occurrence from that class
                KNN[lab] += 1

        # sort KNN in descending order,
        # so that class with most docs in top K results is at starting index
        # e.g: KNN = {"athletics": 2, "tennis": 1} -> 2 docs are from class 'athletics', 1 doc is from class 'tennis'
        KNN = dict(sorted(KNN.items(), key=operator.itemgetter(1), reverse=True))

        # fetch item with max doc occurrences from the start
        # e.g: max_item = ("athletics": 2)
        max_item = list(KNN.items())[0]

        # check if all values are same in KNN
        #   means that in top K results, each class occurred has same number of docs occurrences
        # e.g: KNN = {"athletics": 1`, "tennis": 1, "rugby": 1}
        # all_values_same = {True, False}
        all_values_same = all(x == max_item[1] for x in KNN.values())

        if all_values_same:
            # then select the doc from result_set with max cosine value
            # e.g: doc_key = "003_athletics"
            doc_key = list(result_set.keys())[0]
            predicted_class = doc_key.split("_")[1]
        else:
            # otherwise select the class with most documents in KNN
            # e.g: max_item = ("athletics": 2)
            # max_item[0] = "athletics"
            predicted_class = max_item[0]

        actual_class = test_composite_key.split("_")[1]

        file = open("../out/prediction.txt", "a+")

        if predicted_class == actual_class:
            file.write("test_file: " + test_composite_key + " \t\tpredicted class: " + labels[predicted_class] + "\n")
            correct = 1
        else:
            file.write("test_file: " + test_composite_key + " \t\tpredicted class: " + labels[predicted_class] + "\n")
            correct = 0

        file.close()

        return correct

    def kMeans(self, K, X):

        kmeans = KMeans(n_clusters=K, random_state=0).fit(X)

        print("cluster centers")
        print(kmeans.cluster_centers_)

        # e.g: ['068_athletics', '185_football']
        # seeds_keys = [key for i, key in enumerate(doc_ids) if i < K]

        # u = centroid
        # u = [0] * K
        # total no. of docs
        # N = len(doc_ids)

        # for k in range(0, K):
        #     u[k] = doc_vectors[seeds_keys[k]]
        #     # stopping criteria
        #     while True:
        #         # w = cluster
        #         w = [{}] * K
        #         for k in range(0, K):
        #             for n in range(0, N):
        #                 j = np.sqrt(np.sum((np.array(doc_vectors[doc_ids[n]]) - u[k])**2))
        #                 w[k][j] = doc_vectors[doc_ids[n]]
        #             for k in range(0, K):
        #                 sum_arr = []
        #                 for vec in w[k].values():
        #                     sum_arr = np.sum(np.array(sum_arr), np.array(vec))
        #                 u[k] = (1/len(w[k].keys()))*(np.array(sum_arr))
        #
        # print(u)
