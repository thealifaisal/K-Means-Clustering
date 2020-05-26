# ******************************************
#
#   AUTHOR: ALI FAISAL
#   Github: https://github.com/thealifaisal/
#
# ******************************************

import numpy as np
from datetime import datetime
from src.ml_vsm import MachineLearning
from src.serialization import Serialization

if __name__ == "__main__":

    print(datetime.now().strftime("%H:%M:%S") + ": setting paths...")
    # input paths
    data_folder_path = "../resources/bbcsport/"
    stop_file_path = "../resources/stopword-list.txt"

    # output paths
    json_file_path = "../out/json_out.json"
    # vocab_file_path = "../out/vocab.txt"
    # class_file_path = "../out/class-tf.json"

    print(datetime.now().strftime("%H:%M:%S") + ": serializing raw data...")
    ser = Serialization()
    # imports stoplist
    stop_list = ser.importStopList(stop_file_path)
    ser.preprocessing.stop_word = stop_list

    # returns a serialized data from raw text files
    # e.g: json_list = [{"id": id1, "label": lb, "features": {"term": tf}}, {...}, {...}, ...]
    json_list = ser.readRawData(data_folder_path)

    # randomize all the files for fair seed selection
    ser.shuffleJSONObjects(json_list)
    print(datetime.now().strftime("%H:%M:%S") + ": writing to json files...")
    ser.writeToJSONFile(json_list, json_file_path)

    ml = MachineLearning()

    print(datetime.now().strftime("%H:%M:%S") + ": creating vocabulary...")
    vocabulary = ml.createVocabulary(json_list)
    vocabulary_len = len(vocabulary)
    print(datetime.now().strftime("%H:%M:%S") + ": vocabulary size = " + str(vocabulary_len))

    print(datetime.now().strftime("%H:%M:%S") + ": creating document vectors...")
    # e.g: doc_vectors = {"doc-id": [tf-idf, ...], ...}
    doc_vectors, idf_vector = ml.createTrainVectors(vocabulary, json_list)
    # e.g: ['015_rugby', '126_football', '190_football', '032_football', ...]
    doc_ids = list(doc_vectors.keys())

    print(datetime.now().strftime("%H:%M:%S") + ": started clustering...")

    # since json_list was randomized earlier, we can select any K starting doc-vectors

    doc_tf_idf_vec = [vec for vec in doc_vectors.values()]

    np_doc_tf_idf_vec = np.array(doc_tf_idf_vec)

    K = 5

    ml.kMeans(K, np_doc_tf_idf_vec)



