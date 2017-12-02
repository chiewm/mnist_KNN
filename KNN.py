# @time    : 2017/12/1 21:43
# @Author  : chiew
# @File    : KNN.py

from numpy import *
import operator
from os import listdir


# def create_data_set():
#     group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
#     labels = ['A', 'A', 'B', 'B']
#     return group, labels
#

def img2vector(filename):
    return_vector = zeros((1, 1024))
    with open(filename, 'rt') as fr:
        for i in range(32):
            line_str = fr.readline()
            for j in range(32):
                return_vector[0, 32 * i + j] = int(line_str[j])
    return return_vector


def classify0(inx, data_set, labels, k):
    data_set_size = data_set.shape[0]
    diff_mat = tile(inx, (data_set_size, 1)) - data_set

    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)

    distances = sq_distances ** 0.5

    sorted_dist = distances.argsort()
    class_count = {}

    for i in range(k):
        vote_label = labels[sorted_dist[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    sorted_class_count = sorted(
        class_count.items(),
        key=operator.itemgetter(1),
        reverse=True)

    return sorted_class_count[0][0]


def handwriting_class_test():
    hw_labels = []

    training_file_list = listdir('digits/trainingDigits')
    m = len(training_file_list)

    training_mat = zeros((m, 1024))

    for i in range(m):
        file_name = training_file_list[i]
        file = file_name.split('.')[0]
        class_num = int(file.split('_')[0])
        hw_labels.append(class_num)

        training_mat[i, :] = img2vector('digits/trainingDigits/%s' % file_name)

    test_file_list = listdir('digits/testDigits')

    error_count = 0.0
    m_test = len(test_file_list)

    for i in range(m_test):
        file_name = test_file_list[i]
        file = file_name.split('.')[0]
        class_num = int(file.split('_')[0])

        vector_under_test = img2vector('digits/testDigits/%s' % file_name)

        classifier_result = classify0(
            vector_under_test, training_mat, hw_labels, 3)

        print(
            "the classifier came back with: %d, the real answer is: %d" %
            (classifier_result, class_num))

        if classifier_result != class_num:
            error_count += 1.0

    print("\n the total number of errors is: %d" % error_count)
    print("\n the total error rate is: %f" % (error_count / float(m_test)))


# group,labels = kNN.createDataSet()
# kNN.classify0([0,0], group, labels, 3)


handwriting_class_test()
