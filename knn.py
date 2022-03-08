'''
Question 3.1 Skeleton Code
Here you should implement and evaluate the k-NN classifier.
'''
import load_data
import numpy as np
from sklearn.metrics import accuracy_score
import statistics as stats
from sklearn.model_selection import KFold, train_test_split
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]
        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm
        You should return the digit label provided by the algorithm
        '''

        index = self.l2_distance(test_point).argsort()[:k]
        labels = []
        for i in index:
            labels.append(self.train_labels[i])

        digit = max(set(labels), key = labels.count)
        return digit

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    kf10 = KFold(n_splits=10, shuffle=False)
    k_best_list = []
    accuracy_list = np.zeros(len(k_range))
    for train_index, test_index in kf10.split(train_data):
        X_train, X_test = train_data[train_index], train_data[test_index]
        y_train, y_test = train_labels[train_index], train_labels[test_index]
        best_score = 0
        best_k = 0
        for k in k_range:
            knn = KNearestNeighbor(X_train, y_train)
            score = classification_accuracy(knn, k, X_test, y_test)
            accuracy_list[k-1] += score
            if score > best_score:
                best_score = score
                best_k = k
        k_best_list.append(best_k)
    accuracy_list = accuracy_list/10
    print("accuracy list")
    print(accuracy_list)
    accuracy_list = list(accuracy_list)
    optimal_k = accuracy_list.index(max(accuracy_list))+1
    optimal_k_avg_accuracy = accuracy_list[optimal_k-1]
    print("Optimal K:", optimal_k)


    return optimal_k, optimal_k_avg_accuracy


def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    predicted_label = []
    for i in range(len(eval_data[:])):
        predicted_label.append(knn.query_knn(eval_data[i], k))
    score = accuracy_score(eval_labels, predicted_label)
    return score

def main():
    train_data, train_labels, test_data, test_labels, vld_data, vld_labels = load_data.load_data('corona_tested_individuals_ver_006.english.csv')
    k=1
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_labels)
    k_1_train_score = knn.score(train_data, train_labels)*100
    k_1_test_score = knn.score(test_data, test_labels)*100

    print("k = ", k,", train accuracy is: ",k_1_train_score,", test accuracy is:", k_1_test_score)
    # k = 15
    # k_15_train_score = classification_accuracy(knn, k, train_data, train_labels)
    # k_15_test_score = classification_accuracy(knn, k, test_data, test_labels)
    # print("--------------------------------------------")
    # print("3.1.1(b)")
    # print("k = ", k,", train accuracy is: ",k_15_train_score,", test accuracy is:", k_15_test_score)

    # optimal_k, optimal_k_avg_accuracy = cross_validation(train_data, train_labels, k_range=np.arange(1,16))
    # optimal_k_train_score = classification_accuracy(knn, optimal_k, train_data, train_labels)
    # optimal_k_test_score = classification_accuracy(knn, optimal_k, test_data, test_labels)
    # print("--------------------------------------------")
    # print("3.1.3")
    # print("k = ", optimal_k,", train accuracy is: ",optimal_k_train_score, ", the average accuracy across folds is: ", optimal_k_avg_accuracy,", test accuracy is:", optimal_k_test_score)
    # clf = KNeighborsClassifier(n_neighbors=1).fit(train_data, train_labels)
    # test_prediction = clf.predict(test_data)
    # data.calc(y_true=test_labels,
    #           y_pred=test_prediction, model_name="KNN")
    # probs = clf.predict_proba(test_data)
    # binarized_labels = label_binarize(test_labels, classes=range(10))
    # data.visualize_roc_curve(probs, binarized_labels, name="knn")

main()