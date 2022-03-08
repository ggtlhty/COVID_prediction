import load_data
from sklearn import svm
from sklearn.svm import SVC
from sklearn import svm, metrics
import numpy as np
from sklearn.model_selection import GridSearchCV
import evaluationUtils

def grid_search_fit_and_test(train_data, train_labels, test_data, test_labels):
    gammalist = np.outer(np.logspace(-3, 0, 3), np.array([1, 3])).flatten()
    Clist = np.outer(np.logspace(-1, 1, 5), np.array([1, 4])).flatten()

    clf = SVC()

    grid_classifier = GridSearchCV(
        clf, {'kernel': ['rbf'], 'C': Clist, "gamma": gammalist})

    grid_classifier.fit(train_data, train_labels)

    # return best classifier and best parameters
    best_classifier = grid_classifier.best_estimator_
    best_params = grid_classifier.best_params_
    print(best_params)

    pred = best_classifier.predict(test_data)

    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(test_labels, pred)}\n"
    )

def svm(train_data, train_labels, test_data, test_labels):

    model1 = SVC(C = 0.1,kernel='rbf',gamma = 3.0)
    model1.fit(train_data, train_labels)
    print("\n 'rbf accuracy:", model1.score(test_data,test_labels) * 100)
    return model1



if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels, vld_data, vld_labels = load_data.load_data(
        'corona_tested_individuals_ver_006.english.csv')

    #grid_search_fit_and_test(train_data, train_labels, vld_data, vld_labels)
    #best parameters selected: kernel = rbf, C = 0.1, gamma = 3.0
    svm = svm(train_data, train_labels, test_data, test_labels)
    y_pred = svm.predict(test_data)
    evaluationUtils.evaluate(test_labels,y_pred,svm)