import q3_0
import numpy as np
import data
import collections
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
#Tianyu Hu 1003549564
#Collaborated with Lizhen Qiao 1008651585 and Yi Zhou 1007979190

def grid_search_fit_and_test(train_data, train_labels, test_data, test_labels):
    gammalist = np.outer(np.logspace(-3, 0, 3), np.array([1, 3])).flatten()
    Clist = np.outer(np.logspace(-1, 1, 5), np.array([1, 4])).flatten()

    clf = svm.SVC()

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


def fit_and_test(train_data, train_labels, test_data, test_labels):
    clf = svm.SVC(kernel="rbf", gamma=0.09486832980505137,
                  C=10.0, probability=True)

    clf.fit(train_data, train_labels)

    pred = clf.predict(test_data)

    probs = clf.predict_proba(test_data)
    binarized_labels = label_binarize(test_labels, classes=range(10))
    data.visualize_roc_curve(probs, binarized_labels, name="svm")

    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(test_labels, pred, digits=4)}\n"
    )
    return pred


if __name__ == "__main__":
    test_prediction = []
    train_data, train_labels, test_data, test_labels = data.load_all_data_from_zip(
        "a3digits.zip", "data")
    grid_search_fit_and_test(train_data, train_labels, test_data, test_labels)

    test_prediction = fit_and_test(
        train_data, train_labels, test_data, test_labels)
    data.calc(y_true=test_labels, y_pred=test_prediction, model_name="SVM")
