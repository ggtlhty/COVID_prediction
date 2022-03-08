from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, RocCurveDisplay, auc
import matplotlib.pyplot as plt


def evaluate(y_true, y_pred, model_name):
    print('-------------------------------')
    print(model_name)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred, normalize=True)
    precision = precision_score(
        y_true=y_true, y_pred=y_pred, average="weighted")
    recall = recall_score(y_true=y_true, y_pred=y_pred, average="weighted")
    confusionMatrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("ConfusionMatrix: \n{}".format(confusionMatrix))