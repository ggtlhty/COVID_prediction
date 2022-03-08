import torch
import torch.nn as nn
import torch.nn.functional
import pandas
import load_data
import numpy as np
import time
from evaluationUtils import evaluate


def preprocess_data():
    train_data, train_labels, test_data, test_labels, vld_data, vld_labels = load_data.load_data(
        'corona_tested_individuals_ver_006.english.csv')
    origin_test_labels = test_labels
    train_data = torch.tensor(train_data).float()
    train_labels = pandas.get_dummies(train_labels)
    train_labels = train_labels.values.tolist()
    train_labels = torch.tensor(train_labels).type(torch.float)
    test_data = torch.tensor(test_data).float()
    test_labels = pandas.get_dummies(test_labels)
    test_labels = test_labels.values.tolist()
    test_labels = torch.tensor(test_labels).type(torch.float)

    return train_data, train_labels, test_data, test_labels, origin_test_labels


def setup_model():
    # model declaration
    model = nn.Sequential(
        nn.Linear(8, 4),
        nn.ReLU(),
        #nn.Linear(6, 4),
        #.ReLU(),
        nn.Linear(4, 2)
    )
    # loss function declaration
    loss_fn = nn.MSELoss()
    # optimizer declaration
    optimizer = torch.optim.Adam(model.parameters(), lr=0.131)
    return model, loss_fn, optimizer


def train_model(train_data, train_labels, model, loss_fn, optim, epochs):
    model.train()
    loss_prev = 0
    for epoch in range(epochs):
        # zero out gradients on each training iteration
        optim.zero_grad()
        # forward pass through the network
        y_pred = model(train_data)
        # compute loss
        loss = loss_fn(y_pred, train_labels)
        # compute gradients
        loss.backward()
        #if abs(loss.item() - loss_prev) <= 0.000001:
            #print(
            #    f"Epoch {epoch}: traing loss: {loss.item()}")
            #break
        loss_prev = loss.item()

        print(
            f"Epoch {epoch}: traing loss: {loss.item()}")
        # take a step
        optim.step()


def test(test_data, test_labels, model):
    model.eval()
    test_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        # for each test example
        for i in range(test_data.size()[0]):
            # get prediction
            y_pred = model(test_data[i])
            # compute test loss on this prediction
            test_loss += torch.nn.functional.mse_loss(y_pred,
                                                      test_labels[i], reduction='mean').item()
            # compare the predicted value and test label
            if torch.argmax(y_pred.data).item() == torch.argmax(test_labels[i]).item():
                correct_predictions += 1
            # Store prediction results for q3_3.
            test_pred.append(torch.argmax(y_pred.data).item())

    test_loss = test_loss / len(test_data)
    print('\ntest set loss: {:.7f}, accuracy: {} / {} ({:.2f}%)\n'.format(
        test_loss, correct_predictions, len(test_data),
        100. * correct_predictions / len(test_data)))


if __name__ == "__main__":
    test_pred = []
    train_data, train_labels, test_data, test_labels, origin_test_labels = preprocess_data()
    model, loss_fn, optim = setup_model()
    beforeTrainingTimeStamp = time.time()
    train_model(train_data, train_labels, model, loss_fn, optim, 100)
    afterTrainingTimeStamp = time.time()
    test(test_data, test_labels, model)
    afterPredictingTimeStamp = time.time()
    trainTime = afterTrainingTimeStamp - beforeTrainingTimeStamp
    predictTime = afterPredictingTimeStamp - afterTrainingTimeStamp
    totalTime = afterPredictingTimeStamp - beforeTrainingTimeStamp
    print("Train Time: {}".format(trainTime))
    print("Predict Time: {}".format(predictTime))
    print("Total Time: {}".format(totalTime))
    evaluate(y_true=origin_test_labels, y_pred=np.array(test_pred, dtype=float), model_name="MLP")