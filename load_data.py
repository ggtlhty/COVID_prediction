import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier



def load_data(data_dir):
    df = pd.read_csv (data_dir,low_memory=False)


    df = df[df['corona_result']!='other']
    df = df[df['age_60_and_above']!='None']
    df = df[df['gender']!='None']
    df = df[df['test_indication']!='Abroad']

    e=LabelEncoder()
    df["cough"] = e.fit_transform(df["cough"])
    df["fever"] = e.fit_transform(df["fever"])
    df["sore_throat"] = e.fit_transform(df["sore_throat"])
    df["shortness_of_breath"] = e.fit_transform(df["shortness_of_breath"])
    df["head_ache"] = e.fit_transform(df["head_ache"])
    df["corona_result"] = e.fit_transform(df["corona_result"])
    df["age_60_and_above"] = e.fit_transform(df["age_60_and_above"])
    df["gender"] = e.fit_transform(df["gender"])
    df["test_indication"] = e.fit_transform(df["test_indication"])
    df = df.iloc[:, 1:]



#print(df.dtypes.value_counts())
#print(df.describe(include='all'))
#print(df)

    x = df.drop('corona_result', axis=1)
    y = df['corona_result']

    x = x.to_numpy()
    y = y.to_numpy()
    print(x.shape)
    print(y.shape)
    x = np.reshape(x, (-1, 8))
    y = np.reshape(y, (-1))
    print(x.shape)
    print(y.shape)
    x_train, x_tmp, y_train, y_tmp = train_test_split(x, y, test_size = 0.30)
    x_test, x_vld, y_test, y_vld = train_test_split(x_tmp, y_tmp, test_size = 0.50)

    return(x_train, y_train, x_test, y_test, x_vld, y_vld)