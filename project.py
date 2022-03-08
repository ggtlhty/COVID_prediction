import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
e=LabelEncoder()


df = pd.read_csv (r'D:/2021/Fall2021/2515 Intro to ML/project/corona_tested_individuals_ver_006.english.csv',low_memory=False)
#df = pd.read_csv (r'C:\Users\Yi\Desktop\CSC2515\project/corona_tested_individuals_ver_006.english.csv',low_memory=False)

missing_values=df.isnull().sum() # missing values

percent_missing = df.isnull().sum()/df.shape[0] # missing value %

value = {
    'missing_values ':missing_values,
    'percent_missing %':percent_missing
}
frame=pd.DataFrame(value)
print(frame)
df["cough"] = e.fit_transform(df["cough"])
df["fever"] = e.fit_transform(df["fever"])
df["sore_throat"] = e.fit_transform(df["sore_throat"])
df["shortness_of_breath"] = e.fit_transform(df["shortness_of_breath"])
df["head_ache"] = e.fit_transform(df["head_ache"])

df = df[df['corona_result']!='Other']
df = df[df['age_60_and_above']!='None']
df = df[df['gender']!='None']
df = df[df['test_indication']!='Abroad']


df["corona_result"] = e.fit_transform(df["corona_result"])
df["age_60_and_above"] = e.fit_transform(df["age_60_and_above"])
df["gender"] = e.fit_transform(df["gender"])
df["test_indication"] = e.fit_transform(df["test_indication"])
df = df.iloc[:, 1:]



#print(df.dtypes.value_counts())
#print(df.describe(include='all'))
print(df)

x = df.drop('corona_result', axis=1)
y = df['corona_result']

print(x)
x_train, x_tmp, y_train, y_tmp = train_test_split(x, y, test_size = 0.30)
x_test, x_vld, y_test, y_vld = train_test_split(x_tmp, y_tmp, test_size = 0.50)

knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(x_train.values, y_train.values)
y_pred = knn.predict(x_test.values)
acc_knn=knn.score(x_test.values, y_test.values)
print(acc_knn)