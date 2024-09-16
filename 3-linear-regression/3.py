import pandas as pd
from sklearn.linear_model import LinearRegression

fishes = pd.read_csv('https://courses.openedu.ru/assets/courseware/v1/fed9823d73d2b53f5591d98b87c20b8a/asset-v1:ITMOUniversity+MLDATAN+spring_2024_ITMO_bac+type@asset+block/fish_train.csv')
fishes_reserved = pd.read_csv('https://courses.openedu.ru/assets/courseware/v1/33b24e589714e963ea7081912668c93d/asset-v1:ITMOUniversity+MLDATAN+spring_2024_ITMO_bac+type@asset+block/fish_reserved.csv')

# возведем признаки во 2 степень
fishes[['Length1', 'Length2', 'Length3', 'Height', 'Width']] = fishes[['Length1', 'Length2', 'Length3', 'Height', 'Width']]**2
fishes_reserved[['Length1', 'Length2', 'Length3', 'Height', 'Width']] = fishes_reserved[['Length1', 'Length2', 'Length3', 'Height', 'Width']]**2

# закодируем категориальные признаки
dummies = pd.get_dummies(fishes['Species'], drop_first=True)
fishes[list(dummies.columns)] = dummies
fishes.drop(['Species'], axis=1, inplace=True)

dummies = pd.get_dummies(fishes_reserved['Species'], drop_first=True)
fishes_reserved[list(dummies.columns)] = dummies
fishes_reserved.drop(['Species'], axis=1, inplace=True)

# разделим данные на предикторы и отклики
X_train, y_train = fishes.drop(columns=['Weight']), fishes[['Weight']]
X_test = fishes_reserved

# обучим модель линейной регрессии
lr = LinearRegression().fit(X_train, y_train)

# сделаем предсказание для тестовых данных
prediction = lr.predict(X_test)
list_prediction = [value[0] for value in prediction]
print(list_prediction)
