import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

train = pd.read_csv('https://courses.openedu.ru/assets/courseware/v1/3b5e697be14f493785e3d21577f9fcb3/asset-v1:ITMOUniversity+MLDATAN+spring_2024_ITMO_bac+type@asset+block/adult_data_train.csv')
test = pd.read_csv('https://courses.openedu.ru/assets/courseware/v1/81d9cf5671cf3576fd7776f5165d9cc5/asset-v1:ITMOUniversity+MLDATAN+spring_2024_ITMO_bac+type@asset+block/adult_data_reserved.csv')

# заполним пропуски в данных значениями моды и закодируем категориальные признаки
def clean_data(df):
    df.replace('?', np.nan, inplace=True)

    if 'label' in df.columns:
        for col in df.columns:
            df[col] = df.groupby("label")[col].transform(lambda x: x.fillna(x.mode()[0]))
    else:
        for col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return pd.get_dummies(
        df,
        columns=df.select_dtypes(include=[object]).columns,
        drop_first = True
    )


train = clean_data(train)
test = clean_data(test)
test = test.reindex(columns=train.columns, fill_value=0)

# разделим данные на предикторы и отклики
x_train = train.drop('label', axis=1)
y_train = train['label']
x_test = test.reindex(columns=x_train.columns, fill_value=0)

# масштабируем признаки
scaler = MinMaxScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

# удалим низко коррелирующие с целевой переменной признаки
THRESHOLD = 0.05
corrs = [(y_train.corr(x_train[col]), col) for col in x_train.columns]
to_delete = [col for corr, col in corrs if corr < THRESHOLD]
x_train.drop(to_delete, axis=1, inplace=True)
x_test.drop(to_delete, axis=1, inplace=True)

# строим модель
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 25)}
knn_gscv = GridSearchCV(knn, param_grid, cv=5).fit(x_train, y_train)

# делаем предсказание для тренировочных данных
y_train_pred = knn_gscv.predict(x_train)
print('f1 score:', f1_score(y_train, y_train_pred))

# делаем предсказание для тестовых данных
y_pred = knn_gscv.predict(x_test)
print(list(y_pred))