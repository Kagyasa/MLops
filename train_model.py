import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def scale_frame(mat_df):
    test_mat = mat_df.copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(test_mat)
    test_mat_df = pd.DataFrame(scaled_data, index=test_mat.index, columns=test_mat.columns)
    correlation_matrix_mat = test_mat_df.corr(method='pearson')
    test4_mat_df = test_mat_df.copy()
    correlation_matrix_mat_test4 = correlation_matrix_mat.copy()
    test4_mat_df['Pedu'] = np.where(test4_mat_df['Medu'] > test4_mat_df['Fedu'], test4_mat_df['Medu'],
                                    test4_mat_df['Fedu'])
    test4_mat_df.drop(columns=['Medu', 'Fedu'], inplace=True)
    correlation_matrix_mat_test4.drop(columns=['Medu', 'Fedu'], inplace=True)
    del_col = correlation_matrix_mat_test4.columns[abs(correlation_matrix_mat_test4.iloc[-1]) < 0.105]
    test4_mat_df.drop(columns=del_col, inplace=True)
    correlation_matrix_mat_test4.drop(columns=del_col, inplace=True)
    x = test4_mat_df.drop(columns='G3', inplace=False)
    y = test4_mat_df['G3']

    return x, y

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2



if __name__ == "__main__":
    df_proc = pd.read_csv('mat_df_clear.csv')
    X, Y = scale_frame(df_proc)
    # разбиваем на тестовую и валидационную выборки
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, random_state=42)
    mlflow.set_experiment("linear model cars")
    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Делаем прогноз на валидационной выборке
        y_pred = model.predict(X_val)

        rmse, mae, r2 = eval_metrics(y_val, y_pred)

        # Логируем метрики
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = model.predict(X_train)
        signature = infer_signature(X_train, predictions)

        mlflow.sklearn.log_model(model, "model", signature=signature)

    dfruns = mlflow.search_runs()
    path2model = dfruns.sort_values("metrics.r2", ascending=False).iloc[0]['artifact_uri'].replace("file://",
                                                                                                   "") + '/model'  # путь до эксперимента с лучшей моделью
    print(path2model)
