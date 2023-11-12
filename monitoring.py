import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.svm import NuSVC
from sklearn.metrics import mean_squared_error, r2_score

df = load_breast_cancer()
X = df.data
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(nested=True):
    model = NuSVC()
    mse_list = []

    mlflow.log_params(model.get_params())

    for epoch in range(20):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_pred, y_test)
        mse_list.append(mse)

        mlflow.log_metric("epoch mse", mse, step=epoch)

    mlflow.sklearn.save_model(model, 'model')

mlflow.end_run()