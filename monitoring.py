import mlflow
import mlflow.sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = load_breast_cancer()
X = df.data
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

with mlflow.start_run(nested=True):
    model = MLPRegressor()

    mlflow.log_params(
        model.get_params()
    )

    for step in range(50):
        model.partial_fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = mean_squared_error(y_pred, y_test)
        mlflow.log_metric('epoch accuracy', accuracy, step=step)
    mlflow.sklearn.save_model(model, 'model')
mlflow.end_run()