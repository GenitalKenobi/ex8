import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate some sample data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 * X[:, 0] + 1 + 0.1 * np.random.randn(100)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start an MLflow nested run
with mlflow.start_run(nested=True):

    # Define hyperparameters
    hidden_layer_sizes = (100, 50)  # Two hidden layers with 100 and 50 neurons
    learning_rate_init = 0.001
    max_iter = 500

    # Create and train the MLPRegressor
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate_init, max_iter=max_iter)
    mse_list = []  # List to store MSE values at each epoch

    mlflow.log_params({
        "hidden_layer_sizes": hidden_layer_sizes,
        "learning_rate_init": learning_rate_init,
        "max_iter": max_iter
    })

    for epoch in range(max_iter):
        model.partial_fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_list.append(mse)

        # Log metrics at each epoch
        mlflow.log_metric("epoch_mse", mse, step=epoch)

    # Save the final model
    mlflow.sklearn.save_model(model, "MLPRegressor")

# End the nested MLflow run
mlflow.end_run()