
import numpy as np
from scipy.optimize import minimize
from typing import Optional, Tuple


class LinearModelWithCustomLoss:
    """
    Linear model: Y = XB, fit by minimizing the provided loss_function
    with L2 regularization using scipy minimize, instead of the exact 
    solution provided by OLS that is valid only for MSE loss.

    by Mahdi Torabi Rad, Ph.D. 
    """

    def __init__(
        self,
        loss_function,
        X: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
        coefficients_init: Optional[np.ndarray] = None,
        regularization: float = 1e-3,
    ):
        """
        Initialize the CustomLinearModel object.

        Args:
            loss_function (function): Loss function to be minimized.
            X (numpy.ndarray, optional): Input data matrix. Defaults to None.
            Y (numpy.ndarray, optional): Target variable array. Defaults to None.
            sample_weights (numpy.ndarray, optional): Array of sample weights. Defaults to None.
            beta_init (numpy.ndarray, optional): Initial values for model coefficients. Defaults to None.
            regularization (float, optional): L2 regularization parameter. Defaults to 1e-3.
        """
        self.regularization = regularization
        self.coefficients = None
        self.loss_function = loss_function
        self.sample_weights = sample_weights
        self.initial_coefficients = coefficients_init

        self.input_data = X
        self.target_variable = Y
        self.loss_values = []


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target variable using the linear model.

        Args:
            X (numpy.ndarray): Input data matrix.

        Returns:
            numpy.ndarray: Predicted values.
        """
        predictions = np.matmul(X, self.coefficients)
        return predictions


    def get_model_error(self) -> float:
        """
        Calculate the error of the model's predictions.

        Returns:
            float: Model error.
        """
        if self.sample_weights is not None:
            error = self.loss_function(
                self.target_variable,
                self.predict(self.input_data),
                sample_weights=self.sample_weights,
            )
        else:
            error = self.loss_function(self.target_variable, self.predict(self.input_data))

        return error


    def get_l2_regularized_loss(self, coefficients: np.ndarray) -> float:
        """
        Calculate the regularized loss of the model.

        Args:
            coefficients (numpy.ndarray): Model coefficients.

        Returns:
            float: Regularized loss.
        """
        self.coefficients = coefficients
        model_error = self.get_model_error()
        regularization_term = sum(self.regularization * np.array(self.coefficients) ** 2)
        regularized_loss = model_error + regularization_term

        return regularized_loss

    def fit(self, max_iterations: int = 250):
        """
        Fit the linear model by minimizing the regularized loss.

        Args:
            max_iterations (int, optional): Maximum number of \
            iterations for the optimization. Defaults to 250.
        """
        if self.initial_coefficients is None:
            self.initial_coefficients = np.ones(self.input_data.shape[1])

        if (
            self.coefficients is not None
            and np.all(self.initial_coefficients == self.coefficients)
        ):
            print("There is a fited version of the model already \
                  available; performing more iterations to continue the fit.")

        res = minimize(
            self.get_l2_regularized_loss,
            self.initial_coefficients,
            method="BFGS",
            options={"maxiter": max_iterations},
            callback=self.callback,
        )
        self.coefficients = res.x

    
    def callback(self, x):
        """
            Callback function called during optimization iterations \
            to record the value of loss changes with iterations.
        """

        loss = self.loss_function(self.predict(self.input_data), self.target_variable)
        self.loss_values.append(loss)


