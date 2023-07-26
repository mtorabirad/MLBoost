import numpy as np
from scipy.optimize import minimize, differential_evolution, rosen
from typing import Optional, Tuple


class LinearModelWithCustomLoss:
    """
    Linear model: Y = X Coeffs, fit by minimizing the provided loss_function
    with L2 regularization using scipy minimize, instead of the exact 
    solution provided by OLS that is valid only for MSE loss.

    by Mahdi Torabi Rad, Ph.D. 

     In matrix multiplication of X with Coeff, arrays have the following shapes

    Y: (n_samples, n_fh) where fh stands for forecast horizon
    
    X: (n_samples, n_lb, n_f), where n_lb is the width of look-back window 
                            and n_f is the number of features (dataframe columns)
    
    Coeff: (n_lb*n_f, n_fh).
    
    The multiplication X Coeff will be performed after the last two axes of X are flatted to
    result in shape (n_samples, n_lb*n_f)
    """

    def __init__(
        self,
        loss_function,
        X: Optional[np.ndarray] = None, # Shape (n_samples, n_look_back_window, n_features)
        Y: Optional[np.ndarray] = None, # Shape (n_samples, n_forecast_horizon)
        sample_weights: Optional[np.ndarray] = None,
        coefficients_init: Optional[np.ndarray] = None, # Initial 
        bounds: Optional[np.ndarray] = None, # Bounds for Differential evolution.
        optimizer: str=None,
        regularization: float = 1e-3,
        max_iterations: float = 100,
        tol: float = 1e-3,        
        verboose=False
    ):
        """
        Initialize the LinearModelWithCustomLoss object.       
        """
        self.loss_function = loss_function
        self.X = X
        self.Y = Y
        self.sample_weights = sample_weights
        self.initial_coefficients = coefficients_init
        
        self.bounds = bounds

        self.regularization = regularization
        self.learned_coefficients = None
        
        self.loss_values = []
        self.max_iterations = max_iterations
        self.tol = tol
        self.optimizer = optimizer
        self.function_call_counter = 0
        self.verboose=verboose
        


    def predict(self, X_pred: np.ndarray) -> np.ndarray:
        """
        Predict the target variable using the linear model.

        Args:
            X (numpy.ndarray): Input data matrix.

        Returns:
            numpy.ndarray: Predicted values.
        """
        # print('self.learned_coefficients.shape=', self.learned_coefficients.shape)
        # print('X_pred.shape', X_pred.shape)

        X_2 = X_pred.reshape((X_pred.shape[0], -1))
        #print('X_2.shape=', X_2.shape)
        
        X_2 = np.concatenate((np.ones((X_2.shape[0], 1)), X_2), axis=1)


        #print('X_2.shape=', X_2.shape)

        predictions = np.matmul(X_2, self.learned_coefficients)
        return predictions


    def get_model_error(self) -> float:
        """
        Calculate the error of the model's predictions.

        Returns:
            float: Model error.
        """
        if self.sample_weights is not None:
            error = self.loss_function(
                self.Y,
                self.predict(self.X),
                sample_weights=self.sample_weights,
            )
        else:
            error = self.loss_function(self.Y, self.predict(self.X))

        return error


    def get_l2_regularized_loss(self, optimizer_arg: np.ndarray) -> float:
        """
        This function will be passed to a scipy optimizer to 
        calculate the regularized loss of the model using most recent
        coefficients.

        Args:
            coefficients (numpy.ndarray): Model coefficients.

        Returns:
            float: Regularized loss.
        """
        # To handle vector regression.
        # Update model coefficients
        # print('optimizer_arg.shape = ', optimizer_arg.shape)
        # print('self.initial_coefficients.shape = ', self.initial_coefficients.shape)
        self.learned_coefficients = optimizer_arg.reshape((self.initial_coefficients.shape)) 
        
        model_error = self.get_model_error()
        regularization_term = sum(self.regularization * np.array(optimizer_arg) ** 2)
        regularized_loss = model_error + regularization_term
        
        return regularized_loss

    def fit(self):
        """
        Fit the linear model by minimizing the regularized loss.
        """
        if self.initial_coefficients is None:
            self.initial_coefficients = np.ones(self.X.shape[1] * self.X.shape[2])

        if (
            self.learned_coefficients is not None
            and np.all(self.initial_coefficients == self.learned_coefficients)
        ):
            print("There is a fitted version of the model already available; \
                  performing more iterations to continue the fit.")
        
        """ 
        Differential evolution has to be treated different than other optimizers
        because its API is different

        def differential_evolution(func, bounds, args=()):
        
        def minimize(fun, x0, args=()):
        """
        if (self.optimizer == 'de') or (self.optimizer == 'DE'):
            res = differential_evolution(
                self.get_l2_regularized_loss,
                self.bounds,                
                maxiter=self.max_iterations,
                callback=self.callback,
                tol=self.tol, 
                disp=False, 
                workers=1,
                popsize=50,
                polish=True
            )
        else:
            res = minimize(
                self.get_l2_regularized_loss,
                self.initial_coefficients,
                method=self.optimizer,
                options={"maxiter": self.max_iterations},
                callback=self.callback,
                tol=self.tol
            )
        # To handle vector regression:
        #self.learned_coefficients = res.x.reshape((self.X.shape[1] * self.X.shape[2], self.Y.shape[1])) 

    
    #def callback(self, x):
    def callback(self, xk, convergence=None):
        """
        Callback function called during optimization iterations to \
            record the value of loss changes with iterations.
        """
        self.function_call_counter += 1
        loss = self.loss_function(self.predict(self.X), self.Y)
        self.loss_values.append(loss)
        if self.verboose and self.function_call_counter % 10 == 0:
            print('counter = ', self.function_call_counter, ', loss', loss)
