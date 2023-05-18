import numpy as np
def importance_sampling(pdf, proposal, num_samples):
    """
    Importance sampling algorithm for generating samples from a given probability distribution function.

    Args:
    pdf (function): The probability distribution function to sample from.
    proposal (function): The proposal distribution function to use for generating samples.
    num_samples (int): The number of samples to generate.

    Returns:
    samples (numpy.ndarray): An array of samples drawn from the probability distribution function.
    weights (numpy.ndarray): An array of importance weights for each sample.
    """
    # Generate candidate samples from the proposal distribution
    samples = proposal.rvs(size=num_samples)

    # Calculate the unnormalized weights for each sample
    unnormalized_weights = pdf(samples) / proposal.pdf(samples)

    # Normalize the weights
    weights = unnormalized_weights / np.sum(unnormalized_weights)

    # Resample the samples with replacement according to the weights
    resampled_indices = np.random.choice(np.arange(num_samples), size=num_samples, replace=True, p=weights)
    resampled_samples = samples[resampled_indices]

    return resampled_samples, weights


def calculate_adjusted_MAEP(y_true, y_hat):
    """
    Calculates the adjusted Mean Absolute Percentage Error (adjustedMAPE) of a forecast.

    Parameters:
    y_true (list or array-like): The true values.
    y_hat (list or array-like): The forecasted values.

    Returns:
    float: The adjusted MAPE.
    """
    y_true = np.array(y_true)
    y_hat = np.array(y_hat)

    denom = np.array(y_true) + np.array(y_hat)
    adjustedmape = 200 * np.mean(np.abs(y_true - y_hat) / denom)
    return adjustedmape


def simulate_variable_evolution(num_days, initial_sigma, initial_z):
    """
    Simulates the evolution of a random variable over a period of num_days.
    
    Parameters:
    - num_days: int, the number of days to simulate.
    - initial_sigma: float, the initial standard deviation of the random variable.
    - initial_z: float, the initial value of the random variable.
    
    Returns:
    - A tuple containing three lists:
        1. A list of the values of the random variable.
        2. A list of the standard deviations of the random variable.
        3. A list of the days simulated.
    """
    sigmas = [initial_sigma]
    zs = [initial_z]
    days = [0]
    
    for i in range(num_days-1):
        new_sigma = np.sqrt(0.1 * zs[-1]**2 + 0.1 * sigmas[-1]**2 + 0.8)
        #new_z = np.random.normal(0, new_sigma)
        new_z = new_sigma*np.random.normal(0, 1)
        new_day = days[-1] + 1

        zs.append(new_z)
        sigmas.append(new_sigma)
        days.append(new_day)
        
    return zs, sigmas, days


def prepare_data_for_mapping_func(features_arr:np.ndarray, 
                               use_future_vals_in_pair_cnstruction:list, 
                               series_length:int,
                               n_features:int,
                               n_steps_in:int,
                               n_steps_out:int) -> np.ndarray:	
  '''
  Function map_forecasting_to_supervised that will map forecasting problem
  into a supervised ML problem needs an array with a certain shape.
  That array gets prepared by the current function.
  Let's call the array A.
  
  The first column in A need to be the original time series.
  This column will be used by the map_forecasting_to_supervised to
  create features for the supervised ML problem.
  
  The last column need to be back-ward shifted values of the time series. 
  These are the values we want to forecast.

  The intermediate columns need to be the covariates.
  
  Example:
  Imagine we are given time series 0, 0.1, 0.2, ..., 1. 
  The objective is to forecast the series with a look-back window 
  of size 3 and forecast horizen of 2.
  For example, forecasting [0.3, 0.4] from [0, 0.1, 0.2].

  The first column of array A should be [0, 0.1, ..., 0.8, NaN, NaN].
  Note that 0.9 and 1 should NOT be included in the first column because 
  our last forecasting roll forecasts [0.9, 1] from 
  [0.6, 0.7, 0.8] and NOT from [0.8, 0.9, 1].

  The last column of A should be [Nan, Nan, 0.3, 0.4, ..., 1]. 
  Note that 0.1 and 0.2 should NOT be included in the last column because
  our first forecasting roll uses [0, 0.1, 0.2] to forecast [0.3, 0.4]. 

  Handling of covariates is more tricky because we may want to use 
  future values for some of the covariates. TODO: Add explanation.

  '''
  features_and_trgt_vals = np.empty((series_length-1, n_features+1))
  features_and_trgt_vals[:] = np.NaN

  for curr_col in range(n_features):    
    indx = features_and_trgt_vals.shape[0] - (n_steps_out - 1) # series_length - n_steps_out
    
    if not use_future_vals_in_pair_cnstruction[curr_col]:
      features_and_trgt_vals[:indx, curr_col] = features_arr[:indx, curr_col] # copy past the same rows from features array.
    else:
      shift = n_steps_out
      features_and_trgt_vals[:indx, curr_col]  = features_arr[shift : shift + indx, curr_col]

  features_and_trgt_vals[n_steps_in-1:, n_features] = features_arr[:,0][n_steps_in:]
  return features_and_trgt_vals

from typing import Tuple
def map_forecasting_to_supervised(sequences: list, 
                                  n_steps_in: int, 
								  n_steps_out: int,
								  sliding_width: int, 
								  verbose=False) -> Tuple[np.ndarray, np.ndarray]:
	'''
	This function maps time series forecasting problem into 
	a supervised ML problem. 
	It inputs a time series and creates (X, y) pairs from it.
	For example, lets say our time series is [1, 2, 3, 4, ..., 10] and we want 
	to predic the next two values using the past three values: 
	for example, predicting [4, 5] using [1, 2, 3]. 
	The function creates the following sample pairs:
		x_1 = [1, 2, 3] and y_1 = [4, 5]
		x_2 = [2, 3, 4] and y_2 = [5, 6]
		etc.
		when the width of sliding window is one.
		
		When that width is 2, it creates the following pairs
		x_1 = [1, 2, 3] and y_1 = [4, 5]
		x_2 = [3, 4, 5] and y_2 = [6, 7]
		etc.
	Then returns X, y matrices containing all these pairs.
	'''
	X, y = list(), list()
	#sliding_width = 24
	for start_indx in range(0, len(sequences), sliding_width):
		end_indx = start_indx + n_steps_in
		out_end_ix = end_indx + n_steps_out-1

		if start_indx % 100 == 0:
			if verbose:
				print('from within map_forecasting_to_supervised now generating pair ', i)
		if out_end_ix > len(sequences):
			if verbose:
				print('breaking from within map_forecasting_to_supervised when i is', i)
			break
		
		seq_x = np.empty((n_steps_in, sequences.shape[1] - 1))
		
		for curr_feature in range(sequences.shape[1] - 1):
			seq_x[:, curr_feature] = sequences[start_indx:end_indx, curr_feature]
			
		seq_y = sequences[end_indx-1:out_end_ix, -1]

		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


#def weigthed_mean_absolute_percentage_error(y_pred, y_true, sample_weights=None):
"""  def weigthed_mean_absolute_percentage_error(y_true, y_pred, sample_weights=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true) == len(y_pred)
    
    if np.any(y_true==0):
        print("Found zeroes in y_true. MAPE undefined. Removing from set...")
        idx = np.where(y_true==0)
        y_true = np.delete(y_true, idx)
        y_pred = np.delete(y_pred, idx)
        if type(sample_weights) != type(None):
            sample_weights = np.array(sample_weights)
            sample_weights = np.delete(sample_weights, idx)
        
    if type(sample_weights) == type(None):
        return(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    else:
        sample_weights = np.array(sample_weights)
        assert len(sample_weights) == len(y_true)
        return(100/sum(sample_weights)*np.dot(
                sample_weights, (np.abs((y_true - y_pred) / y_true))
        ))"""