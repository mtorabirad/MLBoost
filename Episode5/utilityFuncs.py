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
        new_z = np.random.normal(0, new_sigma)
        new_day = days[-1] + 1

        zs.append(new_z)
        sigmas.append(new_sigma)
        days.append(new_day)
        
    return zs, sigmas, days
