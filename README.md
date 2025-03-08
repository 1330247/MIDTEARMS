import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters for the log-normal distribution
mu_0 = 0  # true mean of the underlying normal distribution
sigma_0 = 1  # true std of the underlying normal distribution
n = 50  # sample size
num_simulations = 1000  # number of simulations
alpha = 0.05  # significance level

# Function to compute confidence intervals for log-normal parameters
def compute_confidence_intervals(data):
    log_data = np.log(data)
    
    # Estimating parameters for the log-normal (mean and std of log data)
    mu_hat = np.mean(log_data)
    sigma_hat = np.std(log_data, ddof=1)
    
    # Confidence interval for mu using t-distribution
    se_mu = sigma_hat / np.sqrt(n)
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    mu_ci = (mu_hat - t_critical * se_mu, mu_hat + t_critical * se_mu)
    
    # Confidence interval for sigma using chi-squared distribution
    chi2_lower = stats.chi2.ppf(alpha/2, df=n-1)
    chi2_upper = stats.chi2.ppf(1 - alpha/2, df=n-1)
    sigma2_ci = ((n-1) * sigma_hat**2 / chi2_upper, (n-1) * sigma_hat**2 / chi2_lower)
    
    return mu_ci, np.sqrt(sigma2_ci[0]), np.sqrt(sigma2_ci[1])

# Running the simulation
coverage_mu = 0
coverage_sigma = 0
total_length_mu = 0
total_length_sigma = 0

for _ in range(num_simulations):
    # Generate log-normal data
    data = np.random.lognormal(mu_0, sigma_0, n)
    
    # Compute confidence intervals
    mu_ci, sigma_ci_lower, sigma_ci_upper = compute_confidence_intervals(data)
    
    # Check coverage for mu and sigma
    if mu_0 >= mu_ci[0] and mu_0 <= mu_ci[1]:
        coverage_mu += 1
    if sigma_0 >= sigma_ci_lower and sigma_0 <= sigma_ci_upper:
        coverage_sigma += 1
    
    # Calculate average length of confidence intervals
    total_length_mu += (mu_ci[1] - mu_ci[0])
    total_length_sigma += (sigma_ci_upper - sigma_ci_lower)

# Compute coverage probabilities and average lengths
coverage_mu /= num_simulations
coverage_sigma /= num_simulations
avg_length_mu = total_length_mu / num_simulations
avg_length_sigma = total_length_sigma / num_simulations

# Output results
print(f"Coverage Probability for mu: {coverage_mu:.4f}")
print(f"Coverage Probability for sigma: {coverage_sigma:.4f}")
print(f"Average Length of CI for mu: {avg_length_mu:.4f}")
print(f"Average Length of CI for sigma: {avg_length_sigma:.4f}")
