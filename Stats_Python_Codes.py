# Import necessary libraries
def import_all_these():
    # import numpy as np
    # import scipy as sp
    # import pandas as pd
    # import scipy.stats as stats
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # import math
    pass

def stats_z_table():
        # Z-table in python
    import scipy.stats as stats
    # Probabilities upto z-score of 1.5
    print(stats.norm.cdf(1.5))
    # Probabilities greater than z-score
    print (1-stats.norm.cdf(1.5))
# def conf_interval_with_stats_t_critical_score(sample):
# https://github.com/learn-co-curriculum/dsc-2-19-17-confidence-intervals-with-t-distribution-lab/tree/solution
# https://github.com/learn-co-students/dsc-2-19-17-confidence-intervals-with-t-distribution-lab-nyc-career-ds-102218/tree/solution
KP
def conf_interval_two(pop,sample):
    size_of_sample = 500
        # '''
        # Function input: population , sample
        # Function output: z-critical, Margin of error, Confidence interval
        # '''
    n = len(sample)
    x_hat = sample.mean()
    # Calculate the z-critical value using stats.norm.ppf()
    # Note that we use stats.norm.ppf(q = 0.975) to get the desired z-critical value
    # instead of q = 0.95 because the distribution has two tails.
    z = stats.norm.ppf(q = 0.975)  #  z-critical value for 95% confidence
    #Calculate the population std from data
    pop_stdev = pop.std()
    # Calculate the margin of error using formula given above
    moe = z * (pop_stdev/math.sqrt(size_of_sample))
    # Calculate the confidence interval by applying margin of error to sample mean
    # (mean - margin of error, mean+ margin of error)
    conf = (x_hat - moe,x_hat + moe)
    return z, moe, conf
def get_results_from_conf_interval_two():
    # Call above function with sample and population
    z_critical, margin_of_error, confidence_interval = conf_interval(population_ages, sample)
    print("Z-critical value:")
    print(z_critical)
    print ('\nMargin of error')
    print(margin_of_error)
    print("\nConfidence interval:")
    print(confidence_interval)
    return z, moe, conf


def make_confidence_interval_samples_and_plot():
    #Lets set a sample size of 1000 and take 25 samples to calculate the confidence intervals using function above.
    #
    # np.random.seed(12)
    #
    # # Select the sample size
    # sample_size = 1000
    #
    # # Initialize lists to store interval and mean values
    # intervals = []
    # sample_means = []
    #
    # # Run a for loop for sampling 25 times and calculate + store confidence interval and sample mean values
    #
    # for sample in range(25):
    #     # Take a random sample of chosen size
    #     sample = np.random.choice(a= population_ages, size = sample_size)
    #     sample_mean = sample.mean()
    #     sample_means.append(sample_mean)
    #
    #     z_critical = stats.norm.ppf(q = 0.975)  # Get the z-critical value*
    #
    #     # Calculate z_critical, margin_of_error, confidence_interval from function above
    #     # z_critical, margin_of_error, confidence_interval = conf_interval(population_ages, sample)
    #
    #     pop_stdev = population_ages.std()  # Get the population standard deviation
    #
    #     stats.norm.ppf(q = 0.025)
    #
    #     margin_of_error = z_critical * (pop_stdev/math.sqrt(sample_size))
    #
    #     confidence_interval = (sample_mean - margin_of_error,
    #                            sample_mean + margin_of_error)
    #
    #     intervals.append(confidence_interval)
    #
    #     # Calculate and append sample means and conf intervals for each iteration
    # # plot the mean and confidence interval for each sample as error bars
    # # plot the population mean
    #
    # plt.figure(figsize=(15,9))
    #
    # plt.errorbar(x=np.arange(0.1, 25, 1),
    #              y=sample_means,
    #              yerr=[(top-bot)/2 for top,bot in intervals],
    #              fmt='o')
    #
    # plt.hlines(xmin=0, xmax=25,
    #            y=43.0023,
    #            linewidth=2.0,
    #            color="red")
    pass
def make_poisson_distribution_and_take_sample():
    population_ages1 = stats.poisson.rvs(loc=18, mu=35, size=150000)
    population_ages2 = stats.poisson.rvs(loc=18, mu=10, size=100000)
    population_ages = np.concatenate((population_ages1, population_ages2))
    # Take random sample of size 500
    sample_size = 500
    sample = np.random.choice(a= population_ages,size=500)
    # Calculate sample mean and standard deviation
    sample_mean = sample.mean()
    sample_std = sample.std()
    print ("Sample mean:", sample_mean)
    print ("Sample std.:", sample_std)
    print ("Difference between means:", population_ages.mean() - sample_mean)

def get_t_critical_values():
    # # Cal culate the t-critical value for 95% confidence level for sample taken above.
    # t_critical = stats.t.ppf(q=.975,df=sample_size-1)  # Get the t-critical value  by using 95% confidence level and degree of freedom
    # print("t-critical value:")                  # Check the t-critical value
    # print(t_critical)

    # Notes:Confidence interval = (sample_mean + margin of error, sample_mean - margin of error)
    pass

def stats_t_interval():
    # We can verify our calculations by using the Python function stats.t.interval():
    t_interval =stats.t.interval(alpha = 0.95,              # Confidence level
                 df= 24,                    # Degrees of freedom
                 loc = sample_mean,         # Sample mean
                 scale = sigma)             # Standard deviation estimate
    # in order to achieve 95% confidence the average level will have to be between what this formula spits out
    return t_interval

# Function to take in sample data and calculate the confidence interval




def print_metrics(labels, preds):
    from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
    print("Precision Score: {}".format(precision_score(labels, preds)))
    print("Recall Score: {}".format(recall_score(labels, preds)))
    print("Accuracy Score: {}".format(accuracy_score(labels, preds)))
    print("F1 Score: {}".format(f1_score(labels, preds)))
print_metrics(y_test, test_preds)
