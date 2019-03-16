def nyctimesapi():
    !pip install nytimesarticle
    from topstories import TopStoriesAPI
    api = TopStoriesAPI('3dJELzwLusu2ufxoTDgKE7dwjojyBDif')

    stories = api.get_stories("politics") # list of story dicts
    stories_string = api.get_stories("home", return_json_string=True) # json string
    # stories_jsonp = api.get_stories("work", format_type="jsonp") # (string) callback function with data input
    from nytimesarticle import articleAPI
    api = articleAPI('3dJELzwLusu2ufxoTDgKE7dwjojyBDif')
    # articles = api.search(q = 'Obama', fq = {'headline':'Obama', 'source':['Reuters','AP', 'The New York Times']}, begin_date = 20111231, facet_field = ['source','day_of_week'], facet_filter = True )
    articles = api.search( q = 'Obama' )
    return articles


# # Calculate the population mean from pumpkin_dict
# def calculate_mu(x):
#     # USe the formula for mu given above
#     d = float(sum(x.values()))/len(x)
#     return (d)
# mu = calculate_mu(pumpkin_dict)
# mu
def sample_means(sample_size, data):
    # """
    # This function takes in population data as a dictionary along with a chosen sample size
    # to generate all possible combinations of given sample size.
    # The function calculates the mean of each sample and returns:
    # a) a list of all combinations ( as tuples )
    # b) a list of means for all sample
    n = sample_size
    # Calculate the mean of population
    mu = calculate_mu(data)
    #print ("Mean of population is:", mu)
    # Generate all possible combinations using given sample size
    combs = list(itertools.combinations(data, n))
    #print ("Using", n, "samples with a population of size, we can see", len(combs), "possible combinations ")
    # Calculate the mean weight (x_bar) for all the combinations (samples) using the given data
    x_bar_list = []
    # Calculate sample mean for all combinations
    for i in range(len(combs)):
        sum = 0
        for j in range(n):
            key = combs[i][j]
            val =data[str(combs[i][j])]
            sum += val
        x_bar = sum/n
        x_bar_list.append(x_bar)
    #print ("The mean of all sample means mu_x_hat is:", np.mean(x_bar_list))
    #UNHIGHLOGHT BELOW TO RUN
    # return combs, x_bar_list

    #TO PRINT
    # n = 2 #Sample size
    # combs, means = sample_means(n, pumpkin_dict)
    # # Print the sample combinations with their means
    # for c in range(len(combs)):
    #     print (c+1, combs[c], means[c])
    return combs, x_bar_list #Copied to make above not go away
def Calculate_the_standard_error():
    # plt.figure(figsize=(15,10))
    # plt.axvline(x=mu, label = "Population mean")

    # Create empty lists for storing sample means, combinations and standard error for each iteration
    means_list = []
    combs_list = []
    err_list = []
    for n in (1, 2,3,4,5):
        # Calculate combinations, means and probabilities as earlier

        combs, means = sample_means(n, pumpkin_dict)

        combs_list.append(combs)
        means_list.append(means)

        # Calculate the standard error by dividing sample means with square root of sample size
        err = round(np.std(means)/np.sqrt(n), 2)
        err_list.append(err)

        val = n # this is the value where you want the data to appear on the y-axis.
        ar = np.arange(10) # just as an example array
        plt.plot(means, np.zeros_like(means) + val, 'x', label ="Sample size: "+ str(n) + " , Standard Error: "+ str(err) )
        plt.legend()
    plt.show()

def calculate_probability(means):
    import numpy as np
    from collections import Counter
    import matplotlib.pyplot as plt
    import itertools
    '''
    Input: a list of means (x_hats)
    Output: a list of probablitity of each mean value
    '''
    #Calculate the frequency of each mean value
    freq = Counter(means)
    prob = []
    # Calculate and append fequency of each mean value in the prob list.
    for element in means:
        for key in freq.keys():
            if element == key:
                prob.append(str(freq[key])+"/"+str(len(means)))
    return prob
    probs = calculate_probability(means)
    # Print combinations with sample means and probability of each mean value
    for c in range(len(combs)):
        print (c+1, combs[c], means[c], probs[c])
    Counter(means).keys()

def make_combinations():
    # Use itertools.combinations() to generate and print a list of combinations
    print (pumpkin_dict) ={'A': 19, 'B': 14, 'C': 15, 'D': 9, 'E': 10, 'F': 17}

    combs = list(itertools.combinations(pumpkin_dict, n))
    # Identify a sample size n
    n = 2

    # Use itertools.combinations() to generate and print a list of combinations
    combs = list(itertools.combinations(pumpkin_dict, n))
    combs

    print ("Using", n, "samples, we can see", len(combs), "possible combinations as below:")
    print (combs)

    # Using 2 samples, we can see 15 possible combinations as below:
    # [('A', 'B'), ('A', 'C'), ('A', 'D'), ('A', 'E'), ('A', 'F'), ('B', 'C'), ('B', 'D'),
    #  ('B', 'E'), ('B', 'F'), ('C', 'D'), ('C', 'E'), ('C', 'F'), ('D', 'E'), ('D', 'F'),
    #  ('E', 'F')]
    combs[1][1]
def get_html_text_from_link_count_words_in_text():
    import urllib.request
    # def getTopicCount(topic):
    topic="pizza"
    url = "https://en.wikipedia.org/w/api.php?action=parse&section=0&prop=text&format=json&page="
    contents = urllib.request.urlopen(url+topic).read().decode('utf-8')
    count = 0
    pos = contents.find() (topic)
    while pos != -1:
        count += 1
        pos = contents.find(topic, pos+1)
    return count

def get_urllib2():
    import urllib2
    url=""
    response = urllib2.urlopen(url)
    print "Response:", response

    # Get the URL. This gets the real URL.
    print "The URL is: ", response.geturl()

    # Getting the code
    print "This gets the code: ", response.code

    # Get the Headers.
    # This returns a dictionary-like object that describes the page fetched,
    # particularly the headers sent by the server
    print "The Headers are: ", response.info()

    # Get the date part of the header
    print "The Date is: ", response.info()['date']

    # Get the server part of the header
    print "The Server is: ", response.info()['server']

    # Get all data
    html = response.read()
    print "Get all data: ", html

    # Get only the length
    print "Get the length :", len(html)

    # Showing that the file object is iterable
    for line in response:
     print line.rstrip()

def Visualizing_Dispersion_Box_Plots():

    # import matplotlib.pyplot as plt
        # plt.style.use('ggplot') # for viewing a grid on plot
        # x = [54, 54, 54, 55, 56, 57, 57, 58, 58, 60, 81]
        # plt.boxplot(x,  showfliers=False)
        # plt.title ("Retirement Age BoxPlot")
        # plt.show()
    pass


def function_to_extract_links():
        # Function to extract links from webpage
        # If you repeatingly extract links you can use the function below:

        from BeautifulSoup import BeautifulSoup
        import urllib2
        import re
        def getLinks(url):
            html_page = urllib2.urlopen(url)
            soup = BeautifulSoup(html_page)
            links = []

            for link in soup.findAll('a', attrs={'href': re.compile("^http://")}):
                links.append(link.get('href'))

            return links

        print( getLinks("http://arstechnica.com") )


def mac_daddy_all_sklearn_algos():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import mean_squared_error

    # user variables to tune
    folds   = 10
    metric  = "neg_mean_squared_error"

    # hold different regression models in a single dictionary
    models = {}
    models["Linear"]        = LinearRegression()
    models["Lasso"]         = Lasso()
    models["ElasticNet"]    = ElasticNet()
    models["KNN"]           = KNeighborsRegressor()
    models["DecisionTree"]  = DecisionTreeRegressor()
    models["SVR"]           = SVR()
    models["AdaBoost"]      = AdaBoostRegressor()
    models["GradientBoost"] = GradientBoostingRegressor()
    models["RandomForest"]  = RandomForestRegressor()
    models["ExtraTrees"]    = ExtraTreesRegressor()

    # 10-fold cross validation for each model
    model_results = []
    model_names   = []
    for model_name in models:
    	model   = models[model_name]
    	k_fold  = KFold(n_splits=folds, random_state=seed)
    	results = cross_val_score(model, X_train, Y_train, cv=k_fold, scoring=metric)

    	model_results.append(results)
    	model_names.append(model_name)
    	print("{}: {}, {}".format(model_name, round(results.mean(), 3), round(results.std(), 3)))

    # box-whisker plot to compare regression models
    figure = plt.figure()
    figure.suptitle('Regression models comparison')
    axis = figure.add_subplot(111)
    plt.boxplot(model_results)
    axis.set_xticklabels(model_names, rotation = 45, ha="right")
    axis.set_ylabel("Mean Squared Error (MSE)")
    plt.margins(0.05, 0.1)
    plt.show()

def read_pickles():
    raw_path="/Users/powersky/Documents/11FlatIronSchoolFolder/Projects/FinalProject/ETF_Stock_Predictor_FlatironFinal/pkl_files"
    pkl_files = [f for f in listdir(raw_path) if isfile(join(raw_path, f))]
    pkl_files

def run_facebook_prophet():
    figure=model.plot(forecast)
    two_years = forecast.set_index('ds').join(test_ticker)
    two_years = two_years[['AAPL', 'yhat', 'yhat_upper', 'yhat_lower' ]].dropna().tail(800)
    two_years['yhat']=np.exp(two_years.yhat)
    two_years['yhat_upper']=np.exp(two_years.yhat_upper)
    two_years['yhat_lower']=np.exp(two_years.yhat_lower)

    two_years.tail()
    two_years[['AAPL', 'yhat']].plot()
    two_years_AE = (two_years.yhat - two_years.AAPL)
    print (two_years_AE.describe())

    r2_score(two_years.AAPL, two_years.yhat)
    mean_squared_error(two_years.AAPL, two_years.yhat)
    mean_absolute_error(two_years.AAPL, two_years.yhat)
    fig, ax1 = plt.subplots()
    ax1.plot(two_years.AAPL)
    ax1.plot(two_years.yhat)
    ax1.plot(two_years.yhat_upper, color='black',  linestyle=':', alpha=0.5)
    ax1.plot(two_years.yhat_lower, color='black',  linestyle=':', alpha=0.5)

    ax1.set_title('Actual S&P 500 (Orange) vs S&P 500 Forecasted Upper & Lower Confidence (Black)')
    ax1.set_ylabel('Price')
    ax1.set_xlabel('Date')
    full_df = forecast.set_index('ds').join(test_ticker)
    full_df['yhat']=np.exp(full_df['yhat'])
    fig, ax1 = plt.subplots()
    ax1.plot(full_df.AAPL)
    ax1.plot(full_df.yhat, color='black', linestyle=':')
    ax1.fill_between(full_df.index, np.exp(full_df['yhat_upper']), np.exp(full_df['yhat_lower']), alpha=0.5, color='darkgray')
    ax1.set_title('Actual S&P 500 (Orange) vs S&P 500 Forecasted (Black) with Confidence Bands')
    ax1.set_ylabel('Price')
    ax1.set_xlabel('Date')

    L=ax1.legend() #get the legend
    L.get_texts()[0].set_text('S&P 500 Actual') #change the legend text for 1st plot
    L.get_texts()[1].set_text('S&P 5600 Forecasted') #change the legend text for 2nd plot

def function_2_check_memory_of_kernel():
    # import os
    # import pwd
    # import pandas as pd
    # import psutil
    UID   = 1
    EUID  = 2
    pids = [pid for pid in os.listdir('/proc') if pid.isdigit()]
    df = []
    for pid in pids:
        try:
            ret = open(os.path.join('/proc', pid, 'cmdline'), 'rb').read()
        except IOError: # proc has already terminated
            continue

        # jupyter notebook processes
        if len(ret) > 0 and 'share/jupyter/runtime' in ret:
            process = psutil.Process(int(pid))
            mem = process.memory_info()[0]

            # user name for pid
            for ln in open('/proc/%d/status' % int(pid)):
                if ln.startswith('Uid:'):
                    uid = int(ln.split()[UID])
                    uname = pwd.getpwuid(uid).pw_name

            # user, pid, memory, proc_desc
            df.append([uname, pid, mem, ret])

    df = pd.DataFrame(df)
    print(df.columns = ['user', 'pid', 'memory', 'proc_desc'])

#TimeSeries
def stationarity_check(TS): ## Create a function to check for the stationarity of a given timeseries using rolling stats and DF test


    # Import adfuller
    from statsmodels.tsa.stattools import adfuller

    # Calculate rolling statistics
    rolmean = TS.rolling(window = 8, center = False).mean()
    rolstd = TS.rolling(window = 8, center = False).std()

    # Perform the Dickey Fuller Test
    dftest = adfuller(TS['#Passengers']) # change the passengers column as required

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12,6))
    orig = plt.plot(TS, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Print Dickey-Fuller test results
    print ('Results of Dickey-Fuller Test:')
	dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

    return None

def find_null_values_incolumns():
    #create dictionary to easily view number of null values and associated column names for all columns
    #unable to see this for all columns with "isna().sum()"
    null_dict = {}
    for i in range(len(flatiron_df.columns)):
        null_dict[flatiron_df.columns[i]] = flatiron_df.isna().sum()[i]
    null_dict
