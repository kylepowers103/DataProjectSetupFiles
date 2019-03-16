#LOAD compressed data
df = pd.read_csv('creditcard.csv.gz', compression='gzip') #Here we load a compressed csv file.
df.head()

# Create table for missing data analysis
def draw_missing_data_table(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
# Plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation score")
    plt.legend(loc="best")
    return plt
# Plot validation curve
def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=None, cv=None,n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    train_scores, test_scores = validation_curve(estimator, X, y, param_name, param_range, cv)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean, color='r', marker='o', markersize=5, label='Training score')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='r')
    plt.plot(param_range, test_mean, color='g', linestyle='--', marker='s', markersize=5, label='Validation score')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='g')
    plt.grid()
    plt.xscale('log')
    plt.legend(loc='best')
    plt.xlabel('Parameter')
    plt.ylabel('Score')
    plt.ylim(ylim)
    return test_std

### DATA HANDLING

# Import data
df = pd.read_csv('../input/train.csv')
df_raw = df.copy()  # Save original data set, just in case.
# Analyse missing data
draw_missing_data_table(df)

# Drop Column = Cabin
df.drop('Cabin', axis=1, inplace=True)
df.head()

# Fill missing values in Age with a specific value
value = 1000
df['Age'].fillna(1000, inplace=True)
df['Age'].max()

# Delete observations without column entry = Embarked
df.drop(df[pd.isnull(df['Embarked'])].index, inplace=True)  # Get index of points where Embarked is null
df[pd.isnull(df['Embarked'])]

# Data types
df.dtypes

# Drop PassengerId due to wrong dtype
df.drop('PassengerId', axis=1, inplace=True)
df.head()

# Define categorical variables
df['Sex'] = pd.Categorical(df['Sex'])
df['Embarked'] = pd.Categorical(df['Embarked'])

# Transform categorical variables into dummy variables
df = pd.get_dummies(df, drop_first=True)  # To avoid dummy trap
df.head()

# Extract titles from name AKA GET MR, MRS and fill in age if empty
def fill_in_age_by_using_title_andcount_howmany_per_class():
    #  The rule seems to be: *'last name'* + *','* + *'title'* + *'other names'*
    df['Title']=0
    for i in df:
        df['Title']=df_raw['Name'].str.extract('([A-Za-z]+)\.', expand=False)  # Use REGEX to define a search pattern
    df.head()

    # Impute ages based on titles
    idx_nan_age = df.loc[np.isnan(df['Age'])].index
    df.loc[idx_nan_age,'Age'].loc[idx_nan_age] = df['Title'].loc[idx_nan_age].map(map_means)
    df.head()
    return df

    # Count how many people have each of the titles
    df.groupby(['Title'])['PassengerId'].count()

# Create data set to train data imputation methods
X = df[df.loc[:, df.columns != 'Survived'].columns]
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)

# Debug
print('Inputs: \n', X_train.head())
print('Outputs: \n', y_train.head())

# Fit logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Model performance
scores = cross_val_score(logreg, X_train, y_train, cv=10)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# Plot learning curves
title = "Learning Curves (Logistic Regression)"
cv = 10
plot_learning_curve(logreg, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=1);


# Rescale data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_transformed_scaled = scaler.fit_transform(X_train_transformed)
X_test_transformed_scaled = scaler.transform(X_test_transformed)

# Get polynomial features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2).fit(X_train_transformed)
X_train_poly = poly.transform(X_train_transformed_scaled)
X_test_poly = poly.transform(X_test_transformed_scaled)

def feature_selection_using_cross_val_score_and_SelectKBest():
    # Select features using chi-squared test
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2

    ## Get score using original model
    logreg = LogisticRegression(C=1)
    logreg.fit(X_train, y_train)
    scores = cross_val_score(logreg, X_train, y_train, cv=10)
    print('CV accuracy (original): %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    highest_score = np.mean(scores)

    ## Get score using models with feature selection
    for i in range(1, X_train_poly.shape[1]+1, 1):
        # Select i features
        select = SelectKBest(score_func=chi2, k=i)
        select.fit(X_train_poly, y_train)
        X_train_poly_selected = select.transform(X_train_poly)

        # Model with i features selected
        logreg.fit(X_train_poly_selected, y_train)
        scores = cross_val_score(logreg, X_train_poly_selected, y_train, cv=10)
        print('CV accuracy (number of features = %i): %.3f +/- %.3f' % (i,
                                                                         np.mean(scores),
                                                                         np.std(scores)))

        # Save results if best score
        if np.mean(scores) > highest_score:
            highest_score = np.mean(scores)
            std = np.std(scores)
            k_features_highest_score = i
        elif np.mean(scores) == highest_score:
            if np.std(scores) < std:
                highest_score = np.mean(scores)
                std = np.std(scores)
                k_features_highest_score = i
        #### Print the number of features
        # print('Number of features when highest score: %i' % k_features_highest_score)

        return k_features_highest_score

def fit_model_with_above_best_model():
    # Select features
    select = SelectKBest(score_func=chi2, k=k_features_highest_score)
    select.fit(X_train_poly, y_train)
    X_train_poly_selected = select.transform(X_train_poly)

    # Fit model
    logreg = LogisticRegression(C=1)
    logreg.fit(X_train_poly_selected, y_train)

    # Model performance
    scores = cross_val_score(logreg, X_train_poly_selected, y_train, cv=10)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    # Plot learning curves
    title = "Learning Curves (Logistic Regression)"
    cv = 10
    plot_learning_curve(logreg, title, X_train_poly_selected,
                        y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=1);
    # Plot validation curve
    title = 'Validation Curve (Logistic Regression)'
    param_name = 'C'
    param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    cv = 10
    plot_validation_curve(estimator=logreg, title=title, X=X_train_poly_selected, y=y_train,
                          param_name=param_name, ylim=(0.5, 1.01), param_range=param_range);
#learning curve details DONT RUN
def learning_curve_information():
    # Learning curves in a nutshell:
    # * Learning curves allow us to diagnose if the is **overfitting** or **underfitting**.
    # * When the model **overfits**, it means that it performs well on the training set, but not not on the validation set. Accordingly, the model is not able to generalize to unseen data. If the model is overfitting, the learning curve will present a gap between the training and validation scores. Two common solutions for overfitting are reducing the complexity of the model and/or collect more data.
    # * On the other hand, **underfitting** means that the model is not able to perform well in either training or validations sets. In those cases, the learning curves will converge to a low score value. When the model underfits, gathering more data is not helpful because the model is already not being able to learn the training data. Therefore, the best approaches for these cases are to improve the model (e.g., tuning the hyperparameters) or to improve the quality of the data (e.g., collecting a different set of features).
    pass

# Plot validation curve
title = 'Validation Curve (Logistic Regression)'
param_name = 'C'
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
cv = 10
plot_validation_curve(estimator=logreg, title=title, X=X_train, y=y_train, param_name=param_name,
                      ylim=(0.5, 1.01), param_range=param_range);
