html

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(one_hot_df, labels, test_size=0.25)


def sklearn_auto():
    # import autosklearn.classification
    cls = autosklearn.classification.AutoSklearnClassifier()
    cls.fit(X_train, y_train)
    predictions = cls.predict(X_test, y_test)

def knn():
    from sklearn.neighbors import KNeighborsClassifier
    clf1 = KNeighborsClassifier()
    clf1.fit(X_train, y_train)
    test_preds = clf1.predict(X_test)
    return test_preds


def find_best_k(X_train, y_train, X_test, y_test, min_k=1, max_k=25):
    best_k = 0
    best_score = 0.0
    for k in range(min_k, max_k+1, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        f1 = f1_score(y_test, preds)
        if f1 > best_score:
            best_k = k
            best_score = f1
    print("Best Value for k: {}".format(best_k))
    print("F1-Score: {}".format(best_score))
find_best_k(X_train, y_train, X_test, y_test)

def DecisionTree123():
    # from sklearn.model_selection import train_test_split
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn.metrics import accuracy_score, roc_curve, auc
    # from sklearn import tree
    # from sklearn.externals.six import StringIO
    # from IPython.display import Image
    # from sklearn.tree import export_graphviz
    # import pydotplus
    # import pandas as pd
    # import numpy as np
    # %matplotlib inline
    Apply labels to target variable such that yes=1 and no=0
    lb = LabelEncoder()
    # Instantiate a one hot encoder
    enc = preprocessing.OneHotEncoder()

    # Fit the feature set X
    enc.fit(X)

    # Transform X to onehot array
    onehotX = enc.transform(X).toarray()


    # Train a DT classifier
    # train another classifier with complete dataset
    clf= DecisionTreeClassifier(criterion='entropy')
    clf2= DecisionTreeClassifier(criterion='entropy')

    clf2.fit(onehotX,Y) # passing in data pre-split
    y_pred = clf2.predict(X_test)

    # Calculate Accuracy
    acc = accuracy_score(y_test,y_pred) * 100
    print("Accuracy is :{0}".format(acc))

    # Check the AUC for predictions
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print("AUC is :{0}".format(roc_auc))

    print('\nConfusion Matrix')
    print('----------------')
    pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

    Visualize the decision tree using graph viz library
    # dot_data = StringIO()
    # export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,special_characters=True)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # Image(graph.create_png())

    Visualize the tree trained from complete dataset
    # dot_data = StringIO()
    # export_graphviz(clf2, out_file=dot_data, filled=True, rounded=True,special_characters=True)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # Image(graph.create_png())
    return

def decistree_max_Tree_depth():
    # Identify the optimal tree depth for given data
    max_depths = np.linspace(1, 32, 32, endpoint=True)
    train_results = []
    test_results = []
    for max_depth in max_depths:
       dt = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
       dt.fit(x_train, y_train)
       train_pred = dt.predict(x_train)
       false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
       roc_auc = auc(false_positive_rate, true_positive_rate)
       # Add auc score to previous train results
       train_results.append(roc_auc)
       y_pred = dt.predict(x_test)
       false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
       roc_auc = auc(false_positive_rate, true_positive_rate)
       # Add auc score to previous test results
       test_results.append(roc_auc)
    plt.figure(figsize=(12,6))
    plt.plot(max_depths, train_results, 'b', label='Train AUC')
    plt.plot(max_depths, test_results, 'r', label='Test AUC')
    plt.ylabel('AUC score')
    plt.xlabel('Tree depth')
    plt.legend()
    plt.show()

def decistree_min_sample_leaf_splits():
    # Identify the optimal min-samples-split for given data
    min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
    train_results = []
    test_results = []
    for min_samples_split in min_samples_splits:
       dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=min_samples_split)
       dt.fit(x_train, y_train)
       train_pred = dt.predict(x_train)
       false_positive_rate, true_positive_rate, thresholds =    roc_curve(y_train, train_pred)
       roc_auc = auc(false_positive_rate, true_positive_rate)
       train_results.append(roc_auc)
       y_pred = dt.predict(x_test)
       false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
       roc_auc = auc(false_positive_rate, true_positive_rate)
       test_results.append(roc_auc)
    plt.figure(figsize=(12,6))
    plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')
    plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')
    plt.xlabel('Min. Sample splits')
    plt.legend()
    plt.show()

def max_features_decision_tree():
    # Find the best value for optimal maximum feature size
    max_features = list(range(1,x_train.shape[1]))
    train_results = []
    test_results = []
    for max_feature in max_features:
       dt = DecisionTreeClassifier(criterion='entropy', max_features=max_feature)
       dt.fit(x_train, y_train)
       train_pred = dt.predict(x_train)
       false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
       roc_auc = auc(false_positive_rate, true_positive_rate)
       train_results.append(roc_auc)
       y_pred = dt.predict(x_test)
       false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
       roc_auc = auc(false_positive_rate, true_positive_rate)
       test_results.append(roc_auc)


    plt.figure(figsize=(12,6))
    plt.plot(max_features, train_results, 'b', label='Train AUC')
    plt.plot(max_features, test_results, 'r', label='Test AUC')

    plt.ylabel('AUC score')
    plt.xlabel('max features')
    plt.legend()
    plt.show()

def dtree_with_best_features():
    # train a classifier with optimal values identified above
    dt = DecisionTreeClassifier(criterion='entropy',
                               max_features=7,
                               max_depth=3,
                               min_samples_split=0.6,
                               min_samples_leaf=0.25)
    dt.fit(x_train, y_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    roc_auc

def decisiontree_Regressor():
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state = 10, max_depth=3)
    regressor.fit(X_train, y_train)
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import r2_score

    # Make predictions and evaluate
    y_pred = regressor.predict(X_test)
    print ('MSE score:', mse(y_test, y_pred))
    print('R-sq score:',r2_score(y_test,y_pred))
    # Code here
# Visualising the Decision Tree Regression results (higher resolution)
    X_grid = np.arange(min(X), max(X), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))
    # print(X)
    plt.figure(figsize=(15,6))
    plt.scatter(X, y, color = 'red', label='data')
    plt.plot(X_grid, regressor.predict(X_grid), color = 'green', label='Regression function')
    plt.title('Decision Tree Regression')
    plt.xlabel('Features')
    plt.ylabel('Target')
    plt.legend()
    plt.show()

    # Code here
# !pip install # Code here
# Visualising the Decision Tree Regression results (higher resolution)
    X_grid = np.arange(min(X), max(X), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))
    # print(X)
    plt.figure(figsize=(15,6))
    plt.scatter(X, y, color = 'red', label='data')
    plt.plot(X_grid, regressor.predict(X_grid), color = 'green', label='Regression function')
    plt.title('Decision Tree Regression')
    plt.xlabel('Features')
    plt.ylabel('Target')
    plt.legend()
    plt.show()
    from sklearn.externals.six import StringIO
    from IPython.display import Image
    from sklearn.tree import export_graphviz
    import pydotplus
    dot_data = StringIO()
    # export_graphviz(regressor, out_file=dot_data, filled=True, rounded=True,special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.create_png())
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc

def LightGradientBoostingModel(features, test_features, encoding = 'ohe', n_folds = 5):
# https://github.com/WillKoehrsen/kaggle-credit-data-science-competition/blob/master/notebooks/1-Getting%20Started.ipynb
    """Train and test a light gradient boosting model using
    cross validation.

    Parameters
    --------
        features (pd.DataFrame):
            dataframe of training features to use
            for training a model. Must include the TARGET column.
        test_features (pd.DataFrame):
            dataframe of testing features to use
            for making predictions with the model.
        encoding (str, default = 'ohe'):
            method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
            n_folds (int, default = 5): number of folds to use for cross validation

    Return
    --------
        submission (pd.DataFrame):
            dataframe with `SK_ID_CURR` and `TARGET` probabilities
            predicted by the model.
        feature_importances (pd.DataFrame):
            dataframe with the feature importances from the model.
        valid_metrics (pd.DataFrame):
            dataframe with training and validation metrics (ROC AUC) for each fold and overall.

    """

    # Extract the ids
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']

    # Extract the labels for training
    labels = features['TARGET']

    # Remove the ids and target
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])


    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)

        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join = 'inner', axis = 1)

        # No categorical indices to record
        cat_indices = 'auto'

    # Integer label encoding
    elif encoding == 'le':

        # Create a label encoder
        label_encoder = LabelEncoder()

        # List for storing categorical indices
        cat_indices = []

        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)

    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")

    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)

    # Extract feature names
    feature_names = list(features.columns)

    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)

    # Create the kfold object
    k_fold = KFold(n_splits = n_folds, shuffle = True, random_state = 50)

    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))

    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])

    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])

    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []

    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):

        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]

        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary',
                                   class_weight = 'balanced', learning_rate = 0.05,
                                   reg_alpha = 0.1, reg_lambda = 0.1,
                                   subsample = 0.8, n_jobs = -1, random_state = 50)

        # Train the model
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 100, verbose = 200)

        # Record the best iteration
        best_iteration = model.best_iteration_

        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits

        # Make predictions
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits

        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]

        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']

        valid_scores.append(valid_score)
        train_scores.append(train_score)

        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()

    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})

    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)

    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))

    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')

    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})
#to RUN USE BELOW
        #     submission, fi, metrics = model(app_train, app_test)
        # print('Baseline metrics')
        # print(metrics)
        # fi_sorted = plot_feature_importances(fi)
        # submission.to_csv('baseline_lgb.csv', index = False)
        # app_train_domain['TARGET'] = train_labels
        #
        # # Test the domain knolwedge features
        # submission_domain, fi_domain, metrics_domain = model(app_train_domain, app_test_domain)
        # print('Baseline with domain knowledge features metrics')
        # print(metrics_domain)
        # https://github.com/WillKoehrsen/kaggle-credit-data-science-competition/blob/master/notebooks/1-Getting%20Started.ipynb
    return submission, feature_importances, metrics


def xgboost():
    import xgboost as xgb
    boost = xgb.XGBClassifier()
    #print("XGBoost:", boost.score(X_train, Y_train))

    xgb.plot_importance(boost, max_num_features=50, height=0.8)

def plot_dtree_importance():
    tree_clf.feature_importances_
    def plot_feature_importances(model):
    n_features = data_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), data_train.columns.values)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

    plot_feature_importances(tree_clf)

#Model performance
    pred = tree_clf.predict(data_test)
    print(confusion_matrix(target_test, pred))
    print(classification_report(target_test, pred))
    print("Testing Accuracy for Decision Tree Classifier: {:.4}%".format(accuracy_score(target_test, pred) * 100))
def naive_bayes():
    #Fitting Naive_Bayes
    # Code here
    #Fitting Naive_Bayes
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train_scaled, Y_train);

    classifier.class_prior_
    # Make Predictions
    Y_pred = classifier.predict(X_test_scaled)
    Y_pred
    ## Calculate accuracy using formula
    acc=np.mean(Y_test==Y_pred)
    print( acc)
    # Calculate accuracy using scikit learn
    # Code here
    from sklearn.metrics import accuracy_score
    accuracy_score(Y_test, Y_pred)

def bagged_trees():
    bagged_tree =  BaggingClassifier(DecisionTreeClassifier(criterion='gini', max_depth=5), n_estimators=20)
    bagged_tree.fit(data_train, target_train)
    bagged_tree.score(data_train, target_train)
    bagged_tree.score(data_test, target_test)
def randomforests_classifier_model():
    forest = RandomForestClassifier(n_estimators=100, max_depth= 5)
    forest.fit(data_train, target_train)
    forest.score(data_train, target_train)
    forest.score(data_test, target_test)
    plot_feature_importances(forest)
def small_tree_randomforests_classifier_model():
    forest_2 = RandomForestClassifier(n_estimators = 5, max_features= 10, max_depth= 2)
    forest_2.fit(data_train, target_train)
    rf_tree_1 = forest_2.estimators_[0]
    plot_feature_importances(rf_tree_1)
    rf_tree_2 = forest_2.estimators_[1]
    plot_feature_importances(rf_tree_2)
