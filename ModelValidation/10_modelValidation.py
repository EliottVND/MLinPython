# The model is fit using X_train and y_train
model.fit(X_train, y_train)

# Create vectors of predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Train/Test Errors
train_error = mae(y_true=y_train, y_pred=train_predictions)
test_error = mae(y_true=y_test, y_pred=test_predictions)

# Print the accuracy for seen and unseen data
print("Model error on seen data: {0:.2f}.".format(train_error))
print("Model error on unseen data: {0:.2f}.".format(test_error))


# Playing with random forest parameters

# Set the number of trees
rfr.n_estimators = 100

# Add a maximum depth
rfr.max_depth = 6

# Set the random state
rfr.random_state = 1111

# Fit the model
rfr.fit(X_train, y_train)


# Print how important each column is to the model
for i, item in enumerate(rfr.feature_importances_):
      # Use i and item to print out the feature importance of each column
    print("{0:s}: {1:.2f}".format(X_train.columns[i], item))


# Is this course useless?

# Fit the rfc model. 
rfc.fit(X_train, y_train)

# Create arrays of predictions
classification_predictions = rfc.predict(X_test)
probability_predictions = rfc.predict_proba(X_test)

# Print out count of binary predictions
print(pd.Series(classification_predictions).value_counts())

# Print the first value from probability_predictions
print('The first predicted probabilities are: {}'.format(probability_predictions[0]))


# Looking up model parameters

rfc = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=1111)

# Print the classification model
print(rfc)

# Print the classification model's random state parameter
print('The random state is: {}'.format(rfc.random_state))

# Print all parameters
print('Printing the parameters dictionary: {}'.format(rfc.get_params()))


# Dividing into Train, val and test

# Create dummy variables using pandas
X = pd.get_dummies(tic_tac_toe.iloc[:,0:9])
y = tic_tac_toe.iloc[:, 9]

# Create training and testing datasets. Use 10% for the test set
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.1, random_state=1111)

# Create temporary training and final testing datasets
X_temp, X_test, y_temp, y_test  =\
    train_test_split(X, y, test_size=0.2, random_state=1111)

# Create the final training and validation datasets
X_train, X_val, y_train, y_val =\
    train_test_split(X_temp, y_temp, test_size=0.25, random_state=1111)

# Different metrics to valide our model

# MAE : Mean Absolute error (en % d'erreur) : 9.99 = 10% d'erreur
# MSE : Mean Squared Error prend en compte les outliers de manière + importante

    # MAE :

        from sklearn.metrics import mean_absolute_error

        # Manually calculate the MAE
        n = len(predictions)
        mae_one = sum(abs(y_test - predictions)) / n
        print('With a manual calculation, the error is {}'.format(mae_one))

        # Use scikit-learn to calculate the MAE
        mae_two = mean_absolute_error(y_test, predictions)
        print('Using scikit-lean, the error is {}'.format(mae_two))

    # MSE

        from sklearn.metrics import mean_squared_error
        n = len(predictions)
        # Finish the manual calculation of the MSE
        mse_one = sum((y_test - predictions)**2) / n
        print('With a manual calculation, the error is {}'.format(mse_one))

        # Use the scikit-learn function to calculate MSE
        mse_two = mean_squared_error(y_test,predictions)
        print('Using scikit-lean, the error is {}'.format(mse_two))

# Using subsets to see how errors apply specificaly

    # Find the East conference teams
    east_teams = labels == "E"

    # Create arrays for the true and predicted values
    true_east = y_test[east_teams]
    preds_east = predictions[east_teams]

    # Print the accuracy metrics
    print('The MAE for East teams is {}'.format(
        mae(true_east, preds_east)))

    # Print the West accuracy
    print('The MAE for West conference is {}'.format(west_error))


# Metrics : Precision, accuracy and recall

    # Calculate and print the accuracy
    accuracy = (324 + 491) / (953)
    print("The overall accuracy is {0: 0.2f}".format(accuracy))

    # Calculate and print the precision
    precision = (491) / (491 + 15)
    print("The precision is {0: 0.2f}".format(precision))

    # Calculate and print the recall
    recall = (491) / (491 + 123)
    print("The recall is {0: 0.2f}".format(recall))

# Confusion matrix


    from sklearn.metrics import confusion_matrix

    # Create predictions
    test_predictions = rfc.predict(X_test)

    # Create and print the confusion matrix
    cm = confusion_matrix(y_test, test_predictions)
    print(cm)

    # Print the true positives (actual 1s that were predicted 1s)
    print("The number of true positives is: {}".format(cm[1, 1]))

    # Applying precision metrics

    from sklearn.metrics import precision_score

    test_predictions = rfc.predict(X_test)

    # Create precision or recall score based on the metric you imported
    score = precision_score(y_test, test_predictions)

    # Print the final result
    print("The precision value is {0:.2f}".format(score))


#Over-under fitting

    # Update the rfr model
    rfr = RandomForestRegressor(n_estimators=25,
                                random_state=1111,
                                max_features=4)
    rfr.fit(X_train, y_train)

    # Print the training and testing accuracies 
    print('The training error is {0:.2f}'.format(
      mae(y_train, rfr.predict(X_train))))
    print('The testing error is {0:.2f}'.format(
      mae(y_test, rfr.predict(X_test))))

    # Comparing scores 

    from sklearn.metrics import accuracy_score

    test_scores, train_scores = [], []
    for i in [1, 2, 3, 4, 5, 10, 20, 50]:
        rfc = RandomForestClassifier(n_estimators=i, random_state=1111)
        rfc.fit(X_train, y_train)
        # Create predictions for the X_train and X_test datasets.
        train_predictions = rfc.predict(X_train)
        test_predictions = rfc.predict(X_test)
        # Append the accuracy score for the test and train predictions.
        train_scores.append(round(accuracy_score(y_train, train_predictions), 2))
        test_scores.append(round(accuracy_score(y_test, test_predictions), 2))
    # Print the train and test scores.
    print("The training scores were: {}".format(train_scores))
    print("The testing scores were: {}".format(test_scores))


# Chapter 3

# Showing how using only one test set is bad

# Create two different samples of 200 observations 
sample1 = tic_tac_toe.sample(200, random_state=1111)
sample2 = tic_tac_toe.sample(200, random_state=1171)

# Print the number of common observations 
print(len([index for index in sample1.index if index in sample2.index]))

# Print the number of observations in the Class column for both samples 
print(sample1['Class'].value_counts())
print(sample2['Class'].value_counts()) # differs

# Leave One out cross validation
# K-fold


# KFold validation

from sklearn.model_selection import KFold

# Use KFold
kf = KFold(n_splits=5, shuffle=True, random_state=1111)

# Create splits
splits = kf.split(X)

# Print the number of indices
for train_index, validation_index in splits:
    print("Number of training indices: %s" % len(train_index))
    print("Number of validation indices: %s" % len(validation_index))

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

rfc = RandomForestRegressor(n_estimators=25, random_state=1111)

# Access the training and validation indices of splits
for train_index, val_index in splits:
    # Setup the training and validation data
    X_train, y_train = X[train_index], y[train_index]
    X_val, y_val = X[val_index], y[val_index]
    # Fit the random forest model
    rfc.fit(X_train, y_train)
    # Make predictions, and print the accuracy
    predictions = rfc.predict(X_val)
    print("Split accuracy: " + str(mean_squared_error(y_val, predictions)))


# CrossValidation using scikit-learn


# Instruction 1: Load the cross-validation method
from sklearn.model_selection import cross_val_score

# Instruction 2: Load the random forest regression model
from sklearn.ensemble import RandomForestRegressor

# Instruction 3: Load the mean squared error method
# Instruction 4: Load the function for creating a scorer
from sklearn.metrics import mean_squared_error, make_scorer

rfc = RandomForestRegressor(n_estimators=25, random_state=1111)
mse = make_scorer(mean_squared_error)

# Set up cross_val_score
cv = cross_val_score(estimator=rfc,
                     X=X_train,
                     y=y_train,
                     cv=10,
                     scoring=mse)

# Print the mean error
print(cv.mean())


# Leave one out the cross-validation
# K-Fold with every datapoint as a fold and 1 left as validation set

from sklearn.metrics import mean_absolute_error, make_scorer

# Create scorer
mae_scorer = make_scorer(mean_absolute_error)

rfr = RandomForestRegressor(n_estimators=15, random_state=1111)

# Implement LOOCV
scores = cross_val_score(estimator=rfr, X=X, y=y, cv=X.shape[0], scoring=mae_scorer)

# Print the mean and standard deviation
print("The mean of the errors is: %s." % np.mean(scores))
print("The standard deviation of the errors is: %s." % np.std(scores))

# Parameter tuning (yes, again)

# Review the parameters of rfr
print(rfr.get_params())

# Maximum Depth
max_depth = [4, 8, 12]

# Minimum samples for a split
min_samples_split = [2, 5, 10]

# Max features 
max_features =[4, 6, 8, 10]

# Using random.choice to select a random parameter


from sklearn.ensemble import RandomForestRegressor

# Fill in rfr using your variables
rfr = RandomForestRegressor(
    n_estimators=100,
    max_depth=random.choice(max_depth),
    min_samples_split=random.choice(min_samples_split),
    max_features=random.choice(max_features))

# Print out the parameters
print(rfr.get_params())


# Using RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error

# Finish the dictionary by adding the max_depth parameter
param_dist = {"max_depth": [2,4,6,8],
              "max_features": [2, 4, 6, 8, 10],
              "min_samples_split": [2, 4, 8, 16]}

# Create a random forest regression model
rfr = RandomForestRegressor(n_estimators=10, random_state=1111)

# Create a scorer to use (use the mean squared error)
scorer = make_scorer(mean_squared_error)

# Import the method for random RandomizedSearchCVrandom_search
from sklearn.model_selection import RandomizedSearchCV

# Build a random search using param_dist, rfr, and scorer
random_search =\
    RandomizedSearchCV(
        estimator=rfr,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring=scorer)


from sklearn.metrics import precision_score, make_scorer

# Create a precision scorer
precision = make_scorer(precision_score)
# Finalize the random search
rs = RandomizedSearchCV(
  estimator=rfc, param_distributions=param_dist,
  scoring = precision,
  cv=5, n_iter=10, random_state=1111)
rs.fit(X, y)

# print the mean test scores:
print('The accuracy for each run was: {}.'.format(rs.cv_results_['mean_test_score']))
# print the best model score:
print('The best accuracy for a single model was: {}'.format(rs.best_score_))