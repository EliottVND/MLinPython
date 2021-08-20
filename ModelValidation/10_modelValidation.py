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