import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn import svm
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras import metrics, optimizers
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import KFold

# Write a function that imputes median
def impute_median(series):

    return series.fillna(series.median())


# Function that receives two filenames and returns two datasets
def import_data(train_file, prediction_file):
	# import data for train set
	train = pd.read_csv(train_file, index_col=0, header=0)

	# import data for the prediction set
	X_pred = pd.read_csv(prediction_file, index_col=0, header=0)

	return train, X_pred


# Function that receives two datasets, turns the second with the same shape
# of the first, imputes medians to missing values, and return both datasets.
# It is required since we are going to train and predict using the same
# set of features.
def adjust_features(train, X_pred):
	# we start by insertin a 'Survived' collumn into the predictions dataset, in order
	# for it to have exactly the same shape as the train set
	X_pred = pd.concat([X_pred, pd.Series([0 for i in range(len(X_pred))],
		name='Survived', index=[i + len(train) + 1 for i in range(len(X_pred))])], axis=1)


	# the full set is concatenation of train and prediction ones
	full = pd.concat([train, X_pred], axis=0)

	# turn 'name' (last, title, first) column into 'title', 'family', and 'name'
	full['Title'] = full['Name'].str.extract('[A-Za-z\s*]*, ([A-Za-z\s*]*)\.')
	full['Family'] = full['Name'].str.extract('^([A-Za-z\s|\'*|\-]*), [A-Za-z\s*]*\.')
	full['Name'] = full['Name'].str.extract('[A-Z]\w+, \w+\. (.*)$')

	# create two dummy or indicator features from 'sex': 'male' and 'female'
	full = pd.concat([full, pd.get_dummies(full['Sex'])], axis=1)

	# create dummy or indicator features from 'title'
	full = pd.concat([full, pd.get_dummies(full['Title'])], axis=1)

	# create dummy variable for the port of embarkment
	full = pd.concat([full, pd.get_dummies(full['Embarked'].fillna('X'))], axis=1)

	# obtain the deck where the passanger was from the cabin code and turn into dummies
	full = pd.concat([full, pd.get_dummies(full['Cabin'].str[0].fillna('X'))], axis=1)

	# turn family into dummy variables
	full = pd.concat([full, pd.get_dummies(full['Family'].fillna('unknown'))], axis=1)

	# now we split the full back to train and test sets, having the same number of features on
	# both, except the 'Survived' column which is now removed from the test set
	n = len(train)
	train = full[:n].copy(deep=True)
	X_pred = full[n:].drop(['Survived'], axis=1).copy(deep=True)

	# Impute age median by class and title and assign to titanic['age']
	train['Age'] = train.groupby(['Pclass', 'Title'])['Age'].transform(impute_median)
	X_pred['Age'] = X_pred.groupby(['Pclass', 'Title'])['Age'].transform(impute_median)
	# full['Age'] = full.groupby(['Pclass', 'Title'])['Age'].transform(impute_median)

	# Impute age median by class and sex and assign to titanic['age']
	train['Age'] = train.groupby(['Pclass', 'Sex'])['Age'].transform(impute_median)
	X_pred['Age'] = X_pred.groupby(['Pclass', 'Sex'])['Age'].transform(impute_median)
	# full['Age'] = full.groupby(['Pclass', 'Sex'])['Age'].transform(impute_median)

	# Impute fare median by class and sex and assign to titanic['fare']
	train['Fare'] = train.groupby(['Pclass', 'Sex'])['Fare'].transform(impute_median)
	X_pred['Fare'] = X_pred.groupby(['Pclass', 'Sex'])['Fare'].transform(impute_median)
	# full['Fare'] = full.groupby(['Pclass', 'Sex'])['Fare'].transform(impute_median)
	
	# remove columns we don't want and return
	return train.drop([
		'Name',
		'Sex',
		'male',
		'Title',
		'Ticket',
		'Embarked',
		'Cabin',
		'Family'], axis=1), X_pred.drop([
		'Name',
		'Sex',
		'male',
		'Title',
		'Ticket',
		'Embarked',
		'Cabin',
		'Family'], axis=1)


############################
# Main program starts here #
############################

# Import data for train and prediction set
train, X_pred = import_data('train.csv', 'pred.csv')

# Now we have to make both train and prediction datasets compatible (same shape)
# and impute de medians to missing values
train, X_pred = adjust_features(train, X_pred)

# set Survived as target and everything else as features
X = train.drop('Survived', axis=1).values
y = train['Survived'].values

##########################
# linear regression part #
##########################

# Setup the hyperparameter grid
param_grid = {'C': np.logspace(1, 2, 100)}

# Instantiate a logistic regression classifier
logreg = LogisticRegression(max_iter=1000)

# Instantiate the GridSearchCV
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))

# Create the logistic regression classifier using the best parameter found by gridsearch
logreg = LogisticRegression(max_iter=1000, C=logreg_cv.best_params_['C'])

# split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(" ----- metrics for Logistic Regression ----- ")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.show()

# Compute and print AUC score
print("AUC for Log Reg: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')

# Print list of AUC scores
print("AUC scores for Log Reg computed using 5-fold cross-validation: {}".format(cv_auc))

# predict now on kaggle's data set (for submission)
y_pred = logreg.predict(X_pred.values)

# create dataframe for submission and export it to csv
submission = pd.DataFrame(data={'Survived': y_pred}, index=X_pred.index)
submission.to_csv('submission_logreg.csv')

############
# SVM part #
############ 

# Setup the hyperparameter grid
param_grid = {'C': np.logspace(1.5, 4, 20)}

# Instantiate a SVM classifier
clf = svm.SVC()

# Instantiate the GridSearchCV
clf_cv = GridSearchCV(clf, param_grid, cv=5)

# Fit it to the data
clf_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned SVM Parameters: {}".format(clf_cv.best_params_))
print("Best score is {}".format(clf_cv.best_score_))

# Create the logistic regression classifier using the best parameter found by gridsearch
clf = LogisticRegression(C=clf_cv.best_params_['C'])

# split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = clf.predict(X_test)

# Compute and print the confusion matrix and classification report
print(" ----- metrics for SVM ----- ")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Compute predicted probabilities: y_pred_prob
y_pred_prob = clf.predict_proba(X_test)[:, 1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVM')
plt.show()

# Compute and print AUC score
print("AUC for SVM: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')

# Print list of AUC scores
print("AUC scores for SVM computed using 5-fold cross-validation: {}".format(cv_auc))

# predict now on kaggle's data set (for submission)
y_pred = clf.predict(X_pred.values)

# create dataframe for submission and export it to csv
submission = pd.DataFrame(data={'Survived': y_pred}, index=X_pred.index)
submission.to_csv('submission_svm.csv')

######################
# Deep Learning part #
######################

# set Survived as target and everything else as features
X = train.drop('Survived', axis=1).values
y = to_categorical(train['Survived'])

# number of features we're going to use as input
n_cols = X.shape[1]

# to easier testing on number of neurons on hidden layers
n_neurons =  200

# Create model without any layers
model = Sequential()

# Add the first layer
model.add(Dense(n_neurons,activation='relu',input_shape=(n_cols,)))

# Add hidden layer
model.add(Dense(n_neurons, activation='relu'))

# Add hidden layer
model.add(Dense(n_neurons, activation='relu'))

# Add the output layer
model.add(Dense(2,activation='softmax'))

# Compile the model
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.1) #, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# just to see if the model is ok
model.summary()

# fit model
model.fit(X, y, epochs=250, batch_size=10, verbose=True)

# predict as probabilites of true values
y_pred = model.predict(X_pred.values)
probability_true = y_pred[:,1]

y_submit = np.array(probability_true > 0.5, dtype=bool).astype(int)

# create dataframe for submission and export it to csv
submission = pd.DataFrame(data={'Survived': y_submit}, index=X_pred.index)
submission.to_csv('submission_nn.csv')
