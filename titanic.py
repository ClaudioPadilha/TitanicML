import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, roc_curve, roc_auc_score


# Write a function that imputes median
def impute_median(series):

    return series.fillna(series.median())


# do the magic with the DF
def clean(df):

	# turn 'name' (last, title, first) column into 'title', 'family', and 'name'
	df['Title'] = df['Name'].str.extract('[A-Za-z\s*]*, ([A-Za-z\s*]*)\.')
	df['Family'] = df['Name'].str.extract('^([A-Za-z\s|\'*|\-]*), [A-Za-z\s*]*\.')
	df['Name'] = df['Name'].str.extract('[A-Z]\w+, \w+\. (.*)$')

	# Impute age median by class and title and assign to titanic['age']
	df['Age'] = df.groupby(['Pclass', 'Title'])['Age'].transform(impute_median)

	# Impute age median by class and sex and assign to titanic['age']
	df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(impute_median)

	# Impute fare median by class and sex and assign to titanic['fare']
	df['Fare'] = df.groupby(['Pclass', 'Sex'])['Fare'].transform(impute_median)

	# create two dummy or indicator features from 'sex': 'male' and 'female'
	df = pd.concat([df, pd.get_dummies(df['Sex'])], axis=1)

	# create dummy or indicator features from 'title'
	df = pd.concat([df, pd.get_dummies(df['Title'])], axis=1)

	# create dummy variable for the port of embarkment
	df = pd.concat([df, pd.get_dummies(df['Embarked'].fillna('X'))], axis=1)

	# obtain the deck where the passanger was from the cabin code and turn into dummies
	df = pd.concat([df, pd.get_dummies(df['Cabin'].str[0].fillna('X'))], axis=1)

	# turn family into dummy variables
	df = pd.concat([df, pd.get_dummies(df['Family'].fillna('unknown'))], axis=1)

	# remove columns we don't want and return
	return df.drop([
		'Name',
		'Sex',
		'male',
		'Title',
		'Ticket',
		'Embarked',
		'Cabin',
		'Family'], axis=1)


# import data for train set
train = pd.read_csv('train.csv', index_col=0, header=0)

# import data for the prediction set
X_pred = pd.read_csv('pred.csv', index_col=0, header=0)

# add a Survived series to it with all entries = 0, just to have the same shape of train set
X_pred = pd.concat([X_pred, pd.Series([0 for i in range(len(X_pred))],
	name='Survived', index=[i + len(train) + 1 for i in range(len(X_pred))])], axis=1)

# full set is concatenation of train and prediction
full = pd.concat([train, X_pred], axis=0)

# we clean the full dataset. We do so in order not to mess up the number of dummy variables
# in the train and test sets
full = clean(full)

# now we split the full back to train and test sets, having the same number of features on
# both, except the 'Survived' column which is now removed from the test set
n = len(train)
train = full[:n]
X_pred = full[n:].drop(['Survived'], axis=1)

# set Survived as target and everything else as features
X = train.drop('Survived', axis=1).values
y = train['Survived'].values

# Setup the hyperparameter grid
c_space = np.logspace(-2, 2, 80)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression(max_iter=1000)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))

# Create the classifier: logreg
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
submission.to_csv('submission.csv')
