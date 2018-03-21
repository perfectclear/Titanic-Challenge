# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import math

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
#combine = [train_df, test_df]
#g = sns.FacetGrid(train_df, col='Survived')
#g.map(plt.hist, 'Age', bins=10)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
#for dataset in combine:
#    dataset['FamilyName'] = dataset.Name.str.extract('([-A-Za-z- \']+)\,', expand=False)
#pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
#train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

#train_df.head()

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
#train_df.shape, test_df.shape
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#train_df.head()

# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
#grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
#grid.map(plt.hist, 'Age', alpha=.5, bins=20)
#grid.add_legend()

guess_ages = np.zeros((2,3))

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            if not math.isnan(age_guess):
                guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            else:
                guess_ages[i,j] = 0
                
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

#train_df.head()

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
#train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
#train_df.head()
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

#train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for dataset in combine:
    dataset['FamilySmallLarge'] = 0
    dataset.loc[(dataset['FamilySize'] <= 4), 'FamilySmallLarge'] = 0
    dataset.loc[(dataset['FamilySize'] > 4), 'FamilySmallLarge'] = 1
    
train_df[['FamilySmallLarge', 'Survived']].groupby(['FamilySmallLarge'], as_index=False).mean()

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

#for dataset in combine:
#    dataset['Age*Class'] = dataset.Age * dataset.Pclass


freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
train_df = train_df.drop(['Age'], axis=1)
train_df = train_df.drop(['Sex'], axis=1)
test_df = test_df.drop(['Age'], axis=1)
test_df = test_df.drop(['Sex'], axis=1)
combine = [train_df, test_df]

###########################################################################################
############################### DATA IS READY, BEGIN#######################################
###########################################################################################

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape

# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
#acc_log

coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)

## Support Vector Machines
#
#svc = SVC()
#svc.fit(X_train, Y_train)
#Y_pred = svc.predict(X_test)
#acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
#acc_svc
#
#knn = KNeighborsClassifier(n_neighbors = 3)
#knn.fit(X_train, Y_train)
#Y_pred = knn.predict(X_test)
#acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
#acc_knn
#
## Gaussian Naive Bayes
#
#gaussian = GaussianNB()
#gaussian.fit(X_train, Y_train)
#Y_pred = gaussian.predict(X_test)
#acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
#acc_gaussian
#
## Perceptron
#
#perceptron = Perceptron()
#perceptron.fit(X_train, Y_train)
#Y_pred = perceptron.predict(X_test)
#acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
#acc_perceptron
#
## Linear SVC
#
#linear_svc = LinearSVC()
#linear_svc.fit(X_train, Y_train)
#Y_pred = linear_svc.predict(X_test)
#acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
#acc_linear_svc
#
## Stochastic Gradient Descent
#
#sgd = SGDClassifier()
#sgd.fit(X_train, Y_train)
#Y_pred = sgd.predict(X_test)
#acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
#acc_sgd
#
## AdaBoost
#
#Ada_Boost = AdaBoostClassifier(n_estimators=100)
#Ada_Boost.fit(X_train, Y_train)
#Y_pred = Ada_Boost.predict(X_test)
#Ada_Boost.score(X_train, Y_train)
#acc_Ada_Boost = round(Ada_Boost.score(X_train, Y_train) * 100, 2)
#acc_Ada_Boost
#
## Decision Tree
#
#decision_tree = DecisionTreeClassifier()
#decision_tree.fit(X_train, Y_train)
#Y_pred = decision_tree.predict(X_test)
#acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
#acc_decision_tree

# Random Forest

random_forest = RandomForestClassifier(n_estimators=1000, max_depth = 3, min_samples_leaf = 1, min_samples_split = 3, max_features = None, bootstrap = False, criterion = 'gini', n_jobs = -1)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print('acc = ' + str(acc_random_forest))

##hyperparamter?
#param_dist = {"max_depth": [3, None],
#              "max_features": sp_randint(1, 5),
#              "min_samples_split": sp_randint(2, 11),
#              "min_samples_leaf": sp_randint(1, 11),
#              "bootstrap": [True, False],
#              "criterion": ["gini", "entropy"]}
#
#n_iter_search = 20
#random_search = RandomizedSearchCV(random_forest, param_distributions=param_dist,
#                                   n_iter=n_iter_search)
#random_search.fit(X_train, Y_train)
#randsearchresults = random_search.cv_results_
#
#param_grid = {"max_depth": [3, None],
#              "max_features": [1, 3, 5],
#              "min_samples_split": [2, 3, 10],
#              "min_samples_leaf": [1, 3, 10],
#              "bootstrap": [True, False],
#              "criterion": ["gini", "entropy"]}
#
## run grid search
#grid_search = GridSearchCV(random_forest, param_grid=param_grid)
#grid_search.fit(X_train, Y_train)
#gridresults = grid_search.cv_results_

#models = pd.DataFrame({
#    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
#              'Random Forest', 'Naive Bayes', 'Perceptron', 
#              'Stochastic Gradient Decent', 'Linear SVC', 
#              'Decision Tree', 'AdaBoost'],
#    'Score': [acc_svc, acc_knn, acc_log, 
#              acc_random_forest, acc_gaussian, acc_perceptron, 
#              acc_sgd, acc_linear_svc, acc_decision_tree, acc_Ada_Boost]})
#models.sort_values(by='Score', ascending=False)

submission4 = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

submission4.to_csv('submission4.csv', index=False)
