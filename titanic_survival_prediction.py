import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.grid_search import GridSearchCV

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 30)

sns.set_style("darkgrid")
sns.set_context(rc={"figure.figsize": (6, 3)})

####################
# load in datasets #
####################
dfs = {}
for name in ['titanic_train', 'titanic_test']:
    df = pd.read_csv('%s.csv' % name)

    # add a column denoting source (train/test)
    df['data'] = name
    
    # add df to dfs dict
    dfs[name] = df
    
    
# -------------------- basic info about columns in each dataset --------------------
for name, df in dfs.iteritems():
    print 'df: %s\n' % name
    print 'shape: %d rows, %d cols\n' % df.shape
    
    print 'column info:'
    for col in df.columns:
        print '* %s: %d nulls, %d unique vals, most common: %s' % (
            col, 
            df[col].isnull().sum(),
            df[col].nunique(),
            df[col].value_counts().head(2).to_dict()
        )
    print '\n------\n'
    
    
# -------------------- combine train and test data into one df --------------------
df = dfs['titanic_train'].append(dfs['titanic_test'])

df.columns = map(str.lower, df.columns) # lowercase column names

new_col_order = ['data', 'passengerid', 'survived', 'age',
                'cabin', 'embarked', 'fare', 'name', 'parch',
                'pclass', 'sex', 'sibsp', 'ticket']
df = df[new_col_order] # reorder columns


# -------------------- convert sex to ints (male = 1, female = 0) -------------------- 
df['gender'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)


# -------------------- extract title -------------------- 
df['title'] = df['name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
title_mapping = {"Mr": 1, "Miss": 2, "Ms": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Capt": 7, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Dona": 8, "Sir": 9, "Don": 9, "Lady": 10, "Countess": 10, "the Countess": 10, "Jonkheer": 10}
titles = df['title']
for key, value in title_mapping.items():
    titles[titles == key] = value
df['title'] = titles


# -------------------- fill missing age with mean by title -------------------- 
age_by_title = df.groupby('title')['age'].agg(np.mean).to_dict()
df['age'] = df.apply(lambda row: age_by_title.get(row['title']) 
                     if pd.isnull(row['age']) else row['age'], axis=1)


# -------------------- fill missing fare with median by pclass -------------------- 
# -------------------- some people have a fare = 0, so only look at those > 0 -------------------- 
fare_by_pclass = df[df['fare'] > 0].groupby('pclass')['fare'].agg(np.median).to_dict()
df['fare'] = df.apply(lambda r: r['fare'] if r['fare'] > 0 
                      else fare_by_pclass.get(r['pclass']), axis=1)


# -------------------- fill missing embarked with most common port -------------------- 
most_common_port = df['embarked'].mode()[0]
df['embarked'] = df['embarked'].fillna(most_common_port)


# -------------------- transform categorical embarked column to 1/0 dummy columns -------------------- 
dummies = pd.get_dummies(df['embarked'], prefix='port')
df = pd.concat([df, dummies], axis=1)

df.iloc[0]


# -------------------- drop unnecessary columns -------------------- 
drop_cols = ['cabin', 'name', 'ticket', 'title', 'sex','embarked']

df = df.drop(drop_cols, axis=1)


######################
#exploratory analysis#
######################
# -------------------- survival rates across various columns -------------------- 
def plot_survive_rate(df, col, chart='barh'):
    survive_rates = df.groupby(col).agg({
        'survived': lambda x : sum(x) / len(x)
    })
    survive_rates.plot(kind=chart);

def bin_num(x, base=5):
    return int(x - (x % base))
    
# -------------------- get cleaned training data -------------------- 
dg = df[df['data'] == 'titanic_train'].copy()

# discretize age & fare
dg['age_bin'] = dg['age'].apply(lambda x: bin_num(x, 10))
dg['fare_bin'] = dg['fare'].apply(lambda x: bin_num(x, 25))

for col in ['pclass', 'gender', 'age_bin', 'fare_bin']:
    plot_survive_rate(dg, col)
#plt.show()

#######################################
#prepare training data for classifying#
####################################### 
# ----------------- store the columns (features) to be used for classifying survival ------------------
x_cols = df.columns.values[3:]

print '%d total features' % len(x_cols)
print x_cols

# ----------------- separating features and metric we're predicting) ----------------- 
df_train = df[df['data'] == 'titanic_train']

X = df_train[x_cols].as_matrix()
y = df_train['survived'].as_matrix()

print '%d rows, %d features' % (len(df_train), len(x_cols))


####################################################################
# cross validation to evaluate model                               #
# (dividing training data into n chunks and training model n times #
# with a different holdout chunk each time)                        #
####################################################################
cv = StratifiedKFold(y, n_folds=6)
tot_correct, tot_obs = 0, 0

for i, (train, test) in enumerate(cv):
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]

    model = RandomForestClassifier(random_state=321) 
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    correct, obs = (y_test == y_pred).sum(), len(y_test)
    tot_correct += correct
    tot_obs += obs

print 'accuracy: %f' % (tot_correct * 1.0 / tot_obs)


# -------------------- evaluate model over entire training set and look at each feature's importance -------------------- 
model = RandomForestClassifier(random_state=321,n_estimators = 100)
model.fit(X, y)

feature_rank = pd.Series(model.feature_importances_, index=x_cols).order(ascending=False)
print feature_rank


##############################################################################
# model parameter tuning                                                     #
# (there are several input parameters to a random forest model;              #
# grid search automates the process of tweaking these parameters to find the #
# optimal values for these parameters)                                       #
##############################################################################
param_grid = {
    "n_estimators": [100],
    "criterion": ["gini", "entropy"],
    'max_features': [0.5, 1.0, "sqrt"],
    'max_depth': [5, 6, 7, 8, 9, None],
}

model = RandomForestClassifier(random_state=321)
grid_search = GridSearchCV(model, param_grid, cv=12, verbose=0)
grid_search.fit(X, y)

print('best score',grid_search.best_score_)
print('best params',grid_search.best_params_)


# ------------- train model with best parameters from grid search ----------------- 
# ------------- and finally predict survival of people from test data ------------- 
df_train = df[df['data'] == 'titanic_train'].copy()

X_train = df_train[x_cols].as_matrix()
y_train = df_train['survived'].as_matrix()

model = RandomForestClassifier(
    n_estimators=100, 
    criterion=grid_search.best_params_['criterion'], 
    max_features=grid_search.best_params_['max_features'], 
    max_depth=grid_search.best_params_['max_depth'],
    random_state=321,
)

model.fit(X_train, y_train)

df_test = df[df['data'] == 'titanic_test'].copy()

X_test = df_test[x_cols].as_matrix()
y_pred = model.predict(X_test).astype(int)

df_test['survived'] = y_pred

final_df = df_test[['passengerid', 'survived']]
final_df.to_csv('predicted.csv', index=False)









