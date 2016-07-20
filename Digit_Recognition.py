import pandas, sklearn
from sklearn.ensemble          import RandomForestClassifier
from sklearn.naive_bayes       import MultinomialNB


##############
#read in data#
##############
train = pandas.read_csv('train.csv')
test  = pandas.read_csv('test.csv' )
# ------------- check missing value -------------
for col in list(train.columns.values):
    if len(train[col][train[col].isnull()]) > 0:
        missing_percent = len(train[col][train[col].isnull()])/float(train.shape[0])
        print col, missing_percent
predictors = range(1,train.shape[1])
labels     = 0        
      
     
        
####################
#  random forests  #
####################
# -------------- predict on whole data --------------
alg = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)
alg.fit(train.iloc[:,predictors], train.iloc[:,labels])
train_pred = alg.predict(train.iloc[:,predictors])
diff = []
for index in range(len(train_pred)):
    diff.append(int(train_pred[index]!=train.iloc[:,labels][index])) 
print('random forests', 1 - float(sum(diff))/len(diff))


###########################
# predict on the test data#
###########################
# ---------------- random forests ----------------
alg = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)
alg.fit(train.iloc[:,predictors], train.iloc[:,labels])
predictions_rf = alg.predict(test)


####################
# submit to kaggle #
####################
submission = pandas.DataFrame({
        "ImageId": range(1,len(predictions_rf)+1),
        "Label": predictions_rf
    })

submission.to_csv("kaggle.csv", index=False)
