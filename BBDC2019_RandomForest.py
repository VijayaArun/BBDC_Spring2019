#!/usr/bin/env python
# coding: utf-8

# In[29]:


# load a bunch of random py libraries
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


# In[10]:


# Set individual file names
mainFolder = 'E:/DE2020Spring/DataMining/BBDC/bbdc_2019_Bewegungsdaten_mit_referenz/'
trainFile = 'train.csv'
testFile = 'challenge.csv'
new_complete_set = 'C:/Users/SKP/Downloads/BBDC2019_Solution/std_feature_df.csv'
complete_set = 'C:/Users/SKP/Downloads/BBDC2019_Solution/new_feature_df.csv'
interm_complete_set = 'features_dataset_with_test_set_iqr_corr_skew.csv'
original_complete_set = 'C:/Users/SKP/Downloads/BBDC2019_Solution/std_feature_test_df.csv'
entropy_df = 'entropy_df'


# In[11]:


#initialize unlabelled df
complete_df = pd.read_csv(new_complete_set)
#complete_df = complete_df[complete_df['Label'] != 'lay']
#complete_df.drop(columns = ['Subject', 'Datafile', 'Label'], inplace=True)

#drop duplicate rows
complete_df.drop_duplicates(subset ='id', inplace = True) 
#show column names
print(list(complete_df.columns))


# In[12]:


complete_df.shape[0]


# In[13]:


complete_df.replace(np.NaN, 0.0, inplace=True)
complete_df.replace(np.inf, 4.0, inplace=True)

#show whether I missed any
complete_df[complete_df.isin([np.nan, np.inf, -np.inf]).any(1)]


# In[14]:


#traintest_df = final_principal_df
traintest_df = complete_df

# initialize label dataframe
train_labels = pd.read_csv(mainFolder + trainFile)

# merge with training labels to create train_df
train_df = pd.concat([traintest_df, train_labels], axis=1)
train_df = train_df[train_df['Label'] != 'lay']
train_df.drop(columns = ['Subject', 'Datafile', 'id'], inplace = True)

#initialize dataframe with test observations
test_subjects = pd.read_csv(mainFolder + testFile)

#create test_df by merging with testFile
test_df = pd.concat([test_subjects, traintest_df], axis=1)
test_df.drop(columns = ['Label', 'id'], inplace = True)

print(test_df.shape)
print(train_df.shape)


# In[15]:


#check whether lay snuck through
train_df.Label.unique()


# In[16]:


#create new modelling df
model_df = train_df

#factorize label
factor = pd.factorize(train_df['Label'])
model_df.Label = factor[0]

print(factor[0])

#store original labels
definitions = factor[1]
print(factor[1])

# print(model_df.Label.head())
# print(definitions)


# In[17]:


factor[1]


# In[18]:


# insert random packages here
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


# In[84]:


# initializing X and Y array
X_train = model_df.iloc[:,0:-1].values
y_train = model_df.iloc[:,-1].values

#specify classifier
classifier = RandomForestClassifier(n_estimators = 110, random_state = 7, oob_score = True)

# get the cross-validation accuracy score
scores = cross_val_score(classifier, X_train, y_train, cv = 5)


# In[85]:


classifier.fit(X_train, y_train)

#print(classifier.feature_importances_)

#get model score based on full training data
classifier.score(X_train, y_train)


# In[86]:


print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[21]:


# try grid search for number of trees
rf = RandomForestClassifier(n_estimators = 100, random_state = 7)

parameters = {'random_state': list(range(100))}

clf = GridSearchCV(rf, param_grid = parameters, cv = 5)
clf.fit(X_train, y_train)


# In[22]:


score_dict = clf.cv_results_

dict_df = pd.DataFrame(score_dict)
dict_df.to_csv('gridsearch_dict_df.csv', index=False)
dict_df = pd.read_csv('gridsearch_dict_df.csv')
dict_df.head()


# In[23]:


np.argsort(list(dict_df['rank_test_score']))[:20]


# In[24]:



#specify hyperparameters to stay the same during model testing/validation phase and when generating actual predictions

#the list of random states which the classifier uses to generate the Random Forests
#generic:
random_state_range = range(50, 70)
#take top 20(or whatever the last number is) random states from gridsearch
random_state_range = np.argsort(list(dict_df['rank_test_score']))[:20]

# the number of trees in each Random Forest
n_trees = 100


# In[25]:


#train-test split the train set
X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train, 
                                                                            test_size = 0.6, random_state = 2)
#initialize probability array
prob = np.empty((len(X_train_test), 22))

#loop to run Random Forest in different random states and add the probabilities obtained from each to prob
for random_state in random_state_range:
    classifier = RandomForestClassifier(n_estimators = n_trees, random_state = random_state)
    classifier.fit(X_train_train, y_train_train)
    prob += classifier.predict_proba(X_train_test)

#argmax function to choose the label with highest cumulative probability
cum_pred = np.argmax(prob, axis = 1)

#find predicition accuracy by comparing with test set labels
print('Prediction Accuracy=%f' % (sum(1 for i in range(len(cum_pred)) 
                             if cum_pred[i] == y_train_test[i]) / float(len(cum_pred))))


# In[124]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap='Reds'):

    size = (12, 10)
    fig, ax = plt.subplots(figsize=size)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Compute confusion matrix
class_names = set(train_labels['Label'])
cnf_matrix = confusion_matrix(cum_pred, y_train_test)
# cnf_matrix = (normed_c.T / normed_c.astype(np.float).sum(axis=1)).T
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, class_names,title='Confusion matrix, without normalization')


# In[118]:


from sklearn.metrics import classification_report
print(classification_report(y_train_test, cum_pred))


# In[130]:


# import seaborn as sns

# sns.heatmap(cnf_matrix, annot=True, fmt='.2f', xticklabels=class_names, yticklabels=class_names)


# In[33]:


#feature importance for last looped classifier:
feature_importances = pd.DataFrame(classifier.feature_importances_,
                                   index = complete_df.columns[1:],
                                    columns=['importance']).sort_values('importance', ascending=False)
feature_importances.head(10)


# In[36]:


#name submission file
featureImportanceFileName = 'feature_importance'

feature_importances.to_csv(featureImportanceFileName + '.csv')


# In[21]:


from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score


# In[22]:


# initialize and score SVC
svc = LinearSVC(penalty = 'l2', C = 0.01, dual = False, multi_class = 'crammer_singer', random_state = 69, max_iter = 1000)

# get cross val score
cross_val_score(svc, X_train, y_train, cv = 5)


# In[23]:


from sklearn.feature_selection import SelectFromModel

# get new X by eliminating variables
lsvc = svc.fit(X_train, y_train)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X_train)
print(X_new.shape)

# get cross val score with X_new
cross_val_score(svc, X_new, y_train, cv = 5)


# In[24]:


#specify hyperparameters to stay the same during model testing/validation phase and when generating actual predictions

#the list of random states which the classifier uses to generate the Random Forests
# generic:
random_state_range = range(20, 40)
#take top 20(or whatever the last number is) random states from gridsearch
random_state_range = np.argsort(list(dict_df['rank_test_score']))[:20]

# the number of trees in each Random Forest
n_trees = 100


# In[25]:


test_df.head()


# In[26]:


#initialize train array (go agane)
X_train = model_df.iloc[:,0:-1].values
y_train = model_df.iloc[:,-1].values

#initialize test array
X_test = test_df.iloc[:,2:].values

#same procedure as in testing phase above
pred = np.empty((len(X_test), 22))
for random_state in random_state_range:
    classifier = RandomForestClassifier(n_estimators = n_trees, random_state = random_state)
    classifier.fit(X_train, y_train)
    pred += classifier.predict_proba(X_test)

#generate predicted labels
cum_pred = np.argmax(pred, axis = 1)


# In[68]:


# Convert the predictions into labels
reversefactor = dict(zip(range(len(definitions)), definitions))
# print(reversefactor)
pred_label = np.vectorize(reversefactor.get)(cum_pred)

new_series = pd.Series(pred_label)

new_series
#assign the labels to the test_subjects file
test_subjects['Label'] = new_series
test_subjects.head()


# In[76]:


pavg_rf_pred = pred.copy()
pavg_rf_pred = pavg_rf_pred / 20


# In[83]:


pavg_rf_norm_pred_df = pd.DataFrame(pavg_rf_pred, columns = factor[1])
pavg_rf_norm_pred_df.head()


# In[84]:


pavg_rf_pred_df = pd.DataFrame(pavg_rf_pred, columns = factor[1])
pavg_rf_pred_df.head()


# In[85]:


final_avg = pavg_rf_norm_pred_df + pavg_rf_pred_df
final_avg.head()


# In[86]:


final_avg.idxmax(axis = 1)
#assign the labels to the test_subjects file
test_subjects['Label'] = final_avg.idxmax(axis = 1)
test_subjects.head()

