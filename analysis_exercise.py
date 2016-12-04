import os, itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold


def plot_cm(cm, classes, title="Confusion Matrix", cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    print('Confusion matrix')
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True')
    plt.xlabel('Predicted')

# Set up folders and filenames for both, applicants and hires
work_dir = os.getcwd()
data_dir = work_dir + '/data'

files = [(root, dirs, files) for root, dirs, files in os.walk(data_dir)].pop()[-1]
train = dict((f.split('.')[0].split('_')[-1], data_dir + '/' + f) for f in files)

# Read the data into data frames
apps = pd.read_csv(train['applicants'])
hires = pd.read_csv(train['hires'])
both = apps.merge(hires, how='inner', on='user_id')

### DATA EXPLORATION ###

# Find proportions of hires by client
hires.client.value_counts().plot(x=None, y=None, kind = 'pie')

# Find proportions of applicants by client
apps.client_name.value_counts().plot(x=None, y=None, kind = 'pie')

# Calculate applicant-to-hire ratio
apps_prop = apps.client_name.value_counts()
hires_prop = hires.client.value_counts()[apps_prop.index.tolist()]

apps_to_hire = pd.DataFrame(
    dict(applied=apps_prop,
         hired=hires_prop,
         ratio = apps_prop/hires_prop
         )
).sort_values(by='ratio', ascending=False)
print(apps_to_hire)

# Display applicant-to-hire ratio
apps_to_hire.plot(x = apps_to_hire.index, kind='bar', title="Applicant-to-hire ratio", secondary_y = 'ratio')

# Attrition within first 3 months
attr90 = hires.iloc[np.where(hires.tenure_length <= 90)]
attr90.client.value_counts().plot(kind='bar', title='Attrition within first 3 months')

### DATA ANALYSIS ###

# 6-month attrition analysis
attr180 = hires.iloc[np.where(hires.tenure_length <= 180)]
attr180_by_jobs = attr180.groupby('hire_job_category').count().user_id
hired_by_jobs = hires.groupby('hire_job_category').count().user_id[attr180_by_jobs.index.tolist()]
attr_by_jobs = pd.DataFrame(
    dict(
        hired = hired_by_jobs,
        attrition = attr180_by_jobs,
        rate = attr180_by_jobs / hired_by_jobs
    )
).sort_values(by='rate', ascending=False)
attr_by_jobs.plot(kind='bar', title='Attrition rates within 6 month', secondary_y='rate')

# The effect on the tenure's length by device type used to take the questionnaire
# Find columns with missing values
nulls = both.isnull().any()
nulls = nulls.loc[nulls == True].index.tolist()
nans = dict()
for c in nulls:
    nans[c] = round(both[c].isnull().value_counts()[True] / len(both), 4) * 100
nans = pd.Series(nans, index=nans.keys())
nans.plot(kind='barh', title='Missing values by columns, %')

# What the distribution of mean and median tenure's lenght by device type looks like
# replace nulls with 'missing' in device column
both.loc[both.device.isnull(), 'device'] = 'Missing'
device_counts = both.device.value_counts()
device_counts.plot(kind='bar', title="Distribution of device types, count", xlim=(0, 200))

tenure_by_device = both.loc[both.device.notnull(),:].groupby('device').\
    agg({'tenure_length': {'tenure_length_mean': np.mean, 'tenure_length_median': np.median}})

# Drop device columns due to a high number of missing values
both = both.drop('device', axis=1)

# Remove rows with missing tenure_length from the data set
no_tenure = both.loc[both.tenure_length.isnull()]
both = both.loc[both.tenure_length.notnull()]

### Predictive modeling ###
# Create a brand new target column labeling it with accordance to the requirements:
# 1: tenure_lenght < 6 months
# 2: 6 months <= tenure_length <= 12 months
# 3: tenure_length > 12 months
both.loc[both.tenure_length < 180, 'target'] = 1
both.loc[(both.tenure_length >= 180) & (both.tenure_length <= 360), 'target'] = 2
both.loc[both.tenure_length > 360, 'target'] = 3
both.target = both.target.astype(int)

# Get the number of rows with missing values across all the columns
nulls = both.isnull().any()
nulls = nulls.loc[nulls == True].index.tolist()
nans = dict()
for c in nulls:
    nans[c] = both[c].isnull().value_counts()[True]
nans = pd.Series(nans, index=nans.keys())
nans.plot(kind='barh', title='Missing values by columns, count')

# From the graph above it is clear that we can remove rows with missing values as their number is very small
# also we need to cast a few columns into category for further encoding
both = both.dropna()
both.client_name = both.client.astype('category')

# Create train set and the target
y = pd.Series(both.target, index=both.index)
train_features = apps.columns.tolist()[1:-1]
df = pd.DataFrame(both[train_features], index=both.index)
dummies = pd.DataFrame(pd.get_dummies(df.client_name), index=df.index)
X = pd.concat([df.drop('client_name', axis=1), dummies], axis=1)

# Abs to fix answer3 and answer4 negative values
X = X.apply(lambda c: abs(c))

# Fit model using Random Forest
cv = StratifiedKFold(n_splits=5)
rfc = ensemble.RandomForestClassifier()

# Iterrate over a cross-validated set, printing the accuracy score and
# classification report, including precision and recall
iter=0
color_maps = ['Purples', 'Blues', 'Greens', 'Greys', 'Oranges']
for (train, test), color in zip(cv.split(X, y), color_maps):
    X_x, y_y = X.loc[train].dropna(), y.loc[train].dropna()
    iter += 1
    y_score = rfc.fit(X_x, y_y).score(X_x, y_y)
    y_pred = rfc.predict(X_x)
    print("Iteration: {}\tAccuracy score {}\nClassification report:\n".format(iter, round(y_score, 4)))
    print(classification_report(y_y, y_pred))
    plt.figure()
    plot_cm(confusion_matrix(y_y, y_pred), y_y.unique(), title="Iteration: {}".format(iter), cmap=color)
plt.show()
