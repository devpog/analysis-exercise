## Data analysis exercise by Kyrylo Pogrebenko

### 1. Data Exploration
I start my analysis with reading csv-files provide into separate data frames, **apps** and *hires*, which correspond to 
**data_exercise_applicants.csv** and **data_exercise_hires.csv**. In order to do so I create a dictionary to hold data 
frame names and files associated with them:
```python
work_dir = os.getcwd()
data_dir = work_dir + '/data'

files = [(root, dirs, files) for root, dirs, files in os.walk(data_dir)].pop()[-1]
train = dict((f.split('.')[0].split('_')[-1], data_dir + '/' + f) for f in files)

# Read the data into data frames
apps = pd.read_csv(train['applicants'])
hires = pd.read_csv(train['hires'])
both = apps.merge(hires, how='inner', on='user_id')
```
Next, I also merge both data frames using **'user_id'** field.

#### Q1. Which organizations have the greatest applicant-to-hire ratio?
To calculate applicant-to-hire ratio I grouped the number of hires and applicants by client name and get the ratio of
these two:

```python
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
```
As a result I receive a bar chart, with numbers for applied, hired, and the ratio itself, separated into secondary axis 
From thise graph it is clearly visible that **'client11' have the greatest applicant-to-hire ratio**.

#### Q2. Which organizations have the greatest challenge with attrition in the first 3 months of employment?
To find organizations experienced the gratest challenge with attrition within first 3 months, I filtered out everything
else but tenure's length under 90 days and get their counts by client_name, using the following code:
```python
# Attrition within first 3 months
attr90 = hires.iloc[np.where(hires.tenure_length <= 90)]
attr90.client.value_counts().plot(kind='bar', title='Attrition within first 3 months')
```
As a result a bar graph appears where we can see that the most problematic are 'client_5', 'client_11', and 'client2'

### 1. Data Analysis
#### Q1.Do 6 month attrition rates vary significantly across different job categories?
To answer this question, I filtered out the data set leaving out everything, but tenure's length within a 180-day period.
Then I grouped the resulting set by ***hire_job_category***, aggregating it to get counts of each distinct job category,
getting 6-months attrition. To get the ratio I perform similar operations but calculating the actual number of people
which had been hired. See the following code:
```python
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
```
As a result, I display a plot of hires, attrition, and the actual ratio separated into a secondary axes due to a scaling
issue.

#### Q2. How does the type of device used to take the questionnaire impact an applicant’s tenure length, if at all?
The very first issue that caught my eye was a very large number of missing values for **'device'*** column, nearly **50**.
To make sure that my assumption of completely dropping this feature, I assigned a value 'missing' to all null's to see 
the distribution of different device types. Not only, a missing category was large, but **'other'** as well! To convince
myself that I'm on the right track, I calculated mean and median values for the tenure's length by device type and discovered
that some of them were extremely skewed, which didn't came as a suprise, taking into account that the majority of observations
fall into either **'missing'** or **'other'**. Thus, I killed this attriobute completely.
```python
# The effect on the tenure's length by device type used to take the questionnaire
# Find columns with missing values
nulls = both.isnull().any()
nulls = nulls.loc[nulls == True].index.tolist()
nans = dict()
for c in nulls:
    nans[c] = round(both[c].isnull().value_counts()[True] / len(both), 4) * 100
nans = pd.Series(nans, index=nans.keys())
nans.plot(kind='barh', title='Missing values by columns, %')

# How the distribution of mean and median tenure's length by device looks like?
tenure_by_device = both.loc[both.device.notnull(),:].groupby('device').\
    agg({'tenure_length': {'tenure_length_mean': np.mean, 'tenure_length_median': np.median}})
tenure_by_device.plot(kind='bar', title="Mean and median tenure's length by device, days", xlim=(0, 1000))
```

### 3. Predictive modeling
#### Step 1. Building a model
In order to build an effective model, I have done the following:
- **created a brand new column target assigning categories**:
    1. tenure_lenght < 6 months
    2. 6 months <= tenure_length <= 12 months
    3. tenure_length > 12 months
```python
both.loc[both.tenure_length < 180, 'target'] = 1
both.loc[(both.tenure_length >= 180) & (both.tenure_length <= 360), 'target'] = 2
both.loc[both.tenure_length > 360, 'target'] = 3
both.target = both.target.astype(int)
```
- **identified missing values across all columns to see how large are they, if any**:
```python
# Get the number of rows with missing values across all the columns
nulls = both.isnull().any()
nulls = nulls.loc[nulls == True].index.tolist()
nans = dict()
for c in nulls:
    nans[c] = both[c].isnull().value_counts()[True]
nans = pd.Series(nans, index=nans.keys())
nans.plot(kind='barh', title='Missing values by columns, count')
```
Turns out **there were not a lot of missing values** and it was pretty safe to remove those rows, without worrying of
loosing predictive ability:
```python
# From the graph above it is clear that we can remove rows with missing values as their number is very small
# also we need to cast a few columns into category for further encoding
both = both.dropna()
both.client_name = both.client.astype('category')
```
- **created dummy variables for each category of 'client_name' since it is rather nominal then ordinal feature**:
```python
# Create train set and the target
y = pd.Series(both.target, index=both.index)
train_features = apps.columns.tolist()[1:-1]
df = pd.DataFrame(both[train_features], index=both.index)
dummies = pd.DataFrame(pd.get_dummies(df.client_name), index=df.index)
X = pd.concat([df.drop('client_name', axis=1), dummies], axis=1)

# Abs to fix answer3 and answer4 negative values
X = X.apply(lambda c: abs(c))
```
- build a model using **Random Forest** algorithm due to its high interpretability. Which is important for this assignment.
For example, this algorithm allows us to say that when if an applicant's responses to 'answer1', 'answer5', and 'asnwer12'
were '1', '3', '4', respectively, and he/she also responded '1', '1', and '3' to 'answer7', 'answer14', 'answer23',
then we can easily say that his/her tenure's lenght won't exceed 90 days and probably it is better to find a batter fit 
for this particular opening, etc.:
```python
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
```
- From a code snippet above, you can see that I used 5-fold cross validation when trained the model to create pairs of 
train/test sets to account for variation. After each itteration I show the accuracy score, which averaged to ~95%. 
I aslo display additional **metrics, such as precision, recall, f1-score**, and **support** which also allows for a
better evaluation of the model.
- The final step is output of **the confusion matrix**, a contingency table. This matrix allows to calculate **sensitivity,
 specificity, precision**, and **recall**. I use the following function to display the matrix:
```python
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
```

### 4. Conclusion
Assessing the above analysis I can confidently say that this model is able to classify unseen result with sufficiently high 
level of accuracy, providing its simplicity of interpretation. On average its accuracy score is 95% which which leaves Type-I
error within industry allowable threshold.
