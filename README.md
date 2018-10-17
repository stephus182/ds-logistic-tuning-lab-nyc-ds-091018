

```python
import pandas as pd
import numpy as np
import itertools

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE, ADASYN

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

# Predicting Credit Card Fraud
Load the **creditcard.csv** file, split into training and test sets, and fit a logistic regression model to the training data.  
Then plot the ROC curve and confusion matrix for your test sets.


```python
# here we load a compressed csv file.
df = None
# inspect the first few lines
df.head()
```

Count the number of instances in each class


```python
# your code here
```

Seperate the class column (`y`) from the rest of the data set (`X`) and use `train_test_split()` to create a train and a test set.


```python
X = df[df.columns[:-1]]
y = df.Class
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

Use `scikit-learn`s `LogisticRegression()` and get the true positive rate, false positive rate and thresholds using `roc_curve()`.


```python
logreg = None
y_score = None
# get tpr, fpr and thresholds
```

Create an ROC plot using seaborn.


```python
# Create seaborn plot here
```


```python
Plot a confusion matrix here.
```


```python
#Create a function for a confusion matrix here. Make sure to add a normalization option
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    None
```

make `y_hat_test` predictions and create the confusion matrix using `confusion_matrix`. Then use your newly created function.


```python
y_hat_test = None
cnf_matrix = None
# use new plot_confusion_matrix() function
```

# Tuning 
Try some of the various techniques proposed to tune your model. Compare your models using AUC, ROC or another metric. Use different values for normalization weights first and visualize the results.


```python
# Now let's compare a few different regularization performances on the dataset:
```


```python
# plot the result
```

### SMOTE
Repeat what you did before but now using the SMOTE class from the imblearn package in order to improve the model's performance on the minority class.


```python
print(y_train.value_counts()) #Previous original class distribution
# Resample X_train and y_train here
print(pd.Series(y_train_resampled).value_counts()) #Preview synthetic sample class distribution
```


```python
# Now let's compare a few different regularization performances on the dataset using SMOTE
```


```python
# plot the result
```

## Analysis
Describe what is misleading about the AUC score and ROC curves produced by this code.
