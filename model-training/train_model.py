# Importa las bibliotecas necesarias
import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Configura la URI de la base de datos y la dirección del servidor de MLflow
# mlflow.set_tracking_uri("http://mlflow_container:80")
# mlflow.set_experiment('Entrenamiento prueba data iris')

# # Carga los datos de iris
# iris = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# # Entrenamiento del modelo
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Realiza predicciones en el conjunto de prueba
# y_pred = model.predict(X_test)

# # Calcula la precisión del modelo
# accuracy = accuracy_score(y_test, y_pred)

# # Log en MLflow
# with mlflow.start_run():
#     mlflow.log_param("n_estimators", 100)
#     mlflow.log_metric("accuracy", accuracy)
#     mlflow.sklearn.log_model(model, "model")


import boto3 # required in case we store the artifacts on s3
import category_encoders as ce
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import mlflow
import os
import warnings

from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# ## Import dataset

# load dataset
data = 'car_evaluation.csv'
df = pd.read_csv(data, header=None)

# view dimensions of dataset
df.shape

# There are 1728 instances and 7 variables in the data set.

# preview the top of the dataset
df.head()

# rename column names
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names
col_names

# preview the end of the dataset
df.tail()

# We can see that the column names are renamed. Now, the columns have meaningful names with "**class**" as the target variable.

# ## Inspect dataset

df.info()

for col in col_names:    
    print(df[col].value_counts())
    print ("\n")

# ## Declare feature vector and target variable
#

X = df.drop(['class'], axis=1)
y = df['class']

# ## Split data into separate training and test set

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
X_train.shape, X_test.shape

# ## Feature Engineering
#
# **Feature Engineering** is the process of transforming raw data into useful features that help us to understand our model better and increase its predictive power.

# encode variables with ordinal encoding
encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

X_train.head()

X_test.head()

# ## Decision Tree Classifier with criterion gini index

# create classifier
md = 3
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth = md, random_state=0)

# train classifier and predict results
clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)
acc = accuracy_score(y_test, y_pred_gini)

# + tags=[]
# decision-tree visualization
plt.figure(figsize=(12,8))
tree.plot_tree(clf_gini.fit(X_train, y_train))
plt.savefig("tree.jpg")
# -

# register the classifier
mlflow.set_tracking_uri("http://mlflow_container:80")
mlflow.set_experiment('TreeClassifier')

with mlflow.start_run(run_name='blade_runner'):
    mlflow.log_param("max_depth", md)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(sk_model=clf_gini, artifact_path='', registered_model_name='tree_model')
    mlflow.log_artifact("tree.jpg", artifact_path='plots')
# -

# compare accuracy vs. training prediction
print('Model accuracy score with criterion gini index: {0:0.4f}'. format(acc))
y_pred_train_gini = clf_gini.predict(X_train)
print('Training-set accuracy score: {0:0.4f}'. format(acc))

# Here, the training-set accuracy score is 0.7865 while the test-set accuracy to be 0.8021. These two values are quite comparable. So, there is no sign of overfitting. 

# ## Decision Tree Classifier with criterion entropy

# train classifier
clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
clf_en.fit(X_train, y_train)

y_pred_en = clf_en.predict(X_test)
print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test, y_pred_en)))

# We can see that the training-set score and test-set score is same as above. The training-set accuracy score is 0.7865 while the test-set accuracy to be 0.8021. These two values are quite comparable. So, there is no sign of overfitting. 
#

plt.figure(figsize=(12,8))
tree.plot_tree(clf_en.fit(X_train, y_train))


# print the Confusion Matrix and slice it into four pieces
cm = confusion_matrix(y_test, y_pred_en)
print('Confusion matrix\n\n', cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf_en.classes_)
disp.plot()
#plt.show()

# ## Classification report
#
# **Classification report** is another way to evaluate the classification model performance. It displays the  **precision**, **recall**, **f1** and **support** scores for the model. I have described these terms in later.
#
# We can print a classification report as follows:

print(classification_report(y_test, y_pred_en))

