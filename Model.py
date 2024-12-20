from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.ensemble import IsolationForest
import seaborn as sns
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import os


# label encoding codes (1 is normal 0 is malicious)
current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")

testdata = pd.read_csv("KDDTest+.txt")
traindata = pd.read_csv("KDDTrain+.txt")
#Initializing Testdata and Traindata


columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted'
,'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate'
,'srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','attack','level'])

features = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted'
,'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate'
,'srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','level'])
#list of columns

traindata.columns = columns
testdata.columns = columns
#initializing a columns array to columns in the dataset

#DATA CLEANING and Preprocessing

traindata = traindata.drop_duplicates()

#dropping duplicates in training data set

# print(traindata.isnull().sum())
# print(traindata.isnull().values.any())
#Checking for Null values across data set there are 0

# print(traindata.dtypes)
#checking for datatypes, object data type exists and must be encoded

traindata.loc[traindata['attack'] != 'normal', 'attack'] = 'malicious'
testdata.loc[testdata['attack'] != 'normal', 'attack'] = 'malicious'
#Converts all that is not normal to malicious in both test and training data sets

object_datatypes = list(traindata[columns].select_dtypes('object').columns.values)
#Creating a list of object datatypes for preprocessing/label encoding

for col in object_datatypes:
    le = LabelEncoder()
    traindata[col] = le.fit_transform(traindata[col].values)
    testdata[col] = le.transform(testdata[col].values)
#   Label encoding all object datatypes across datasets.


#Outlier detection and checking box plots:
# plt.figure(figsize=(20, 40))
# traindata.plot(kind='box', subplots=True, layout=(8, 6), figsize=(20, 40))
# plt.show()

# traindata.hist(bins=43,figsize=(20,30))
# plt.show()


#ensuring correct mapping

X = traindata[features]
y = traindata['attack']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#Creating a train-test split


#Feature Selection
train_index = X_train.columns
mutual_info = mutual_info_classif(X_train, y_train)
mutual_info = pd.Series(mutual_info)
mutual_info.index = train_index
mutual_info = mutual_info.sort_values(ascending=False)

# print(mutual_info)
# mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 5));
# plt.show()
#checking importance of columns to attack column (visual and numerical representation)

selectFeatures = SelectKBest(mutual_info_classif, k=27)
selectFeatures.fit(X_train, y_train)
selected_features = train_index[selectFeatures.get_support()]

X_train = selectFeatures.transform(X_train)
X_test = selectFeatures.transform(X_test)

#selecting top 27 features as rest barely appear on the bar graph plot

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#scaling the values down

#RANDOM FOREST CLASSIFIER Model
# model_RFC = RandomForestClassifier()
#
# model_RFC.fit(X_train, y_train)
#
# y_pred_RFC = model_RFC.predict(X_test)

# Logistic Regression Model
# model_LR = LogisticRegression(max_iter=500)
#
# model_LR.fit(X_train, y_train)
#
# y_pred_LR = model_LR.predict(X_test)

#Gradient Boosting Classifier Model

# model_GBC = GradientBoostingClassifier()
#
# model_GBC.fit(X_train, y_train)
#
# y_pred_GBC = model_GBC.predict(X_test)

#XGB Classifier Model Found to return the best RocAuc Score

model_XGB = XGBClassifier(eval_metric='logloss')

model_XGB.fit(X_train, y_train)

y_pred_XGB = model_XGB.predict(X_test)

# y_pred_proba_RFC = model_RFC.predict_proba(X_test)[:, 1]
# y_pred_proba_LR = model_LR.predict_proba(X_test)[:, 1]
# y_pred_proba_GBC = model_GBC.predict_proba(X_test)[:, 1]
# y_pred_proba_XGB = model_XGB.predict_proba(X_test)[:, 1]

# roc_auc_RFC = roc_auc_score(y_test, y_pred_proba_RFC)
# roc_auc_LR = roc_auc_score(y_test, y_pred_proba_LR)
# roc_auc_GBC = roc_auc_score(y_test, y_pred_proba_GBC)
# roc_auc_XGB = roc_auc_score(y_test, y_pred_proba_XGB)

# print(f"ROC AUC RFC Score: {roc_auc_RFC}")
# print(f"ROC AUC LR Score: {roc_auc_LR}")
# print(f"ROC AUC GBC Score: {roc_auc_GBC}")
# print(f"ROC AUC XGB Score: {roc_auc_XGB}")

# Training predictions (Testing Overfitting, no large difference between RocAucs of X-train and X-test)
# y_pred_proba_train = model_XGB.predict_proba(X_train)[:, 1]
# roc_auc_train = roc_auc_score(y_train, y_pred_proba_train)

# Test predictions
# y_pred_proba_test = model_XGB.predict_proba(X_test)[:, 1]
# roc_auc_test = roc_auc_score(y_test, y_pred_proba_test)
#
# print(f"Training ROC AUC: {roc_auc_train}")
# print(f"Test ROC AUC: {roc_auc_test}")

#Hyper Parameter Tuning Using Bayesian Optimization

def optimizer(trial):
    param = {
        "objective": "binary:logistic",
        "n_estimators": 1000,
        "verbosity": 0,
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),

    }

    # Training model
    model_XGB = XGBClassifier(**param, random_state=1)
    model_XGB.fit(X_train, y_train)

    y_pred_proba_XGB = model_XGB.predict_proba(X_test)[:, 1]
    roc_auc_XGB = roc_auc_score(y_test, y_pred_proba_XGB)

    return roc_auc_XGB

study = optuna.create_study(direction='maximize')

study.optimize(optimizer, n_trials=50)

best_trial = study.best_trial

print(f"Best ROC AUC Score: {best_trial.value}")

print("Best parameters found:", best_trial.params)

best_params = best_trial.params

#   Returns best parameters based on the train-test split


X_testdata = testdata[features]
y_testdata = testdata['attack']

X_testdata = selectFeatures.transform(X_testdata)

X_testdata = scaler.transform(X_testdata)

#   Preparing Testdata set for Model prediction

optimized_XGB = XGBClassifier(**best_params, random_state=1)
optimized_XGB.fit(X_train, y_train)

#   initializing Optimal Model based on train-test split

y_pred_trainsplit = optimized_XGB.predict(X_test)
y_pred_proba_trainsplit = optimized_XGB.predict_proba(X_test)[:, 1]
#   creating predictions for the train-test split

y_pred_testdata = optimized_XGB.predict(X_testdata)
y_pred_proba_testdata = optimized_XGB.predict_proba(X_testdata)[:, 1]
#   creating predictions for the unseen test data

# Calculate and print the final ROC AUC score for this model including train-test split results and real results on an unseen dataset
roc_auc_XGB_trainsplit = roc_auc_score(y_test, y_pred_proba_trainsplit)
print(f"\n Final ROC AUC Score (Train-Test Split): {roc_auc_XGB_trainsplit}\n")

accuracy_XGB_trainsplit = accuracy_score(y_test, y_pred_trainsplit)
print(f"XGBoost Accuracy (Train-Test Split): {accuracy_XGB_trainsplit * 100:.2f}% \n")

roc_auc_XGB_testdata = roc_auc_score(y_testdata, y_pred_proba_testdata)
print(f"Final ROC AUC Score (Test Data): {roc_auc_XGB_testdata}\n")

accuracy_XGB_testdata = accuracy_score(y_testdata, y_pred_testdata)
print(f"XGBoost Accuracy (Test Data): {accuracy_XGB_testdata * 100:.2f}%\n")