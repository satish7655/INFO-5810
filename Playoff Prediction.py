# Playoff Prediction Based on  LogisticRegression()),RandomForestClassifier()),& KNN )

# Importing necessary modules
import pandas as pd
import pylab as pl
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix, plot_roc_curve, \
    plot_confusion_matrix
import numpy as np

# Importing the data Read the Csv files
NBA_2021_df = pd.read_csv('C:/Users/I24253.VERISK/OneDrive - Verisk Analytics/Desktop/INFO/Masters/NBA_2000_2021.csv')
NBA_2022_df = pd.read_csv('C:/Users/I24253.VERISK/OneDrive - Verisk Analytics/Desktop/INFO/Masters/NBA_2022_ED.csv')

# Load the data and check the values for each dataset
print("\n", NBA_2021_df.head(10))
print("\n", NBA_2022_df.head(10))
print(NBA_2021_df.columns)

# drop few unwanted columns

# Cleaning Data
NBA_2021_df = NBA_2021_df.drop(
    columns=['RK', 'FG', 'FGA', '3PA', 'FGA', 'FT', 'FTA', '2P', 'MP_scalnorm', 'FG_scalnorm',
             'FGA_scalnorm',
             'FG%_scalnorm', '3P_scalnorm', '3PA_scalnorm', '3P%_scalnorm',
             '2P_scalnorm', '2PA_scalnorm', '2P%_scalnorm', 'FT_scalnorm',
             'FTA_scalnorm', 'FT%_scalnorm', 'ORB_scalnorm', 'DRB_scalnorm',
             'AST_scalnorm', 'STL_scalnorm', 'BLK_scalnorm', 'TOV_scalnorm',
             'PF_scalnorm', 'PTS_scalnorm', 'W%_scalnorm', ])
NBA_2022_df = NBA_2022_df.drop(
    columns=['RK', 'FG', 'FGA', '3PA', 'FGA', 'FT', 'FTA', '2P', 'MP_scalnorm', 'FG_scalnorm',
             'FGA_scalnorm',
             'FG%_scalnorm', '3P_scalnorm', '3PA_scalnorm', '3P%_scalnorm',
             '2P_scalnorm', '2PA_scalnorm', '2P%_scalnorm', 'FT_scalnorm',
             'FTA_scalnorm', 'FT%_scalnorm', 'ORB_scalnorm', 'DRB_scalnorm',
             'AST_scalnorm', 'STL_scalnorm', 'BLK_scalnorm', 'TOV_scalnorm',
             'PF_scalnorm', 'PTS_scalnorm', 'W%_scalnorm', ])

print("Value Counts\n:", NBA_2021_df.value_counts(dropna=False))

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print("Printing Summary\n:", NBA_2021_df.info(10))

print("Describe the dataset:\n", NBA_2021_df.describe())

# Displaying Relationship between filtered features
NBA_2021_df.dropna(axis=0, inplace=True)
corr = NBA_2021_df.corr().abs()
corrMatrix = corr.loc[corr['W'] > .25]
print(corrMatrix.index)
variables = list(corr.index)
# EDA
plt.figure(figsize=(25, 25))
sn.heatmap(corrMatrix, cmap='Purples', annot=True, linecolor='Red', linewidths=1.0)
plt.show()

ax = sn.regplot(x=NBA_2021_df['PTS'], y=NBA_2021_df['W'], label='PTS')
sn.regplot(x=NBA_2021_df['TOV'], y=NBA_2021_df['W'], label='TOV')
sn.regplot(x=NBA_2021_df['STL'], y=NBA_2021_df['W'], label='STL')
sn.regplot(x=NBA_2021_df['FG%'], y=NBA_2021_df['W'], label='FG%')
sn.regplot(x=NBA_2021_df['DRB'], y=NBA_2021_df['W'], label='DRB')
sn.regplot(x=NBA_2021_df['DRB'], y=NBA_2021_df['W'], label='DRB')
ax.set(xlabel="League Standing in various performance indicators")
plt.legend()
plt.title("Correlation of various variables and Team Wins")
plt.show()

# relationship between different variables
plt.title("Relationship between BLK and DRBs")
aax = sn.regplot(x=NBA_2021_df['DRB'], y=NBA_2021_df['BLK'])
ax.set(xlabel="Defensive rebound", ylabel="Blocks per game")
plt.show()

Team_Stats_over_years = NBA_2021_df.loc[NBA_2021_df['W'] > 16]

plt.figure(figsize=(5, 5))
plt.title("League performances over the years")
ax = sn.lineplot(x=Team_Stats_over_years.Year, y=Team_Stats_over_years['3P'], label="3P")
sn.lineplot(x=Team_Stats_over_years.Year, y=Team_Stats_over_years['TOV'], label="TOV")
sn.lineplot(x=Team_Stats_over_years.Year, y=Team_Stats_over_years['DRB'], label="DRB")
sn.lineplot(x=Team_Stats_over_years.Year, y=Team_Stats_over_years['AST'], label="AST")
sn.lineplot(x=Team_Stats_over_years.Year, y=Team_Stats_over_years['BLK'], label="BLK")

pl.ylim(0, 30)
ax.set(ylabel="Season League Ranking in various performance factors over the year")
plt.show()

frames = [NBA_2021_df, NBA_2022_df]
NBA_2021_df = pd.concat(frames)
print(NBA_2021_df.head(20))

# apply machine learning modeling technique

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report
import pandas as pd

NBA_2021_df.dropna(axis=0, inplace=True)
corr = NBA_2021_df.corr().abs()
corrMatrix = corr.loc[corr['playoffs_y_n'] == 1]
corrMatrix.index

variables = list(corr.index)

print(NBA_2021_df.head(5))

# Modeling
X = NBA_2021_df[variables].drop(['playoffs_y_n', 'W', 'L', 'MP',
                                 'W%'], 1)
y = NBA_2021_df['playoffs_y_n']

pred_X = NBA_2022_df[variables].drop('playoffs_y_n', 1)

print("Predictor:\n", pred_X)
print("selected Variables:\n", X.head(5))

# split your dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=58)

print(f'\nX train shape: {X_train.shape}')
print(f'\nX test shape: {X_test.shape}')

#Choosing and applying to the  model
def evaluate_Models(X_train, y_train, X_test, y_test):
    dfs = []

    models = [
        ('LogReg', LogisticRegression()),
        ('RF', RandomForestClassifier()),
        ('KNN', KNeighborsClassifier())
    ]

    results = []

    names = []

    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']

    target_names = ['Makes Playoff', 'No Playoff']

    for name, model in models:
        # Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).
        kfold = model_selection.KFold(n_splits=3, shuffle=True, random_state=42)
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
        clf = model.fit(X_train, y_train)
        pred_X = clf.predict(X_test)

        print(name)
        print(classification_report(y_test, pred_X, target_names=target_names))

        results.append(cv_results)
        names.append(name)

        this_df = pd.DataFrame(cv_results)
        this_df['model'] = name
        dfs.append(this_df)
        plot_roc_curve(model, X_test, y_test)
        print('Confusion Matrix:\n')
        plot_confusion_matrix(model, X_test, y_test)

    final = pd.concat(dfs, ignore_index=True)

    return final


final = evaluate_Models(X_train, y_train, X_test, y_test)

# AEvaluate the Results -Model Performance
bootstraps = []
for model in list(set(final.model.values)):
    model_df = final.loc[final.model == model]
    bootstrap = model_df.sample(n=30, replace=True)
    bootstraps.append(bootstrap)

bootstrap_df = pd.concat(bootstraps, ignore_index=True)
results_long = pd.melt(bootstrap_df, id_vars=['model'], var_name='metrics', value_name='values')
time_metrics = ['fit_time', 'score_time']  # fit time metrics
## PERFORMANCE METRICS
results_long_nofit = results_long.loc[~results_long['metrics'].isin(time_metrics)]  # get df without fit data
results_long_nofit = results_long_nofit.sort_values(by='values')
## TIME METRICS
results_long_fit = results_long.loc[results_long['metrics'].isin(time_metrics)]  # df with fit data
results_long_fit = results_long_fit.sort_values(by='values')

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 15))
sns.set(font_scale=3.0)
g = sns.boxplot(x="model", y="values", hue="metrics", data=results_long_nofit, palette="Set2")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Comparison of Model by Classification Metric')
plt.show()

metrics = list(set(results_long_nofit.metrics.values))
print(bootstrap_df.groupby(['model'])[metrics].agg([np.std, np.mean]))
