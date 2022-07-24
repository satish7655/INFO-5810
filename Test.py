import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model

data = pd.read_csv('C:/Users/I24253.VERISK/OneDrive - Verisk Analytics/Desktop/INFO/Masters/NBA_2000_2021.csv')
pred_data = pd.read_csv('C:/Users/I24253.VERISK/OneDrive - Verisk Analytics/Desktop/INFO/Masters/NBA_2022_ED.csv')

#check for the null value
#print(pd.isnull(data.sum()))
print(data[pd.isnull(data["playoffs_y_n"])])


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(data.head(10))



#inspecting the different team stat categories
print("\nDisplaying Categories from the dataset:\n",data.columns)
'''
# Exploratory Data Analysis
print("\n**EDA**")
print(data.head(10))
print(data['Team'].value_counts())

fig = plt.figure(figsize=(100, 100))
sns.countplot(x="Team", data=data, palette='Paired')
plt.title("Average Win Per Team")
plt.show()

# display Team PPG and count from the given years
Team = data.groupby('Team')['PTS'].agg(
    ['count', 'mean', 'std'])
Team['std'] = round(Team['std'], 1)
Team['mean'] = round(Team['mean'], 1)
Team.sort_values('mean', ascending=False)
print("\n Displaying team count, mean and std\n",Team.head(10))

# Create histogram to show distribution of Team blocks per game Rating

sns.set_theme(style = 'whitegrid')
f, ax = plt.subplots(figsize=(10,10))
sns.set(font_scale = 3)


sns.histplot(data = data, x="BLK", bins = 20)
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
plt.xlabel( "Team block rating", fontsize = 20)
plt.ylabel("Team Count", fontsize=20)
plt.title("Distribution of Team blocks per game Rating")
ax.tick_params(labelsize = 20)
plt.show()


# Distribution of Team  3 Pointer Percentage
sns.set_theme(style = 'whitegrid')
f, ax = plt.subplots(figsize=(10,10))
sns.set(font_scale = 3)
# Create histogram
sns.histplot(data = data, x="3P%", bins = 20)
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
plt.xlabel( "Turnover %", fontsize = 20)
plt.ylabel("Team Count", fontsize=20)
plt.title("Distribution of Team  3 Pointer Percentage")
ax.tick_params(labelsize = 20)
plt.show()

'''
# finding variables that have a Pearson's Correlation Coefficient of at least medium strength with number of W
corr = data.corr().abs()
corr = corr.loc[corr['W'] > .25]
print(corr.index)

variables = list(corr.index)
corr_df = data[variables].groupby('W').mean()
print(corr_df)

fig = plt.figure(figsize=(20, 20))
plt.title("Average League Ranking in Each Category, by W")
sns.heatmap(data=corr_df,annot=True)
plt.xlabel("Regular Season Stat Rankings")
plt.show()


predictors = ['Unnamed: 0', 'RK', 'Year', 'G',
       'L', 'MP', 'FGA', '3P', '3PA', '2P', '2PA', '2P%',
       'FT', 'PF','PTS', 'W%', 'MP_scalnorm', 'FG_scalnorm', 'FGA_scalnorm',
       'FG%_scalnorm', '3P_scalnorm', '3PA_scalnorm', '3P%_scalnorm',
       '2P_scalnorm', '2PA_scalnorm', '2P%_scalnorm', 'FT_scalnorm',
       'FTA_scalnorm', 'FT%_scalnorm', 'ORB_scalnorm', 'DRB_scalnorm',
       'AST_scalnorm', 'STL_scalnorm', 'BLK_scalnorm', 'TOV_scalnorm',
       'PF_scalnorm', 'PTS_scalnorm', 'W%_scalnorm']
variables = [item for item in variables if item not in predictors]
print("Selected varibales only:",variables)


X = data[variables].drop('W',1)
y = data['W']

pred_X = pred_data[variables].drop("W", 1)

print(X.head())


#splitting data into training and testing sets
X_train, X_valid, y_train, y_valid = train_test_split(X,y)


model = linear_model.LinearRegression()
model.fit(X_train, y_train)
model_predicted_wins = model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(model_predicted_wins, y_valid)))

model_predicted_wins = model.predict(pred_X)
print(model_predicted_wins)

pred_data.rename(columns={'W': 'last Year Wins'}, inplace = True)


model_predicted_wins_df = pred_data[['Team','Year']]
i=0
while i<30:
    model_predicted_wins_df.at[i, 'Predicted Wins'] = model_predicted_wins[i]
    i += 1
model_predicted_wins_df.sort_values(by='Predicted Wins',ascending=False)
print("\n******Prediction for the up-coming NBA season*******\n",model_predicted_wins_df)




