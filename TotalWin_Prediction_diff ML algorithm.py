import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

NBA_2021_df = pd.read_csv('C:/Users/I24253.VERISK/OneDrive - Verisk Analytics/Desktop/INFO/Masters/NBA_2000_2021.csv')
NBA_2022_df = pd.read_csv('C:/Users/I24253.VERISK/OneDrive - Verisk Analytics/Desktop/INFO/Masters/NBA_2022_ED.csv')

# check for the null value
# print(pd.isnull(NBA_2021_df.sum()))
print(NBA_2021_df[pd.isnull(NBA_2021_df["playoffs_y_n"])])

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(NBA_2021_df.head(10))

# inspecting the different team stat categories
print("\nDisplaying Categories from the NBA_2021_dfset:\n", NBA_2021_df.columns)

# Exploratory NBA_2021_df Analysis
print("\n**EDA**")
print(NBA_2021_df['Team'].value_counts())
fig = plt.figure(figsize=(100, 100))
sns.countplot(x="Team", data=NBA_2021_df, palette='hls', )
plt.title("Average Win count Per Team from all the seasons")
plt.show()

# display Team PPG and count from the given years
Team = NBA_2021_df.groupby('Team')['PTS'].agg(
    ['count', 'mean', 'std'])
Team['std'] = round(Team['std'], 1)
Team['mean'] = round(Team['mean'], 1)
Team.sort_values('mean', ascending=False)
print("\n Displaying team count, mean and std\n", Team.head(10))

# Create histogram to show distribution of Team blocks per game Rating
sns.set_theme(style='whitegrid')
f, ax = plt.subplots(figsize=(10, 10))
sns.set(font_scale=3)
sns.histplot(data=NBA_2021_df, x="BLK", bins=20)
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
plt.xlabel("Team block rating", fontsize=20)
plt.ylabel("Team Count", fontsize=20)
plt.title("Distribution of Team blocks per game Rating")
ax.tick_params(labelsize=20)
plt.show()

# Distribution of Team  3 Pointer Percentage
sns.set_theme(style='darkgrid')
f, ax = plt.subplots(figsize=(10, 10))
sns.set(font_scale=3)
# Create histogram
sns.histplot(data=NBA_2021_df, x="3P%", bins=20)
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
plt.xlabel("Turnover %", fontsize=20)
plt.ylabel("Team Count", fontsize=20)
plt.title("Distribution of Team  3 Pointer Percentage")
ax.tick_params(labelsize=20)
plt.show()

Ranking = NBA_2021_df.loc[NBA_2021_df['W%'] >= 0.50]
plt.figure(figsize=(7, 7))
plt.title("Team Offense Rankings")
ax = sns.lineplot(x=Ranking.Year, y=Ranking['3P'], label="3P")
sns.lineplot(x=Ranking.Year, y=Ranking['FG'], label="FG")
sns.lineplot(x=Ranking.Year, y=Ranking['BLK'], label="BPP")
plt.ylim(0, 100)
ax.set(ylabel="League Ranking")
plt.show()

# finding variables that have a Pearson's Correlation Coefficient of at least medium strength with number of W
corr = NBA_2021_df.corr().abs()
corr = corr.loc[corr['W'] > .25]
print(corr.index)

variables = list(corr.index)
corr_df = NBA_2021_df[variables].groupby('W').mean()
print(corr_df)

# correlation between wins and other variables

ax = sns.regplot(x=NBA_2021_df['2P%'] * 100, y=NBA_2021_df['W'], label='2P%')
sns.regplot(x=NBA_2021_df['FG%'] * 100, y=NBA_2021_df['W'], label='FG')
sns.regplot(x=NBA_2021_df['3P%'] * 100, y=NBA_2021_df['W'], label='3P')
sns.regplot(x=NBA_2021_df['TOV'], y=NBA_2021_df['W'], label='TOV')
ax.set(xlabel="League Rank")
plt.legend()
plt.title("Correlation of different varibales and Wins")
plt.show()

# preparing for the machine learning
predictors = ['Unnamed: 0', 'RK', 'Year', 'G',
              'L', 'MP', 'FGA', '3P', '3PA', '2P', '2PA', '2P%',
              'FT', 'PF', 'PTS', 'W%', 'MP_scalnorm', 'FG_scalnorm', 'FGA_scalnorm',
              'FG%_scalnorm', '3P_scalnorm', '3PA_scalnorm', '3P%_scalnorm',
              '2P_scalnorm', '2PA_scalnorm', '2P%_scalnorm', 'FT_scalnorm',
              'FTA_scalnorm', 'FT%_scalnorm', 'ORB_scalnorm', 'DRB_scalnorm',
              'AST_scalnorm', 'STL_scalnorm', 'BLK_scalnorm', 'TOV_scalnorm',
              'PF_scalnorm', 'PTS_scalnorm', 'W%_scalnorm']
selected_variables = [item for item in variables if item not in predictors]
print("Selected variables only:", variables)

X = NBA_2021_df[selected_variables].drop('W', 1)
y = NBA_2021_df['W']

pred_X = NBA_2022_df[selected_variables].drop("W", 1)

print(X.head())

# splitting the dataframe into training and testing sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

# instantiate the LinearRegression model
model1 = linear_model.LinearRegression()

# fit the model using the training data
model1.fit(X_train, y_train)
model_predicted_wins = model1.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(model_predicted_wins, y_valid)))

model_predicted_wins = model1.predict(pred_X)
print(model_predicted_wins)

NBA_2022_df.rename(columns={'W': 'last Year Wins'}, inplace=True)
model_predicted_wins_df = NBA_2022_df[['Team', 'Year']]
i = 0
while i < 30:
    model_predicted_wins_df.at[i, 'Predicted Wins'] = model_predicted_wins[i]
    i += 1
model_predicted_wins_df.sort_values(by='Predicted Wins', ascending=False)
print("\n******Prediction for the up-coming NBA season using LinearRegression model *******\n", model_predicted_wins_df)

# create regressor object and Training the Algorithm to see how it performs
model2 = RandomForestRegressor(n_estimators=100, random_state=0)
model2.fit(X_train, y_train)

# Evaluating the Algorithm and Predicting a new result
model2_predicted_wins = model2.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(model2_predicted_wins, y_valid)))
print('Mean Squared Error:' + str(mean_squared_error(model2_predicted_wins, y_valid)))

model2_predicted_wins = model2.predict(pred_X)
print(model2_predicted_wins)

# convert the array to panda dataframe
model2predicted_wins_df = NBA_2022_df[['Team', 'Year']]
i = 0
while i < 30:
    model2predicted_wins_df.at[i, 'Predicted Wins'] = model2_predicted_wins[i]
    i += 1
model2predicted_wins_df.sort_values(by='Predicted Wins', ascending=False)
print("\n******Prediction for the up-coming NBA season using RandomForestRegressor model *******\n",
      model2predicted_wins_df)
