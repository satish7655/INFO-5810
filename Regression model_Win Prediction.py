import matplotlib
import pandas as pd
from sklearn.linear_model import Ridge

data = pd.read_csv('C:/Users/I24253.VERISK/OneDrive - Verisk Analytics/Desktop/INFO/Masters/NBA_2000_2021.csv')
pred_data = pd.read_csv('C:/Users/I24253.VERISK/OneDrive - Verisk Analytics/Desktop/INFO/Masters/NBA_2022.csv')

# check for the null value
print(pd.isnull(data.sum()))

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(data.head(10))

# inspecting the different team stat categories
print("\nDisplaying Categories from the dataset:\n", data.columns)

# just use numerical values to train the
predictors = (['playoffs_y_n', 'G', 'L', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%',
               'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
               'PTS', 'W%'])
train = data[data["Year"] < 2022]
test = pred_data[pred_data["Year"] > 2021]

reg = Ridge(alpha=.1)

reg.fit(train[predictors], train["W"])
predictions = reg.predict(test[predictors])
print(predictions)
# convert the array to panda dataframe
predictions = pd.DataFrame(predictions, columns=["predictions"])
print(predictions)

combination = pd.concat([test[["Team", "W", "Year"]], predictions], axis=1)
# print(combination)
print("\n", combination.sort_values("W", ascending=False).head(10))
combination["predicted Ranks"] = list(range(1, combination.shape[0] + 1))
print("\nWin Prediction in up coming NBA Season-2022:\n", combination.head(15))
