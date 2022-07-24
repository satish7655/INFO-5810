import pandas as pd
from sklearn.linear_model import LogisticRegression

# Read the Csv files
NBA_2021_df = pd.read_csv('C:/Users/I24253.VERISK/OneDrive - Verisk Analytics/Desktop/INFO/Masters/NBA_2000_2021.csv')
NBA_2022_df = pd.read_csv('C:/Users/I24253.VERISK/OneDrive - Verisk Analytics/Desktop/INFO/Masters/NBA_2022_ED.csv')

# Load the data and check the values for each dataset
print("\n", NBA_2021_df.head(10))
print("\n", NBA_2022_df.head(10))

print(NBA_2021_df.columns)

# Drop multiple columns from the dataframe
Remove = ['playoffs_y_n', 'RK', 'Team', 'Year', 'Conf',
          'G', 'W', 'L',
          'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%',
          'ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS',
          'W%', ]

frames = [NBA_2021_df, NBA_2022_df]
combined_df = pd.concat(frames)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(combined_df.head(20))

# select the conference to predict the playoff team in the up coming season
selected_conference = combined_df.loc[combined_df["Conf"] == "East"]

train_1 = selected_conference.loc[selected_conference["Year"] < 2022]
test_1 = selected_conference.loc[selected_conference["Year"] >= 2022]

# define the predictor variables and the response variable
X_train_1 = train_1.drop(columns=Remove)
y_train_1 = train_1[['playoffs_y_n']]

X_test_1 = test_1.drop(columns=Remove)
y_test_1 = test_1[['playoffs_y_n']]

# columns included in training the model
print(X_train_1.columns)

print("\n Logistic Regression Model to predict team from East Conf. who will make Playoffs next season")

# instantiate the model
log_regression = LogisticRegression()

# fit the model using the training data
log_regression.fit(X_train_1, y_train_1)

Training_Score = log_regression.score(X_train_1, y_train_1)
Testing_Score = log_regression.score(X_test_1, y_test_1)

print(f"Training Data Score: {Training_Score}")
print(f"Testing Data Score: {Testing_Score}")

print("\n")
Conf_scores = [["Train Score", Training_Score],
               ["Test Score", Testing_Score]]
selected_conference_scores = pd.DataFrame(Conf_scores, columns=['Type', 'Logistic'])
print(selected_conference_scores)

print("\n**********Predicted team that will make Playoff along w/ their details**************")
Playoff_Prediction_2022 = test_1[["Team", "Year", "playoffs_y_n", "W%"]]

log_probability = log_regression.predict_proba(X_test_1)[:, 1].tolist()
log_prediction = log_regression.predict(X_test_1).tolist()
Playoff_Prediction_2022["prediction_log"] = log_prediction
Playoff_Prediction_2022["probability_log"] = log_probability

Playoff_Prediction_2022.loc[Playoff_Prediction_2022["playoffs_y_n"] == 1]
print(Playoff_Prediction_2022.sort_values("probability_log", ascending=False))
