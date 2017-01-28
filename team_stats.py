import sys
from plots import *

# method for getting the number of actual wins vs predicted wins per team
def plot_win_stats(dataframe1, pred_target, chartname):
    teams = list(set(dataframe1['HomeTeam']) | set(dataframe1['AwayTeam']))
    actual_team_win = {}
    predicted_team_win = {}
    for team in teams:
        actual_team_win[team] = 0
        predicted_team_win[team] = 0

    if len(dataframe1) != len(pred_target):
        print("The length is different\n")
        sys.exit(0)

    for index, row in dataframe1.iterrows():
        #actual win stats per team
        if row['FTR'] == 'H':
            actual_team_win[row['HomeTeam']] += 1
        if row['FTR'] == 'A':
            actual_team_win[row['AwayTeam']] += 1

        #predicted win stats per team
        if pred_target[index] == 1:
            predicted_team_win[row['HomeTeam']] += 1
        if pred_target[index] == 2:
            predicted_team_win[row['AwayTeam']] += 1
    # print("Actual team win stats")
    # print(actual_team_win)
    # print("Predicted team win stats")
    # print(predicted_team_win)
    # output = open(chartname, 'w')
    # output.write("Actual win stats for team\n")
    # output.write(str(actual_team_win))
    # output.write("\n\nPredicted win stats for team\n")
    # output.write(str(predicted_team_win))
    # output.close()
    plot_hist(actual_team_win, predicted_team_win, chartname)