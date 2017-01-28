import datetime as time

#method to get last x matches
def get_lastmatches_data(dataframe, date1, team, x=5):
    prev_matches = dataframe[(dataframe['HomeTeam'] == team) | (dataframe['AwayTeam'] == team)]
    prev_matches = prev_matches.reset_index(drop=True)
    i = 0
    # get the last x games
    index1 = 0
    for index, row in prev_matches.iterrows():
        date = str(row['Date'])
        if time.datetime.strptime(date, "%d/%m/%y") \
                < time.datetime.strptime(date1, "%d/%m/%y"):
            continue
        else:
            index1 = index
            break
    df_temp = prev_matches.iloc[index1 - x:index1]
    return df_temp

# method to get last x matches
def get_lastmatchesH2H_data(dataframe, date1, hometeam, awayteam, x=5):
   prev_matches = dataframe[(dataframe['HomeTeam'] == hometeam) & (dataframe['AwayTeam'] == awayteam) \
      | (dataframe['AwayTeam'] == hometeam) & (dataframe['HomeTeam'] == awayteam)]
   prev_matches = prev_matches.reset_index(drop=True)
   i = 0
   index1 = prev_matches.shape[0]
   for index, row in prev_matches.iterrows():
       date = str(row['Date'])
       if time.datetime.strptime(date, "%d/%m/%y") \
               < time.datetime.strptime(date1, "%d/%m/%y"):
           continue
       else:
           index1 = index
           break
   if index1 - x >= 0:
       df_temp = prev_matches.iloc[index1 - x:index1]
   else:
       df_temp = prev_matches.iloc[prev_matches.shape[0]-x:prev_matches.shape[0]]
   return df_temp


# helper functions


#number of losses in last 5 matches
def get_number_of_losses(matches, date, team, x=5):
    last_matches_df = get_lastmatches_data(matches, date, team, x)
    count = 0
    for _, row in last_matches_df.iterrows():
        # if team is home team
        if row['HomeTeam'] == team:
            if row['FTR'] == 'A':
                count += 1
        else:
            # team is away team
            if row['FTR'] == 'H':
                count += 1
    return count

#number of draws in last 5 matches
def get_number_of_draws(matches, date, team, x=5):
    last_matches_df = get_lastmatches_data(matches, date, team, x)
    count = 0
    for _, row in last_matches_df.iterrows():
        if row['FTR'] == 'D':
            count += 1
    return count

#number of wins in last 5 matches
def get_number_of_wins(matches, date, team, x=5):
    last_matches_df = get_lastmatches_data(matches, date, team, x)
    count = 0
    for _, row in last_matches_df.iterrows():
        # if team is home team
        if row['HomeTeam'] == team:
            if row['FTR'] == 'H':
                count += 1
        else:
            # team is away team
            if row['FTR'] == 'A':
                count += 1
    return count

#number of head to head losses in last 5 matches
def get_number_of_h2h_losses(matches, date, home_team, away_team, target_team, x=5):
    last_matches_df = get_lastmatchesH2H_data(matches, date, home_team, away_team, x)
    count = 0
    for _, row in last_matches_df.iterrows():
        # if team is home team
        if row['HomeTeam'] == target_team:
            if row['FTR'] == 'A':
                count += 1
        else:
            # team is away team
            if row['FTR'] == 'H':
                count += 1
    return count

#number of head to head draws in last 5 matches
def get_number_of_h2h_draws(matches, date, home_team, away_team, target_team=None, x=5):
    last_matches_df = get_lastmatchesH2H_data(matches, date, home_team, away_team, x)
    count = 0
    for _, row in last_matches_df.iterrows():
        if row['FTR'] == 'D':
            count += 1
    return count

#number of the head to head wins in last 5 matches for the target team
def get_number_of_h2h_wins(matches, date, home_team, away_team, target_team, x=5):
    last_matches_df = get_lastmatchesH2H_data(matches, date, home_team, away_team, x)
    count = 0
    for _, row in last_matches_df.iterrows():
        # if team is home team
        if row['HomeTeam'] == target_team:
            if row['FTR'] == 'H':
                count += 1
        else:
            # team is away team
            if row['FTR'] == 'A':
                count += 1
    return count

#number of the goals scored in the last  matches overall
def get_number_of_goals_scored(matches, date, team, x=5):
    last_matches_df = get_lastmatches_data(matches, date, team, x)
    count = 0
    for _, row in last_matches_df.iterrows():
        # if team is home team
        if row['HomeTeam'] == team:
            count += row['FTHG']
        else:
            # team is away team
            count += row['FTAG']
    return count

#number of goals conceded in last 5 matches overall
def get_number_of_goals_conceded(matches, date, team, x=5):
    last_matches_df = get_lastmatches_data(matches, date, team, x)
    count = 0
    for _, row in last_matches_df.iterrows():
        # if team is home team
        if row['HomeTeam'] == team:
            count += row['FTAG']
        else:
            # team is away team
            count += row['FTHG']
    return count

#number of shots on target overall
def get_number_of_shots_on_target(matches, date, team, x=5):
    last_matches_df = get_lastmatches_data(matches, date, team, x)
    count = 0
    for _, row in last_matches_df.iterrows():
        # if team is home team
        if row['HomeTeam'] == team:
            count += row['HST']
        else:
            # team is away team
            count += row['AST']
    return count

#number of corners overall
def get_number_of_corners(matches, date, team, x=5):
    last_matches_df = get_lastmatches_data(matches, date, team, x)
    count = 0
    for _, row in last_matches_df.iterrows():
        # if team is home team
        if row['HomeTeam'] == team:
            count += row['HC']
        else:
            # team is away team
            count += row['AC']
    return count

# number of red cards overall
def get_number_of_red_cards(matches, date, team, x=5):
    last_matches_df = get_lastmatches_data(matches, date, team, x)
    count = 0
    for _, row in last_matches_df.iterrows():
        # if team is home team
        if row['HomeTeam'] == team:
            count += row['HR']
        else:
            # team is away team
            count += row['AR']
    return count

#head to head goals scored by the home team
def get_number_of_h2h_home_goals_scored(matches, date, home_team, away_team, target_team, x=5):
    last_matches_df = get_lastmatchesH2H_data(matches, date, home_team, away_team, x)
    count = 0
    # print last_matches_df
    for _, row in last_matches_df.iterrows():
        # if team is home team
        if row['HomeTeam'] == target_team:
            count += row['FTHG']
    return count

#head to head goals conceded bby the home team
def get_number_of_h2h_home_goals_conceded(matches, date, home_team, away_team, target_team, x=5):
    last_matches_df = get_lastmatchesH2H_data(matches, date, home_team, away_team, x)
    count = 0
    for _, row in last_matches_df.iterrows():
        # if team is home team
        if row['HomeTeam'] == target_team:
            count += row['FTAG']
    return count

#head to head goals scored by the away team
def get_number_of_h2h_away_goals_scored(matches, date, home_team, away_team, target_team, x=5):
    last_matches_df = get_lastmatchesH2H_data(matches, date, home_team, away_team, x)
    count = 0
    for _, row in last_matches_df.iterrows():
        # if team is away team
        if row['AwayTeam'] == target_team:
            count += row['FTAG']
    return count

#head to head goals conceded by the away team
def get_number_of_h2h_away_goals_conceded(matches, date, home_team, away_team, target_team, x=5):
    last_matches_df = get_lastmatchesH2H_data(matches, date, home_team, away_team, x)
    count = 0
    for _, row in last_matches_df.iterrows():
        # if team is away team
        if row['AwayTeam'] == target_team:
            count += row['FTHG']
    return count

#head to head shots on target for the home team
def get_number_of_h2h_home_shot_on_target(matches, date, home_team, away_team, target_team, x=5):
    last_matches_df = get_lastmatchesH2H_data(matches, date, home_team, away_team, x)
    count = 0
    for _, row in last_matches_df.iterrows():
        # if team is home team
        if row['HomeTeam'] == target_team:
            count += row['HST']
    return count

#head to head shots on target for away team
def get_number_of_h2h_away_shot_on_target(matches, date, home_team, away_team, target_team, x=5):
    last_matches_df = get_lastmatchesH2H_data(matches, date, home_team, away_team, x)
    count = 0
    for _, row in last_matches_df.iterrows():
        # if team is away team
        if row['AwayTeam'] == target_team:
            count += row['AST']
    return count

#head to head corners for the home team
def get_number_of_h2h_home_corners(matches, date, home_team, away_team, target_team, x=5):
    last_matches_df = get_lastmatchesH2H_data(matches, date, home_team, away_team, x)
    count = 0
    for _, row in last_matches_df.iterrows():
        # if team is home team
        if row['HomeTeam'] == target_team:
            count += row['HC']
    return count

#head to head corners for the away team
def get_number_of_h2h_away_corners(matches, date, home_team, away_team, target_team, x=5):
    last_matches_df = get_lastmatchesH2H_data(matches, date, home_team, away_team, x)
    count = 0
    for _, row in last_matches_df.iterrows():
        # if team is away team
        if row['AwayTeam'] == target_team:
            count += row['AC']
    return count


#head to head red card for home team
def get_number_of_h2h_home_red_cards(matches, date, home_team, away_team, target_team, x=5):
    last_matches_df = get_lastmatchesH2H_data(matches, date, home_team, away_team, x)
    count = 0
    for _, row in last_matches_df.iterrows():
        # if team is home team
        if row['HomeTeam'] == target_team:
            count += row['HR']
    return count

#head to head away red cards for away team
def get_number_of_h2h_away_red_cards(matches, date, home_team, away_team, target_team, x=5):
    last_matches_df = get_lastmatchesH2H_data(matches, date, home_team, away_team, x)
    count = 0
    for _, row in last_matches_df.iterrows():
        # if team is away team
        if row['AwayTeam'] == target_team:
            count += row['AR']
    return count

# get the last 5 matches performance using (2*win+1*draw+ -2*loss) overall
def get_past_performance(dataframe, date1, team):
    win = get_number_of_wins(dataframe, date1, team)
    draw = get_number_of_draws(dataframe, date1, team)
    loss = get_number_of_losses(dataframe, date1, team)
    return 2*win+1*draw-2*loss

# get the last 5 matches performance using (2*win+1*draw+ -2*loss) head to head
def get_past_h2h_performance(dataframe, date1, hometeam, awayteam, target_team):
    win = get_number_of_h2h_wins(dataframe, date1, hometeam, awayteam, target_team)
    draw = get_number_of_h2h_draws(dataframe, date1, hometeam, awayteam, target_team)
    loss = get_number_of_h2h_losses(dataframe, date1, hometeam, awayteam, target_team)
    return 2*win+1*draw-2*loss

# Testing APIS
#print(get_lastmatches_data(dataframe,'24/04/16', 'Arsenal'))
# print()
# print("Performance overall = " + str(get_past_performance(dataframe,'24/04/16', 'Arsenal')))
# print()
# print("Performance  H2H = " + str(get_past_h2h_performance(dataframe,'24/04/16', 'Arsenal', 'Chelsea', 'Arsenal')))
# print("Performance  H2H = " + str(get_past_h2h_performance(dataframe,'24/04/16', 'Arsenal', 'Chelsea', 'Chelsea')))
