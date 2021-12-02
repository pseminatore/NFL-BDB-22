import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
from operator import countOf

# 1.0 for clean catch, 0.7 for clean field, 0.5 for bobble, 0 for muff, None for Touchback, OOB, etc
contact_mappings = {'BB': 0.72, 'BC': 0.57, 'BF': 0.77, 'BOG': 0.52}

def read_inputs():
    ## Read input data
    
    scouting_df = pd.read_csv('input/nfl-big-data-bowl-2022/PFFScoutingData.csv')
    games_df = pd.read_csv('input/nfl-big-data-bowl-2022/games.csv')
    plays_df = pd.read_csv('input/nfl-big-data-bowl-2022/plays.csv')
    tracking_df = pd.DataFrame()
    #tracking_2018 = pd.read_csv('../input/nfl-big-data-bowl-2022/tracking2018.csv')
    #tracking_2019 = pd.read_csv('../input/nfl-big-data-bowl-2022/tracking2019.csv')
    tracking_2020 = pd.read_csv('input/nfl-big-data-bowl-2022/tracking2020.csv')
    #tracking_df = tracking_df.append(tracking_2018)
    #tracking_df = tracking_df.append(tracking_2019)
    tracking_df = tracking_df.append(tracking_2020)
    return scouting_df, games_df, plays_df, tracking_df

def encode_columns(plays_df, scouting_df, tracking_df):
    ## Preprocessing step for encoding inputs
    
    # Get playIDs for all punts, as well as other raw columns that require no further manipulation
    params_df = plays_df[['gameId', 'playId', 'specialTeamsPlayType', 'kickerId', 'returnerId', 'kickLength', 'kickReturnYardage']].query('specialTeamsPlayType == "Punt"')
    scouting_df = scouting_df[['gameId', 'playId', 'snapDetail', 'operationTime', 'hangTime', 'kickType', 'kickDirectionIntended', 'kickDirectionActual', 'returnDirectionIntended', 'returnDirectionActual', 'gunners', 'puntRushers', 'specialTeamsSafeties', 'vises', 'kickContactType']]
    params_df.set_index(['gameId', 'playId'], inplace=True)
    scouting_df.set_index(['gameId', 'playId'], inplace=True)
    params_df = params_df.join(scouting_df, on=['gameId', 'playId'])
    
    catch_df = getFrameIDOfPuntCatch(tracking_df, plays_df)
    
    # Derived Columns
    params_df['snapQuality'] = params_df['snapDetail'].apply(lambda x: 1 if x == 'OK' else 0)
    params_df['kickDirDiff'] = params_df[['kickDirectionIntended', 'kickDirectionActual']].apply(lambda x: 1 if x[0] == x['kickDirectionActual'] else (0.5 if (x['kickDirectionIntended'] == 'C' or x['kickDirectionActual'] == 'C') else 0), axis=1)
    params_df['retDirDiff'] = params_df[['returnDirectionIntended', 'returnDirectionActual']].apply(lambda x: 1 if x['returnDirectionIntended'] == x['returnDirectionActual'] else (0.5 if (x['returnDirectionIntended'] == 'C' or x['returnDirectionActual'] == 'C') else 0), axis=1)
    params_df['numGunners'] = params_df['gunners'].apply(lambda x: 0 if pd.isna(x) else (len(x.split(';'))))
    params_df['numVises'] = params_df['vises'].apply(lambda x: 0 if pd.isna(x) else (len(x.split(';'))))
    params_df['numSafeties'] = params_df['specialTeamsSafeties'].apply(lambda x: 0 if pd.isna(x) else (len(x.split(';'))))
    params_df['numRushers'] = params_df['puntRushers'].apply(lambda x: 0 if pd.isna(x) else (len(x.split(';'))))
    params_df.drop(['gunners', 'vises', 'specialTeamsSafeties', 'puntRushers', 'returnDirectionIntended', 'returnDirectionActual', 'kickDirectionIntended', 'kickDirectionActual', 'snapDetail'], axis=1, inplace=True)
    
    return params_df


def getFrameIDOfPuntCatch(tracking_df, plays_df):
    plays_df['season'] = plays_df.apply(lambda x: int(str(x['gameId'])[:4]), axis=1)
    punts_df = plays_df[['gameId', 'playId', 'season', 'specialTeamsPlayType', 'returnerId']].query('specialTeamsPlayType == "Punt" & (season == 2020 | season == 2021)')
    temp_df = tracking_df[['event', 'gameId', 'playId', 'frameId']]
    temp_df = temp_df.query('event == "kick_received" | event == "punt_land" | event == "punt_downed" | event == "punt_received" | event == "punt_muffed" | event == "fair_catch"')
    #temp_df = temp_df.query('event == "kick_received"').drop_duplicates()
    temp_df.set_index(['gameId', 'playId'], inplace=True)
    punts_df.set_index(['gameId', 'playId'], inplace=True)
    catch_df = temp_df.join(punts_df, on=['gameId', 'playId'], how='inner')
    catch_df = catch_df.drop_duplicates()
    print(catch_df['event'].drop_duplicates())
    catch_df.head()
    return catch_df




def main():
    scouting_df, games_df, plays_df, tracking_df = read_inputs()
    params_df = encode_columns(plays_df, scouting_df, tracking_df)
    print(params_df.head())
    

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    main()