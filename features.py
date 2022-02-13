import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from pandas.core.groupby.generic import ScalarResult
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPRegressor
import warnings


# 1.0 for clean catch, 0.7 for clean field, 0.5 for bobble, 0 for muff, None for Touchback, OOB, etc
contact_mappings = {'CC': 1.0, 'CFFG': 0.9, 'BB': 0.72, 'BC': 0.57, 'BF': 0.77, 'BOG': 0.52, 'ICC': 0.25, 'MBC': 0.0, 'MBDR': 0.0}

def read_inputs():
    ## Read input data
    
    scouting_df = pd.read_csv('input/nfl-big-data-bowl-2022/PFFScoutingData.csv')
    games_df = pd.read_csv('input/nfl-big-data-bowl-2022/games.csv')
    plays_df = pd.read_csv('input/nfl-big-data-bowl-2022/plays.csv')
    tracking_df = pd.DataFrame()
    tracking_2018 = pd.read_csv('input/nfl-big-data-bowl-2022/tracking2018.csv')
    tracking_2019 = pd.read_csv('input/nfl-big-data-bowl-2022/tracking2019.csv')
    tracking_2020 = pd.read_csv('input/nfl-big-data-bowl-2022/tracking2020.csv')
    tracking_df = tracking_df.append(tracking_2018)
    tracking_df = tracking_df.append(tracking_2019)
    tracking_df = tracking_df.append(tracking_2020)
    return scouting_df, games_df, plays_df, tracking_df

def encode_columns(plays_df, scouting_df, tracking_df, games_df):
    ## Preprocessing step for encoding inputs
    games_df = games_df.copy()
    games_df = games_df.reset_index(drop=True)
    # Get playIDs for all punts, as well as other raw columns that require no further manipulation
    params_df = plays_df[['gameId', 'playId', 'specialTeamsPlayType', 'kickerId', 'returnerId', 'kickLength', 'kickReturnYardage', 'possessionTeam']].query('specialTeamsPlayType == "Punt"')
    scouting_df = scouting_df[['gameId', 'playId', 'snapDetail', 'operationTime', 'hangTime', 'kickType', 'kickDirectionIntended', 'kickDirectionActual', 'returnDirectionIntended', 'returnDirectionActual', 'gunners', 'puntRushers', 'specialTeamsSafeties', 'vises', 'kickContactType']]
    params_df.set_index(['gameId', 'playId'], inplace=True)
    scouting_df.set_index(['gameId', 'playId'], inplace=True)
    params_df = params_df.join(scouting_df, on=['gameId', 'playId'])
    games_df = games_df.reset_index()
    params_df = params_df.reset_index()

    params_df['season'] = params_df['gameId'].apply(lambda x: games_df.loc[x == games_df['gameId'], ['season']].values[0][0])
    params_df.set_index(['gameId', 'playId'], inplace=True)
    
    catch_df = getFrameIDOfPuntCatch(tracking_df, plays_df)
    temp_df = getDistancesAtTimeOfCatch(catch_df.copy(), tracking_df)
    temp_df.set_index(['gameId', 'playId'], inplace=True)
    
    # Derived Columns
    params_df['snapQuality'] = params_df['snapDetail'].apply(lambda x: 1 if x == 'OK' else 0)
    params_df['kickDirDiff'] = params_df[['kickDirectionIntended', 'kickDirectionActual']].apply(lambda x: 1 if x[0] == x['kickDirectionActual'] else (0.5 if (x['kickDirectionIntended'] == 'C' or x['kickDirectionActual'] == 'C') else 0), axis=1)
    params_df['retDirDiff'] = params_df[['returnDirectionIntended', 'returnDirectionActual']].apply(lambda x: 1 if x['returnDirectionIntended'] == x['returnDirectionActual'] else (0.5 if (x['returnDirectionIntended'] == 'C' or x['returnDirectionActual'] == 'C') else 0), axis=1)
    params_df['numGunners'] = params_df['gunners'].apply(lambda x: 0 if pd.isna(x) else (len(x.split(';'))))
    params_df['numVises'] = params_df['vises'].apply(lambda x: 0 if pd.isna(x) else (len(x.split(';'))))
    params_df['numSafeties'] = params_df['specialTeamsSafeties'].apply(lambda x: 0 if pd.isna(x) else (len(x.split(';'))))
    params_df['numRushers'] = params_df['puntRushers'].apply(lambda x: 0 if pd.isna(x) else (len(x.split(';'))))
    params_df['catchQuality'] = params_df['kickContactType'].apply(lambda x: contact_mappings[x] if x in contact_mappings.keys() else None)
    params_df['kickReturnYardage'].fillna(0, inplace=True)
    params_df['catchQuality'].fillna(0.0, inplace=True)
    
    returner_df = getReturnerAvg(params_df)
    params_df = params_df.join(returner_df, on=['returnerId'], how='left')
    params_df = params_df.join(temp_df, on=['gameId', 'playId'], how='inner')
    params_df['isReturn'] = params_df[['event', 'kickReturnYardage', 'catchQuality']].apply(lambda x: 0 if (x['event'] == 'fair_catch' or x['event'] == 'punt_downed' or (x['kickReturnYardage'] == 0 and x['event'] == 'punt_land' and x['catchQuality'] == 0)) else 1, axis=1)
    params_df.drop(['gunners', 'vises', 'specialTeamsSafeties', 'puntRushers', 'returnDirectionIntended', 'returnDirectionActual', 'kickDirectionIntended', 'kickDirectionActual', 'snapDetail', 'kickType', 'kickContactType'], axis=1, inplace=True)
    
    params_df.reset_index(inplace=True)
    
    return params_df, returner_df


def getReturnerAvg(df):
    df = df.copy()[['returnerId', 'kickReturnYardage']]
    returner_df = df.groupby(['returnerId']).mean()
    returner_df['returnAvg'] = returner_df['kickReturnYardage']
    returner_df.drop(['kickReturnYardage'], axis=1, inplace=True)
    return returner_df
    


def getDistancesAtTimeOfCatch(catch_df, tracking_df):
    catch_df.reset_index(inplace=True)
    
    catch_df.dropna(subset=['returnerId'], axis=0, inplace=True)
    
    tracking_df.set_index(['gameId', 'playId', 'frameId'], inplace=True)
    catch_df.set_index(['gameId', 'playId', 'frameId'], inplace=True)
    
    full_df = catch_df.join(tracking_df, on=['gameId', 'playId', 'frameId'], how='inner', rsuffix='r_')
    
    full_df.reset_index(inplace=True)
    catch_df.reset_index(inplace=True)
    
    punts_returners = catch_df[['gameId', 'playId', 'returnerId']].values.tolist()
    
    col_arr = []
    yd_arr = []
    in_radius_arr = []
    for punt in punts_returners:
        play = full_df.query('gameId == @punt[0] & playId == @punt[1]').dropna(subset=['nflId'])
        returner_id = int(punt[2].split(';')[0])
        returner_loc = play.query('nflId == @returner_id')[['x','y', 'team']].values.tolist()
        closest_loc = 1000.0
        play_in_rad = 0
        for row in play.itertuples():
            if int(row.nflId) == returner_id or row.team == returner_loc[0][2]:
                pass
            else:
                this_loc = distanceBetween(returner_loc[0][0], returner_loc[0][1], row.x, row.y)
                if this_loc < closest_loc:
                    closest_loc = this_loc
                if this_loc < 8.0:
                    play_in_rad += 1
        col_arr.append(closest_loc)
        yd_arr.append(returner_loc[0][1])
        in_radius_arr.append(play_in_rad)
        
    col = pd.Series(data=col_arr)
    yd = pd.Series(data=yd_arr)
    in_rad = pd.Series(data=in_radius_arr)
    yd = yd.apply(lambda x: 0 if (x < 10.0 or x > 110.0) else (int(120.0 - x) if x > 60.0 else int(x)))
    catch_df['closest_gunner'] = col
    catch_df['catch_yardline'] = yd
    catch_df['defenders_within_radius'] = in_rad
    catch_df.drop(['frameId', 'season', 'specialTeamsPlayType', 'returnerId'], axis=1, inplace=True)
    
    return catch_df
 
 
    
    
def getFrameIDOfPuntCatch(tracking_df, plays_df):
    plays_df['season'] = plays_df.apply(lambda x: int(str(x['gameId'])[:4]), axis=1)
    punts_df = plays_df[['gameId', 'playId', 'season', 'specialTeamsPlayType', 'returnerId']].query('specialTeamsPlayType == "Punt"')
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


def distanceBetween(x1, y1, x2, y2):
    a = (x1, y1)
    b = (x2, y2)
    c = math.dist(a, b)
    return c

def test_logistic_model(df):
    df = df.copy()
    df.drop(['gameId', 'playId', 'event', 'specialTeamsPlayType', 'kickerId', 'returnerId', 'kickReturnYardage'], axis=1, inplace=True)
    model = LogisticRegression(solver='newton-cg')
    df = df.sample(frac=1).reset_index(drop=True)
    y = df['isReturn'].to_numpy()
    X = df.drop(['isReturn'], axis=1)
    X.fillna(0.0, inplace=True)
    X = normalize(X, axis=0)
    Xone = X[:-200]
    yone = y[:-200]
    Xtwo = X[-199:]
    ytwo = y[-199:]

    #X = X.to_numpy()
    model.fit(Xone, yone)
    print('individual testing')
    total = len(ytwo)
    correct = 0
    for i in range(len(ytwo)):
        yres = model.predict([Xtwo[i]])
        confidence = model.predict_proba([Xtwo[i]])
        manual = 0
        if confidence[0][1] > 0.6:
            manual = 1
        if manual == ytwo[i]:
            correct += 1
        print('predicted: {}'.format(yres))
        print('confidence: {}'.format(confidence))
        print('actual: {}'.format(ytwo[i]))
        print('manual: {}'.format(manual))
        print('------------------')
    print("isReturn Score:")
    print(model.score(X, y))
    manual_score = correct/total
    print("manual score: {}".format(manual_score))


def test_linear_model(df):
    df = df.copy()
    df.drop(['gameId', 'playId', 'event', 'specialTeamsPlayType', 'kickerId', 'returnerId', 'isReturn', 'Unnamed: 0', 'possessionTeam', 'season'], axis=1, inplace=True)
    model = SGDRegressor(alpha=0.00002, penalty='elasticnet')
    scores = []
    
    y = df['kickReturnYardage'].to_numpy()
    X = df.drop(['kickReturnYardage'], axis=1)
    X.fillna(0.0, inplace=True)
    
    #X = normalize(X, axis=0)[:-1]
    X = X.to_numpy()
    
    test_x = X[-1]
    test_y = y[-1]
    
        
    model.fit(X, y)
    score = model.score(X, y)
    print(score)
    print(test_y)
    
def fit_logistic_model(df):
    df = df.copy()
    df.drop(['gameId', 'playId', 'event', 'specialTeamsPlayType', 'kickerId', 'returnerId', 'kickReturnYardage', 'Unnamed: 0', 'possessionTeam', 'season'], axis=1, inplace=True)
    model = LogisticRegression(solver='newton-cg')
    df = df.sample(frac=1).reset_index(drop=True)
    y = df['isReturn'].to_numpy()
    X = df.drop(['isReturn'], axis=1)
    X.fillna(0.0, inplace=True)
    X = normalize(X, axis=0)
    model.fit(X, y)
    return model
    
    
def fit_linear_model(df):
    df = df.copy()
    df.drop(['gameId', 'playId', 'event', 'specialTeamsPlayType', 'kickerId', 'returnerId', 'isReturn', 'Unnamed: 0', 'possessionTeam', 'season'], axis=1, inplace=True)
    model = SGDRegressor(alpha=0.00002, penalty='elasticnet')
    
    y = df['kickReturnYardage'].to_numpy()
    X = df.drop(['kickReturnYardage'], axis=1)
    X.fillna(0.0, inplace=True)
    
    X = normalize(X, axis=0)    
        
    model.fit(X, y)
    return model

def fit_neural_model(df):
    df = df.copy()
    df.drop(['gameId', 'playId', 'event', 'specialTeamsPlayType', 'kickerId', 'returnerId', 'isReturn'], axis=1, inplace=True)
    model = MLPRegressor(activation='identity', learning_rate='adaptive', solver='sgd', alpha=0.001, max_iter=900)
    
    y = df['kickReturnYardage'].to_numpy()
    X = df.drop(['kickReturnYardage'], axis=1)
    X.fillna(0.0, inplace=True)
    
    X = normalize(X, axis=0)
    #X = X.to_numpy()
    
        
    model.fit(X, y)
    return model

def tune_neural_model(df):
    df = df.copy()
    df = df.query("isReturn == 1")
    df.drop(['gameId', 'playId', 'event', 'specialTeamsPlayType', 'kickerId', 'returnerId', 'isReturn'], axis=1, inplace=True)
    scores = []
    df = df.sample(frac=1).reset_index(drop=True)
    
    train_df = df.iloc[:-200]
    test_df = df.iloc[-199:]
    
    ytrain = train_df['kickReturnYardage'].to_numpy()
    ytest = test_df['kickReturnYardage'].to_numpy()
    
    Xtrain = train_df.drop(['kickReturnYardage'], axis=1)
    Xtest = test_df.drop(['kickReturnYardage'], axis=1)
    Xtrain.fillna(0.0, inplace=True)
    Xtest.fillna(0.0, inplace=True)
    Xtrain = normalize(Xtrain, axis=0)
    Xtest = normalize(Xtest, axis=0)
    
    hidden_layer_options = [(100,), (50,100), (50, 100, 50)]
    activation_options = ['identity', 'logistic', 'tanh', 'relu']
    solver_options = ['lbfgs', 'sgd', 'adam']
    alpha_options = [0.0001, 0.00005, 0.001]
    learning_rate_options = ['constant', 'invscaling', 'adaptive']
    max_iter_options = [200, 500, 700]
    
    best_score = -10000
    best_options = []
    for activation in tqdm(activation_options):
        for solver in solver_options:
            for alpha in alpha_options:
                for learning_rate in learning_rate_options:
                    for max_iter in max_iter_options:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            scores = []
                            model = MLPRegressor(activation=activation, solver=solver, alpha=alpha, learning_rate=learning_rate, max_iter=max_iter)
                            model.fit(Xtrain, ytrain)
                            for i in range(len(ytest)):
                                yards_res = model.predict([Xtest[i]])[0]
                                yards_diff = (1.0 - abs(((yards_res/2 + int(yards_res)) / yards_res) - ((yards_res/2 + int(ytest[i])) / yards_res))) + 1 / (max(yards_res, ytest[i]))
                                scores.append(yards_diff)
                            overall_acc = sum(scores) / len(scores)
                            if overall_acc > best_score:
                                best_score = overall_acc
                                best_options = [activation, solver, alpha, learning_rate, max_iter]
                                print('New best: {}'.format(best_score))
    print('Best configuration: {}'.format(best_options))
    print('Best score: {}'.format(best_score))
           
def l2_normalize(col):
    col = col.to_numpy()
    col = col.reshape(-1,1)
    col = normalize(col, axis=0)
    col = col.flatten()
    return pd.Series(col)
    
def evaluate_players(params_df, returners_df, return_model, yardage_model):
    pyroe_df = pd.DataFrame(columns=['returnerId', 'PRYOE', 'PRYOE_AVG', 'returnerAvg', 'numReturns'])
    headers = ['gameId', 'playId', 'event', 'specialTeamsPlayType', 'kickerId', 'returnerId', 'isReturn', 'kickReturnYardage', 'possessionTeam', 'season']
    features = [col for col in params_df.columns if col not in headers]
    features.pop(0)
    params_df.fillna(0.0, inplace=True)
    for feature in features:
        params_df[feature] = l2_normalize(params_df[feature])
    for returner in returners_df.iterrows():
        returnerId = returner[1].returnerId
        if not ';' in returnerId:
            returns = params_df.query('returnerId == @returnerId') 
            returner_pryoe = 0
            returner_yds = []
            for _return in returns.iterrows():
                yret = _return[1]['isReturn']
                yyrd = _return[1]['kickReturnYardage']
                X = _return[1].drop(headers)
                X = X.to_numpy()[1:]
                
                predReturn = return_model.predict_proba([X])
                if predReturn[0][1] > 0.6:
                    predYds = yardage_model.predict([X])[0]
                    pryoe = yyrd - predYds
                    returner_pryoe += pryoe
                    returner_yds.append(yyrd)
            try:
                returnerAvg = sum(returner_yds) / len(returner_yds)
                pryoeAvg = returner_pryoe / len(returner_yds)
            except:
                returnerAvg = 0
                pryoeAvg = 0
            row = {'returnerId': returnerId, 'PRYOE': returner_pryoe, 'PRYOE_AVG': pryoeAvg, 'returnerAvg': returnerAvg, 'numReturns': len(returner_yds)}
            pyroe_df = pyroe_df.append(row, ignore_index=True)
    pyroe_df.sort_values(by='PRYOE', axis=0, inplace=True, ascending=False, na_position='last')
    return pyroe_df
    
def build_units_df(games_df, plays_df):
    units_df = games_df[['gameId', 'season']]
    plays_df = plays_df[['gameId', 'possessionTeam']]
    
    units_df = units_df.reset_index(drop=True)
    plays_df = plays_df.reset_index(drop=True)
    
    units_df = units_df.set_index('gameId')
    plays_df = plays_df.set_index('gameId')

    units_df = plays_df.join(units_df, on='gameId', how='inner', rsuffix='_r')
    units_df = units_df.drop_duplicates()
    return units_df

    
def evaluate_units(params_df, return_model, yardage_model):
    unit_pyroe_df = pd.DataFrame(columns=['Team', 'season', 'PRYOE', 'PRYOE_AVG', 'returnAvg', 'numReturns'])
    games_df = pd.read_csv('input/nfl-big-data-bowl-2022/games.csv')
    plays_df = pd.read_csv('input/nfl-big-data-bowl-2022/plays.csv')
    units_df = build_units_df(games_df, plays_df)
    headers = ['gameId', 'playId', 'event', 'specialTeamsPlayType', 'kickerId', 'returnerId', 'isReturn', 'kickReturnYardage', 'possessionTeam', 'season']
    features = [col for col in params_df.columns if col not in headers]
    features.pop(0)
    params_df.fillna(0.0, inplace=True)
    for feature in features:
        params_df[feature] = l2_normalize(params_df[feature])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for unit in tqdm(units_df.iterrows()):
            unit_team = unit[1].possessionTeam
            unit_season = unit[1].season
            returns = params_df.query('possessionTeam == @unit_team & season == @unit_season')
            unit_pryoe = 0
            unit_yds = []
            for _return in returns.iterrows():
                yyrd = _return[1]['kickReturnYardage']
                X = _return[1].drop(headers)
                X = X.to_numpy()[1:]
                predReturn = return_model.predict_proba([X])
                if predReturn[0][1] > 0.6:
                        predYds = yardage_model.predict([X])[0]
                        pryoe = yyrd - predYds
                        unit_pryoe += pryoe
                        unit_yds.append(yyrd)
            try:
                unitAvg = sum(unit_yds) / len(unit_yds)
                pryoeAvg = unit_pryoe / len(unit_yds)
            except:
                unitAvg = 0
                pryoeAvg = 0
            row = {'Team': unit_team, 'season': unit_season,'PRYOE': unit_pryoe, 'PRYOE_AVG': pryoeAvg, 'returnAvg': unitAvg, 'numReturns': len(unit_yds)}
            unit_pyroe_df = unit_pyroe_df.append(row, ignore_index=True)
    unit_pyroe_df.sort_values(by='PRYOE', axis=0, inplace=True, ascending=True, na_position='last', ignore_index=True)
    return unit_pyroe_df
    

def linear_two_step(df):
    df = df.copy()
    scores = []
    df = df.sample(frac=1).reset_index(drop=True)
    train_df = df.iloc[:-300]
    test_df = df.iloc[-299:]
    return_model = fit_logistic_model(train_df)
    yardage_model = fit_linear_model(train_df)
    ytest = test_df[['isReturn', 'kickReturnYardage']].to_numpy()
    Xtest = test_df.drop(['gameId', 'playId', 'event', 'specialTeamsPlayType', 'kickerId', 'returnerId', 'isReturn', 'kickReturnYardage'], axis=1)
    Xtest.fillna(0.0, inplace=True)
    Xtest = normalize(Xtest, axis=0)
    linear_scores = []
    for i in range(len(ytest)):
        return_res = return_model.predict_proba([Xtest[i]])
        return_manual = 0
        if return_res[0][1] > 0.6:
            return_manual = 1
        if return_manual == 0 and ytest[i][0] == 0:
            scores.append(1)
        elif return_manual == 1 and ytest[i][0] == 0:
            scores.append(0)
        elif return_manual == 0 and ytest[i][0] == 1:
            scores.append(0)
        else:
            yards_res = yardage_model.predict([Xtest[i]])[0]
            if ytest[i][1] == 0:
                ytest[i][1] = 0.1
            yards_score = (1.0 - abs(((yards_res/2 + int(yards_res)) / yards_res) - ((yards_res/2 + int(ytest[i][1])) / yards_res))) + 1 / (max(yards_res, ytest[i][1]))
            if yards_score < 0:
                yards_score = 0
            print('predicted: {}'.format(yards_res))
            print('actual: {}'.format(ytest[i][1]))
            print('score: {}'.format(yards_score))
            scores.append(yards_score)
            linear_scores.append(yards_score)
            print('--------------------')
    linear_acc = sum(linear_scores) / len(linear_scores)
    print('Linear Score: {}'.format(linear_acc))
    overall_acc = sum(scores) / len(scores)
    print('Overall Score: {}'.format(overall_acc))
            
        
def write_player_results(pryoe_df, players_df):
    players_df = players_df[['nflId', 'displayName']]
    players_df['returnerId'] = players_df['nflId']
    players_df = players_df.reset_index()
    players_df = players_df.drop(['nflId'], axis=1)
    pryoe_df['returnerId'] = pryoe_df['returnerId'].astype(int)
    players_df['returnerId'] = players_df['returnerId'].astype(int)
    pryoe_df.set_index(['returnerId'], inplace=True)
    players_df.set_index(['returnerId'], inplace=True)
    pryoe_df = pryoe_df.join(players_df, on='returnerId', how='left')
    print(pryoe_df.head())
    pryoe_df.to_csv('player_pryoe.csv')


def main():
    #scouting_df, games_df, plays_df, tracking_df = read_inputs()
    #print(tracking_df.shape[0])
    #params_df, returners_df = encode_columns(plays_df, scouting_df, tracking_df, games_df)
    #params_df.to_csv('features.csv')
    #returners_df.to_csv('returners.csv')
    params_df = pd.read_csv('features.csv')
    returers_df = pd.read_csv('returners.csv')
    #tune_neural_model(params_df)
    return_model = fit_logistic_model(params_df)
    yardage_model = fit_linear_model(params_df)
    #linear_two_step(params_df)
    #pryoe_df = evaluate_players(params_df, returers_df, return_model, yardage_model)
    units_df = evaluate_units(params_df, return_model, yardage_model)
    units_df.to_csv('units.csv')
    #players_df = pd.read_csv('input/nfl-big-data-bowl-2022/players.csv')
    #write_player_results(pryoe_df, players_df)
   
    
    

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    main()