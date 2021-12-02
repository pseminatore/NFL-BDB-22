import pandas as pd
import plotly.express as px



df = pd.read_csv('data/PFFScoutingData.csv')
punt_df = df[['gameId', 'playId', 'operationTime', 'hangTime', 'kickType']].query('kickType == "N" | kickType == "A" | kickType == "R"')
punt_avgs = punt_df[['kickType', 'operationTime', 'hangTime']].groupby('kickType').mean()
print(punt_df)
print(punt_avgs)
temp = punt_df.mean(axis=0, numeric_only=True)
opFig = px.violin(punt_df, x="kickType", y='operationTime')
hangFig = px.violin(punt_df, x="kickType", y='hangTime')

opFig.show()
hangFig.show()