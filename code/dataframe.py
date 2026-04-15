import pandas as pd
import os

# Get the absolute path of the directory where dataframe.py is found
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Finds /data directory by going up one level from /code
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))

def get_path(filename):
    return os.path.join(DATA_DIR, filename)

# Cleaning and pre-processing function
def clean_stat_df(filename, prefix):
    path = get_path(filename)
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None
    df = pd.read_csv(path)
    
    # Handle Pro-Football-Reference header offset
    if 'Tm' not in df.columns:
        df.columns = df.iloc[0]
        df = df[1:].copy()
    
    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]
    df = df[df['Tm'].notna()]
    df = df[~df['Tm'].isin(['Tm', 'Avg Team', 'League Total', 'Avg Tm/G'])]
    df['Tm'] = df['Tm'].str.strip()
    
    # Prefix columns
    cols_to_prefix = [c for c in df.columns if c not in ['Tm', 'G', 'Rk']]
    df.rename(columns={c: f"{prefix}_{c}" for c in df.columns if c in cols_to_prefix}, inplace=True)
    
    return df[['Tm'] + [c for c in df.columns if c != 'Tm']]

# Combine all data files together

stats_files = {
    "Passing": "2024-Team-Passing-Stats.csv",
    "Rushing": "2024-Team-Rushing-Stats.csv",
    "Scoring": "2024-Team-ScoringOffense-Stats.csv",
    "Offense": "2024-Team-Offense-Stats.csv",
    "Returns": "2024-Team-Returns-Stats.csv",
    "Punting": "2024-Team-Punting-Stats.csv",
    "Kicking": "2024-Team-Kicking-Stats.csv"
}

team_stats = None
for prefix, file in stats_files.items():
    df = clean_stat_df(file, prefix)
    if df is not None:
        if team_stats is None:
            team_stats = df
        else:
            cols_to_drop = [c for c in ['G', 'Rk'] if c in team_stats.columns and c in df.columns]
            team_stats = pd.merge(team_stats, df.drop(columns=cols_to_drop), on='Tm', how='outer')

if team_stats is None:
    print("Not files were loaded") # If this msg appears check file structure
    exit()

# Processing Season Results
results_path = get_path("2024-Season-Results.csv")

if not os.path.exists(results_path):
    print(f"Error: {results_path} not found.")
    exit()

results = pd.read_csv(results_path)
results = results[results['Winner/tie'].notna() & results['Loser/tie'].notna()]
results = results[results['Winner/tie'] != 'Winner/tie']

def get_home_away(row):
    if row['Unnamed: 5'] == '@':
        return pd.Series({
            'Home': row['Loser/tie'], 'Away': row['Winner/tie'],
            'Home_Score': row['Pts.1'], 'Away_Score': row['Pts']
        })
    else:
        return pd.Series({
            'Home': row['Winner/tie'], 'Away': row['Loser/tie'],
            'Home_Score': row['Pts'], 'Away_Score': row['Pts.1']
        })

home_away_info = results.apply(get_home_away, axis=1)
results = pd.concat([results, home_away_info], axis=1)

# Create final dataset
final_df = pd.merge(results, team_stats.add_prefix('Home_'), left_on='Home', right_on='Home_Tm', how='left')
final_df = pd.merge(final_df, team_stats.add_prefix('Away_'), left_on='Away', right_on='Away_Tm', how='left')

final_df['Home_Win'] = (final_df['Home_Score'] > final_df['Away_Score']).astype(int)

output_path = get_path("ML_Ready_NFL_2024.csv")
final_df.to_csv(output_path, index=False)
print(f"\nUnified dataset saved to: {output_path}")

# Prepare ML objects
leakage_cols = [
    'Week', 'Day', 'Date', 'Time', 'Winner/tie', 'Unnamed: 5', 'Loser/tie', 
    'Date.1', 'Pts', 'Pts.1', 'YdsW', 'TOW', 'YdsL', 'TOL', 'Home', 'Away', 
    'Home_Score', 'Away_Score', 'Home_Tm', 'Away_Tm', 'Home_Win'
]

X = final_df.drop(columns=leakage_cols)
y = final_df['Home_Win']
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
print(f"Total Matches Processed: {len(final_df)}")
print(f"Features for Training: {X.shape[1]}")