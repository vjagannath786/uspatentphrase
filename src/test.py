import pandas as pd
import numpy as np
import config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold

df = pd.read_csv(config.train_file)

'''
df['score_map'] = df['score'].map({0.00: 0, 0.25: 1, 0.50: 2, 0.75: 3, 1.00: 4})

encoder = LabelEncoder()
df['anchor_map'] = encoder.fit_transform(df['anchor'])

kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for n, (_, valid_index) in enumerate(kf.split(df, df['score_map'], groups=df['anchor_map'])):
    df.loc[valid_index, 'fold'] = int(n)

df['fold'] = df['fold'].astype(int)

df.to_csv('../train_gskffolds.csv', index=False)
'''
df1 = pd.read_csv(config.titles_file)

final_df = df.merge(df1, left_on='context', right_on='code', how='left')

final_df = final_df.reset_index(drop=True)

print(final_df.loc[final_df['title'].isnull()])