import pandas as pd
import numpy as np
import config

df = pd.read_csv(config.train_file)

df1 = pd.read_csv(config.titles_file)

final_df = df.merge(df1, left_on='context', right_on='code', how='left')

final_df = final_df.reset_index(drop=True)


print(final_df.loc[(final_df['anchor'] == 'overall weight') & (final_df['target'] == 'total of weight'),['anchor','title','target','score']])