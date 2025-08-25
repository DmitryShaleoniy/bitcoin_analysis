import pandas as pd

hash_rate_df = pd.read_json('hash-rate.json')
hash_rate_df = hash_rate_df.drop(columns=['metric1', 'metric2', 'timespan', 'average', 'type'])

hash_rate_df['date'] = hash_rate_df['hash-rate'].apply(lambda x: x['x'])
hash_rate_df['hash-rate'] = hash_rate_df['hash-rate'].apply(lambda x: x['y'])
hash_rate_df['market-price'] = hash_rate_df['market-price'].apply(lambda x: x['y'])
hash_rate_df['date'] = pd.to_datetime(hash_rate_df['date'], unit='ms')
df1 = hash_rate_df


hash_rate_df = pd.read_json('hash-rate(1).json')
hash_rate_df = hash_rate_df.drop(columns=['metric1', 'metric2', 'timespan', 'average', 'type'])

hash_rate_df['date'] = hash_rate_df['hash-rate'].apply(lambda x: x['x'])
hash_rate_df['hash-rate'] = hash_rate_df['hash-rate'].apply(lambda x: x['y'])
hash_rate_df['market-price'] = hash_rate_df['market-price'].apply(lambda x: x['y'])
hash_rate_df['date'] = pd.to_datetime(hash_rate_df['date'], unit='ms')
df2 = hash_rate_df


df1 = df1.merge(df2, on='date', how='outer').sort_values(by='date')
df1['hash-rate'] = df1['hash-rate_x'].fillna(df1['hash-rate_y']).astype(float)
df1['market-price'] = df1['market-price_x'].fillna(df1['market-price_y']).astype(float)
df1 = df1.drop(columns=['hash-rate_x', 'hash-rate_y'])
df1 = df1.drop(columns=['market-price_x', 'market-price_y'])

print(df1.info())
print(df1.head(10))
print(df1.tail(10))


