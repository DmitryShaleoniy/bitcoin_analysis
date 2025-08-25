import pandas as pd

hash_rate_df = pd.read_json('hash-rate.json')
hash_rate_df = hash_rate_df.drop(columns=['metric1', 'metric2', 'timespan', 'average', 'type'])

hash_rate_df['date'] = hash_rate_df['hash-rate'].apply(lambda x: x['x'])
hash_rate_df['hash-rate'] = hash_rate_df['hash-rate'].apply(lambda x: x['y'])
hash_rate_df['market-price'] = hash_rate_df['market-price'].apply(lambda x: x['y'])
hash_rate_df['date'] = pd.to_datetime(hash_rate_df['date'], unit='ms')

print(hash_rate_df.info())
print(hash_rate_df.head(10))
print(hash_rate_df.tail(10))


