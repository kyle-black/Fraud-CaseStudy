import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

def get_data(datafile):
    df = pd.read_json(datafile)
    df['fraud'] = df['acct_type'].str.contains('fraud')
    clean_df = prepare_data(df)
    scaler = StandardScaler()
    X = scaler.fit_transform(clean_df.drop(columns='fraud').values)
    y = clean_df['fraud'].values.astype(int)
    return X, y, scaler

def prepare_data(df):
    num_cols = (df.dtypes != object).values
    num_df = df.iloc[:,num_cols].copy()
    
    num_df['fraud'] = df['fraud']
    num_df['previous_payouts'] = df['previous_payouts'].apply(lambda x: sum([payout['amount'] for payout in x]))
    
    num_df2 = pd.get_dummies(num_df, columns=['delivery_method', 'has_header', 'user_type'], dummy_na=True, drop_first=True, dtype=np.int64)
    num_df2['org_facebook'].fillna(np.mean(num_df2['org_facebook']), inplace=True)
    num_df2['org_twitter'].fillna(np.mean(num_df2['org_twitter']), inplace=True)
    num_df2['sale_duration'].fillna(np.mean(num_df2['sale_duration']), inplace=True)
    num_df2['event_published'].fillna(np.mean(num_df2['event_published']), inplace=True)

    num_df2['has_lat_long'] = num_df2.apply(lambda x: has_lat_long(x['venue_latitude'], x['venue_longitude']), axis=1)

    num_df3 = num_df2.drop(columns=['venue_latitude', 'venue_longitude'])

    return num_df3

def has_lat_long(lat, long):
    if lat == 0 and long == 0:
        return 0
    if lat == np.nan and long == np.nan:
        return 0
    else:
        return 1


if __name__ == '__main__':
    X, y, scaler = get_data('data/data.json')
    model = GradientBoostingClassifier(subsample=0.7, n_estimators=1000, max_depth=10, learning_rate=0.03, max_features='auto')
    model.fit(X, y)
    with open('models/GBCmodel.pkl', 'wb') as f:
        # Write the model to a file.
        pickle.dump(model, f)
    
    with open('models/GBCmodelScaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)