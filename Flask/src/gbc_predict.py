import pickle
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score

def get_example_X_y(row, scaler):
    #df = pd.read_csv(datafile, index_col=0)
    clean_df = prepare_data(row)
    X = scaler.transform(clean_df)
    return X

def prepare_data(df):
    num_cols = (df.dtypes != object).values
    num_df = df.iloc[:,num_cols].copy()
    
    if 'previous_payouts' in df:
        num_df['previous_payouts'] = df['previous_payouts'].apply(lambda x: sum([payout['amount'] for payout in x]))
    else:
        num_df['previous_payouts'] = 5145.
        
    num_df2 = one_hot(num_df)
    
    num_df2['org_facebook'].fillna(np.mean(num_df2['org_facebook']), inplace=True)
    num_df2['org_twitter'].fillna(np.mean(num_df2['org_twitter']), inplace=True)
    num_df2['sale_duration'].fillna(np.mean(num_df2['sale_duration']), inplace=True)
    num_df2['event_published'].fillna(np.mean(num_df2['event_published']), inplace=True)

    if 'venue_latitude' in num_df2:
        num_df2['has_lat_long'] = num_df2.apply(lambda x: has_lat_long(x['venue_latitude'], x['venue_longitude']), axis=1)
        num_df3 = num_df2.drop(columns=['venue_latitude', 'venue_longitude'])
    else:
        num_df2['has_lat_long'] = 0
        num_df3 = num_df2
    
    column_list = ['approx_payout_date', 'body_length', 'channels', 'event_created',
       'event_end', 'event_published', 'event_start', 'fb_published', 'gts',
       'has_analytics', 'has_logo', 'name_length', 'num_order', 'num_payouts',
       'object_id', 'org_facebook', 'org_twitter', 'sale_duration',
       'sale_duration2', 'show_map', 'user_age', 'user_created',
       'previous_payouts', 'delivery_method_1.0', 'delivery_method_3.0',
       'delivery_method_nan', 'has_header_1.0', 'has_header_nan',
       'user_type_2.0', 'user_type_3.0', 'user_type_4.0', 'user_type_5.0',
       'user_type_103.0', 'user_type_nan', 'has_lat_long']
    
    for col in column_list:
        if col not in num_df3:
            num_df3[col] = 0

    final_df = num_df3[column_list]
    return final_df

def has_lat_long(lat, long):
    if lat == 0 and long == 0:
        return 0
    if lat == np.nan and long == np.nan:
        return 0
    else:
        return 1

def one_hot(df):
    if 'delivery_method' in df:
        df['delivery_method_1.0'] = (df['delivery_method'] == 1).astype(int)
        df['delivery_method_3.0'] = (df['delivery_method'] == 3).astype(int)
        df['delivery_method_nan'] = (df['delivery_method'] == np.nan).astype(int)
        df = df.drop(columns='delivery_method')
    else:
        df['delivery_method_1.0'] = 0
        df['delivery_method_3.0'] = 0
        df['delivery_method_nan'] = 1

    if 'has_header' in df:
        df['has_header_1.0'] = (df['has_header'] == 1).astype(int)
        df['has_header_nan'] = (df['has_header'] == np.nan).astype(int)
        df = df.drop(columns='has_header')
    else:
        df['has_header_1.0'] = 0
        df['has_header_nan'] = 1

    if 'user_type' in df:
        df['user_type_2.0'] = (df['user_type'] == 2).astype(int)
        df['user_type_3.0'] = (df['user_type'] == 3).astype(int)
        df['user_type_4.0'] = (df['user_type'] == 4).astype(int)
        df['user_type_5.0'] = (df['user_type'] == 5).astype(int)
        df['user_type_103.0'] = (df['user_type'] == 103).astype(int)
        df['user_type_nan'] = (df['user_type'] == np.nan).astype(int)
        df = df.drop(columns='user_type')
    else:
        df['user_type_2.0'] = 0
        df['user_type_3.0'] = 0
        df['user_type_4.0'] = 0
        df['user_type_5.0'] = 0
        df['user_type_103.0'] = 0
        df['user_type_nan'] = 1
    return df



if __name__ == "__main__":
    with open('models/GBCmodel.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('models/GBCmodelScaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    X = get_example_X_y('data/test_script_examples.csv', scaler)
    print(model.predict_proba(X))
    
    
    # y_pred = model.predict(X)
    # print(f'Accuracy Score: {accuracy_score(y,y_pred)}')
    # print(f'Confusion Matrix: \n{confusion_matrix(y, y_pred)}')
    # print(f'Area Under Curve: {roc_auc_score(y, y_pred)}')
    # print(f'Recall score: {recall_score(y,y_pred)}')
    # print(f'Precision score: {precision_score(y,y_pred)}')
    # print(f'F1 score: {f1_score(y,y_pred)}')