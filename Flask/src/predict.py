import pickle
import pandas as pd
import numpy as np
import model
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve


class LogReg():

    def __init__(self, penalty='l2', class_weight={0:9, 1:91}, C=1.0, fit_intercept=True):
        self.model = LogisticRegression(penalty=penalty, class_weight=class_weight, C=C, fit_intercept=fit_intercept, solver='liblinear')

    def fit(self, X, y):
        self.model.fit(X,y)
        self.coef_ = self.model.coef_
        self.classes_ = self.model.classes_
        self.class_weight = self.model.class_weight

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)

    def get_scores(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        acc = accuracy_score(y_test,y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        recall = recall_score(y_test,y_pred)
        return acc, conf_mat, roc_auc, recall

def get_example_X_y(datafile, scaler):
    df = pd.read_csv(datafile, index_col=0)
    df['fraud'] = df['acct_type'].str.contains('fraud')
    clean_df = prepare_data(df)
    X = scaler.transform(clean_df.drop(columns='fraud').values)
    y = clean_df['fraud'].values.astype(int)
    return X, y

def prepare_data(df):
    num_cols = (df.dtypes != object).values
    num_df = df.iloc[:,num_cols].copy()
    
    num_df['fraud'] = df['fraud']
    num_df['previous_payouts'] = df['previous_payouts'].apply(lambda x: ast.literal_eval(x))
    num_df['previous_payouts'] = num_df['previous_payouts'].apply(lambda x: sum([payout['amount'] for payout in x]))
        
    num_df2 = one_hot(num_df)
    
    num_df2['org_facebook'].fillna(np.mean(num_df2['org_facebook']), inplace=True)
    num_df2['org_twitter'].fillna(np.mean(num_df2['org_twitter']), inplace=True)
    num_df2['sale_duration'].fillna(np.mean(num_df2['sale_duration']), inplace=True)
    num_df2['event_published'].fillna(np.mean(num_df2['event_published']), inplace=True)

    num_df2['has_lat_long'] = num_df2.apply(lambda x: has_lat_long(x['venue_latitude'], x['venue_longitude']), axis=1)

    num_df3 = num_df2.drop(columns=['venue_latitude', 'venue_longitude'])

    final_df = num_df3[['approx_payout_date', 'body_length', 'channels', 'event_created',
       'event_end', 'event_published', 'event_start', 'fb_published', 'gts',
       'has_analytics', 'has_logo', 'name_length', 'num_order', 'num_payouts',
       'object_id', 'org_facebook', 'org_twitter', 'sale_duration',
       'sale_duration2', 'show_map', 'user_age', 'user_created', 'fraud',
       'previous_payouts', 'delivery_method_1.0', 'delivery_method_3.0',
       'delivery_method_nan', 'has_header_1.0', 'has_header_nan',
       'user_type_2.0', 'user_type_3.0', 'user_type_4.0', 'user_type_5.0',
       'user_type_103.0', 'user_type_nan', 'has_lat_long']]
    return final_df

def has_lat_long(lat, long):
    if lat == 0 and long == 0:
        return 0
    if lat == np.nan and long == np.nan:
        return 0
    else:
        return 1

def one_hot(df):
    df['delivery_method_1.0'] = (df['delivery_method'] == 1).astype(int)
    df['delivery_method_3.0'] = (df['delivery_method'] == 3).astype(int)
    df['delivery_method_nan'] = (df['delivery_method'] == np.nan).astype(int)

    df['has_header_1.0'] = (df['has_header'] == 1).astype(int)
    df['has_header_nan'] = (df['has_header'] == np.nan).astype(int)

    df['user_type_2.0'] = (df['user_type'] == 2).astype(int)
    df['user_type_3.0'] = (df['user_type'] == 3).astype(int)
    df['user_type_4.0'] = (df['user_type'] == 4).astype(int)
    df['user_type_5.0'] = (df['user_type'] == 5).astype(int)
    df['user_type_103.0'] = (df['user_type'] == 103).astype(int)
    df['user_type_nan'] = (df['user_type'] == np.nan).astype(int)

    return df.drop(columns=['delivery_method', 'has_header', 'user_type'])



if __name__ == "__main__":
    with open('models/LRmodel.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('models/LRmodelScaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    X, y = get_example_X_y('data/test_script_examples.csv', scaler)
    print(model.predict_proba(X)[np.random.randint(low=0, high=25)])