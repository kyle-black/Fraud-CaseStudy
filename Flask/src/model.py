import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve
from sklearn.metrics import f1_score

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
    model = LogReg()
    model.fit(X, y)
    with open('models/LRmodel.pkl', 'wb') as f:
        # Write the model to a file.
        pickle.dump(model, f)
    
    with open('models/LRmodelScaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)