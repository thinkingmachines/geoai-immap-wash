import pandas as pd
from math import sqrt
import numpy as np
from tqdm import tqdm

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.externals import joblib
from sklearn.base import clone

data_dir = '../data/'
model_dir = data_dir + 'models/'
scaler_dir = data_dir + 'scalers/'
output_dir = data_dir + 'outputs/'

def fit_test(clf, indicator, test_area, scale = True):
    global df_train, y_true, y_pred
    raw_ = raw.copy()

    df_train = raw_.query(f"adm1_name != '{test_area}'")
    df_test = raw_.query(f"adm1_name == '{test_area}'")
    y = df_train[indicator]
    y_true = df_test[indicator]
    if scale:
        scaler = RobustScaler()
        scaler.fit(df_train[features])
        df_train = scaler.transform(df_train[features])
        df_test = scaler.transform(df_test[features])
    X = df_train
    X_test = df_test
    
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    metrics = {
        'correlation': np.corrcoef(y_true, y_pred)[0,1],
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': sqrt(mean_squared_error(y_true, y_pred)),
    }
    return clf, metrics, scaler

def fit_models():
    recs = []
    lr = Ridge(alpha=1.0)
    rf = RandomForestRegressor(random_state=42)
    clfs = [lr, rf]

    for indicator in tqdm(indicators):
        for model in clfs:
            for area in test_areas:
                unfitted = clone(model)
                fitted, metrics, scaler = fit_test(unfitted, indicator, area)
                model_type = str(type(fitted)).split('.')[-1].replace("'>", '')
                rec = (
                    indicator,
                    area,
                    model_type,
                    metrics['correlation'],
                    metrics['r2'],
                    metrics['mae'],
                    metrics['rmse']
                )
                recs.append(rec)
                joblib.dump(fitted, model_dir + indicator + '_' + area + '_' + model_type + '.pkl')
                joblib.dump(scaler, scaler_dir + indicator + '_' + area + '_' + model_type + '.pkl')

    cols = ['indicator', 'area', 'model', 'correlation', 'r2_score', 'mae', 'rmse']
    results = pd.DataFrame(recs, columns = cols)

    return results

def predict_on_holdout(results, indicator, area, model_type):
    row = (results
        .query(f"indicator == '{indicator}'")
        .query(f"model == '{model_type}'")
        .query(f"area == '{area}'"))
    model = joblib.load(model_dir + indicator + '_' + area + '_' + model_type + '.pkl')
    scaler = joblib.load(scaler_dir + indicator + '_' + area + '_' + model_type + '.pkl')
    sub = raw.query(f"adm1_name == '{area}'")
    X = sub[features]
    X = scaler.transform(X)
    sub['pred_' + indicator] = model.predict(X)
    keep_cols = ['adm1_name', 'id', 'geometry_x', indicator, 'pred_' + indicator]
    sub[keep_cols].to_csv(output_dir + indicator + '_' + area + '_' + model_type + '.csv', index = False)
