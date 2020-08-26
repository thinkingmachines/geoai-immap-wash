import pandas as pd
from math import sqrt
import numpy as np
from tqdm import tqdm
import os
import glob
import re

from scipy.stats import pearsonr, spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.externals import joblib
from sklearn.base import clone

from settings import data_dir, model_dir, scaler_dir, preds_dir

def calculate_metrics(y_true, y_pred):
    return {
        'correlation': pearsonr(y_true, y_pred)[0],
        'r2': pearsonr(y_true, y_pred)[0]**2,#r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': sqrt(mean_squared_error(y_true, y_pred)),
    }

def _fit_with_one_holdout(df, clf, features, indicator, test_area, scale = True):
    '''Returns fitted model, metrics, and scaler'''
    global df_train, y_true, y_pred
    raw_ = df.copy()

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
    metrics = calculate_metrics(y_true, y_pred)
    return clf, metrics, scaler

def fit_models(df, features, indicators, test_areas, prefix = ''):
    '''
    Loops model fitting over test areas
    Saves models and scalers as pkl files
    '''
    recs = []
    lr = Ridge(alpha=1.0, random_state=42)
    rf = RandomForestRegressor(random_state=42)
    clfs = [lr, rf]
    
    print('Fitting using the following features:')
    print(features)
    
    print('Evaluating on the following test areas:')
    print(test_areas)
    
    # remove previous results
    fileList = glob.glob(f'{preds_dir}{prefix}_perc_hh_no*.csv')
    for filePath in fileList:
        os.remove(filePath)

    for indicator in tqdm(indicators):
        for model in clfs:
            for area in test_areas:
                unfitted = clone(model)
                fitted, metrics, scaler = _fit_with_one_holdout(df, unfitted, features, indicator, area)
                model_type = str(type(fitted)).split('.')[-1].replace("'>", '')
                rec = (
                    indicator,
                    area,
                    model_type,
                    metrics['r2'],
                    metrics['correlation'],
                    metrics['mae'],
                    metrics['rmse']
                )
                recs.append(rec)
                joblib.dump(fitted, model_dir + prefix + '_' + indicator + '_' + area + '_' + model_type + '.pkl')
                joblib.dump(scaler, scaler_dir + prefix + '_' + indicator + '_' + area + '_' + model_type + '.pkl')

    cols = ['indicator', 'area', 'model', 'r2_score', 'correlation', 'mae', 'rmse']
    results = pd.DataFrame(recs, columns = cols)

    return results

def _predict_one_holdout_area(df, features, indicator, area, model_type, prefix = ''):
    '''
    Predicts using a saved model and scaler from pkl files
    Outputs a csv file of grid-level predictions
    '''
    model = joblib.load(model_dir + prefix + '_' + indicator + '_' + area + '_' + model_type + '.pkl')
    scaler = joblib.load(scaler_dir + prefix + '_' + indicator + '_' + area + '_' + model_type + '.pkl')
    sub = df.query(f"adm1_name == '{area}'")
    X = sub[features]
    X = scaler.transform(X)
    sub['pred_' + indicator] = model.predict(X)
    keep_cols = ['adm1_name', 'id', 'geometry', indicator, 'pred_' + indicator]
    sub[keep_cols].to_csv(preds_dir + prefix + '_' + indicator + '_' + area + '_' + model_type + '.csv', index = False)

def find_ind(text):
    "finds indicator text in filename"
    #text = 'all_perc_hh_no_toilet_bogot_dc_RandomForestRegressor.csv'
    m1 = re.search('water', text)
    m2 = re.search('toilet', text)
    m3 = re.search('sewage', text)

    if m1 is None:
        if m2 is None:
            ind = m3[0]
        else:
            ind = m2[0]
    else:
        ind = m1[0]
    return ind

def predict_on_holdout_areas(df, test_areas, features, indicators, prefix = ''):
    '''
    Loops prediction across holdout areas
    Consolidates to one csv file of grid-level predictions
    '''
    out_file = f'{prefix}_predictions.csv'
    for indicator in tqdm(indicators):
        for model_type in ['RandomForestRegressor']:
            for area in test_areas:
                _predict_one_holdout_area(df, features, indicator, area, model_type, prefix)

    # consolidate hold out results to one df
    files_ = glob.glob(f'{preds_dir}{prefix}_perc_hh_no*.csv')

    dfs = []
    for f in files_:
        df = pd.read_csv(f)
        df.columns = ['adm1_name', 'id', 'geometry', 'y_true', 'y_pred']
        df['indicator'] = find_ind(f)
        dfs.append(df)

    mega_df = pd.concat(dfs, axis = 0)
    mega_df['absdiff'] = abs(mega_df['y_true'] - mega_df['y_pred'])
    mega_df.to_csv(data_dir + out_file, index= False)
    
def evaluate_results(indicators, prefix = ''):
    out_file = f'{prefix}_predictions.csv'
    for indicator in indicators:
        print(indicator)
        ind = indicator.split('_')[3]
        df = pd.read_csv(data_dir + out_file).query(f"indicator == '{ind}'")
        print(calculate_metrics(df['y_true'], df['y_pred']))
        df.plot(x = 'y_true', y = 'y_pred', kind = 'scatter', figsize = (5,5), xlim = (0,1), ylim = (0,1), title = indicator)