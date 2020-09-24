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
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import seaborn as sns

from settings import data_dir, model_dir, scaler_dir, preds_dir

def calculate_metrics(y_true, y_pred):
    '''
    Calculates metrics of accuracy between actual values and model predicted values.
    
    Args
        y_true (list): Actual WASH indicator values
        y_pred (list): Model predicted values
    Returns
        (dict): Dictionary of correlation, r-squared, mean absolute error, and root mean squared error
    '''
    return {
        'correlation': pearsonr(y_true, y_pred)[0],
        'r2': pearsonr(y_true, y_pred)[0]**2,#r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': sqrt(mean_squared_error(y_true, y_pred)),
    }

def _fit_with_one_holdout(df, clf, features, indicator, test_area, scale = True):
    '''
    Trains input model for input indicator using input features, holding out input test_area
    
    Args
        df (dataframe): dataset source of training data
        clf (sklearn model): unfitted model
        features (list of str): list of features used for training
        indicator (str): indicator being modelled
        test_area (str): area to serve as testing data
        scale (bool): scale features based sklearn RobustScaler
    Returns 
        clf (sklearn model): fitted model
        metrics (dict): dictionary of accuracies from calculate_metrics()
        scaler (sklearn object): fitted scaler object on train data
    '''
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
    Wrapper function that:
    - Runs _fit_with_one_holdout() over test areas
    - Creates sklearn model objects, 1 Ridge Regressor and 1 Random Forest Regressor
    - Saves models and scalers as pkl files to model and scaler dirs specified in settings.py
    
    Args
        df (dataframe): dataset source of training data
        features (list of str): list of features used for training
        indicators (list of str): list of indicators being modelled
        test_areas (list of str): list of areas to serve as testing data
        prefix (str): prefix added to filename used in saving pkl files
    Returns
        results (DataFrame): aggregated table of accuracies, broken down based on indicator, model type, and test area
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
    Wrapper function that
    - Predicts using a saved model and scaler from pkl files
    - Outputs a csv file of grid-level predictions
    
    Args
        df (dataframe): dataset source of training data
        features (list of str): list of features used for training
        indicator (str): indicator being modelled
        area (str): area to serve as testing data
        model_type (str): type of sklearn model e.g. RandomForestRegressor
        prefix (str): prefix added to filename used in saving pkl files
    Returns
        None
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
    '''
    Finds indicator text in filename 
    
    Args
        text (str): filename, e.g. 'all_perc_hh_no_toilet_bogot_dc_RandomForestRegressor.csv'
    Returns
        ind (str): indicator text, e.g. 'toilet'
    '''
    #text = 
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
    Wrapper function that
    - Runs _predict_one_holdout_area() across holdout areas
    - Consolidates to one csv file of grid-level predictions
    
    Args
        df (dataframe): dataset source of training data
        test_areas (list of str): list of areas to serve as testing data
        features (list of str): list of features used for training
        indicators (list of str): list of indicators being modelled
        prefix (str): prefix added to filename used in saving pkl files
    Returns
        None
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
    '''
    Prints out accuracy metrics and scatter plots of actual value vs predicted
    
    Args
        indicators (list of str): list of indicators to evaluate on
    Returns
        None
    '''
    out_file = f'{prefix}_predictions.csv'
    for indicator in indicators:
        print(indicator)
        ind = indicator.split('_')[3]
        df = pd.read_csv(data_dir + out_file).query(f"indicator == '{ind}'")
        print(calculate_metrics(df['y_true'], df['y_pred']))
        df.plot(x = 'y_true', y = 'y_pred', kind = 'scatter', figsize = (5,5), xlim = (0,1), ylim = (0,1), title = indicator)
        
def check_nulls_and_outliers(df, columns = None):
    '''
    1. Find rows missing some values
    2. Find outliers
    '''
    if columns is None:
        columns = df.columns
    
    sub = df[columns]
    rows, cols = df.shape
    print(f'Total rows: {rows}')
    print('Variables with missing values:')
    print(sub.describe().transpose().query(f"count < {rows}")[['count', 'mean', 'std', 'min', 'max']])
    scaler = MinMaxScaler()
    scaler.fit(sub)
    scaler.transform(sub)
    sub.boxplot(figsize=(12,4))

def average_results(df, iterate_over = 'adm1_name', prefix = 'all'):
    '''
    Calculate accuracies by calculating per subgroup, then averaging across all subgroups
    
    Args
        df (DataFrame): source of training data
        iterate_over (str): column over which to iterate, i.e. subgroups used for calculation
        prefix (str): string prepended to saved csv file
    Returns
        None
    '''
    inds = list(df.indicator.unique())
    dfs = []
    for ind in inds:
        ind_ = ind.replace('perc_hh_no_', '').split('_')[0]
        print(f"Access to {ind_}")
        sub1 = df.query(f"indicator == '{ind}'")
        list_ = list(sub1[iterate_over].unique())
        recs = []
        for item in list_:
            sub2 = sub1.query(f"{iterate_over} == '{item}'")
            metrics_ = calculate_metrics(sub2['y_true'], sub2['y_pred'])
            recs.append((ind_, item, metrics_['correlation'], metrics_['r2'], metrics_['rmse']))
        df_ = pd.DataFrame(recs, columns = ['indicator', iterate_over, 'correlation', 'r2', 'rmse']).set_index(iterate_over)
        print(df_.mean())
        dfs.append(df_)
    res = pd.concat(dfs, axis = 0)
    # res.to_csv(data_dir + prefix + '_' + indicator + '_grouped_results.csv', index = False)

def consolidate_results(df, prefix = 'all'):
    '''
    Calculate accuracies by consolidating all predictions
    
    Args
        df (DataFrame): source of training data
        iterate_over (str): column over which to iterate, i.e. subgroups used for calculation
        prefix (str): string prepended to saved csv file
    Returns
        None
    '''
    inds = list(df.indicator.unique())
    for ind in inds:
        ind_ = ind.replace('perc_hh_no_', '')
        print(f"Access to {ind_}")
        sub = df.query(f"indicator == '{ind}'")
        print(calculate_metrics(sub['y_true'], sub['y_pred']))

def fit_with_randomsplit(df, clf, features, indicators, scale = True, n_splits = 5, prefix = 'all'):
    '''
    Trains input model for input indicator using input features, using randomly selected 20% of rows
    
    Args
        df (dataframe): dataset source of training data
        clf (sklearn model): unfitted model
        features (list of str): list of features used for training
        indicators (list of str): indicators being modelled
        scale (bool): scale features based sklearn RobustScaler
        n_splits (int): 
    Returns 
        None
    '''
    global X
    raw_ = df.copy()
    dfs = []
    
    for indicator in tqdm(indicators):
        #print(indicator)
        X = raw_[features]
        y = raw_[indicator]

        kf = KFold(n_splits=n_splits, shuffle = True, random_state = 42)
        c = 0
        for train_index, test_index in kf.split(X):
            #print(c)
            c+=1
            clf_ = clone(clf)
            X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
            y_train, y_test = y[train_index], y[test_index]
            if scale:
                scaler = RobustScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

            clf_.fit(X_train, y_train)
            y_pred = clf_.predict(X_test)
            df_ = pd.DataFrame({
                'split_id': str(c),
                'indicator': indicator,
                'grid_id': raw_.loc[test_index, 'id'],
                'adm1_name': raw_.loc[test_index, 'adm1_name'],
                'y_true': y_test,
                'y_pred': y_pred,
            })
            dfs.append(df_)
    
    cons_df = pd.concat(dfs, axis = 0)
    cons_df.to_csv(data_dir + prefix + '_randomsplit_results.csv', index = False)
    return cons_df

def model_rollout(train_df, test_df, fit = False, save = False):
    """
    Fit model and return test_df with predictions
    
    Args
        train_df (dataframe): data to train model on (2018)
        test_df (dataframe): data to predict on (2019/2020)
        fit (bool): if False, load saved pkl file of model; else fit on train_df
        save (bool): if True, save model and scaler as pkl files
    Returns
        test_df (dataframe): original test_df but with predictions per indicator
        top_features (dataframe): top features sorted by random forest importance
    """

    global clf
    clf = RandomForestRegressor(random_state=42)
    
    feats = []
    for indicator in tqdm(indicators):

        avg_metrics = {'correlation':[], 'r2':[], 'mae':[], 'rmse':[]}
        X_train, y_train = train_df[features], train_df[indicator]
        X_test = test_df[features]
        scaler = RobustScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        if fit:
            clf.fit(X_train, y_train)
        else:
            clf = joblib.load(model_dir + 'model_' + indicator + '_2018_250mv2.pkl')
        
        y_pred = clf.predict(X_test)
        test_df['pred_' + indicator] = y_pred
        
        feature_importances = pd.DataFrame({'feature': list(train_df[features].columns)
                                            , 'importance': list(clf.feature_importances_)})
        top_features = (feature_importances
                            .sort_values(by=['importance'], ascending = False))
        top_features['indicator'] = indicator
        feats.append(top_features)
        
        if save:
            joblib.dump(clf, model_dir + 'model_' + indicator + '_2018_250mv2.pkl')
            joblib.dump(scaler, scaler_dir + 'scaler_2018_250mv2.pkl') # writes 3 times, but might be cleaner to read
    
    return test_df, pd.concat(feats, axis = 0).reset_index(drop = True)