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

from settings import data_dir, model_dir, scaler_dir, preds_dir, features, indicators

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

    
def _aggregate_by_metro_area():
    '''
    Aggregates predictions by metro area. Content originally from 20200914_check_trends.ipynb, Section 'check trends'
    '''
    
    import pandas as pd
    import re

    def clean_name(text):
        return (re.sub('[^a-z ]','', text.lower()).replace(' ', '_')
                .replace('area_metropolitana_de_', '')
                .replace('area_metropolitana_del_', ''))

    wash18 = pd.read_csv(data_dir + '20200916_dataset.csv').drop_duplicates('id')
    grid_in_metro = pd.read_csv(data_dir + 'grids_in_metro_areas.csv')
    metro19 = pd.read_csv(data_dir + '20200831_GEIH_Metro_Areas.csv')
    metro20 = pd.read_csv(data_dir + '20200908_GEIH_Metro_Areas_2020.csv')
    metro_name = pd.read_csv(data_dir + 'metro_areas_id_name.csv')
    pred_metro18 = pd.read_csv(data_dir + 'metro_area_predictions_2018.csv')
    pred_metro19 = pd.read_csv(data_dir + 'metro_area_predictions.csv')
    pred_metro20 = pd.read_csv(data_dir + 'metro_area_predictions_2020.csv')

    metro_name['a_mtro'] = metro_name['a_mtro'].apply(clean_name)
    metro_name = metro_name.rename(columns = {'OBJECTID': 'metro_id'})

    # Actual
    spanish = {
        'd_hogares': 'population',
        'd_c_acuedu': 'hh_no_water_supply',
        'd_c_alcant': 'hh_no_sewage',
        'd_c_sanita': 'hh_no_toilet',
    }

    df1 = (pd.merge(grid_in_metro, wash18[['id'] + list(spanish.keys())], how = 'left', on = 'id')
        .rename(columns = spanish))
    df2 = df1.groupby('metro_id').agg('sum').reset_index()
    for indicator in indicators:
        df2[indicator] = 100*df2[indicator.replace('perc_', '')] / df2['population']

    metro18 = df2

    spanish = {
        'OBJECTID': 'metro_id',
        'personas': 'population',
        'c_acueduct': 'hh_no_water_supply',
        'c_alcantar': 'hh_no_sewage',
        'c_sanitari': 'hh_no_toilet',
        'mc_acueduc': 'perc_hh_no_water_supply',
        'mc_alcanta': 'perc_hh_no_sewage',
        'mc_sanitar': 'perc_hh_no_toilet',
    }

    metro19 = metro19.rename(columns = spanish)
    metro20 = metro20.rename(columns = spanish)

    cols = ['metro_id', 'year'] + indicators

    metro18['year'] = 2018
    metro19['year'] = 2019
    metro20['year'] = 2020

    df3 = pd.concat([
        metro18[cols],
        metro19[cols],
        metro20[cols],   
    ], axis = 0)

    df4 = pd.merge(metro_name, df3, how = 'left', on = 'metro_id')
    df5 = df4.set_index(['metro_id', 'a_mtro', 'year']).stack().reset_index()
    df5.columns = ['metro_id', 'a_mtro', 'year', 'indicator', 'value']

    # Predicted
    rnm = {
        'pred_perc_hh_no_water_supply': 'perc_hh_no_water_supply', 
        'pred_perc_hh_no_toilet': 'perc_hh_no_toilet', 
        'pred_perc_hh_no_sewage': 'perc_hh_no_sewage'
    }

    pred_metro18['year'] = 2018
    pred_metro19['year'] = 2019
    pred_metro20['year'] = 2020

    cols2 = ['metro_id', 'year'] + list(rnm.keys())
    df6 = pd.concat([
        pred_metro18[cols2].rename(columns = rnm),#metro18[cols],
        pred_metro19[cols2].rename(columns = rnm),
        pred_metro20[cols2].rename(columns = rnm)
    ], axis = 0)

    df7 = pd.merge(metro_name, df6, how = 'left', on = 'metro_id')
    df8 = df7.set_index(['metro_id', 'a_mtro', 'year']).stack().reset_index()
    df8.columns = ['metro_id', 'a_mtro', 'year', 'indicator', 'value']

    df5['val_type'] = 'actual'
    df8['val_type'] = 'pred'
    df9 = pd.concat([df5, df8], axis = 0)
    # df9.to_csv(data_dir + 'metro_trends.csv', index = False)

    return df9

def _aggregate_by_department():
    '''
    Aggregates predictions by department. Content originally from 03_Rollout.ipynb, Section 'what changed'
    '''

    scaler = joblib.load(scaler_dir + 'scaler_2018_250mv2.pkl')

    agg_level = 'adm1_name'
    keep_cols = [agg_level] + features + indicators

    def clean_name(text):
        return re.sub('[^a-z ]','', text.lower()).replace(' ', '_')

    raw = pd.read_csv(data_dir + '20200830_dataset.csv').drop_duplicates('id')
    raw['adm1_name'] = raw['adm1_name'].apply(clean_name)

    feats_2020 = pd.read_csv(data_dir + '20200914_dataset_2020.csv')
    preds_2020 = pd.read_csv(data_dir + '20200914_predictions2020.csv').rename(columns = {
        'pred_perc_hh_no_water_supply': 'perc_hh_no_water_supply',
        'pred_perc_hh_no_toilet': 'perc_hh_no_toilet',
        'pred_perc_hh_no_sewage': 'perc_hh_no_sewage',
    })[['id', 'perc_hh_no_water_supply', 'perc_hh_no_toilet', 'perc_hh_no_sewage']]

    # join
    wash_grid_2018_ = raw
    wash_grid_2020_ = pd.merge(feats_2020, preds_2020, on = 'id')

    # filter to 2018 grids only for comparability
    wash_grid_2020_ = wash_grid_2020_[wash_grid_2020_['id'].isin(list(wash_grid_2018_['id'].unique()))]

    # scale features except population
    wash_grid_2018 = wash_grid_2018_[keep_cols].copy()
    wash_grid_2020 = wash_grid_2020_[keep_cols].copy()

    wash_grid_2018.loc[:,features] = scaler.transform(wash_grid_2018[features])
    wash_grid_2020.loc[:,features] = scaler.transform(wash_grid_2020[features])

    wash_grid_2018['population'] = wash_grid_2018_['population']
    wash_grid_2020['population'] = wash_grid_2020_['population']

    # standardize naming
    to_replace = {'laguajira': 'la_guajira','valledelcauca': 'valle_del_cauca'}
    wash_grid_2018['adm1_name'] = wash_grid_2018['adm1_name'].replace(to_replace)
    wash_grid_2020['adm1_name'] = wash_grid_2020['adm1_name'].replace(to_replace)

    # get median for everything except population
    agg_type = {
        'vegetation': 'median',
        'aridity_cgiarv2': 'median',
        'temperature': 'median',
        'nighttime_lights': 'median',
        'population': 'sum',
        'elevation': 'median',
        'urban_index': 'median',
        'nearest_waterway': 'median',
        'nearest_commercial': 'median',
        'nearest_restaurant': 'median',
        'nearest_hospital': 'median',
        'nearest_airport': 'median',
        'nearest_highway': 'median',
        'perc_hh_no_water_supply': 'median',
        'perc_hh_no_toilet': 'median',
        'perc_hh_no_sewage': 'median',
    }
    wash_metro_2018 = wash_grid_2018.groupby(agg_level).agg(agg_type).reset_index()
    wash_metro_2020 = wash_grid_2020.groupby(agg_level).agg(agg_type).reset_index()

    # combine (wide format)
    wash_agg = pd.merge(
        wash_metro_2018, wash_metro_2020, left_on = agg_level, right_on = agg_level, suffixes = ['', '_2020']
        , how = 'left'
    )

    # convert to long
    df_ = wash_agg.set_index('adm1_name').stack().reset_index()
    df_.columns = ['adm1_name', 'feature', 'value']
    df_['year'] = 2018
    for i, row in df_.iterrows():
        if row.feature[-5:] == '_2020':
            df_.loc[i, 'year'] = 2020
            df_.loc[i, 'feature'] = df_.loc[i, 'feature'][:-5]

    # df_.to_csv('wash_agg.csv', index = False)

    return df_

def aggregate_predictions(by = 'department'):
    '''
    Aggregates predictions according to the level specified.

    Returns
        (DataFrame): aggregated features and metrics per year (long format)
    '''
    if by == 'department':
        return _aggregate_by_department()
    elif by == 'metro_area':
        return _aggregate_by_metro_area()
    else:
        print('Unrecognized aggregation level.')
        
