# !pip install -q rasterio geopandas
# !pip install scikit-learn==0.22.2.post1
import re
import numpy as np
import pandas as pd
from math import sqrt
import geopandas as gpd
import rasterio as rio
from shapely.wkt import loads
from tqdm import tqdm
import re

from bqutils import bq, run_sql
from settings import *

project = 'immap-colombia-270609'
dataset = 'wash_prep'

def generate_blocks_geopackage(gdf, adm):
    '''
    From raw blocks dataset, add admin boundaries, rename relevant wash indicators
    Output: wash_indicators_by_block.csv, the raw blocks dataset converted to geopackage format
    
    Steps:
    1. Generate grid boxes for whole Colombia using QGIS (grid_1x1km.gpkg)
    2. Run function to get wash_indicators_by_block.csv
    3. Upload wash_indicators_by_block to BQ then process using indicator_labelled_grid.sql
    
    Input:
    gdf - GeoDataFrame of raw blocks dataset
    adm - GeoDataFrame of admin boundaries in Colombia
    '''
    
    def sub(text, start,end): 
        return text[start:end];
    
    adm['adm2_code'] = adm['admin2Pcod'].apply(sub, args=(2,None)).astype(int)
    gdf['adm2_code'] = gdf['cod_dane'].apply(sub, args=(0,5)).astype(int)

    gdf2 = pd.merge(gdf, adm[['adm2_code', 'admin1Name', 'admin2RefN']], on = 'adm2_code', how = 'left')
    rnm = {
        'd_mc_acued': 'perc_hh_no_water_supply',
        'd_mc_alcan': 'perc_hh_no_sewage',
        'd_mc_sanit': 'perc_hh_no_toilet',
        'admin1Name': 'adm1_name', 
        'admin2RefN': 'adm2_name',
    }

    gdf2.rename(columns=rnm, inplace = True)
    gdf2.dropna(inplace = True)
    gdf2 = gdf2.to_crs('EPSG:4326')

    gdf2.to_file('wash_indicators_by_block.gpkg', driver = 'GPKG')
    gdf2.to_csv('wash_indicators_by_block.csv', index = False)
    # !gsutil cp wash_indicators_by_block.gpkg gs://immap-wash-training/indicators/
    # !gsutil cp wash_indicators_by_block.csv gs://immap-wash-training/indicators/
    
    print('wash_indicators_by_block.csv generated and transferred to GCS.')

def generate_indicator_labelled_grid(for_ = 'u'):
    "for_ can be 'u' or 'r', urban or rural"
    # blocks = gpd.read_file(feats_dir + 'Manzanas_urbano/Manzanas_urbano.shp')
    # adm = gpd.read_file(feats_dir + 'admin_bounds.gpkg', driver = 'GPKG')
    # geoutils.generate_blocks_geopackage(blocks, adm)
    # bqutils.run_sql(sql_dir + 'indicator_labelled_grid.sql')
    # bqutils.run_sql(sql_dir + 'indicator_labelled_grid_rural.sql')
    
    if for_ == 'u':
        df = pd.read_csv(inds_dir + 'indicator_labelled_grid.csv')
    else:
        df = pd.read_csv(inds_dir + 'indicator_labelled_grid_rural.csv')
    df['centroid_geometry'] = df['centroid_geometry'].apply(loads)
    gdf = gpd.GeoDataFrame(df, geometry='centroid_geometry').set_crs('EPSG:4326')
    return gdf

def generate_satellite_features(gdf):
    # satellite image derived - pierce through rasters
    geom_col = 'centroid_geometry'
    satellite_features_ = satellite_features + ['nearest_highway']
    pois_ = ['waterway', 'commercial', 'restaurant', 'hospital', 'airport']
    poi_features_ = ['clipped_nearest_' + poi for poi in pois_]
    for feature in tqdm(poi_features_ + satellite_features_):
        tif_file = feats_dir + f'2018_{area}_{feature}.tif'
        raster = rio.open(tif_file)

        # Perform point sampling
        pxl = []
        for index, row in gdf.iterrows():
            for val in raster.sample([(row[geom_col].x, row[geom_col].y)]):
                pxl.append(val[0])

        # Add column to geodataframe
        col_name = feature.replace('clipped_','')
        gdf[col_name] = pxl
    return gdf

def distance_to_nearest(
    poi_type = 'restaurant',
    department = 'bogot_dc',
    project = 'immap-colombia-270609', 
    dataset = 'wash_prep', 
    origin = 'grid_1x1km_wadmin', 
    destination = 'wash_pois_osm', 
    origin_id = 'id', 
    poi_id = 'osm_id', 
    km_dist = 20
):
    """
    Creates BQ query that finds distance nearest to a table of pois (destination)
    origin dataset coords column vs destination dataset poi_coords column - formatted as WK
    """
    query = f"""WITH all_dists AS (
                WITH destination AS (
                    SELECT
                    {poi_id},
                    geometry
                    FROM
                    `{dataset}.{destination}`
                    where fclass = '{poi_type}')
                SELECT
                    {origin_id},
                    origin.geometry,
                    ST_DISTANCE(st_centroid(st_geogfromtext(origin.geometry)), destination.geometry) AS dist
                FROM
                    (select * from `{dataset}.{origin}` where adm1_name = '{department}') AS origin,
                    destination
                WHERE
                    ST_DWITHIN(st_centroid(st_geogfromtext(origin.geometry)), destination.geometry, {km_dist}*1000))
                SELECT
                ARRAY_AGG(all_dists
                    ORDER BY
                    dist
                    LIMIT
                    1)[OFFSET(0)]
                FROM
                all_dists
                GROUP BY
                {origin_id}
            """
    
    return re.sub('[ ]+', ' ', query).replace('\n', ' ')

def fill(poi):
    bq(f'nearest_{poi}',f"""
    select
    base.id, base.geometry, coalesce(cons.f0_.dist, 40000) dist,
    adm1_name, adm2_name
    from `wash_prep.grid_1x1km_wadmin` base
    left join `temp.__cons` cons on base.id = cons.f0_.id
    """,
    dataset = 'wash_prep',
    create_view = True)

def get_depts():
    df = pandas_gbq.read_gbq(
            'SELECT adm1_name, count(id) cnt FROM `wash_prep.grid_1x1km_wadmin` group by 1 order by 2 desc', 
        project_id = 'immap-colombia-270609')
    return list(df.adm1_name)

def generate_poi_features_by_dept(poi):
    # apply distance to nearest, 1 department at a time
    cons_query = []

    for dept in tqdm(depts):
        # create temp table with distances
        bq(f'_{dept}', distance_to_nearest(
                poi_type = poi, department = dept
            ), dataset = 'temp', create_view = False)
        cons_query.append(f'select * from temp._{dept}')

    bq(f'__cons', ' union all '.join(cons_query), dataset = 'temp', create_view = False)
    fill(poi)

# overrides GeoPandasBase method
# https://github.com/martinfleis/geopandas/blob/0a749f95f8476f145788a8a2a14b71418232912e/geopandas/geodataframe.py
def explode(self):
    """
    Explode muti-part geometries into multiple single geometries.
    Each row containing a multi-part geometry will be split into
    multiple rows with single geometries, thereby increasing the vertical
    size of the GeoDataFrame.
    The index of the input geodataframe is no longer unique and is
    replaced with a multi-index (original index with additional level
    indicating the multiple geometries: a new zero-based index for each
    single part geometry per multi-part geometry).
    Returns
    -------
    GeoDataFrame
        Exploded geodataframe with each single geometry
        as a separate entry in the geodataframe.
    """
    df_copy = self.copy()

    df_copy["__order"] = range(len(df_copy))

    exploded_geom = df_copy.geometry.explode().reset_index(level=-1)
    exploded_index = exploded_geom.columns[0]

    df = pd.merge(
        df_copy.drop(df_copy._geometry_column_name, axis=1),
        exploded_geom,
        left_index=True,
        right_index=True,
        sort=False,
        how="left",
    )
    # reset to MultiIndex, otherwise df index is only first level of
    # exploded GeoSeries index.
    df.set_index(exploded_index, append=True, inplace=True)
    df.index.names = list(self.index.names) + [None]
    df.sort_values("__order", inplace=True)
    geo_df = df.set_geometry(self._geometry_column_name).drop("__order", axis=1)
    return geo_df

def dissolve_via_code():
    df = pd.read_csv(data_dir + 'aggregated_blocks_to_grid_unfiltered.csv')
    df['geometry'] = df['geometry'].apply(wkt.loads)
    df['agg'] = 'colombia'
    gdf = gpd.GeoDataFrame(df, geometry = 'geometry')

    multipart = gdf.dissolve(by = 'agg')
    singleparts = explode(multipart)

    singleparts = singleparts.reset_index().reset_index().rename(columns = {'index': 'id'})
    singleparts.to_csv(data_dir + 'prep/01_dissolve_code.csv', index = False)

def explode_manual_dissolved():
    gdf = gpd.read_file(data_dir + 'prep/02_dissolve_manual.gpkg', driver = 'GPKG')
    singleparts = explode(gdf.drop(labels = ['id', 'id_1', 'level_1'], axis = 1))
    singleparts = singleparts.reset_index().reset_index().rename(columns = {'index': 'id'})
    singleparts.to_csv(data_dir + '20200810_pixelated_urban_areas.csv', index = False)
    
def generate_urban_area_features():
    '''
    Generates BQ table of urban area features
    
    Steps:
    1. From Urban_Areas_COL, dissolve via QGIS
    2. dissolve_via_code() - first dissolve grids to pixelated urban area polygons, still has lines inside polygons
    3. dissolve manually - remove points inside polygons via QGIS
    4. explode_manual_dissolved() - explode QGIS output
    5. Upload to BQ table as mgn_urban_areas
    '''
    dissolve_via_code()
    explode_manual_dissolved()
    bqutils.run_sql(sql_dir + 'urban_area_features.sql')


# code for spatial lagging
# https://colab.research.google.com/drive/1Sahrwon3N4kvrNln7q68_577PZiCJ0uD?usp=sharing

# shift all elements in matrix, 1 step along direction
def right(mat): return np.roll(mat,1,axis=1);
def left(mat): return np.roll(mat,-1,axis=1);
def down(mat): return np.roll(mat,1,axis=0);
def up(mat): return np.roll(mat,-1,axis=0);

def spatial_lag(grid_ids, vals):
    '''Averages neighboring values of a grid based on Queen contiguity'''
    global neighbors, padded, grid_inds, matrix, matrix_of_ids

    # fill gaps of grid ids with 0
    max_ind = height*width
    extended = np.arange(max_ind)
    filled = np.zeros(max_ind)
    # filled = filled - 999.999
    filled[grid_ids] = np.array(vals) + 0.00001 # ensure values are nonzero
    matrix = filled.reshape((height,width))
    filled2 = np.zeros(max_ind)
    filled2[grid_ids] = grid_ids
    matrix_of_ids = filled2.reshape((height,width))

    # add a boundary of 0s to surround the whole matrix
    padded = np.zeros((height+2,width+2))
    # padded = padded - 999.999
    padded[1:height+1, 1:width+1] = matrix
    grid_inds = np.argwhere(padded)# != -999.999)

    neighbors = [
        # straights
        right(padded),
        left(padded),
        down(padded),
        up(padded),

        # diagonals
        right(down(padded)),
        left(down(padded)),
        right(up(padded)),
        left(up(padded)),
    ]
    sum_ = np.zeros((height+2,width+2))
    count_ = np.zeros((height+2,width+2))
    for neighbor in neighbors:
        sum_ = sum_ + neighbor
        count_ = count_ + (neighbor != 0)*1 #-999.999)*1
    avg_ = sum_/count_

    return avg_[grid_inds[:,0],grid_inds[:,1]]

def check_val(grid_id, padded, neighbors):
    '''input grid_id, output index in matrix + neighbors + average calculated'''
    # grid_id = 1077163
    inds = np.unravel_index(grid_id, shape = (height,width), order = 'C')
    print(f'For grid_id: {grid_id}')
    print('located in reshaped matrix at')
    print(inds)
    print('Indices of surrounding tiles here')
    print(matrix_of_ids[inds[0]-1:inds[0]+2, inds[1]-1:inds[1]+2])
    print('Values of surrounding tiles here')
    print(matrix[inds[0]-1:inds[0]+2, inds[1]-1:inds[1]+2])
    print('Value in padded matrix:')
    print(padded[inds])
    print('Values of Neighbors:')
    for neighbor in neighbors:
        print(neighbor[inds])

def generate_spatial_lag_features():
    '''
    Generate spatial lagged features. This function performs steps 4 and 5, 1-3 are done manually via QGIS
    
    Steps
    1. create unclipped grid from QGIS using GEE vegetation.tif
    2. create unclipped_id from unclipped grid via
        !gsutil cp gs://immap-wash-training/grid/grid_1x1km_unclipped.gpkg .
        gdf = gpd.read_file('grid_1x1km_unclipped.gpkg', driver = 'GPKG').set_crs('EPSG:4326')
        gdf = gdf.reset_index().rename(columns = {'index': 'unclipped_id'})
        gdf.to_csv('grid_1x1km_unclipped.csv', index = False)
        !gsutil cp grid_1x1km_unclipped.csv gs://immap-wash-training/grid/
        
    3. add features to grid via raster pierce code in 01_Data_Preprocessing (grid_1x1km_wfeatures)
    4. join unclipped id to features
    5. calculate spatial lags based on edge adjacency of grids
    '''

    run_sql(sql_dir + 'grid_1x1km_wfeatures_wunclippedid.sql')
    # !gsutil cp gs://immap-wash-training/grid/grid_1x1km_wfeatures_wunclippedid.csv {feats_dir}
    raw = pd.read_csv(feats_dir + 'grid_1x1km_wfeatures_wunclippedid.csv').sort_values('unclipped_id')

    df = raw.copy()#[raw.adm1_name == dept]
    for feature in tqdm(features):
        grid_ids = list(df['unclipped_id'])
        vals = list(df[feature])
        df['lag_' + feature] = spatial_lag(grid_ids, vals)
        
    df.to_csv(feats_dir + 'grid_1x1km_wfeatures_lagged.csv')

def generate_training_data(gdf):
    ua_feats = pd.read_csv(feats_dir + 'urban_area_features.csv').drop(labels = ['geometry'], axis = 1)
    lag_feats = pd.read_csv(feats_dir + 'grid_1x1km_wfeatures_lagged.csv')
    cols = ['id'] + [text for text in list(lag_feats.columns) if re.search('lag_*', text) is not None]
    lag_feats = lag_feats[cols]
    
    # master table
    df_ = pd.merge(gdf, ua_feats, how = 'left', on = 'id')
    df = pd.merge(df_, lag_feats, how = 'left', on = 'id')

    # add night time lights mean
    mean_col = df.groupby('pixelated_urban_area_id')['nighttime_lights'].mean() # don't reset the index!
    df = df.set_index('pixelated_urban_area_id') # make the same index here
    df['nighttime_lights_area_mean'] = mean_col

    # format for R-INLA
    df['x'] = df['centroid_geometry'].x
    df['y'] = df['centroid_geometry'].y

    train_df = df.reset_index()
    return train_df