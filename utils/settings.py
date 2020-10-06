
data_dir = '../data/'
feats_dir = data_dir + 'features/'
inds_dir = data_dir + 'indicators/'
model_dir = data_dir + 'models/'
scaler_dir = data_dir + 'scalers/'
preds_dir = data_dir + 'predictions/'
sql_dir = '../sql/'
dept_dir = data_dir + 'by_dept/'
grid250_dir = dept_dir + 'grid_250x250m/'
feats250_dir = dept_dir + 'features/'
preds250_dir = dept_dir + 'predictions/'

area = 'colombia'
BBOX = [-73.17020892181104, 11.560920839000062, -72.52724612099996, 10.948171764015513]

# dimensions of grid from qgis
height = 1979
width = 1660

indicators = [
    'perc_hh_no_water_supply',
    'perc_hh_no_toilet',
    'perc_hh_no_sewage'
]

satellite_features = [
    'vegetation',
    'aridity_cgiarv2',
    'temperature',
    'nighttime_lights',
    'population',
    'elevation',
    'urban_index',
]

pois = ['waterway', 'commercial', 'restaurant', 'hospital', 'airport', 'highway']
poi_features = ['nearest_' + poi for poi in pois]

urban_features = [
    #'pixelated_urban_area_id',
    'distance_from_capital',
    'distance_from_capital_outskirts',
    #'distance_between_muni_centers',
    'pixelated_urban_area_size',
    #'distance_to_nearest_pixelated_urban_area',
    'nighttime_lights_area_mean',
]

lag_features = [
    'lag_nearest_waterway',
    'lag_nearest_commercial',
    'lag_nearest_restaurant',
    'lag_nearest_hospital',
    'lag_nearest_airport',
    'lag_vegetation',
    'lag_aridity_cgiarv2',
    'lag_temperature',
    'lag_nighttime_lights',
    'lag_population',
    'lag_elevation',
    'lag_urban_index',
    'lag_nearest_highway',
]

features = satellite_features + poi_features # + urban_features + lag_features
