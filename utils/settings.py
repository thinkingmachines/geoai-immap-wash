
data_dir = '../data/'
feats_dir = data_dir + 'features/'
inds_dir = data_dir + 'indicators/'
model_dir = data_dir + 'models/'
scaler_dir = data_dir + 'scalers/'
output_dir = data_dir + 'outputs/'
sql_dir = '../sql/'

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
    'nearest_highway',
]

pois = ['waterway', 'commercial', 'restaurant', 'hospital', 'airport']
poi_features = ['clipped_nearest_' + poi for poi in pois]

features = [
    'vegetation',
    'temperature',
    'population',
    'nighttime_lights',
    'aridity_cgiarv2',
    'elevation',
    'urban_index',

    'nearest_waterway',
    'nearest_commercial',
    'nearest_restaurant',
    'nearest_hospital',
    'nearest_airport',
    'nearest_highway',

    #'pixelated_urban_area_id',
    'distance_from_capital',
    'distance_from_capital_outskirts',
    #'distance_between_muni_centers',
    'pixelated_urban_area_size',
    #'distance_to_nearest_pixelated_urban_area',
    'nighttime_lights_area_mean',
    
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
    'lag_nearest_highway'
]

test_areas = [
    'bogot_dc',
    'norte_de_santander',
    'la_guajira',
    'nario'
]