from bqutils import bq
import re

project = 'immap-colombia-270609'
dataset = 'wash_prep'

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

def process_by_dept(poi):
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