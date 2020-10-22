# How to Start Using BigQuery GIS
Taken from the [[BQ docs]](https://cloud.google.com/bigquery/docs/sandbox)

1. Log-in to your google account.
2. Go to the [[BQ console]](https://console.cloud.google.com/bigquery).
3. Follow the prompts to create your new project.
4. Test GIS capabilities by running the query below.

```
-- output table: join_block_centroid_to_grid
with grids as (
    select 1 as grid_id, 
    'POLYGON ((-73.70061740234379 3.5458327102661, -73.6983651523438 3.5458327102661, -73.6983651523438 3.5435804602661, -73.70061740234379 3.5435804602661, -73.70061740234379 3.5458327102661))'
    as wkt
),
blocks as (
    select 2 as block_id,
    'POLYGON ((-73.69865984099999 3.544908443999987, -73.69874805199998 3.544861092000019, -73.69882686900002 3.544946835000019, -73.69883395699998 3.544963026000005, -73.69887401800003 3.54505453500002, -73.69885491500003 3.545081837999987, -73.69867157800002 3.545209345999979, -73.69826254200001 3.545493822000026, -73.69818155899998 3.545390365999992, -73.69816109599998 3.545355486000005, -73.69814196700003 3.545322879000025, -73.698103984 3.545258134999983, -73.69809780200001 3.545247598000003, -73.69842963100001 3.545032038999977, -73.69847407399999 3.545008178999979, -73.69862016500002 3.544929749999994, -73.69865984099999 3.544908443999987))'
    as wkt
)

select grid_id, block_id, grids.wkt
from grids, blocks
where st_within(
    st_centroid(st_geogfromtext(blocks.wkt)),
    st_geogfromtext(grids.wkt)
)
```

5. Upload table of geometries to bigquery by selecting the project > Create Dataset > Select Dataset > Create Table > Create Table from: Upload
Screenshots of this stream of steps can be seen [[here]](https://cloud.google.com/bigquery/docs/loading-data-local#loading_data_from_a_local_data_source)
