-- output table: mgn_urban_areas_wadmin
SELECT
urb.id,
urb.geometry,
st_centroid(SAFE.st_geogfromtext(urb.geometry)) centroid_geometry,
st_area(SAFE.st_geogfromtext(urb.geometry)) urban_area_size,
replace(regexp_replace(lower(admin1Name), '[^a-z ]',''), ' ', '_') adm1_name,
replace(regexp_replace(lower(admin2RefN), '[^a-z ]',''), ' ', '_') adm2_name
FROM 
`wash_prep.mgn_urban_areas` urb
, `wash_prep.admin_bounds` adm
where st_within(st_centroid(SAFE.st_geogfromtext(urb.geometry)), st_geogfromtext(adm.geometry));

-- output table: mgn_urban_areas_dept_capital
with filtered as (
    SELECT a.*
    FROM `wash_prep.mgn_urban_areas_wadmin` a
    left join wash_prep.dept_capital_lookup b
    on concat(a.adm1_name, a.adm2_name) = concat(b.adm1_name, b.adm2_name)
    where b.adm1_name is not null
    order by adm1_name, urban_area_size desc
),
instanced as (
    select *, row_number() over(partition by adm1_name order by urban_area_size desc) instance from filtered
)

select * except(instance) from instanced where instance = 1;

-- output table: mgn_urban_areas_distance_to_dept_capital
select
grid.id,
grid.geometry,
ST_DISTANCE(grid.centroid_geometry,area.centroid_geometry) distance_from_capital,
ST_DISTANCE(grid.centroid_geometry,st_geogfromtext(area.geometry)) distance_from_capital_outskirts,
grid.adm1_name,
grid.adm2_name,
area.adm1_name area_adm1_name
from
`wash_prep.indicator_labelled_grid` grid
left join
`wash_prep.mgn_urban_areas_dept_capital` area
on grid.adm1_name = area.adm1_name;

-- output table: pixelated_urban_areas_wadmin
SELECT
urb.id,
urb.geometry,
st_centroid(SAFE.st_geogfromtext(urb.geometry)) centroid_geometry,
st_area(SAFE.st_geogfromtext(urb.geometry)) urban_area_size,
replace(regexp_replace(lower(admin1Name), '[^a-z ]',''), ' ', '_') adm1_name,
replace(regexp_replace(lower(admin2RefN), '[^a-z ]',''), ' ', '_') adm2_name
FROM 
`wash_prep.pixelated_urban_areas` urb
, `wash_prep.admin_bounds` adm
where st_within(st_centroid(SAFE.st_geogfromtext(urb.geometry)), st_geogfromtext(adm.geometry));

--output table: pixelated_urban_areas_area_level_nearest
WITH
  all_dists AS (
  SELECT
    origin.id,
    origin.geometry,
    origin.centroid_geometry,
    origin.adm1_name,
    origin.adm2_name,
    destination.id d_id,
    destination.adm1_name d_adm1_name,
    destination.adm2_name d_adm2_name,
    ST_DISTANCE(origin.centroid_geometry,
      destination.centroid_geometry) AS dist
  FROM `wash_prep.pixelated_urban_areas_wadmin` AS origin,
    `wash_prep.pixelated_urban_areas_wadmin` AS destination
    where origin.id <> destination.id)
SELECT
  ARRAY_AGG(all_dists
  ORDER BY
    dist
  LIMIT
    1)[
OFFSET
  (0)]
FROM
  all_dists
GROUP BY
  id;

-- output table: pixelated_urban_areas_distance_between_muni_centers
SELECT f0_.adm1_name adm1_name, avg(f0_.dist) distance_between_muni_centers FROM `wash_prep.pixelated_urban_areas_area_level_nearest` group by 1 order by 2;

-- output table: pixelated_urban_areas_nearest_area_to_grid
select
grid.id,
grid.geometry,
grid.centroid_geometry,
grid.adm1_name,
grid.adm2_name,
f0_.id pixelated_urban_area_id,
f0_.geometry pixelated_urban_area_geometry,
st_area(st_geogfromtext(f0_.geometry)) pixelated_urban_area_size,
f0_.d_id nearest_pixelated_urban_area_id,
f0_.dist distance_to_nearest_pixelated_urban_area

from wash_prep.indicator_labelled_grid grid
, wash_prep.pixelated_urban_areas_area_level_nearest near
where st_within(grid.centroid_geometry, st_geogfromtext(near.f0_.geometry));

-- output table: urban_area_features
select a.id,
a.geometry geometry,
pixelated_urban_area_id,
distance_from_capital,
distance_from_capital_outskirts,
distance_between_muni_centers,
pixelated_urban_area_size,
distance_to_nearest_pixelated_urban_area
from
wash_prep.indicator_labelled_grid a
left join wash_prep.mgn_urban_areas_distance_to_dept_capital b on a.id = b.id
left join wash_prep.pixelated_urban_areas_distance_between_muni_centers c on a.adm1_name = c.adm1_name
left join wash_prep.pixelated_urban_areas_nearest_area_to_grid d on a.id = d.id;

