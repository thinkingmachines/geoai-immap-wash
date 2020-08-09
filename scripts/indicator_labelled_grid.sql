
-- output table: grid_1x1km_wadmin
SELECT
grid.id,
grid.geometry,
replace(regexp_replace(lower(admin1Name), '[^a-z ]',''), ' ', '_') adm1_name,
replace(regexp_replace(lower(admin2RefN), '[^a-z ]',''), ' ', '_') adm2_name
FROM 
`wash_prep.grid_1x1km` grid
, `wash_prep.admin_bounds` adm
where st_within(st_centroid(st_geogfromtext(grid.geometry)), st_geogfromtext(adm.geometry));

-- output table: join_blocks_to_grid
select
grid.id,
grid.geometry,
inds.geometry block_geometry,
st_area(st_geogfromtext(inds.geometry)) block_area,
st_area(st_geogfromtext(grid.geometry)) grid_area,
inds.* except(geometry)
from
wash_prep.wash_indicators_by_block inds
, wash_prep.grid_1x1km grid
where st_within(st_centroid(st_geogfromtext(inds.geometry)), st_geogfromtext(grid.geometry));

-- output table: grids_with_mostly_null_blocks
with classify as (
    select *,
    CASE WHEN (perc_hh_no_water_supply IS NULL or perc_hh_no_sewage IS NULL or perc_hh_no_toilet IS NULL) THEN 'HAS NO DATA' ELSE 'HAS DATA' END tile
    FROM `wash_prep.join_blocks_to_grid`
),
aggregate as (
    SELECT
    id,
    st_astext(geometry) geometry,
    adm1_name,
    adm2_name,
    sum(case when tile = 'HAS NO DATA' then 1 else 0 end) block_cnt_no_data,
    sum(case when tile = 'HAS NO DATA' then block_area else 0 end) block_area_no_data,
    sum(case when tile = 'HAS DATA' then 1 else 0 end) block_cnt_has_data,
    sum(case when tile = 'HAS DATA' then block_area else 0 end) block_area_has_data,
    AVG(grid_area) grid_area
    FROM classify
    GROUP BY
    1,
    2,
    3,
    4
)

select *, 
case 
when block_area_has_data = 0 then -999
else block_area_no_data / block_area_has_data end
null_ratio
from aggregate
ORDER BY
  10 DESC;

-- output table: aggregated_blocks_to_grid
select 
id, st_astext(geometry) geometry,

sum(d_c_sanita)/sum(d_hogares) perc_hh_no_toilet,
sum(d_c_acuedu)/sum(d_hogares) perc_hh_no_water_supply,
sum(d_c_alcant)/sum(d_hogares) perc_hh_no_sewage,
sum(d_c_basura)/sum(d_hogares) d_mc_basur,
sum(d_c_aguaco)/sum(d_hogares) d_mc_aguac,
sum(d_c_freq_b)/sum(d_hogares) d_mc_freq_,
sum(d_c_pared)/sum(d_hogares) d_mc_pare,
sum(d_c_piso)/sum(d_hogares) d_mc_piso,
sum(d_c_electr)/sum(d_hogares) d_mc_elect,
sum(d_c_hacina)/sum(d_hogares) d_mc_hacin,
sum(d_c_cocina)/sum(d_hogares) d_mc_cocin,
sum(d_c_gas)/sum(d_hogares) d_mc_gas,

sum(d_hogares) d_hogares,
sum(d_c_sanita) d_c_sanita,
sum(d_c_acuedu) d_c_acuedu,
sum(d_c_alcant) d_c_alcant,
sum(d_c_basura) d_c_basura,
sum(d_c_aguaco) d_c_aguaco,
sum(d_c_freq_b) d_c_freq_b,
sum(d_c_pared) d_c_pared,
sum(d_c_piso) d_c_piso,
sum(d_c_electr) d_c_electr,
sum(d_c_hacina) d_c_hacina,
sum(d_c_cocina) d_c_cocina,
sum(d_c_gas) d_c_gas,

from wash_prep.join_blocks_to_grid
where id in (select id from wash_prep.grids_with_mostly_null_blocks where null_ratio between 0 and 1)
group by 2,1;

-- output table: indicator_labelled_grid
select
grid.*,
st_centroid(st_geogfromtext(grid.geometry)) centroid_geometry,
adm.admin1Name adm1_name,
adm.admin2RefN adm2_name
from
wash_prep.aggregated_blocks_to_grid grid
, wash_prep.admin_bounds adm
where st_within(st_centroid(st_geogfromtext(grid.geometry)), st_geogfromtext(adm.geometry));
