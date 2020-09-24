
-- output table: grid_250x250m_wadmin
SELECT
grid.id,
grid.geometry,
replace(regexp_replace(lower(admin1Name), '[^a-z ]',''), ' ', '_') adm1_name,
replace(regexp_replace(lower(admin2RefN), '[^a-z ]',''), ' ', '_') adm2_name
FROM 
`wash_prep.grid_250x250m` grid
, `wash_prep.admin_bounds` adm
where st_within(st_centroid(st_geogfromtext(grid.geometry)), st_geogfromtext(adm.geometry));

-- output table: join_blocks_to_grid_250
select
grid.id,
grid.geometry,
inds.geometry block_geometry,
st_area(SAFE.st_geogfromtext(inds.geometry)) block_area,
st_area( st_geogfromtext(grid.geometry)) grid_area,
inds.* except(geometry)
from
wash_prep.wash_indicators_by_block inds
, wash_prep.grid_250x250m_wadmin grid
where st_within(st_centroid(SAFE.st_geogfromtext(inds.geometry)),  st_geogfromtext(grid.geometry));

-- output table: aggregated_blocks_to_grid_unfiltered_250
select 
id, geometry,

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

from wash_prep.join_blocks_to_grid_250
group by 2,1;

-- output table: aggregated_blocks_to_grid_250
select * from wash_prep.aggregated_blocks_to_grid_unfiltered_250
where id in (select id from wash_prep.grids_with_mostly_null_blocks where null_ratio between 0 and 1);

## indicator_labelled_grid_250
-- output table: grid_250x250m_wadmin
SELECT
grid.id,
grid.geometry,
replace(regexp_replace(lower(admin1Name), '[^a-z ]',''), ' ', '_') adm1_name,
replace(regexp_replace(lower(admin2RefN), '[^a-z ]',''), ' ', '_') adm2_name
FROM 
`wash_prep.grid_250x250m` grid
, `wash_prep.admin_bounds` adm
where st_within(st_centroid(st_geogfromtext(grid.geometry)), st_geogfromtext(adm.geometry));

-- output table: join_rural_to_grid_250
select
grid.id,
grid.geometry,
grid.adm1_name,
grid.adm2_name,
inds.ID siasar_id,
inds.longitud longitude,
inds.latitud latitude,
inds.hogares,
inds.c_acueducto,
inds.c_alcantarillado,
inds.c_sanitario,
cast(round(inds.hogares * inds.c_acueducto, 0) as int64) hh_no_water_supply,
cast(round(inds.hogares * inds.c_alcantarillado, 0) as int64) hh_no_sewage,
cast(round(inds.hogares * inds.c_sanitario, 0) as int64) hh_no_toilet

from
wash_prep.siasar_data inds
, wash_prep.grid_250x250m_wadmin grid
where st_within(st_geogpoint(inds.longitud, inds.latitud), st_geogfromtext(grid.geometry));

-- output table: aggregated_rural_to_grid_250
select 
id, geometry,
adm1_name,
adm2_name,

sum(hh_no_toilet)/sum(hogares) perc_hh_no_toilet,
sum(hh_no_water_supply)/sum(hogares) perc_hh_no_water_supply,
sum(hh_no_sewage)/sum(hogares) perc_hh_no_sewage,

0.0001 d_mc_basur,
0.0001 d_mc_aguac,
0.0001 d_mc_freq_,
0.0001 d_mc_pare,
0.0001 d_mc_piso,
0.0001 d_mc_elect,
0.0001 d_mc_hacin,
0.0001 d_mc_cocin,
0.0001 d_mc_gas,

sum(hogares) d_hogares,
sum(hh_no_toilet) d_c_sanita,
sum(hh_no_water_supply) d_c_acuedu,
sum(hh_no_sewage) d_c_alcant,
0.0001 d_c_basura,
0.0001 d_c_aguaco,
0.0001 d_c_freq_b,
0.0001 d_c_pared,
0.0001 d_c_piso,
0.0001 d_c_electr,
0.0001 d_c_hacina,
0.0001 d_c_cocina,
0.0001 d_c_gas

from wash_prep.join_rural_to_grid_250
group by 4,3,2,1;

-- output table: indicator_labelled_grid_rural_250
select
* except(adm1_name, adm2_name),
st_centroid(st_geogfromtext(geometry)) centroid_geometry,
adm1_name,
adm2_name
from
wash_prep.aggregated_rural_to_grid_250;