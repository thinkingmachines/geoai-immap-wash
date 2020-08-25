-- output table: join_rural_to_grid
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
, wash_prep.grid_1x1km_wadmin grid
where st_within(st_geogpoint(inds.longitud, inds.latitud), st_geogfromtext(grid.geometry));

-- output table: aggregated_rural_to_grid
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

from wash_prep.join_rural_to_grid
group by 4,3,2,1;

-- output table: indicator_labelled_grid_rural
select
* except(adm1_name, adm2_name),
st_centroid(st_geogfromtext(geometry)) centroid_geometry,
adm1_name,
adm2_name
from
wash_prep.aggregated_rural_to_grid;
