-- output table: join_blocks_to_grid
select
grid.id,
grid.geometry,
grid.left,
grid.top,
inds.* except(geometry)
from
wash_prep.wash_indicators_by_block_geom inds
, wash_prep.grid_1x1km grid
where st_within(st_centroid(inds.geometry), st_geogfromtext(grid.geometry));
    
-- output table: aggregated_blocks_to_grid
select 
id,geometry,`left`,top,

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
group by 4,3,2,1;

-- output table: indicator_labelled_grid
select
grid.*,
adm.admin1Name adm1_name,
adm.admin2RefN adm2_name
from
wash_prep.aggregated_blocks_to_grid grid
, wash_prep.admin_bounds_geom adm
where st_within(st_centroid(st_geogfromtext(grid.geometry)), adm.geometry);