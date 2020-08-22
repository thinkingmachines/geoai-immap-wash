-- output table: grid_1x1km_wunclippedid
select unclipped.unclipped_id, orig.*
from
wash_prep.grid_1x1km_wadmin orig
, wash_prep.grid_1x1km_unclipped unclipped
where st_within(st_centroid(st_geogfromtext(orig.geometry)), st_geogfromtext(unclipped.geometry));

-- output table: grid_1x1km_wfeatures_wunclippedid
select unclipped.unclipped_id, feats.*
from
wash_prep.grid_1x1km_wfeatures feats
left join
wash_prep.grid_1x1km_wunclippedid unclipped
on feats.id = unclipped.id;