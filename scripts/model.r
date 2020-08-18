# -------------------------------------------------------------------------------
# Geostats model via R-INLA
#
# input: satellite features tif files, cropped prediction area tif file
# output: tif file of predictions on out of sample area
# ran on Windows 10, R v4.0.2
# takes 41 minutes in predicting over a 66x67 pixel grid
# -------------------------------------------------------------------------------

# install.packages('rgdal')
# install.packages('geoR')
# install.packages('sp')
# install.packages('foreach')
# install.packages('shiny')
# install.packages("INLA", repos = "https://inla.r-inla-download.org/R/stable", dep = TRUE) #719pm
# install.packages('optparse')
# install.packages('tidyverse')
# install.packages('raster')

library(optparse)
library(tidyverse)
library(dplyr)
library(geoR)
library(sp)
library(rgdal)
library(raster)
library(INLA)
library(purrr)

data_dir = 'C:/Users/ncdejito/Downloads/geoai-immap-wash/data/'#'../data/'
feats_dir = paste0(data_dir, 'features/')

# gsutil cp gs://immap-wash-training/20200811_dataset.csv .
# gsutil cp gs://immap-wash-training/features/2018_colombia_*.tif data/features/

merged = 'bogot_dc.tif';#'2018_colombia_merged_satellite_features_v2.tif';
indicator = 'perc_hh_no_water_supply';
area = 'bogot_dc';

inds_file = '20200811_dataset.csv'

features = c(
  'vegetation',
  'aridity_cgiarv2',
  'temperature',
  'nighttime_lights',
  'population',
  'elevation',
  'urban_index'
)
test_areas = c(
  'bogot_dc',
  'norte_de_santander',
  'la_guajira',
  'nario'
)

# read df, excluding test area
count_vars = c(
  perc_hh_no_toilet = 'd_c_sanita',
  perc_hh_no_water_supply = 'd_c_acuedu',
  perc_hh_no_sewage = 'd_c_alcant'
)
count_var = count_vars[[indicator]]
cols = c('x', 'y', count_var, 'd_hogares')
data = read.csv(paste0(data_dir,inds_file), stringsAsFactors = F) %>% 
  filter(adm1_name != area) %>%
  dplyr::select(cols) %>% 
  dplyr::rename(count = count_var, total = 'd_hogares')

# convert counts to multi-rowed binary
# https://www.r-bloggers.com/expanding-binomial-counts-to-binary-0-1-with-purrrpmap/
binary_dat = pmap_dfr(data, 
                      function(x, y, count, total) {
                        data.frame(x = x,
                                   y = y,
                                   pos = c( rep(1, count),
                                            rep(0, total - count) ) )
                      }
)

d <- group_by(binary_dat, x, y) %>%
  summarize(
    total = n(),
    positive = sum(pos),
    prev = positive / total
  )
print(dim(d))

# set coords
spst <- SpatialPoints(d[, c("x", "y")],
                      proj4string = CRS("+proj=longlat +datum=WGS84")
)

d[, c("long", "lat")] <- coordinates(spst)

# read rasters
for (f in features){
  fname = paste0(feats_dir,'2018_colombia_',f,'.tif')
  r <- raster(fname)
  d[[f]] <- raster::extract(r, d[, c("long", "lat")])
}
d = d[complete.cases(d),]
print(dim(d))

# get stats from training data
df = data.frame()
for (f in features){
  qs = quantile(d[[f]],c(0.25, 0.75))
  df_ = data.frame(
    feature = f, 
    median = median(d[[f]]), 
    p25 = qs[1], 
    p75 = qs[2], 
    mad = mad(d[[f]])
  )
  df = rbind(df, df_)
}
stats = df
stats

robustscaler_sklearn <- function(df){
  for (f in features){
    row = filter(stats, feature == f)
    df[[f]] = (df[[f]] - row$median)/(row$p75 - row$p25)
  }
  return(df)
}

# read data to predict on
# reading multiband rasters: https://www.neonscience.org/dc-multiband-rasters-r
fname = paste0(feats_dir,merged)
r = stack(fname)
# r <- aggregate(stack(fname), fact = 5, fun = mean) # aggregates 5 cells in each direction
dp <- data.frame(rasterToPoints(r))
dp = dp[complete.cases(dp),]
colnames(dp) = append(c('x', 'y'), features) # rename columns

d = robustscaler_sklearn(d)
dp = robustscaler_sklearn(dp)

# affects how many traingles are made inside the mesh, increased by x10 to reduce triangles
coo <- cbind(d$x, d$y)
mesh <- inla.mesh.2d(
  loc = coo, max.edge = c(1, 50),
  cutoff = 0.1
  #   loc = coo, max.edge = c(0.1, 5),
  #   cutoff = 0.01
)

mesh$n

plot(mesh)
points(coo, col = "red")

spde <- inla.spde2.matern(mesh = mesh, alpha = 2, constr = TRUE)

indexs <- inla.spde.make.index("s", spde$n.spde)
lengths(indexs)

A <- inla.spde.make.A(mesh = mesh, loc = coo)

coop <- cbind(dp$x, dp$y)

Ap <- inla.spde.make.A(mesh = mesh, loc = coop)

df_e = data.frame(
  b0 = 1, 
  vegetation = d$vegetation,
  aridity_cgiarv2 = d$aridity_cgiarv2,
  temperature = d$temperature,
  nighttime_lights = d$nighttime_lights,
  population = d$population,
  elevation = d$elevation,
  urban_index = d$urban_index
)
df_p = data.frame(
  b0 = 1, 
  vegetation = dp$vegetation,
  aridity_cgiarv2 = dp$aridity_cgiarv2,
  temperature = dp$temperature,
  nighttime_lights = dp$nighttime_lights,
  population = dp$population,
  elevation = dp$elevation,
  urban_index = dp$urban_index
)

stk.e <- inla.stack(
  tag = "est",
  data = list(y = d$positive, numtrials = d$total),
  A = list(1, A),
  effects = list(df_e, s = indexs)
)

stk.p <- inla.stack(
  tag = "pred",
  data = list(y = NA, numtrials = NA),
  A = list(1, Ap),
  effects = list(df_p, s = indexs)
)

stk.full <- inla.stack(stk.e, stk.p)

formula <- (
  y ~ 0 + b0 + 
    vegetation +
    aridity_cgiarv2 +
    temperature +
    nighttime_lights +
    population +
    elevation +
    urban_index +
    f(s, model = spde)
)

# 17 minutes predicting on bogota
res <- inla(formula,
            family = "binomial", Ntrials = numtrials,
            control.family = list(link = "logit"),
            data = inla.stack.data(stk.full),
            control.predictor = list(
              compute = TRUE, link = 1,
              A = inla.stack.A(stk.full)
            )
            , verbose = TRUE
)

# output coefficients to a text file
out <- capture.output(summary(res))
cat("My title", out, file="summary_of_my_very_time_consuming_regression.txt", sep="\n", append=TRUE)

## 9.4 Mapping malaria prevalence

index <- inla.stack.index(stack = stk.full, tag = "pred")$data

prev_mean <- res$summary.fitted.values[index, "mean"]
prev_ll <- res$summary.fitted.values[index, "0.025quant"]
prev_ul <- res$summary.fitted.values[index, "0.975quant"]

r_prev_mean <- rasterize(
  x = coop, y = r, field = prev_mean,
  fun = mean
)

# r_prev_mean

rf <- writeRaster(r_prev_mean, filename=paste0("teston_", area, ".tif"), format="GTiff", overwrite=TRUE)

