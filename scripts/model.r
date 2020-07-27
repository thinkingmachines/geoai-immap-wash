# # # !sudo apt-get install gdal-bin proj-bin libgdal-dev libproj-dev
# # install.packages('rgdal') # for linux vm install
# # install.packages('foreach') # for linux vm install
# install.packages('geoR')
# install.packages("INLA",repos = "https://inla.r-inla-download.org/R/stable", dep = TRUE) #829pm
# install.packages('optparse')

library(optparse)
library(tidyverse)
library(dplyr)
library(geoR)
library(sp)
library(rgdal)
library(raster)
library(INLA)
library(purrr)

# addresses glibc_2.27 not found: https://www.mn.uio.no/math/english/services/it/help/status/2018-07-26-inla-r.html
INLA:::inla.dynload.workaround()
# reduce memory in a 4cpu vm
inla.setOption( num.threads = 2 )

# gsutil cp gs://immap-wash-training/20200725_dataset.csv .
# gsutil cp gs://immap-wash-training/features/2018_colombia_aridity.tif .

# data(gambia)
# r0 <- getData(name = "alt", country = "GMB", mask = TRUE)

indicator = 'perc_hh_no_water_supply';
area = 'bogot_dc';

inds_file = '20200725_dataset.csv'
feats_file = "2018_colombia_aridity.tif"
test_areas = c(
    'bogot_dc',
    'norte_de_santander',
    'la_guajira',
    'nario'
)
count_vars = c(
    perc_hh_no_toilet = 'd_c_sanita',
    perc_hh_no_water_supply = 'd_c_acuedu',
    perc_hh_no_sewage = 'd_c_alcant'
)
count_var = count_vars[[indicator]]
cols = c('x', 'y', count_var, 'd_hogares')

r <- raster(feats_file)
data = read.csv(inds_file, stringsAsFactors = F) %>% 
    filter(adm1_name != area) %>%
    dplyr::select(cols) %>% 
    dplyr::rename(count = count_var, total = 'd_hogares')

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

spst <- SpatialPoints(d[, c("x", "y")],
  proj4string = CRS("+proj=longlat +datum=WGS84")
)

d[, c("long", "lat")] <- coordinates(spst)

d$alt <- raster::extract(r, d[, c("long", "lat")])
dim(d)
d <- d[complete.cases(d), ] 
dim(d)

# *10
coo <- cbind(d$long, d$lat)
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

dp <- rasterToPoints(r)

ra <- aggregate(r, fact = 5, fun = mean) # aggregates 5 cells in each direction

dp <- rasterToPoints(ra)

coop <- dp[, c("x", "y")]

Ap <- inla.spde.make.A(mesh = mesh, loc = coop)

stk.e <- inla.stack(
  tag = "est",
  data = list(y = d$positive, numtrials = d$total),
  A = list(1, A),
  effects = list(data.frame(b0 = 1, altitude = (d$alt-67)/(503-67)), s = indexs)
)

stk.p <- inla.stack(
  tag = "pred",
  data = list(y = NA, numtrials = NA),
  A = list(1, Ap),
  effects = list(data.frame(b0 = 1, altitude = (dp[, 3]-67)/(503-67)),
    s = indexs
  )
)

stk.full <- inla.stack(stk.e, stk.p)

formula <- y ~ 0 + b0 + altitude + f(s, model = spde)

# takes a minute before crashing
res <- inla(formula,
  family = "binomial", Ntrials = numtrials,
  control.family = list(link = "logit"),
  data = inla.stack.data(stk.full),
  control.predictor = list(
    compute = TRUE, link = 1,
    A = inla.stack.A(stk.full)
  ),
#   control.fixed=list(prec.intercept=1),
  verbose = TRUE,
)

## 9.4 Mapping malaria prevalence

index <- inla.stack.index(stack = stk.full, tag = "pred")$data

prev_mean <- res$summary.fitted.values[index, "mean"]
# prev_ll <- res$summary.fitted.values[index, "0.025quant"]
# prev_ul <- res$summary.fitted.values[index, "0.975quant"]

r_prev_mean <- rasterize(
  x = coop, y = ra, field = prev_mean,
  fun = mean
)

# r_prev_mean

rf <- writeRaster(r_prev_mean, filename=paste0("teston_", area, ".tif"), format="GTiff", overwrite=TRUE)

