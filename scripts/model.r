# -------------------------------------------------------------------------------
# Geostats model via R-INLA
#
# Rscript --vanilla model.R -i perc_hh_no_water_supply -a bogot_dc
#
# input: satellite features tif files, cropped prediction area tif file
# output: tif file of predictions on out of sample area
# ran on Windows 10, R v4.0.2, 8cpu GCE VM
# takes 10 minutes in predicting over a 66x67 pixel grid
# -------------------------------------------------------------------------------

# install.packages('rgdal')
# install.packages('geoR')
# install.packages('sp')
# install.packages('foreach')
# install.packages('shiny')
# install.packages("INLA", repos = "https://inla.r-inla-download.org/R/stable", dep = TRUE)
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

work_dir = 'C:/Users/cholo/Downloads/'
setwd(work_dir)
data_dir = paste0(work_dir, 'geoai-immap-wash/data/')#'../data/'
feats_dir = paste0(data_dir, 'features/')

# gsutil cp gs://immap-wash-training/20200811_dataset.csv .
# gsutil cp gs://immap-wash-training/features/2018_colombia_*.tif data/features/

# parser
option_list = list(
  make_option(c("-i", "--indicator"), type="character", default="perc_hh_no_water_supply", 
              help="indicator to model", metavar="character"),
	make_option(c("-a", "--area"), type="character", default="bogot_dc", 
              help="area to hold out", metavar="character")
); 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);
indicator = opt$indicator;
area = opt$area;

merged = paste0(area,'.tif');#'2018_colombia_merged_satellite_features_v2.tif';
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
count_vars = c(
  perc_hh_no_toilet = 'd_c_sanita',
  perc_hh_no_water_supply = 'd_c_acuedu',
  perc_hh_no_sewage = 'd_c_alcant'
)

robustscaler_sklearn <- function(df){
    for (f in features){
        row = filter(stats, feature == f)
        df[[f]] = (df[[f]] - row$median)/(row$p75 - row$p25)
    }
    return(df)
}

now <- function(){
  return(format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
}

read_ <- function(){
    print(paste0(now(),' Reading data'))
    # read indicators df, excluding test area
    count_var = count_vars[[indicator]]
    cols = c('x', 'y', count_var, 'd_hogares')
    d = read.csv(paste0(data_dir,inds_file), stringsAsFactors = F) %>% 
        filter(adm1_name != area) %>%
        dplyr::select(all_of(cols)) %>% 
        dplyr::rename(positive = all_of(count_var), total = 'd_hogares') %>% 
        dplyr::mutate(prev = positive / total) %>% 
        dplyr::arrange(x)

    print(dim(d))

    # read feature raster values of indicator labelled grid
    for (f in features){
        fname = paste0(feats_dir,'2018_colombia_',f,'.tif')
        r <- raster(fname)
        d[[f]] <- raster::extract(r, d[, c("x", "y")])
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
    stats <<- df

    # read data to predict on
    fname = paste0(feats_dir,merged)
    r <- stack(fname)
    dp = data.frame(rasterToPoints(r))
    dp = dp[complete.cases(dp),]
    colnames(dp) = append(c('x', 'y'), features) # rename columns

    d <<- robustscaler_sklearn(d)
    dp <<- robustscaler_sklearn(dp)
    return(r)
}

stack_ <- function(d, dp) {
    print(paste0(now(),' Stacking and making meshes'))
    # affects how many traingles are made inside the mesh, increased by x10 to reduce triangles
    coo <<- cbind(d$x, d$y)
    mesh <<- inla.mesh.2d(
        loc = coo, max.edge = c(1, 50),
        cutoff = 0.1
        #   loc = coo, max.edge = c(0.1, 5),
        #   cutoff = 0.01
    )

    # sense checks
    # mesh$n
    # plot(mesh)
    # points(coo, col = "red")

    spde <<- inla.spde2.matern(mesh = mesh, alpha = 2, constr = TRUE)

    indexs <<- inla.spde.make.index("s", spde$n.spde)
    lengths(indexs)

    A <<- inla.spde.make.A(mesh = mesh, loc = coo)

    coop <<- cbind(dp$x, dp$y)

    Ap <<- inla.spde.make.A(mesh = mesh, loc = coop)

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

    stk.full <<- inla.stack(stk.e, stk.p)

}

fit_ <- function(
    subset = FALSE,
    theta0 = NULL,
    x0 = NULL
){
    print(paste0(now(),' Fitting INLA model'))
    if (subset == TRUE) {
        control_mode_list = list();
        control_compute_list = list();
    } else {
        control_mode_list = list(theta = theta0, x = x0);
        control_compute_list = list(openmp.strategy="huge");
        # control_compute_list = list();
    }

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
                , control.mode = control_mode_list
                , control.compute = control_compute_list
                # , verbose = TRUE
    )
    return(res)
}

predict_ <- function(res, r){
    print(paste0(now(),' Predicting using fitted INLA model'))

    ## 9.4 Mapping malaria prevalence

    index <- inla.stack.index(stack = stk.full, tag = "pred")$data

    prev_mean <- res$summary.fitted.values[index, "mean"]
    prev_ll <- res$summary.fitted.values[index, "0.025quant"]
    prev_ul <- res$summary.fitted.values[index, "0.975quant"]

    r_prev_mean <- rasterize(
        x = coop, y = r, field = prev_mean,
        fun = mean
    )

    rf <- writeRaster(r_prev_mean, filename=paste0("preds_", indicator, "_", area, ".tif"), format="GTiff", overwrite=TRUE)

    # output coefficients to a text file
    out <- capture.output(summary(res))
    cat("Model results", out, file=paste0("model_summary_",indicator,"_",area,".txt"), sep="\n", append=TRUE)
}

main_ <- function() {
    print(paste0('Processing ',indicator,' for ',area))
    r = read_()
    
    # subset to get initial params
    set.seed(42)
    n_samples = round(dim(d)[1]*0.1)
    sub = d[sample(nrow(d), n_samples), ]
    subp = dp[0:5,]
    
    stack_(sub, subp)
    res0 = fit_(subset = TRUE)
    theta0 = res0$mode$theta
    x0 = res0$mode$x
    print(paste0(now(),' Fitting to subset done.'))

    stack_(d, dp)
    res = fit_(subset = FALSE)
    predict_(res, r)
}

main_()
