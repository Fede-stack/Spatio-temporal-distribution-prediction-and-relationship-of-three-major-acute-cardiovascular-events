rm(list = ls())
set.seed(12345)
library(tidyverse)
library(readxl)   # for reading excel files
library(raster)   # for reading shape file of Canton Ticino
library(spatstat) # for tools of spatial statistics
library(RColorBrewer) # creates nice color schemes 
library(classInt) # finds class intervals for continuous variables 
library(sf)  # update and load package udunits2 before in case of problems
library(spdep) # for tools of spatial statistics
library(splines)
library(INLA)
library(viridis)

# Loading Data ------------------------------------------------------------

setwd('/Users/federico.ravenda/Documents/PROGETTO OHCA/data/ohca') # set the directory
data = readRDS("PAPER_ohca_stroke_stemi/stroke_paper_data.rds") %>%
  filter(Year < 2022)# dataset with covariates and response already cleaned
x = sf::read_sf("limiti_comuni_2017_MN95.shp") # shape file in which Ticino geographical information are synthesized
municipalities = x$NOME_COMUN
n.mun = length(municipalities)
#Create the graph for INLA
zzz = spdep::poly2nb(x)
spdep::nb2INLA("Ticino.graph", zzz)
Ticino.adj = paste(getwd(),"/Ticino.graph",sep="")
data.Ticino = attr(x, "data")
data.Ticino = inla.read.graph("Ticino.graph")

#---- naive prediction ----
# previous value
lagged_value = lag(data$stroke, n.mun)[(nrow(data)-n.mun+1):nrow(data)]
observed = as.vector(data %>% filter(Year == 2021) %>% dplyr::select(stroke))$stroke
mae_np = Metrics::mae(observed, lagged_value)
err0a = observed - lagged_value
chi2_np = sum(err0a^2/(lagged_value+0.5))
# exponential average of order 3
TT = length(unique(data$Year)); years = sort(unique(data$Year))
Y = matrix(data$stroke,n.mun,TT); dimnames(Y) = list(municipalities,years)
tmp = rowMeans(Y[,1:3])
for(t in 4:(TT-1)) tmp = tmp/2+Y[,t]/2
mYh0b = tmp
err0b = Y[,TT]-mYh0b
chi2_ema = sum(err0b^2/(mYh0b+0.5))
mae_ema = mean(abs(err0b))

# INLA FOR ALL THE (cardiac) DATA -----------------------------------------
interval = 1
real.values = data$stroke
to.NA = (length(real.values)-n.mun*interval+1):length(real.values) # NAs for 2021
data[to.NA, 'stroke'] = NA

inla_data = data %>% 
  mutate(ID.area = as.numeric(as.factor(Municipality)), 
         year1 = Year, 
         year2 = (Year)^2, 
         E = Pop, 
         ID.area2 = as.numeric(as.factor(Municipality)))
inla_data = as.data.frame(inla_data)

formula.inla = stroke ~ 1 + f(ID.area, model = "bym2", graph = Ticino.adj, scale.model = TRUE, 
                            constr = TRUE, hyper = list(prec = list(prior = "pc.prec", 
                                                                    param = c(1, 0.01)))) + ns(year1, df = 3) + 
  proportion.f + proportion.6 + proportion.5 + proportion.4 + proportion.3 + proportion.2 + big.city 

model.inla = inla(formula.inla,family="poisson",
                  data=inla_data,
                  E=data$Pop, 
                  control.predictor=list(link=1,compute=TRUE), 
                  control.compute=list(dic=TRUE,waic=TRUE,config=TRUE),
                  verbose = FALSE)
model.inla$summary.hyperpar
summary(model.inla)

to.pred    = which(is.na(inla_data$stroke))
prediction = round(inla_data$Pop[to.pred]*model.inla$summary.fitted.values$mean[to.pred])
lowerb     = round(inla_data$Pop[to.pred]*model.inla$summary.fitted.values$`0.025quant`[to.pred])
upperb     = round(inla_data$Pop[to.pred]*model.inla$summary.fitted.values$`0.975quant`[to.pred])
real_pred = data.frame(real_values = tail(real.values, n.mun*interval), denominazione_regione = tail(inla_data$Municipality,  n.mun*interval))
real_pred['pred'] = prediction; real_pred['lowerb'] = lowerb; real_pred['upperbb'] = upperb

prediction = round(inla_data$Pop*model.inla$summary.fitted.values$mean) #interpolation + extrapolation
lowerb = round(inla_data$Pop*model.inla$summary.fitted.values$`0.025quant`)
upperb = round(inla_data$Pop*model.inla$summary.fitted.values$`0.975quant`)

# Predictions with credible intervals
prediction = data.frame(data = inla_data$Year, 
                        denominazione_regione = inla_data$Municipality,
                        prediction = prediction, 
                        lowerb = lowerb, 
                        upperb = upperb, 
                        real_values = real.values) #interpolation + extrapolation

#Predictions for 2021 Year
predictions = real_pred %>%
  dplyr::select(denominazione_regione, pred, real_values) %>%
  rename(Municipality = denominazione_regione, 
         `Predicted Value` = pred, 
         `Observed Value` = real_values)


# Evaluation --------------------------------------------------------------
mae_inla = Metrics::mae(predictions$`Observed Value`, predictions$`Predicted Value`)
err_inla = as.vector(predictions %>% mutate(err = `Observed Value` - `Predicted Value`) )$err
chi2_inla = sum(err_inla^2/(predictions$`Predicted Value`+0.5))

# Visualization Results ---------------------------------------------------
#Big City
big.city = c('BELLINZONA', 'LUGANO', 'LOCARNO', 'MENDRISIO', 'CHIASSO' )
df4plot = reshape2::melt(predictions %>%
                           filter(Municipality %in% big.city) , id.vars='Municipality')
ggplot(df4plot, aes(x = Municipality, y= value, fill = variable)) +
  geom_bar(stat="identity", width=.5, position = "dodge") +
  scale_fill_hue(l=40, c=35) +
  xlab('Municipality') +
  ylab('# of strokes') +
  theme_minimal()
diff.plot = predictions %>%
  mutate(difff = as.factor(abs(`Predicted Value`-`Observed Value`))) %>%
  group_by(difff) %>%
  summarise(Discrepancies = n())
ggplot(diff.plot, aes(x = difff, y= Discrepancies)) +
  geom_bar(stat="identity", width=.5, position = "dodge", fill = 'deepskyblue4') +
  ylab('Counts') +
  xlab('Discrepancies Between Predictions and Real Values') +
  geom_text( aes(label=Discrepancies), vjust=-1) +
  theme_minimal()

lug = prediction %>% filter(denominazione_regione == 'LUGANO')
ggplot(lug) +
  geom_line(mapping = aes(x = data, y = upperb), col = 'indianred') +
  geom_line(mapping = aes(x = data, y = lowerb), col = 'indianred') +
  geom_ribbon(aes(x = data,
                  ymin = lowerb,
                  ymax = upperb),
              fill = "indianred", alpha = .4) +
  geom_line(mapping = aes(x = data, y = prediction), col = 'navyblue') +
  geom_point(mapping = aes(x = data, y = real_values), col = 'navyblue') +
  geom_point(tail(lug, 1), mapping = aes(data, prediction), col = '#0099CC', size = 10, shape = 3) +
  ylab('# of STROKEs') +
  xlab('Date') +
  theme_minimal()

results = data.frame(Model = c('Previous Value', 'Exponential Moving Average', 'INLA'), 
                     MAE = c(mae_np, mae_ema, mae_inla), 
                     Chisq = c(chi2_np, chi2_ema, chi2_inla))
results

results$Model = factor(results$Model, 
                       levels = c('Previous Value', 'Exponential Moving Average', 'INLA')) # in questo ordine
ggplot(results, mapping = aes(Model, Chisq, group=1)) +
  geom_point(size = 6, col = 'navyblue'  ) +
  stat_summary(fun.y=sum, geom="line", col = 'indianred', size = 2, lty = 2)+
  xlab('') +
  theme_minimal()


# Prediction Map ----------------------------------------------------------
x_new = readOGR("limiti_comuni_2017_MN95.shp")
poly.mun.tic = x_new
poly.mun.tic@data = poly.mun.tic@data %>%
  mutate(PREDICTIONS = predictions$`Predicted Value`,
         content = paste0(NOME_COMUN,': ',PREDICTIONS))
predictions = predictions[match(poly.mun.tic@data$NOME_COMUN, predictions$Municipality), ]
poly.mun.tic_sf = st_as_sf(poly.mun.tic)
color_scheme = leaflet::colorBin(palette = 'RdBu', domain = poly.mun.tic_sf$PREDICTIONS)
reversed_colors = rev(brewer.pal(11, 'RdBu'))
ggplot(data = poly.mun.tic_sf) +
  geom_sf(aes(fill = PREDICTIONS)) +
  scale_fill_gradientn(colors =reversed_colors, name = "PREDICTIONS") +
  theme_minimal()
