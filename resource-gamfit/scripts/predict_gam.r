#!/usr/bin/env Rscript
options(width=200) 

#########################
#
# Loading Data and Libraries
#
#########################

args <- commandArgs(trailingOnly=TRUE)
cliargs <- args[-(1:2)]
options('warn'=1) #report non-convergences, etc

library(mgcv)
# library(languageR)
library(optimx)
library(ggplot2)
#The below scripts cannot be distributed with Modelblocks
# Relative path code from Suppressingfire on StackExchange
initial.options <- commandArgs(trailingOnly = FALSE)
file.arg.name <- "--file="
script.name <- sub(file.arg.name, "", initial.options[grep(file.arg.name, initial.options)])
script.basename <- dirname(script.name)
wd = getwd()
setwd(script.basename)
source('../../resource-rhacks/scripts/mer-utils.R') #obtained from https://github.com/aufrank
source('../../resource-rhacks/scripts/regression-utils.R') #obtained from https://github.com/aufrank
source('../../resource-gamfit/scripts/gamtools.r')
setwd(wd)

model_data <- get(load(args[1]))
model <- model_data$model
input <- file(args[2], open='rt', raw=TRUE)
df <- read.table(input, header=TRUE, sep=' ', quote='', comment.char='') 
close(input)
df <- cleanupData(df, stdout=FALSE)
df <- recastEffects(df, stdout=FALSE)

f <- model_data$f
y <- model.frame(paste0(toString(f[2]), '~ 1'), data=df)
# print(y)

colnames(y) = c('y')
y_hat <- predict(model, newdata=df, type='response', allow.new.levels=TRUE)
# colnames(y_hat) = c('y_hat')
# err = y-y_hat
# colnames(err) = c('err')
# ae = abs(err)
# colnames(ae) = c('ae')
# se = err^2
# colnames(se) = c('se')
# # write.table(cbind(y,y_hat,err,ae,se), file=outfile, quote=FALSE, row.names=FALSE)
# # write.table(se, file=stdout(), quote=FALSE, col.names=FALSE, row.names=FALSE)
# write.table(cbind(df,y,y_hat,err,se), file=stdout(), quote=FALSE, col.names=TRUE, row.names=FALSE)

# code from https://github.com/coryshain/cdrgam
# gaulss and shash family not implemented
family <- model$family$family
if (family == 'gaussian') {
    mu <- y_hat
    sigma <- sqrt(mean(residuals(model)^2))
    ll <- dnorm(y$y, mean=mu, sd=sigma, log=TRUE)
} else if (family == 'gaulss') {
    mu <- response[, 1]
    sigma <- 1 / response[, 2]
    ll <- dnorm(y, mean=mu, sd=sigma, log=TRUE)
} else if (family == 'shash') {
    mu <- response[, 1]
    sigma <- exp(response[, 2])  # mgcv represents on a log scale
    nu <- response[, 3]
    tau <- exp(response[, 4])  # mgcv represents on a log scale
    ll <- gamlss.dist::dSHASHo2(y, mean=mu, sd=sigma, nu=nu, tau=tau, log=TRUE)
} else {
    stop(paste0('Unknown family: ', family))
}

cat(sprintf("%.6f", sum(ll)))
cat("\n")