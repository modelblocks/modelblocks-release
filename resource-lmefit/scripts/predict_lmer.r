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

library(lme4)
library(languageR)
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
source('../../resource-lmefit/scripts/lmetools.r')
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
colnames(y) = c('y')
y_hat <- data.frame(list(y_hat=predict(model, newdata=df, type='response', allow.new.levels=TRUE)))
colnames(y_hat) = c('y_hat')
err = y-y_hat
colnames(err) = c('err')
ae = abs(err)
colnames(ae) = c('ae')
se = err^2
colnames(se) = c('se')
# write.table(cbind(y,y_hat,err,ae,se), file=outfile, quote=FALSE, row.names=FALSE)
write.table(se, file=stdout(), quote=FALSE, row.names=FALSE)

