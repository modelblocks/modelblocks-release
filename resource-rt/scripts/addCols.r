#!/usr/bin/Rscript

########################################################
#
# Reusable evaluation containing typical columns
# used in experiments using reading latency as a
# dependent variable with memory and syntactic
# predictors.
#
########################################################

########################################################
#
# Load Data and Libraries
#
########################################################

# Import packages
library(lme4)
library(languageR)
library(optimx)
library(ggplot2)
library(optparse)
# The below scripts cannot be distributed with Modelblocks
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
options('warn'=1) #report non-convergences, etc

data <- read.table(file("stdin"), header=TRUE, quote='', comment.char='')
data <- addColumns(data)
write.table(data, stdout(), sep=' ', quote=FALSE, row.names=FALSE, na="nan")
