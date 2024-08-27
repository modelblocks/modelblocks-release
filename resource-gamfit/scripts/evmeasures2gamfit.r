#!/usr/bin/env Rscript
options(width=200,digits=7)

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
options('warn'=1) #report non-convergences, etc

########################################################
#
# Process CLI Arguments
#
########################################################

opts <- processLMEArgs()
params <- opts$options
input <- opts$args[1] # positional arg, input file specification
input <- file(input, open='rt', raw=TRUE)
output <- opts$args[2] # positional arg, output file specification

cat('Regression Modeling Log\n')
cat('=======================\n\n')

smartPrint('Reading data from file')
data <- read.table(input, header=TRUE, sep=' ', quote='', comment.char='')
close(input)
if (!params$entire) {
    data <- computeSplitIDs(data, params$splitcols)
}
data <- cleanupData(data, params$filterfiles, params$filterlines, params$filtersents, params$filterscreens, params$filterpunc, params$restrdomain, params$upperbound, params$lowerbound, params$mincorrect)
data <- recastEffects(data, params$splitcols, params$indicatorlevel, params$groupingfactor)
data_full <- data

if (params$dev) {
    data <- create.dev(data, params$partitionmod, params$partitiondevindices)
} else if (params$test) {
    data <- create.test(data, params$partitionmod, params$partitiondevindices)
}

if (!is.null(params$restrict)) {
    data <- data[data[[params$restrict$col]] == params$restrict$val,]
    smartPrint(paste0('Domain-restricted data rows: ', nrow(data)))
}

if (params$boxcox) {
    f <- file(description=params$bformfile, open='r')
    flines <- readLines(f)
    close(f)
    DV <- flines[1]
    bc <- MASS:::boxcox(as.formula(paste0(DV, '~ 1')), data=data)
    params$lambda <- bc$x[which.max(bc$y)]
    smartPrint(paste0('Box & Cox lambda: ', params$lambda))
}



########################################################
#
# Main Program
#
########################################################

for (effect in params$addEffects) {
    smartPrint(paste0('SD of main effect ', effect, ': ', sd(data[[effect]])))
    smartPrint(paste0('Range of main effect ', effect, ': ', max(data[[effect]])-min(data[[effect]])))
}

if (params$totable) {
   smartPrint('Writing experiment data table to file') 
   write.table(data, output, sep=' ', quote=FALSE, row.names=FALSE, na="nan")
   quit()
}

if (length(params$groupingfactor) > 0) {
    for (e in params$addEffects) {
        smartPrint(paste0('Within-group statistics for main effect ', e, ' by grouping factor ', params$groupingfactor, ':'))
        for (g in levels(as.factor(data[[params$groupingfactor]]))) {
            smartPrint(paste0('    Group: ', g))
            g_data = data[data[[params$groupingfactor]] == g,]
            smartPrint(paste0('        Number of events: ', nrow(g_data)))
            smartPrint(paste0('        Within-group ' , e , ' mean: ', mean(g_data[[e]])))
            smartPrint(paste0('        Within-group ' , e , ' SD: ', sd(g_data[[e]])))
            smartPrint(paste0('        Within-group ' , e , ' range: ', max(g_data[[e]]-min(g_data[[e]]))))
            smartPrint(paste0('        Within-group ' , e , ' 5th percentile: ', quantile(g_data[[e]], c(0.05))))
            smartPrint(paste0('        Within-group ' , e , ' 95th percentile: ', quantile(g_data[[e]], c(0.95))))
        }
    }
}

fit <- fitModel(data, output, params$bformfile, params$fitmode,
                     params$logmain, params$logdepvar, params$lambda,
                     params$addEffects, params$extraEffects, params$ablEffects,
                     params$groupingfactor, params$indicatorlevel, params$crossfactor,
                     params$interact, params$corpus, params$suppress_nlminb,
                     params$noMainRandomEffect)

# residuals <- resid(fit$m)
# saveRDS(residuals, file = paste("residuals",format(Sys.time()),"rds", sep = "."))


if (fit$fitmode != 'bme') {
    m = fit$m
    f = fit$f
    f = as.formula(paste0(toString(f[2]), ' ~ 1'))
     
    if (params$trainmse) {
        data <- create.dev(data, params$partitionmod, params$partitiondevindices)
        y <- model.frame(f, data=data)[toString(f[2])]
        colnames(y) = c('y')
        y_hat <- data.frame(list(y_hat=predict(m, newdata=data, type='response', allow.new.levels=TRUE)))
        err = y-y_hat
        colnames(err) = c('err')
        ae = abs(err)
        colnames(ae) = c('ae')
        se = err^2
        colnames(se) = c('se')
        outfile = gsub('.rdata', '.train.mse.txt', output)
        write.table(cbind(y,y_hat,err,ae,se), file=outfile, quote=FALSE, row.names=FALSE)
    }
    
    if (params$devmse) {
        data <- create.dev(data_full, params$partitionmod, params$partitionmod-2)
        y <- model.frame(f, data=data)[toString(f[2])]
        colnames(y) = c('y')
        y_hat <- data.frame(list(y_hat=predict(m, newdata=data, type='response', allow.new.levels=TRUE)))
        colnames(y_hat) = c('y_hat')
        err = y-y_hat
        colnames(err) = c('err')
        ae = abs(err)
        colnames(ae) = c('ae')
        se = err^2
        colnames(se) = c('se')
        outfile = gsub('.rdata', '.dev.mse.txt', output)
        write.table(cbind(y,y_hat,err,ae,se), file=outfile, quote=FALSE, row.names=FALSE)
    }
    
    if (params$testmse) {
        data <- create.dev(data_full, params$partitionmod, params$partitionmod-1)
        y <- model.frame(f, data=data)[toString(f[2])]
        colnames(y) = c('y')
        y_hat <- data.frame(list(y_hat=predict(m, newdata=data, type='response', allow.new.levels=TRUE)))
        colnames(y_hat) = c('y_hat')
        err = y-y_hat
        colnames(err) = c('err')
        ae = abs(err)
        colnames(ae) = c('ae')
        se = err^2
        colnames(se) = c('se')
        outfile = gsub('.rdata', '.test.mse.txt', output)
        write.table(cbind(y,y_hat,err,ae,se), file=outfile, quote=FALSE, row.names=FALSE)
    }
} 
