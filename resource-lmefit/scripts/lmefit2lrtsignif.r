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

basefile <- load(args[1])
base <- get(basefile)
corpus <- base$corpus

mainfile <- load(args[2])
main <- get(mainfile)

if ('-p' %in% args) {
    ix = match('-p', args) + 1
    if (args[ix] == 'train') {
        permutation_test_data='train'
    } else if(args[ix] == 'dev') {
        permutation_test_data='dev'
    } else if(args[ix] == 'test') {
        permutation_test_data='test'
    } else {
        stop('Permutation test requested on invalid partition name')
    }
} else {
    permutation_test_data=NULL
}

#########################
#
# Definitions
#
#########################

printSignifSummary <- function(mainName, mainEffect, basemodel, testmodel, signif, base_diff=NULL, aov=NULL) {
    cat(paste0('Main effect: ', mainName), '\n')
    cat(paste0('Corpus: ', corpus), '\n')
    cat(paste0('Effect estimate (',mainEffect,'): ', summary(testmodel)[[10]][mainEffect,'Estimate'], sep=''), '\n')
    cat(paste0('t value (',mainEffect,'): ', summary(testmodel)[[10]][mainEffect,'t value'], sep=''), '\n')
    if (!is.null(base_diff)) {
        cat(paste0('MSE loss improvement over baseline model (', permutation_test_data, ' set): ', base_diff, '\n'))
    }
    cat(paste0('p value: ', signif), '\n')
    cat(paste0('Baseline loglik: ', logLik(basemodel), '\n'))
    cat(paste0('Full loglik: ', logLik(testmodel), '\n'))
    cat(paste0('Log likelihood ratio: ', logLik(testmodel)-logLik(basemodel), '\n'))
    cat(paste0('Relative gradient (baseline): ',  max(abs(with(basemodel@optinfo$derivs,solve(Hessian,gradient)))), '\n'))
    convWarnBase <- is.null(basemodel@optinfo$conv$lme4$messages)
    cat(paste0('Converged (baseline): ', as.character(convWarnBase), '\n'))
    cat(paste0('Relative gradient (main effect):',  max(abs(with(testmodel@optinfo$derivs,solve(Hessian,gradient)))), '\n'))
    convWarnMain <- is.null(testmodel@optinfo$conv$lme4$messages)
    cat(paste0('Converged (main effect): ', as.character(convWarnMain), '\n'))
    if (!is.null(aov)) {
        print(aov)
    }
}

cat('Likelihood Ratio Test (LRT) Summary\n')
cat('===================================\n\n')
cat('Correlation of numeric variables in model:\n')
print(main$correlations)
cat('\n\n')
cat('SDs of numeric variables in model:\n')
for (c in names(main$sd_vals)) {
    cat(paste0(c, ': ', main$sd_vals[[c]], '\n'))
}
cat('\n')

smartPrint('Summary of baseline model')
printSummary(base$model)
smartPrint(paste('Summary of main effect (', setdiff(base$abl,main$abl), ') model', sep=''))
printSummary(main$model)

if (!is.null(permutation_test_data)) {
    aov = NULL
    if (permutation_test_data == 'train') {
        mse1_file = gsub('.rdata', '.train.mse.txt', args[1])
        mse2_file = gsub('.rdata', '.train.mse.txt', args[2])
    } else if (permutation_test_data == 'dev') {
        mse1_file = gsub('.rdata', '.dev.mse.txt', args[1])
        mse2_file = gsub('.rdata', '.dev.mse.txt', args[2])
    } else if (permutation_test_data == 'test') {
        mse1_file = gsub('.rdata', '.test.mse.txt', args[1])
        mse2_file = gsub('.rdata', '.test.mse.txt', args[2])
    }
    mse1 = read.csv(mse1_file, sep=' ', quote='', comment.char='', header=TRUE)
    mse2 = read.csv(mse2_file, sep=' ', quote='', comment.char='', header=TRUE)
    ptest_out = permutation_test(mse1$se, mse2$se, n_iter=10000, n_tail=2)
    signif = ptest_out$p
    base_diff = ptest_out$base_diff
} else {
    aov = anova(base$model, main$model)
    base_diff = NULL
    signif = aov[['Pr(>Chisq)']][[2]]
}


cat('Significance results\n')
smartPrint(paste('Baseline vs. Main Effect (', setdiff(base$abl,main$abl), ')', sep=''))
printSignifSummary(setdiff(base$abl,main$abl),
                   setdiff(base$ablEffects,main$ablEffects),
                   base$model,
                   main$model,
                   signif,
                   base_diff,
                   aov)

if (!is.null(main$lambda)) {
    printBoxCoxInvBetas(main$beta_ms, main$lambda, main$y_mu, main$sd_vals)
}

