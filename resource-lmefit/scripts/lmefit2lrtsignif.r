#!/usr/bin/Rscript

#########################
#
# Loading Data and Libraries
#
#########################

smartPrint <- function(string) {
    print(string)
    write(string, stderr())
}

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
setwd(wd)

basefile <- load(args[1])
base <- get(basefile)
corpus <- base$corpus

mainfile <- load(args[2])
main <- get(mainfile)

#########################
#
# Definitions
#
#########################

# Output a summary of model fit
printSummary <- function(reg) {
    print(paste('LME Fit Summary (',reg@optinfo$optimizer,')',sep=''))
    print(summary(reg))
    relgrad <- with(reg@optinfo$derivs,solve(Hessian,gradient))
    print('Relative Gradient (<0.002?)') #check for convergence even if warned that convergence failed
    smartPrint(max(abs(relgrad)))
    smartPrint('AIC:')
    smartPrint(AIC(logLik(reg)))
}

printSignifSummary <- function(mainName, mainEffect, basemodel, testmodel, signif) {
    print(paste('Main effect:', mainName))
    print(paste('Corpus:', corpus))
    print(paste('Effect estimate (',mainEffect,'): ', summary(testmodel)[[10]][mainEffect,'Estimate'], sep=''))
    print(paste('t value (',mainEffect,'): ', summary(testmodel)[[10]][mainEffect,'t value'], sep=''))
    print(paste('Significance (Pr(>Chisq)):', signif[['Pr(>Chisq)']][[2]]))
    print(paste('Relative gradient (baseline):',  max(abs(with(basemodel@optinfo$derivs,solve(Hessian,gradient))))))
    print(paste('Relative gradient (main effect):',  max(abs(with(testmodel@optinfo$derivs,solve(Hessian,gradient))))))
    print(signif)
}

smartPrint('Summary of baseline model')
printSummary(base$model)
smartPrint(paste('Summary of main effect (', setdiff(base$abl,main$abl), ') model', sep=''))
printSummary(main$model)

print('Significance testing')
smartPrint(paste('Baseline vs. Main Effect (', setdiff(base$abl,main$abl), ')', sep=''))
printSignifSummary(setdiff(base$abl,main$abl),
                   setdiff(base$ablEffects,main$ablEffects),
                   base$model,
                   main$model,
                   anova(base$model, main$model))

print(signif)

warnings()
