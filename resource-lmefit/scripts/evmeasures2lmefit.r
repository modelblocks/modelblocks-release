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


########################################################
#
# Method definitions
#
########################################################

processArgs <- function(cliargs) {
    # -abl, -dev, -test, -fl, -fse, -fsc, -ff, -logfdur, -fp, -gp, #-subj, sentid
    opt_list <- list(
        make_option(c('-b', '--bformfile'), type='character', default='../resource-rt/scripts/mem.lmeform', help='Path to LME formula specification file (<name>.lmeform'),
        make_option(c('-a', '--abl'), type='character', default=NULL, help='Effect(s) to ablate, delimited by "+". Effects that are not already in the baseline specification will be ignored (to add new effects to the baseline formula in order to ablate them, use the -A (--all) option.'),
        make_option(c('-A', '--all'), type='character', default=NULL, help='All main effects, delimited by "+". Effects that are not already in the baseline specification will be added as fixed and random effects.'),
        make_option(c('-x', '--extra'), type='character', default=NULL, help='All main effects, delimited by "+". Effects that are not already in the baseline specification will be added as fixed and random effects.'),
        make_option(c('-c', '--corpus'), type='character', default='corpus', help='Name of corpus (for output labeling).'),
        make_option(c('-R', '--restrdomain'), type='character', default=NULL, help='Basename of *.restrdomain.txt file (must be in modelblocks-repository/resource-lmefit/scripts/) containing key-val pairs for restricting domain (see file "noNVposS1.restrdomain.txt" in this directory for formatting).'),
        make_option(c('-d', '--dev'), type='logical', action='store_true', default=FALSE, help='Run evaluation on dev dataset.'),
        make_option(c('-t', '--test'), type='logical', action='store_true', default=FALSE, help='Run evaluation on test dataset.'),
        make_option(c('-e', '--entire'), type='logical', action='store_true', default=FALSE, help='Run evaluation on entire dataset.'),
        make_option(c('-s', '--splitcols'), type='character', default='subject+sentid', help='"+"-delimited list of columns to intersect in order to create a single ID for splitting dev and test (default="subject+sentid")'),
        make_option(c('-P', '--partition'), type='numeric', default=3, help='Skip to use in dev/test partition (default = 3).'),
        make_option(c('-N', '--filterlines'), type='logical', action='store_true', default=FALSE, help='Filter out events at line boundaries.'),
        make_option(c('-S', '--filtersents'), type='logical', action='store_true', default=FALSE, help='Filter out events at sentence boundaries.'),
        make_option(c('-C', '--filterscreens'), type='logical', action='store_true', default=FALSE, help='Filter out events at screen boundaries.'),
        make_option(c('-F', '--filterfiles'), type='logical', action='store_true', default=FALSE, help='Filter out events at file boundaries.'),
        make_option(c('-p', '--filterpunc'), type='logical', action='store_true', default=FALSE, help='Filter out events containing phrasal punctuation.'),
        make_option(c('-f', '--firstpass'), type='logical', action='store_true', default=FALSE, help='Use first-pass durations as the dependent variable.'),
        make_option(c('-g', '--gopast'), type='logical', action='store_true', default=FALSE, help='Use go-past durations as the dependent variable.'),
        make_option(c('-l', '--logfdur'), type='logical', action='store_true', default=FALSE, help='Log transform fixation durations.'),
        make_option(c('-X', '--boxcox'), type='logical', action='store_true', default=FALSE, help='Use Box & Cox (1964) to find and apply the best power transform of the dependent variable.'),
        make_option(c('-L', '--logmain'), type='logical', action='store_true', default=FALSE, help='Log transform main effect.'),
        make_option(c('-G', '--groupingfactor'), type='character', default=NULL, help='A grouping factor to run as an interaction with the main effect (if numeric, will be coerced to categorical).'),
        make_option(c('-n', '--indicatorlevel'), type='character', default=NULL, help='If --groupingfactor has been specified, creates an indicator variable for a particular factor level to test for interaction with the main effect.'),
        make_option(c('-i', '--crossfactor'), type='character', default=NULL, help='An interaction term to cross with (and add to) the main effect (if numeric, remains numeric, otherwise identical to --groupingfactor).'),
        make_option(c('-r', '--restrict'), type='character', default=NULL, help='Restrict the data to a subset defined by <column>+<value>. Example usage: -r pos+N.'),
        make_option(c('-I', '--interact'), type='logical', action='store_false', default=TRUE, help="Do not include interaction term between random slopes and random intercepts.")
    )
    
    opt_parser <- OptionParser(option_list=opt_list)
    opts <- parse_args(opt_parser, positional_arguments=2)
    params <- opts$options

    if (!is.null(params$all)) {
        opts$options$addEffects <- strsplit(params$all,'+',fixed=T)[[1]]
    } else opts$options$addEffects <- c()
    
    if (!is.null(params$abl)) {
        opts$options$ablEffects <- strsplit(params$abl,'+',fixed=T)[[1]]
    } else opts$options$ablEffects <- c()

    if (!is.null(params$extra)) {
        opts$options$extraEffects <- strsplit(params$extra,'+',fixed=T)[[1]]        
    } else opts$options$extraEffects <- c()
   
    if (!is.null(params$restrict)) {
        smartPrint('Restricting!')
        smartPrint(params$restrict)
        restrictor = strsplit(params$restrict, '+', fixed=T)[[1]]
        opts$options$restrict = list(col = restrictor[1], val = restrictor[2])
        smartPrint(paste0('Restricting data to ', opts$options$restrict$col,'=', opts$options$restrict$val))
    }
 
    if (params$test) {
        smartPrint('Evaluating on confirmatory (test) data')
    } else if (params$entire) {
        smartPrint('Evaluating on complete data')
    } else {
       opts$options$dev <- TRUE
        smartPrint("Evaluating on exploratory (dev) data")
    }

    if (!params$entire) {
        opts$options$splitcols <- strsplit(params$splitcols, '+', fixed=T)[[1]]  
        smartPrint(paste0('Splitting dev/test on ', paste(opts$options$splitcols, collapse=' + ')))
    } 

    if (params$firstpass) {
        smartPrint('Evaluating on first-pass fixation durations')
    } else if (params$gopast) {
        smartPrint('Evaluating on go-past fixation durations')
    }

    if (length(params$groupingfactor) > 0) {
       smartPrint(paste0('Grouping the main effect by factor ', params$groupingfactor))
    }
    
    if (params$logfdur && params$boxcox) {
        stop('Incompatible options: cannot apply logarithmic and power transformations simultaneously')
    }
    if (length(params$groupingfactor) > 0 && length(params$crossfactor) > 0) {
        stop('Incompatible options: cannot simultaneously apply --groupingfactor and --crossfactor')
    }
    if (length(params$indicatorlevel) > 0) {
        if (length(params$groupingfactor) <= 0) {
            stop('Incompatible options: --indicatorlevel requires a specification for --groupingfactor')
        } else smartPrint(paste0('Using indicator variable for ', params$groupingfactor, '=', params$indicatorlevel, '.'))
    } 

    if (params$logfdur) {
        smartPrint('Log-transforming fdur')
    }
    if (params$boxcox) {
        smartPrint('Using Box & Cox (1964) to find and apply the best power transform of the dependent variable.')
    }

    return(opts)
}


########################################################
#
# Process CLI Arguments
#
########################################################

opts <- processArgs()
params <- opts$options
input <- opts$args[1] # positional arg, input file specification
output <- opts$args[2] # positional arg, output file specification

smartPrint('Reading data from file')
data <- read.table(input, header=TRUE, quote='', comment.char='')
data <- cleanupData(data, params)
data <- recastEffects(data, params)

if (params$dev) {
    data <- create.dev(data, params$partition)
} else if (params$test) {
    data <- create.test(data, params$partition)
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


smartPrint(paste0('SD of main effect z.(', params$addEffects, '): ', sd(data[[params$addEffects]])))
smartPrint(paste0('Range of main effect z.(', params$addEffects, '): ', max(data[[params$addEffects]])-min(data[[params$addEffects]])))


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


lmefit(data, output, params)
