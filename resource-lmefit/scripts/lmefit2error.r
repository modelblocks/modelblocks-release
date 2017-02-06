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
source('../resource-lmefit/scripts/lmetools.r')
# The below scripts cannot be distributed with Modelblocks
source('../resource-rhacks/scripts/mer-utils.R') #obtained from https://github.com/aufrank
source('../resource-rhacks/scripts/mtoolbox.R')  
source('../resource-rhacks/scripts/regression-utils.R') #obtained from https://github.com/aufrank
options('warn'=1) #report non-convergences, etc

########################################################
#
# Method definitions
#
########################################################

processArgs <- function(cliargs) {
    opt_list <- list(
        make_option(c('-b', '--base'), type='character', help='Path to LME baseline object file (<name>.rdata)'),
        make_option(c('-m', '--main'), type='character', help='Path to LME main effect object file (<name>.rdata)'),
        make_option(c('-i', '--input'), type='character', help='Path to input data table (if not specified, defaults to stdin)'),
        make_option(c('-o', '--output'), type='character', help='Path to output data table (if not specified, defaults to stdout)'),
        make_option(c('-d', '--dev'), type='logical', action='store_true', default=FALSE, help='Run evaluation on dev dataset.'),
        make_option(c('-t', '--test'), type='logical', action='store_true', default=FALSE, help='Run evaluation on test dataset.'),
        make_option(c('-e', '--entire'), type='logical', action='store_true', default=FALSE, help='Run evaluation on entire dataset.'),
        make_option(c('-s', '--splitcols'), type='character', default='subject+sentid', help='"+"-delimited list of columns to intersect in order to create a single ID for splitting dev and test (default="subject+sentid")'),
        make_option(c('-P', '--partition'), type='numeric', default=3, help='Skip to use in dev/test partition (default = 3).'),
        make_option(c('-N', '--filterlines'), type='logical', action='store_true', default=FALSE, help='Filter out events at line boundaries.'),
        make_option(c('-S', '--filtersents'), type='logical', action='store_true', default=FALSE, help='Filter out events at sentence boundaries.'),
        make_option(c('-C', '--filterscreens'), type='logical', action='store_true', default=FALSE, help='Filter out events at screen boundaries.'),
        make_option(c('-F', '--filterfiles'), type='logical', action='store_true', default=FALSE, help='Filter out events at file boundaries.'),
        make_option(c('-f', '--firstpass'), type='logical', action='store_true', default=FALSE, help='Use first-pass durations as the dependent variable.'),
        make_option(c('-g', '--gopast'), type='logical', action='store_true', default=FALSE, help='Use go-past durations as the dependent variable.'),
        make_option(c('-l', '--logfdur'), type='logical', action='store_true', default=FALSE, help='Log transform fixation durations.'),
        make_option(c('-X', '--boxcox'), type='logical', action='store_true', default=FALSE, help='Use Box & Cox (1964) to find and apply the best power transform of the dependent variable.')
    )
    
    opt_parser <- OptionParser(option_list=opt_list)
    opts <- parse_args(opt_parser, positional_arguments=0)
    params <- opts$options

    if (!is.null(params$base)) {
        opts$options$base_obj <- get(load(params$base))
    } else stop('No baseline *.rdata object provided. Use the -b (--base) option.')

    if (!is.null(params$main)) {
        opts$options$main_obj <- get(load(params$main))
    } else stop('No main effect *.rdata object provided. Use the -m (--main) option.')
    
    if (is.null(params$input)) opts$options$input <- 'stdin'
    if (is.null(params$output)) stop('No output file path specified. Use the -o (--output) option.')

    if (params$test) {
        smartPrint('Analyzing predictions on confirmatory (test) data')
    } else if (params$entire) {
        smartPrint('Evaluating on complete data')
    } else {
       opts$options$dev <- TRUE
        smartPrint("Analyzing predictions on exploratory (dev) data")
    }
    if (!params$entire) {
        opts$options$splitcols <- strsplit(params$splitcols, '+', fixed=T)[[1]]  
        smartPrint(paste0('Splitting dev/test on ', paste(opts$options$splitcols, collapse=' + ')))
    } 

    if (params$firstpass) {
        smartPrint('Predicting first-pass fixation durations')
    } else if (params$gopast) {
        smartPrint('Predicting go-past fixation durations')
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

smartPrint('Reading data from file')
data <- read.table(params$input, header=TRUE, quote='', comment.char='')
data <- cleanupData(data, params)
data <- recastEffects(data, params)

if (params$dev) {
    data <- create.dev(data, params$partition)
} else if (params$test) {
    data <- create.test(data, params$partition)
}


########################################################
#
# Main Program
#
########################################################

errData <- error_anal(data, params)
write.table(errData,file=params$output,quote=FALSE,row.names=FALSE)
