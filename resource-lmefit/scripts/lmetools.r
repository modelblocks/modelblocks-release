########################################################
#
# Commonly used methods for LME regressions
#
########################################################

'%ni%' <- Negate('%in%')

# Requires the optparse library
processLMEArgs <- function() {
    library(optparse)
    opt_list <- list(
        make_option(c('-b', '--bformfile'), type='character', default='../resource-rt/scripts/mem.lmeform', help='Path to LME formula specification file (<name>.lmeform'),
        make_option(c('-a', '--abl'), type='character', default=NULL, help='Effect(s) to ablate, delimited by "+". Effects that are not already in the baseline specification will be added as a random slope, unless --noMainRandomEffect is used.'),
        make_option(c('-A', '--all'), type='character', default=NULL, help='Effect(s) to add, delimited by "+". Effects that are not already in the baseline specification will be added as fixed effects, and also as random slopes unless --noMainRandomEffect is used.'),
        make_option(c('-x', '--extra'), type='character', default=NULL, help='Additional (non-main) effect(s) to add, delimited by "+". Effects that are not already in the baseline specification will be added as fixed and random effects.'),
        make_option(c('-c', '--corpus'), type='character', default=NULL, help='Name of corpus (for output labeling). If not specified, will try to infer from output filename.'),
        make_option(c('-m', '--fitmode'), type='character', default='lme', help='Fit mode. Currently supports "lme" (linear mixed effects), "bme" (Bayesian mixed effects), and "lm" (simple linear regression, which discards all random terms). Defaults to "lme".'),
        make_option(c('-R', '--restrdomain'), type='character', default=NULL, help='Basename of *.restrdomain.txt file (must be in modelblocks-repository/resource-lmefit/scripts/) containing key-val pairs for restricting domain (see file "noNVposS1.restrdomain.txt" in this directory for formatting).'),
        make_option(c('-d', '--dev'), type='logical', action='store_true', default=FALSE, help='Run evaluation on dev dataset.'),
        make_option(c('-t', '--test'), type='logical', action='store_true', default=FALSE, help='Run evaluation on test dataset.'),
        make_option(c('-e', '--entire'), type='logical', action='store_true', default=FALSE, help='Run evaluation on entire dataset.'),
        make_option(c('-s', '--splitcols'), type='character', default='subject+sentid', help='"+"-delimited list of columns to intersect in order to create a single ID for splitting dev and test (default="subject+sentid")'),
        make_option(c('-M', '--partitionmod'), type='numeric', default=3, help='Modulus to use in dev/test partition (default = 4)'),
        make_option(c('-K', '--partitiondevindices'), type='character', default=0, help='Comma-delimited list of indices to retain in dev set (default = "0,1")'),
        make_option(c('-N', '--filterlines'), type='logical', action='store_true', default=FALSE, help='Filter out events at line boundaries.'),
        make_option(c('-S', '--filtersents'), type='logical', action='store_true', default=FALSE, help='Filter out events at sentence boundaries.'),
        make_option(c('-C', '--filterscreens'), type='logical', action='store_true', default=FALSE, help='Filter out events at screen boundaries.'),
        make_option(c('-F', '--filterfiles'), type='logical', action='store_true', default=FALSE, help='Filter out events at file boundaries.'),
        make_option(c('-p', '--filterpunc'), type='logical', action='store_true', default=FALSE, help='Filter out events containing phrasal punctuation.'),
        make_option(c('-U', '--upperbound'), type='numeric', default=NULL, help='Filter out events with response value >= n.'),
        make_option(c('-B', '--lowerbound'), type='numeric', default=NULL, help='Filter out events with response value <= n.'),
        make_option(c('-o', '--mincorrect'), type='numeric', default=NULL, help='Filter out events with number correct < n.'),
        make_option(c('-l', '--logdepvar'), type='logical', action='store_true', default=FALSE, help='Log transform fixation durations.'),
        make_option(c('-X', '--boxcox'), type='logical', action='store_true', default=FALSE, help='Use Box & Cox (1964) to find and apply the best power transform of the dependent variable.'),
        make_option(c('-L', '--logmain'), type='logical', action='store_true', default=FALSE, help='Log transform main effect.'),
        make_option(c('-G', '--groupingfactor'), type='character', default=NULL, help='A grouping factor to run as an interaction with the main effect (if numeric, will be coerced to categorical).'),
        make_option(c('-n', '--indicatorlevel'), type='character', default=NULL, help='If --groupingfactor has been specified, creates an indicator variable for a particular factor level to test for interaction with the main effect.'),
        make_option(c('-i', '--crossfactor'), type='character', default=NULL, help='An interaction term to cross with (and add to) the main effect (if numeric, remains numeric, otherwise identical to --groupingfactor).'),
        make_option(c('-r', '--restrict'), type='character', default=NULL, help='Restrict the data to a subset defined by <column>+<value>. Example usage: -r pos+N.'),
        make_option(c('-I', '--interact'), type='logical', action='store_false', default=TRUE, help="Do not include interaction term between random slopes and random intercepts."),
        make_option(c('-u', '--trainmse'), type='logical', action='store_true', default=FALSE, help='Generate error table for train partition.'),
        make_option(c('-v', '--devmse'), type='logical', action='store_true', default=FALSE, help='Generate error table for dev partition.'),
        make_option(c('-w', '--testmse'), type='logical', action='store_true', default=FALSE, help='Generate error table for test partition.'),
        make_option(c('-T', '--totable'), type='logical', action='store_true', default=FALSE, help="Preprocess data and output table only (do not regress)."),
        make_option(c('--seed'), type='numeric', default=NULL, help='Set random seed.'),
        make_option(c('--suppress_nlminb'), type='logical', action='store_true', default=FALSE, help='If BOBYQA fails, do not attempt to use NLMINB.'),
        make_option(c('--noMainRandomEffect'), type='logical', action='store_true', default=FALSE, help='Remove per-subject random slopes for the main effects provided by the -a and -A options')
    )
    opt_parser <- OptionParser(option_list=opt_list)
    opts <- parse_args(opt_parser, positional_arguments=2)
    params <- opts$options

    if (!is.null(params$seed)) {
        set.seed(params$seed)
    }
    
    if (is.null(params$corpus)) {
        filename = strsplit(opts$args[2], '/', fixed=T)[[1]]
        corpus = strsplit(filename[length(filename)], '.', fixed=T)[[1]][1]
        opts$options$corpus = corpus
        smartPrint(paste0('Corpus: ', opts$options$corpus))
    }

    if (!is.null(params$all)) {
        opts$options$addEffects <- strsplit(params$all,'+',fixed=T)[[1]]
    } else opts$options$addEffects <- c()

    if (!is.null(params$abl)) {
        opts$options$ablEffects <- strsplit(params$abl,'+',fixed=T)[[1]]
    } else opts$options$ablEffects <- c()

    opts$options$addEffects = c(opts$options$addEffects, opts$options$ablEffects)

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

    opts$options$partitiondevindices <- as.numeric(strsplit(params$partitiondevindices, ',', fixed=T)[[1]])

    if (length(params$groupingfactor) > 0) {
       smartPrint(paste0('Grouping the main effect by factor ', params$groupingfactor))
    }

    if (params$logdepvar && params$boxcox) {
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

    if (params$logdepvar) {
        smartPrint('Log-transforming fdur')
    }
    if (params$boxcox) {
        smartPrint('Using Box & Cox (1964) to find and apply the best power transform of the dependent variable.')
    }

    return(opts)
}

computeSplitIDs <- function(data, splitcols) {
    ## Exploratory/confirmatory partition utility column
    data$splitID <- 0
    for (col in splitcols) {
        ## Convert to factor and then zero-index for consistent handling of all forms of ID
        x <- as.numeric(as.factor(data[[col]])) - 1
        data$splitID <- data$splitID + x
    }
    return(data)
}

cleanupData <- function(data, filterfiles=FALSE, filterlines=FALSE, filtersents=FALSE, filterscreens=FALSE, filterpunc=FALSE, restrdomain=NULL, upperbound=NULL, lowerbound=NULL, mincorrect=NULL, stdout=TRUE) {
    smartPrint(paste('Number of data rows (raw):', nrow(data)), stdout=stdout)
    
    if (!is.null(data$wdelta)) {
        # Remove outliers
        data <- data[data$wdelta <= 4,]
        smartPrint(paste('Number of data rows (no saccade lengths > 4):', nrow(data)), stdout=stdout)
    }
    # Filter tokens
    if (filterfiles) {
        if (!is.null(data$startoffile) && !is.null(data$endoffile)) {
            smartPrint('Filtering file boundaries', stdout=stdout)
            data <- data[data$startoffile != 1,]
            data <- data[data$endoffile != 1,]
            smartPrint(paste('Number of data rows (no file boundaries)', nrow(data)), stdout=stdout)
        } else smartPrint('No file boundary fields to filter', stdout=stdout)
    } else {
        smartPrint('File boundary filtering off', stdout=stdout)
    }
    if (filterlines) {
        if (!is.null(data$startoffile) && !is.null(data$endoffile)) {
            smartPrint('Filtering line boundaries', stdout=stdout)
            data <- data[data$startofline != 1,]
            data <- data[data$endofline != 1,]
            smartPrint(paste('Number of data rows (no line boundaries)', nrow(data)), stdout=stdout)
        } else smartPrint('No line boundary fields to filter', stdout=stdout)
    } else {
        smartPrint('Line boundary filtering off', stdout=stdout)
    }
    if (filtersents) {
        if (!is.null(data$startofsentence) && !is.null(data$endofsentence)) {
            smartPrint('Filtering sentence boundaries', stdout=stdout)
            data <- data[data$startofsentence != 1,]
            data <- data[data$endofsentence != 1,]
            smartPrint(paste('Number of data rows (no sentence boundaries)', nrow(data)), stdout=stdout)
        } else smartPrint('No sentence boundary fields to filter', stdout=stdout)
    } else {
        smartPrint('Sentence boundary filtering off', stdout=stdout)
    }
    if (filterscreens) {
        if (!is.null(data$startofscreen) && !is.null(data$endofscreen)) {
            smartPrint('Filtering screen boundaries', stdout=stdout)
            data <- data[data$startofscreen != 1,]
            data <- data[data$endofscreen != 1,]
            smartPrint(paste('Number of data rows (no screen boundaries)', nrow(data)), stdout=stdout)
        } else smartPrint('No screen boundary fields to filter', stdout=stdout)
    } else {
        smartPrint('Screen boundary filtering off', stdout=stdout)
    }
    if (filterpunc) {
        if (!is.null(data$punc)) {
            smartPrint('Filtering screen boundaries', stdout=stdout)
            data <- data[data$punc != 1,]
            smartPrint(paste('Number of data rows (no phrasal punctuation)', nrow(data)), stdout=stdout)
        } else smartPrint('No phrasal punctuation field to filter', stdout=stdout)
    } else {
        smartPrint('Phrasal punctuation filtering off', stdout=stdout)
    }
    if (!is.null(upperbound)) {
        smartPrint(paste0('Filtering out rows with response variable >= ', toString(upperbound)), stdout=stdout)
        data <- data[data$fdur < upperbound,]
        smartPrint(paste0('Number of data rows (fdur < ', upperbound, '): ', nrow(data)), stdout=stdout)
    }
    if (!is.null(lowerbound)) {
        smartPrint(paste0('Filtering out rows with response variable <= ', toString(lowerbound)), stdout=stdout)
        data <- data[data$fdur > lowerbound,]
        smartPrint(paste0('Number of data rows (fdur > ', lowerbound, '): ', nrow(data)), stdout=stdout)
    }

    if (!is.null(mincorrect) & 'correct' %in% colnames(data)) {
        smartPrint(paste0('Filtering out rows with correct < ', toString(mincorrect)), stdout=stdout)
        data <- data[data$correct >= mincorrect,]
        smartPrint(paste0('Number of data rows (min correct): ', nrow(data)), stdout=stdout)
    }

    # Remove any incomplete rows
    # data <- data[complete.cases(data),]
    # smartPrint(paste('Number of data rows (complete cases):', nrow(data)), stdout=stdout)

    if (!is.null(restrdomain)) {
        restr = file(description=paste0('scripts/', restrdomain, '.restrdomain.txt'), open='r')
        rlines = readLines(restr)
        close(restr)
        for (l in rlines) {
            l = gsub('^\\s*|\\s*$', '', l)
            if (!(l == "" || substr(l, 1, 1) == '#')) {
                filter = strsplit(l, '\\s+')[[1]]
                if (filter[1] == 'only') {
                    smartPrint(paste0('Filtering out all rows with ', filter[2], ' != ', filter[3]), stdout=stdout)
                    data = data[data[[filter[2]]] == filter[3],]
                    smartPrint(paste0('Number of data rows after filtering out ', filter[2], ' != ', filter[3], ': ', nrow(data)), stdout=stdout)
                } else if (filter[1] == 'noneof') {
                    smartPrint(paste0('Filtering out all rows with ', filter[2], ' = ', filter[3]), stdout=stdout)
                    data = data[data[[filter[2]]] != filter[3],]
                    smartPrint(paste0('Number of data rows after filtering out ', filter[2], ' = ', filter[3], ': ', nrow(data)), stdout=stdout)
                } else smartPrint(paste0('Unrecognized filtering instruction in ', restrdomain, '.restrdomain.txt'), stdout=stdout)
            }
        }
    }

    return(data)
}

addColumns <- function(data) {
    for (x in colnames(data)[grepl('prob',colnames(data))]) {
        data[[paste(x, 'surp', sep='')]] <- as.numeric(as.character(-data[[x]]))
    }
    data$wlen <- as.integer(nchar(as.character(data$word)))
    for (x in colnames(data)[grepl('Ad|Bd', colnames(data))]) {
        data[[paste0(x, 'prim')]] <- substr(data[[x]], 1, 1)
    }
    if ('wdelta' %in% colnames(data)) {
        data$prevwasfix = as.integer(as.logical(data$wdelta == 1))
    }
    return(data)
}

recastEffects <- function(data, splitcols=NULL, indicatorlevel=NULL, groupingfactor=NULL, stdout=TRUE) {
    ## Ensures that data columns are interpreted with the correct dtype, since R doesn't always infer this correctly
    smartPrint("Recasting Effects", stdout=stdout)

    ## DEPENDENT VARIABLES
    ## Reading times
    for (x in colnames(data)[grepl('^fdur', colnames(data))]) {
        data[[x]] <- as.numeric(as.character(data[[x]]))
    }
    ## BOLD levels (fMRI)    
    for (x in colnames(data)[grepl('^bold', colnames(data))]) {
        data[[x]] <- as.numeric(as.character(data[[x]]))
    }

    ## NUISANCE VARIABLES
    for (x in colnames(data)[grepl('^sentid', colnames(data))]) {
        data[[x]] <- as.numeric(as.character(data[[x]]))
    }
    for (x in colnames(data)[grepl('^sentpos', colnames(data))]) {
        data[[x]] <- as.integer(as.character(data[[x]]))
    }
    for (x in colnames(data)[grepl('^wdelta', colnames(data))]) {
        data[[x]] <- as.integer(as.character(data[[x]]))
    }
    for (x in colnames(data)[grepl('^prevwasfix', colnames(data))]) {
        data[[x]] <- as.integer(as.character(data[[x]]))
    }
    for (x in colnames(data)[grepl('^word',colnames(data))]) {
        data[[x]] <- as.character(data[[x]])
    }
    for (x in colnames(data)[grepl('^wlen',colnames(data))]) {
        data[[x]] <- as.integer(as.character(data[[x]]))
    }
    for (x in colnames(data)[grepl('^rolled',colnames(data))]) {
        data[[x]] <- as.logical(as.character(data[[x]]))
    }
    for (x in colnames(data)[grepl('^pos',colnames(data))]) {
        data[[x]] <- as.character(data[[x]])
    }

    ## MAIN EFFECTS
    for (x in colnames(data)[grepl('^embd', colnames(data))]) {
        data[[x]] <- as.numeric(as.character(data[[x]]))
    }
    for (x in colnames(data)[grepl('^startembd', colnames(data))]) {
        data[[x]] <- as.numeric(as.character(data[[x]]))
    }
    for (x in colnames(data)[grepl('^endembd', colnames(data))]) {
        data[[x]] <- as.numeric(as.character(data[[x]]))
    }
    for (x in colnames(data)[grepl('^dlt',colnames(data))]) {
        data[[x]] <- as.numeric(as.character(data[[x]]))
    }
    for (x in colnames(data)[grepl('^noF',colnames(data))]) {
        data[[x]] <- as.numeric(as.character(data[[x]]))
    }
    for (x in colnames(data)[grepl('^yesJ',colnames(data))]) {
        data[[x]] <- as.numeric(as.character(data[[x]]))
    }
    for (x in colnames(data)[grepl('^coref',colnames(data))]) {
        data[[x]] <- as.numeric(as.character(data[[x]]))
    }
    for (x in colnames(data)[grepl('^reinst',colnames(data))]) {
        data[[x]] <- as.numeric(as.character(data[[x]]))
    }
    for (x in colnames(data)[grepl('surp',colnames(data))]) {
        data[[x]] <- as.numeric(as.character(data[[x]]))
    }
    for (x in colnames(data)[grepl('entropy',colnames(data))]) {
        data[[x]] <- as.numeric(as.character(data[[x]]))
    }
    for (x in colnames(data)[grepl('entred',colnames(data))]) {
        data[[x]] <- as.numeric(as.character(data[[x]]))
    }
    for (x in colnames(data)[grepl('prob',colnames(data))]) {
        data[[x]] <- as.numeric(as.character(data[[x]]))
    }
    if ('correct' %in% colnames(data)) {
        data$correct <- as.numeric(as.character(data$correct))
    }

    ## Columns if using categorical grouping variables
    if (length(indicatorlevel) > 0) {
        for (level in levels(as.factor(data[[groupingfactor]]))) {
            data[[paste0(groupingfactor, 'Yes', level)]] = data[[groupingfactor]] == level
            hits = sum(data[[paste0(groupingfactor, 'Yes', level)]])
            smartPrint(paste0('Indicator variable for level ', level, ' of ', groupingfactor, ' has ', hits, ' TRUE events.'))
        }
    }

    smartPrint('The data frame contains the following columns:', stdout=stdout)
    smartPrint(paste(colnames(data), collapse=' '), stdout=stdout)

    ## NAN removal
    na_cols <- colnames(data)[colSums(is.na(data)) > 0]
    if (length(na_cols) > 0) {
        smartPrint('The following columns contain NA values:', stdout=stdout)
        smartPrint(paste(na_cols, collapes=' '), stdout=stdout)
    }

    return(data)
}

smartPrint <- function(string,stdout=TRUE,stderr=TRUE) {
    if (stdout) cat(paste0(string, '\n'))
    if (stderr) write(string, stderr())
}

# Partition data
create.dev <- function(data, i, devindices) {
    dev <- data[(data$splitID %% i) %in% devindices,]
    smartPrint('Dev dimensions')
    smartPrint(dim(dev))
    return(dev)
}

create.test <- function(data, i, devindices) {
    test <- data[(data$splitID %% i) %ni% devindices,]
    smartPrint('Test dimensions')
    smartPrint(dim(test))
    return(test)
}

# Generate LMER formulae
baseFormula <- function(bformfile, logdepvar=FALSE, lambda=NULL) {
    f <- file(description=bformfile, open='r')
    flines <- readLines(f)
    depvar <- flines[1]
    if (!is.null(lambda)) {
        smartPrint('Boxcoxing')
        depvar <- paste0('((', depvar, '^', lambda, '-1)/', lambda, ')')
    }
    else if (logdepvar) {
        depvar <- paste('log(', depvar,')', sep='')
    }
    # depvar <- paste('c.(', depvar, ')', sep='')
    bform <- list(
        dep=depvar,
        fixed=flines[2],
        by_subject=flines[3]
    )
    if (length(flines) > 3) {
        bform$other = flines[4]
    }
    close(f)
    return(bform)
}

processForm <- function(formList, addEffects=NULL, extraEffects=NULL, ablEffects=NULL,
                        groupingfactor=NULL, indicatorlevel=NULL, crossfactor=NULL,
                        logmain=FALSE, interact=TRUE, include_random=TRUE, noMainRandomEffect=FALSE) {
    formList <- addEffects(formList, addEffects, groupingfactor, indicatorlevel, crossfactor, logmain, noMainRandomEffect)
    formList <- addEffects(formList, extraEffects, groupingfactor, indicatorlevel, crossfactor, FALSE)
    formList <- ablateEffects(formList, ablEffects, groupingfactor, indicatorlevel, crossfactor, logmain)
    return(formlist2form(formList,interact,include_random))
}

processEffects <- function(effectList, data, logtrans) {
    srcList <- effectList
    if (logtrans) {
        for (i in 1:length(effectList)) {
            tryCatch({
                log1p(data[[srcList[i]]])
                effectList[i] <- paste('log1p(',effectList[i],')',sep='')
            }, error = function (e) {
                return
            })
        }
    }
    for (i in 1:length(effectList)) {
        tryCatch({
            z.(data[[srcList[i]]])
            effectList[i] <- paste('z.(',effectList[i],')',sep='')
        }, error = function (e) {
            return
        })
    }
    return(effectList)
}

update.formStr <- function(x, new) {
    if (x != '') {   
        return(gsub('~','',paste(update.formula(as.formula(paste('~',x)), paste('~.',new,sep='')),collapse='')))
    } else {
        return(new)
    }
}

addEffect <- function(formList, newEffect, groupingfactor=NULL, indicator=NULL, crossfactor=NULL, noMainRandomEffect=FALSE) {
    smartPrint(paste0('Adding effect: ', newEffect))
    if (length(groupingfactor) > 0) {
        if (length(indicator) > 0) {
            formList$fixed <- update.formStr(formList$fixed, paste('+', newEffect, '+as.factor(', paste0(groupingfactor, 'Yes', indicator), ')+', paste0(newEffect, ':as.factor(', paste0(groupingfactor, 'Yes', indicator), ')')))
            if (!noMainRandomEffect) {
                formList$by_subject <- update.formStr(formList$by_subject, paste('+', newEffect, '+as.factor(', paste0(groupingfactor, 'Yes', indicator), ')+', paste0(newEffect, ':as.factor(', paste0(groupingfactor, 'Yes', indicator), ')')))
            }
            
        } else {
            formList$fixed <- update.formStr(formList$fixed, paste('+', newEffect, '+ as.factor(', groupingfactor, ')+', paste0(newEffect, ':as.factor(', groupingfactor, ')')))
            if (!noMainRandomEffect) {
                formList$by_subject <- update.formStr(formList$by_subject, paste('+', newEffect, '+as.factor(', groupingfactor, ')'))
            }
    }
    } else if (length(crossfactor) > 0) {
        formList$fixed <- update.formStr(formList$fixed, paste('+', newEffect, '+', crossfactor, '+', paste0(newEffect, ':', crossfactor)))
        if (!noMainRandomEffect) {
            formList$by_subject <- update.formStr(formList$by_subject, paste('+', newEffect, '+', crossfactor))
        }
    } else {
        formList$fixed <- update.formStr(formList$fixed, paste('+', newEffect))
        if (!noMainRandomEffect) {
            formList$by_subject <- update.formStr(formList$by_subject, paste('+', newEffect))
        }
    }
    return(formList)
}

addEffects <- function(formList, newEffects, groupingfactor=NULL, indicator=NULL, crossfactor=NULL, logtrans, noMainRandomEffect=FALSE) {
    newEffects <- processEffects(newEffects, data, logtrans)
    for (effect in newEffects) {
        formList <- addEffect(formList, effect, groupingfactor, indicator, crossfactor, noMainRandomEffect)
    }
    return(formList)
}

ablateEffect <- function(formList, ablEffect, groupingfactor=NULL, indicator=NULL, crossfactor=NULL) {
    smartPrint(paste0('Ablating effect: ', ablEffect))
    if (length(groupingfactor) > 0) {
        if (length(indicator) > 0) {
            formList$fixed <- update.formStr(formList$fixed, paste('-', paste0(ablEffect, ':as.factor(', paste0(groupingfactor, 'Yes', indicator), ')')))
        } else {
            formList$fixed <- update.formStr(formList$fixed, paste('-', paste0(ablEffect, ':as.factor(', groupingfactor, ')')))
        }
    } else if (length(crossfactor) > 0) {
        formList$fixed <- update.formStr(formList$fixed, paste('-', ablEffect, '-', crossfactor, '-', paste0(ablEffect, ':', crossfactor)))
    } else formList$fixed <- update.formStr(formList$fixed, paste('-', ablEffect))
    return(formList)
}

ablateEffects <- function(formList, ablEffects, groupingfactor=NULL, indicator=NULL, crossfactor=NULL, logtrans) {
    ablEffects <- processEffects(ablEffects, data, logtrans)
    for (effect in ablEffects) {
        formList <- ablateEffect(formList, effect, groupingfactor, indicator, crossfactor)
    }
    return(formList)
}

formlist2form <- function(formList, interact, include_random=TRUE) {
    if (interact) coef <- 1 else coef <- 0
    if (include_random) {
        formStr <- paste0(formList$dep, ' ~ ', formList$fixed, ' + (', coef, ' + ',
                   formList$by_subject, ' | subject)')
    } else {
        formStr <- paste(formList$dep, ' ~ ', formList$fixed)
    }
    formList[c('dep', 'fixed', 'by_subject')] <- NULL
    if (include_random) {
        if (!interact) formStr <- paste(formStr, '+ (1 | subject)')
        if ('other' %in% names(formList)) {
            other <- paste(formList, collapse=' + ')
            formStr <- paste(formStr, '+', other)
        }
    }
    form <- as.formula(formStr)
    return(form)
}

# Compare convergence between two regressions
minRelGrad <- function(reg1, reg2) {
    relgrad1 <- max(abs(with(reg1@optinfo$derivs,solve(Hessian,gradient))))
    relgrad2 <- max(abs(with(reg2@optinfo$derivs,solve(Hessian,gradient))))
    if (relgrad1 < relgrad2) {
        smartPrint(paste('Best convergence with optimizer ', reg1@optinfo$optimizer, ', relgrad = ', relgrad1, sep=""))
        return(reg1)
    } else {
        smartPrint(paste('Best convergence with optimizer ', reg2@optinfo$optimizer, ', relgrad = ', relgrad2, sep=""))
        return(reg2)
    }
}

# Fit a model formula with bobyqa, try again with nlminb on convergence failure
regressLinearModel <- function(dataset, form, params, suppress_nlminb=FALSE) {
    library(optimx)
    library(lme4)
    bobyqa <- lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=50000))
    nlminb <- lmerControl(optimizer="optimx",optCtrl=list(method=c("nlminb"),maxit=50000))
   
    smartPrint('-----------------------------')
    smartPrint('Fitting linear mixed-effects model with bobyqa')
    smartPrint(paste(' ', date()))
    m <- lmer(form, dataset, REML=F, control = bobyqa)
    smartPrint('-----------------------------')
    smartPrint('SUMMARY:')
    printSummary(m)
    convWarn <- m@optinfo$conv$lme4$messages
    convWarnN <- NULL
    
    if (!is.null(convWarn) && suppress_nlminb) {
        m1 <- m
        smartPrint('Fitting linear mixed-effects model with nlminb')
        smartPrint(paste(' ', date()))
        m2 <- lmer(form, dataset, REML=F, control = nlminb)
        convWarnN <- m2@optinfo$conv$lme4$messages
        printSummary(m2)
        if (is.null(convWarnN)) {
            m <- m2
        } else {
            m <- minRelGrad(m1, m2)            
        }
    }
    
    if (!is.null(convWarn) && !is.null(convWarnN)) {
        smartPrint('Model failed to converge under both bobyqa and nlminb');
    }
    return(m)
}

regressSimpleLinearModel <- function(dataset, form) {
    smartPrint('-----------------------------')
    smartPrint('Fitting linear model')
    m <- lm(form, dataset)
    smartPrint('-----------------------------')
    smartPrint('SUMMARY:')
    printLMSummary(m)
    smartPrint('logLik:')
    smartPrint(logLik(m))
    return(m)
}

regressBayesianModel <- function(dataset, form, nchains=4, algorithm='sampling') {
    library(rstanarm)
    options(mc.cores = parallel::detectCores())
    attach(dataset)
    depVar <- eval(parse(text=as.character(form)[[2]]))
    detach(dataset)    
    #bound = as.numeric(quantile(depVar, .95))

    smartPrint('-------------=---------------')
    smartPrint('Fitting (MCMC) with stan_lmer')
    smartPrint(paste(' ', date()))

    if (FALSE) {
        m <- stan_lmer(formula = form,
                       prior_intercept = normal(mean(depVar), 1),
                       prior = normal(0, 1),
                       prior_covariance = decov(),
                       data = dataset,
                       algorithm = 'meanfield',
                       QR = FALSE
                       )
        cat('PRE-TRAINING SUMMARY:\n')
        printBayesSummary(m)
        m <- update(m,
                    chains = nchains,
                    cores = nchains,
                    algorithm = algorithm,
                    iter = 2000,
                    QR = FALSE,
                    refresh = 1
                    )
    } else {
        m <- stan_lmer(formula = form,
                       prior_intercept = normal(mean(depVar), 1),
                       prior = normal(0, 1),
                       prior_covariance = decov(),
                       data = dataset,
                       algorithm = algorithm,
                       QR = FALSE
                       )
    }

    smartPrint('-----------------------------')
    
    smartPrint('SUMMARY:')
    printBayesSummary(m)
    return(m)
}

# Output a summary of model fit
printSummary <- function(reg) {
    cat(paste0('LME Summary (',reg@optinfo$optimizer,'):\n'))
    print(summary(reg))
    cat('Convergence Warnings:\n')
    convWarn <- reg@optinfo$conv$lme4$messages
    if (is.null(convWarn)) {
        convWarn <- 'No convergence warnings.'
    }
    cat(paste0(convWarn,'\n'))
    relgrad <- with(reg@optinfo$derivs,solve(Hessian,gradient))
    smartPrint('Relgrad:')
    smartPrint(max(abs(relgrad)))
    smartPrint('AIC:')
    smartPrint(AIC(logLik(reg)))
}

printLMSummary <- function(m) {
    cat(paste0('LM Summary:\n'))
    print(summary(m))
}

printBayesSummary <- function(m) {
    # Get fixed effect names
    cat(paste0('BME Summary:\n'))
    cols = names(m$coefficients)
    fixed = cols[substr(cols, 1, 2) != 'b[']
    print(summary(m, pars=fixed, digits=5))
    cat('\nError terms:\n')
    print(VarCorr(m))
}


# Generate logarithmically binned categorical effect
# from discrete/continouous effect
binEffect <- function(x) {
    if (x == 0) return(0) else
    if (x <= 1) return(1) else
    if (x <= 2) return(2) else
    if (x > 2 && x <= 4) return(3) else
    if (x > 4 && x <= 8) return(4) else
    if (x > 8) return(5) else
    return ("negative")
}

getModelVars <- function(bform) {
    if (bform == '') {
        return('')
    } else {
        
    }
}

getCorrelations <- function(data, bform) {
    vars = all.vars(bform)
    vars = vars[vars %ni% c('subject', 'word')]
    return(cor(data[,vars]))
}

getSDs <- function(data, bform) {
    vars = all.vars(bform)
    vars = vars[vars %ni% c('subject', 'word')]
    sd_vals = list()
    for (c in vars) {
        sd_vals[[c]] = sd(data[[c]])
    }
    return(sd_vals)
}

# Fit mixed-effects regression
fitModel <- function(dataset, output, bformfile, fitmode='lme',
                   logmain=FALSE, logdepvar=FALSE, lambda=NULL,
                   addEffects=NULL, extraEffects=NULL, ablEffects=NULL, groupingfactor=NULL,
                   indicatorlevel=NULL, crossfactor=NULL, interact=TRUE,
                   corpusname='corpus',suppress_nlminb=FALSE, noMainRandomEffect=FALSE) {
   
    if (fitmode == 'lm') {
        bform <- processForm(baseFormula(bformfile, logdepvar, lambda),
                             addEffects, extraEffects, ablEffects,
                             groupingfactor, indicatorlevel,
                             crossfactor, logmain, interact,
                             include_random=FALSE)
    } else { 
        bform <- processForm(baseFormula(bformfile, logdepvar, lambda),
                             addEffects, extraEffects, ablEffects,
                             groupingfactor, indicatorlevel,
                             crossfactor, logmain, interact,
                             noMainRandomEffect=noMainRandomEffect)
    }
    
    correlations = getCorrelations(dataset, bform)
    cat('\n')
    cat('Correlation of numeric variables in model:\n')
    print(correlations) 
    cat('\n\n')
    sd_vals = getSDs(dataset, bform)
    cat('Standard deviations of numeric variables in model:\n')
    for (c in names(sd_vals)) {
        cat(paste0(c, ': ', sd_vals[[c]], '\n'))
    }
    cat('\n')

    smartPrint('Regressing model:')
    smartPrint(deparse(bform))

    if (fitmode=='bme') {
        outputModel <- regressBayesianModel(dataset, bform)
    } else if (fitmode=='lm') {
        outputModel <- regressSimpleLinearModel(dataset, bform)
    } else {
        outputModel <- regressLinearModel(dataset, bform, suppress_nlminb=suppress_nlminb)
    }
    if (params$boxcox) {
        mixed = fitmode %in% c('lme', 'bme')
        bc_inv_out = getBoxCoxInvBetas(dataset, bform, lambda, outputModel, mixed=mixed) 
        beta_ms = bc_inv_out$beta_ms 
        y_mu = bc_inv_out$y_mu 
        printBoxCoxInvBetas(beta_ms, lambda, y_mu, sd_vals) 
    } else { 
        beta_ms = fixef(outputModel) 
        y_mu = NULL 
    } 
    fitOutput <- list(
        f = bform,
        fitmode = fitmode,
        abl = ablEffects,
        ablEffects = processEffects(ablEffects, data, logmain),
        corpus = corpusname,
        model = outputModel,
        logmain = logmain,
        logdepvar = logdepvar,
        lambda = lambda,
        beta_ms = beta_ms,
        y_mu = y_mu,
        correlations = correlations,
        sd_vals = sd_vals
    )
    save(fitOutput, file=output)
    return(list(m=outputModel, f=bform, fitmode=fitmode))
}

getBoxCoxInvBetas <- function(dataset, bform, lambda, outputModel, mixed=True) {
    attach(dataset)
    response = as.character(bform)[[2]]
    if (substr(response, 1, 3) %in% c('c.(', 'z.(')) {
        response = substr(response, 4, nchar(response)-1)
        print(response)
    }
    y_mu = mean(eval(parse(text=response)))
    detach(dataset)
    if (mixed) {
        fixednames = names(fixef(outputModel))
        fixedbetas = fixef(outputModel)
    } else {
        fixednames = names(outputModel$coefficients)
        fixedbetas = outputModel$coefficients
    }
    beta_ms = list()
    for (f in fixednames) {
        beta = fixedbetas[[f]]
        beta_ms[[f]] = boxcox_inv(lambda, beta, y_mu)
    }
    return(list(beta_ms=beta_ms, y_mu=y_mu))
}

printBoxCoxInvBetas <- function(beta_ms, lambda, y_mu, sd_vals=NULL) {
    cat(paste0('\nInverse Box-Cox estimates (ms) using lambda = ', lambda, ' and mean y = ', y_mu, '\n'))
    for (f in names(beta_ms)) {
        name = f
        beta = beta_ms[[f]]
        cat(paste0('Beta (ms) of effect ', name, ': ', beta, '\n'))
        if (!is.null(sd_vals) && substr(f, 1, 3) == 'z.(') {
            name = substr(f, 4, nchar(f)-1)
            beta = beta_ms[[f]] / sd_vals[[name]]
            cat(paste0('Beta (ms) of effect ', name, ': ', beta, '\n'))
        }
    }
}

# LME error analysis
error_anal <- function(data, params) {
    name <- setdiff(params$base_obj$abl,params$main_obj$abl)[[1]]
    errData <- data[c('word','sentid','sentpos','subject','fdur', name)]
    if (params$logdepvar) {
        errData[[paste0(name,'BaseErr')]] <- c.(log1p(errData$fdur)) - predict(params$base_obj$model, data)
        errData[[paste0(name,'MainErr')]] <- c.(log1p(errData$fdur)) - predict(params$main_obj$model, data)
    } else if (params$boxcox) {
        bc <- MASS:::boxcox(as.formula('fdur ~ 1'), data=data)
        l <- bc$x[which.max(bc$y)]
        smartPrint(paste0('Box & Cox lambda: ', l))
        errData[[paste0(name,'BaseErr')]] <- c.((errData$fdur^l-1)/l) - predict(params$base_obj$model, data)
        errData[[paste0(name,'MainErr')]] <- c.((errData$fdur^l-1)/l) - predict(params$main_obj$model, data)        
    } else {
        errData[[paste0(name,'BaseErr')]] <- c.(errData$fdur) - predict(params$base_obj$model, data)
        errData[[paste0(name,'MainErr')]] <- c.(errData$fdur) - predict(params$main_obj$model, data)
    }
    errData[[paste0(name,'SqErrReduc')]] <- errData[paste0(name,'BaseErr')]^2 - errData[paste0(name,'MainErr')]^2
    errData[[paste0(name,'BaseErr')]] <- NULL
    errData[[paste0(name,'MainErr')]] <- NULL
    errData <- errData[order(errData$sentid,errData$sentpos),]
    smartPrint(paste0('Error Reduction values calculated for ',name))
    return(errData)
}

boxcox_inv <- function(lambda, beta, y_mu) {
    return((lambda*(y_mu + beta) + 1)^(1/lambda) - (lambda*y_mu + 1)^(1/lambda))
}

permutation_test <- function(err1, err2, n_iter=10000, n_tails=2) {
    base_diff = mean(err1) - mean(err2)
    err_matrix = matrix(c(err1, err2), length(err1), 2)
    hits = 0
    cat('Permutation testing...\n', file=stderr())
    cat(paste0('base_diff: ', base_diff, '\n'))
    for (i in 1:n_iter) {
        cat(paste0('\r',i, '/', n_iter), file=stderr())
        shuffle = runif(length(err1)) > 0.5
        shuffle = as.numeric(shuffle)
        m1 = err_matrix[cbind(seq_along(shuffle), shuffle+1)]
        m2 = err_matrix[cbind(seq_along(shuffle), 2-shuffle)]
        curr_diff = mean(m1) - mean(m2)
        cat(paste0('curr_diff: ', curr_diff, '\n'))
        if (n_tails == 1) {
            if ((base_diff < 0) & (curr_diff <= base_diff)) {
                hits = hits + 1
	        cat(paste0('Hit! Count: ', hits, '\n'))
            } else if ((base_diff > 0) & (curr_diff >= base_diff)) {
                hits = hits + 1
	        cat(paste0('Hit! Count: ', hits, '\n'))
            } else if (base_diff == 0) {
                hits = hits + 1
	        cat(paste0('Hit! Count: ', hits, '\n'))
            }    
        } else if (n_tails == 2) {
            if (abs(curr_diff) >= abs(base_diff)) {
                hits = hits + 1
	        cat(paste0('Hit! Count: ', hits, '\n'))
            }
        } else {
            stop('Error: n_tails must be in {1,2}')
        }
    }
    p = (hits + 1) / (n_iter + 1)
    
    cat(paste0('\np=',p,'\n'), file=stderr())
    return(list(p=p, base_diff=base_diff))
}
