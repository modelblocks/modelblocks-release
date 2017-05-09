########################################################
#
# Commonly used methods for LME regressions
#
########################################################

cleanupData <- function(data, filterfiles=FALSE, filterlines=FALSE, filtersents=FALSE, filterscreens=FALSE, filterpunc=FALSE, restrdomain=NULL) {
    smartPrint(paste('Number of data rows (raw):', nrow(data)))
    
    if (!is.null(data$wdelta)) {
        # Remove outliers
        data <- data[data$wdelta <= 4,]
        smartPrint(paste('Number of data rows (no saccade lengths > 4):', nrow(data)))
    }
    # Filter tokens
    if (filterfiles) {
        if (!is.null(data$startoffile) && !is.null(data$endoffile)) {
            smartPrint('Filtering file boundaries')
            data <- data[data$startoffile != 1,]
            data <- data[data$endoffile != 1,]
            smartPrint(paste('Number of data rows (no file boundaries)', nrow(data)))
        } else smartPrint('No file boundary fields to filter')
    } else {
        smartPrint('File boundary filtering off')
    }
    if (filterlines) {
        if (!is.null(data$startoffile) && !is.null(data$endoffile)) {
            smartPrint('Filtering line boundaries')
            data <- data[data$startofline != 1,]
            data <- data[data$endofline != 1,]
            smartPrint(paste('Number of data rows (no line boundaries)', nrow(data)))
        } else smartPrint('No line boundary fields to filter')
    } else {
        smartPrint('Line boundary filtering off')
    }
    if (filtersents) {
        if (!is.null(data$startofsentence) && !is.null(data$endofsentence)) {
            smartPrint('Filtering sentence boundaries')
            data <- data[data$startofsentence != 1,]
            data <- data[data$endofsentence != 1,]
            smartPrint(paste('Number of data rows (no sentence boundaries)', nrow(data)))
        } else smartPrint('No sentence boundary fields to filter')
    } else {
        smartPrint('Sentence boundary filtering off')
    }
    if (filterscreens) {
        if (!is.null(data$startofscreen) && !is.null(data$endofscreen)) {
            smartPrint('Filtering screen boundaries')
            data <- data[data$startofscreen != 1,]
            data <- data[data$endofscreen != 1,]
            smartPrint(paste('Number of data rows (no screen boundaries)', nrow(data)))
        } else smartPrint('No screen boundary fields to filter')
    } else {
        smartPrint('Screen boundary filtering off')
    }
    if (filterpunc) {
        if (!is.null(data$punc)) {
            smartPrint('Filtering screen boundaries')
            data <- data[data$punc != 1,]
            smartPrint(paste('Number of data rows (no phrasal punctuation)', nrow(data)))
        } else smartPrint('No phrasal punctuation field to filter')
    } else {
        smartPrint('Phrasal punctuation filtering off')
    }

    # Remove any incomplete rows
    data <- data[complete.cases(data),]
    smartPrint(paste('Number of data rows (complete cases):', nrow(data)))

    if (!is.null(restrdomain)) {
        restr = file(description=paste0('scripts/', restrdomain, '.restrdomain.txt'), open='r')
        rlines = readLines(restr)
        close(restr)
        for (l in rlines) {
            l = gsub('^\\s*|\\s*$', '', l)
            if (!(l == "" || substr(l, 1, 1) == '#')) {
                filter = strsplit(l, '\\s+')[[1]]
                if (filter[1] == 'only') {
                    smartPrint(paste0('Filtering out all rows with ', filter[2], ' != ', filter[3]))
                    data = data[data[[filter[2]]] == filter[3],]
                    smartPrint(paste0('Number of data rows after filtering out ', filter[2], ' != ', filter[3], ': ', nrow(data)))
                } else if (filter[1] == 'noneof') {
                    smartPrint(paste0('Filtering out all rows with ', filter[2], ' = ', filter[3]))
                    data = data[data[[filter[2]]] != filter[3],]
                    smartPrint(paste0('Number of data rows after filtering out ', filter[2], ' = ', filter[3], ': ', nrow(data)))
                } else smartPrint(paste0('Unrecognized filtering instruction in ', restrdomain, '.restrdomain.txt'))
            }
        }
    }

    return(data)
}

addColumns <- function(data) {
    for (x in colnames(data)[grepl('dlt',colnames(data))]) {
        data[[paste(x, 'bin', sep='')]] <- sapply(data[[x]], binEffect)
    }
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

recastEffects <- function(data, splitcols=NULL, indicatorlevel=NULL, groupingfactor=NULL) {
    ## Ensures that data columns are interpreted with the correct dtype, since R doesn't always infer this correctly
    smartPrint("Recasting Effects")

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
    for (x in colnames(data)[grepl('^subject', colnames(data))]) {
        data[[x]] <- as.numeric(as.factor(as.character(data[[x]])))
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
    for (x in colnames(data)[grepl('^endembd', colnames(data))]) {
        data[[x]] <- as.logical(as.character(data[[x]]))
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
    for (x in colnames(data)[grepl('prob',colnames(data))]) {
        data[[x]] <- as.numeric(as.character(data[[x]]))
    }

    ## Exploratory/confirmatory partition utility column
    data$splitID <- 0
    for (col in splitcols) {
        data$splitID <- data$splitID + as.numeric(data[[col]])
    }

    ## Columns if using categorical grouping variables
    if (length(indicatorlevel) > 0) {
        for (level in levels(as.factor(data[[groupingfactor]]))) {
            data[[paste0(groupingfactor, 'Yes', level)]] = data[[groupingfactor]] == level
            hits = sum(data[[paste0(groupingfactor, 'Yes', level)]])
            smartPrint(paste0('Indicator variable for level ', level, ' of ', groupingfactor, ' has ', hits, ' TRUE events.'))
        }
    }

    smartPrint('The data frame contains the following columns:')
    smartPrint(paste(colnames(data), collapse=' '))

    ## NAN removal
    na_cols <- colnames(data)[colSums(is.na(data)) > 0]
    if (length(na_cols) > 0) {
        smartPrint('The following columns contain NA values:')
        smartPrint(paste(na_cols, collapes=' '))
    }

    return(data)
}

smartPrint <- function(string,stdout=TRUE,stderr=TRUE) {
    if (stdout) cat(paste0(string, '\n'))
    if (stderr) write(string, stderr())
}

# Partition data
create.dev <- function(data, i) {
    dev <- data[(data$splitID %% i) == 0,]
    smartPrint('Dev dimensions')
    smartPrint(dim(dev))
    return(dev)
}

create.test <- function(data, i) {
    test <- data[(data$splitID %% i) != 0,]
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
        depvar <- paste0('((', depvar, '^', lambda, ' - 1)/', lambda, ')')
    }
    else if (logdepvar) {
        depvar <- paste('log1p(', depvar,')', sep='')
    }
    depvar <- paste('c.(', depvar, ')', sep='')
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
                        groupingfactor=NULL, indicatorlevel=NULL, crossfactor=NULL, logmain=FALSE, interact=TRUE) {
    formList <- addEffects(formList, addEffects, groupingfactor, indicatorlevel, crossfactor, logmain)
    formList <- addEffects(formList, extraEffects, groupingfactor, indicatorlevel, crossfactor, FALSE)
    formList <- ablateEffects(formList, ablEffects, groupingfactor, indicatorlevel, crossfactor, logmain)
    return(formlist2form(formList,interact))
}

processEffects <- function(effectList, data, logtrans) {
    srcList <- effectList
    for (i in 1:length(effectList)) {
        tryCatch({
            z.(data[[srcList[i]]])
            effectList[i] <- paste('z.(',effectList[i],')',sep='')
        }, error = function (e) {
            return
        })
    }
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
    return(effectList)
}

update.formStr <- function(x, new) {
    return(gsub('~','',paste(update.formula(as.formula(paste('~',x)), paste('~.',new,sep='')),collapse='')))
}

addEffect <- function(formList, newEffect, groupingfactor=NULL, indicator=NULL, crossfactor=NULL) {
    smartPrint(paste0('Adding effect: ', newEffect))
    if (length(groupingfactor) > 0) {
        if (length(indicator) > 0) {
            formList$fixed <- update.formStr(formList$fixed, paste('+', newEffect, '+as.factor(', paste0(groupingfactor, 'Yes', indicator), ')+', paste0(newEffect, ':as.factor(', paste0(groupingfactor, 'Yes', indicator), ')')))
            formList$by_subject <- update.formStr(formList$by_subject, paste('+', newEffect, '+as.factor(', paste0(groupingfactor, 'Yes', indicator), ')+', paste0(newEffect, ':as.factor(', paste0(groupingfactor, 'Yes', indicator), ')')))
            
        } else {
            formList$fixed <- update.formStr(formList$fixed, paste('+', newEffect, '+ as.factor(', groupingfactor, ')+', paste0(newEffect, ':as.factor(', groupingfactor, ')')))
            formList$by_subject <- update.formStr(formList$by_subject, paste('+', newEffect, '+as.factor(', groupingfactor, ')'))
    }
    } else if (length(crossfactor) > 0) {
        formList$fixed <- update.formStr(formList$fixed, paste('+', newEffect, '+', crossfactor, '+', paste0(newEffect, ':', crossfactor)))
        formList$by_subject <- update.formStr(formList$by_subject, paste('+', newEffect, '+', crossfactor))
    } else {
        formList$fixed <- update.formStr(formList$fixed, paste('+', newEffect))
        formList$by_subject <- update.formStr(formList$by_subject, paste('+', newEffect))
    }
    return(formList)
}

addEffects <- function(formList, newEffects, groupingfactor=NULL, indicator=NULL, crossfactor=NULL, logtrans) {
    newEffects <- processEffects(newEffects, data, logtrans)
    for (effect in newEffects) {
        formList <- addEffect(formList, effect, groupingfactor, indicator, crossfactor)
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

formlist2form <- function(formList, interact) {
    if (interact) coef <- 1 else coef <- 0
    formStr <- paste(formList$dep, ' ~ ', formList$fixed, ' + (', coef, ' + ',
               formList$by_subject, ' | subject)', sep='')
    formList[c('dep', 'fixed', 'by_subject')] <- NULL
    if (!interact) formStr <- paste(formStr, '+ (1 | subject)')
    if ('other' %in% names(formList)) {
        other <- paste(formList, collapse=' + ')
        formStr <- paste(formStr, '+', other)
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
regressModel <- function(dataset, form) {
    bobyqa <- lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=50000))
    nlminb <- lmerControl(optimizer="optimx",optCtrl=list(method=c("nlminb"),maxit=50000))
    
    smartPrint('Regressing with bobyqa')
    smartPrint(paste(' ', date()))
    regressionOutput <- lmer(form, dataset, REML=F, control = bobyqa)
    printSummary(regressionOutput)
    convWarn <- regressionOutput@optinfo$conv$lme4$messages
    
    if (!is.null(convWarn)) {
        regressionOutputO <- regressionOutput
        smartPrint('Regressing with nlminb')
        smartPrint(paste(' ', date()))
        regressionOutputN <- lmer(form, dataset, REML=F, control = nlminb)
        convWarnN <- regressionOutputN@optinfo$conv$lme4$messages
        printSummary(regressionOutputN)
        if (is.null(convWarnN)) {
            regressionOutput <- regressionOutputN
        } else {
            regressionOutput <- minRelGrad(regressionOutputO, regressionOutputN)            
        }
    }
    
    if (!is.null(convWarn) && !is.null(convWarnN)) {
        smartPrint('Model failed to converge under both bobyqa and nlminb');
    }
    return(regressionOutput)
}

# Output a summary of model fit
printSummary <- function(reg) {
    cat(paste0('LME Fit Summary (',reg@optinfo$optimizer,')\n'))
    print(summary(reg))
    cat('Convergence Warnings:\n')
    convWarnN <- regressionOutputN@optinfo$conv$lme4$messages
    if (is.null(convWarnN)) {
        convWarnN <- ''
    }
    cat(paste0(convWarnN,'\n'))
    relgrad <- with(reg@optinfo$derivs,solve(Hessian,gradient))
    smartPrint('Relgrad:')
    smartPrint(max(abs(relgrad)))
    smartPrint('AIC:')
    smartPrint(AIC(logLik(reg)))
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

# Run lineary regression
lmefit <- function(dataset, output, bformfile,
                   logmain=FALSE, logdepvar=FALSE, lambda=NULL,
                   addEffects=NULL, extraEffects=NULL, ablEffects=NULL, groupingfactor=NULL,
                   indicatorlevel=NULL, crossfactor=NULL, interact=TRUE,
                   corpusname='corpus') {
    
    bform <- processForm(baseFormula(bformfile, logdepvar, lambda),
                         addEffects, extraEffects, ablEffects,
                         groupingfactor, indicatorlevel,
                         crossfactor, logmain, interact)
    
    smartPrint('Regressing model:')
    smartPrint(deparse(bform))

    outputModel <- regressModel(dataset, bform)
    fitOutput <- list(
        abl = ablEffects,
        ablEffects = processEffects(ablEffects, data, logmain),
        corpus = corpusname,
        model = outputModel
    )
    smartPrint(fitOutput$ablEffects)
    save(fitOutput, file=output)
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
        errData[[paste0(name,'BaseErr')]] <- c.((errData$fdur^l - 1)/l) - predict(params$base_obj$model, data)
        errData[[paste0(name,'MainErr')]] <- c.((errData$fdur^l - 1)/l) - predict(params$main_obj$model, data)        
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

boxcox_rev_estimate <- function(l, beta, intercept, x_0 = 0) {
    y_0 = (l * (x_0 * beta + intercept) + 1) ^ (1/l)
    y_1 = (l * ((x_0 + 1) * beta + intercept) + 1) ^ (1/l)
    print(y_0)
    print(y_1)
    return(y_1 - y_0)
}
