#!/usr/bin/Rscript

argoffset = 5 #this is the offset for sys.argv for running an rscript from cli

#########################
#
# Possible cli args: S 'Find simplest model fit', R 'Reresidualize using simplest model fit', V 'Verbose', d 'Depth-specific analyses',
#    D 'Distance-weighted analyses', A 'Test even F+L+ and F-L-', 
#    cmcl2013 'Run the non/canonical dundee eval given in vanschijndeletal2013:cmcl',
#    Pxxx (where xxx are integers) 'use xxx as significance threshold (100 is p=1, 001 is p=.01)'
#
#########################


#########################
#
# Loading Data and Libraries
#
#########################

args <- commandArgs()
cliargs <- args[argoffset+1]
data <- read.table(args[argoffset+2],header=TRUE,quote='',comment.char='')
options('warn'=1) #report non-convergences, etc

library(lme4)
library(languageR)
source("scripts/rtools.R")
source("extraScripts/mer-utils.R")
source("extraScripts/mtoolbox.R")
source("extraScripts/regression-utils.R")

#########################
#
# Data Columns
#
#########################

#subject word sentpos nrchar previsfix nextisfix locfreq bakfreq uprob prevlogprob fwprob bwprob laundist landpos \
#  fdur prevfdur nextfdur \
#  totsurp cumtotsurp lexsurp cumlexsurp synsurp cumsynsurp entred cumentred \
#  embedep cumembedep aveembedep embedif cumembedif fp cumfp fm cumfm dfp cumdfp dfm cumdfm \
#  lp cumlp lm cumlm dlp cumdlp dlm cumdlm Dlp cumDlp Dlm cumDlm \

#  fmlm cumfmlm fplm cumfplm fmlp cumfmlp fplp cumfplp \
#  dfplp cumdfplp dfplm cumdfplm dfmlp cumdfmlp dfmlm cumdfmlm \
#  Dfmlp cumDfmlp dDfmlp cumdDfmlp \

#  bp cumbp badd cumbadd bsto cumbsto bcdr cumbcdr bm cumbm \
#  dbp cumdbp dbm cumdbm Dbp cumDbp wdelta cumwdelta \
#  fmlmba cumfmlmba fmlmbc cumfmlmbc fmlmbo cumfmlmbo fmlmbp cumfmlmbp fmlpba cumfmlpba fmlpbc cumfmlpbc fmlpbo cumfmlpbo fmlpbp cumfmlpbp \
#  fplmba cumfplmba fplmbc cumfplmbc fplmbo cumfplmbo fplmbp cumfplmbp fplpba cumfplpba fplpbc cumfplpbc fplpbo cumfplpbo fplpbp cumfplpbp

#  lagsentpos lagnrchar lagprevisfix lagnextisfix laglaundist laglandpos laglocfreq lagbakfreq \
#  laguprob lagfwprob lagbwprob \
#  lagtotsurp lagcumtotsurp lagcumlexsurp lagcumsynsurp lagcumentred lagembedep lagcumembedep lagaveembedep lagcumembedif \
#  lagcumfp lagcumfm lagcumdfp lagcumdfm \
#  lagcumlp lagcumlm lagcumdlp lagcumdlm lagcumDlp lagcumDlm \
#  lagcumbp lagcumbsto lagcumbm lagcumdbp lagcumdbm lagcumDbp lagcumwdelta \
#  lagcumfmlmba lagcumfmlmbc lagcumfmlmbo lagcumfmlmbp lagcumfmlpba lagcumfmlpbc lagcumfmlpbo lagcumfmlpbp \
#  lagcumfplmba lagcumfplmbc lagcumfplmbo lagcumfplmbp lagcumfplpba lagcumfplpbc lagcumfplpbo lagcumfplpbp

#########################
#
# Definitions
#
#########################

create.dev <- function(nooutliers, partix, ixid){
  out <- data.frame()
  for (s in unique(nooutliers[,partix])){ if (s != ixid) { out <- rbind(out,nooutliers[nooutliers[,partix] == s,])}}
  ## This is R-ease for:
  #for (s in unique(nooutliers$partix)){ if (s != ixid) { out <- rbind(out,subset(nooutliers, partix == s))}}
  ## Why does R have to make everything more difficult than it needs to be?
  ## And why won't it stick to its own notation?
  return(out)
}

create.test <- function(nooutliers,dev){
  out <- nooutliers[setdiff(rownames(nooutliers),rownames(dev)),]
  return(out)
}

run.test <- function(dev,test) {
  #######
  ## This is the main function
  ## Note that residualization occurs on testdata since the resid vector is the residual for each observation
  #######

  if (length(grep('S',cliargs,fixed=T)) > 0 ) SIMPLIFY <- T
  else SIMPLIFY <- F
  if (length(grep('A',cliargs,fixed=T)) > 0 ) ALL <- T
  else ALL <- F
  if (length(grep('R',cliargs,fixed=T)) > 0 ) RERESID <- T
  else RERESID <- F
  if (length(grep('V',cliargs,fixed=T)) > 0 ) VERBOSE <- T
  else VERBOSE <- F
  if (length(grep('d',cliargs,fixed=T)) > 0 ) DEPTH <- T
  else DEPTH <- F
  if (length(grep('cmcl2013',cliargs,fixed=T)) > 0 ) CMCL2013 <- T
  else CMCL2013 <- F
  if (length(grep('D',cliargs,fixed=T)) > 0 ) DIST <- T
  else DIST <- F
  if (length(grep('P',cliargs,fixed=T)) > 0 ) {
    PVAL <- as.integer(substr(a,regexpr('P',cliargs,fixed=T)[1]+1,regexpr('P',cliargs,fixed=T)[1]+3))/100
  }
  else PVAL <- .05


  print("Building Baseline")

  #clfdur <- c.(test$fdur) #standard fdur
  clfdur <- c.(log(test$fdur)) #log-transformed fdur
  #clfdur <- c.((((test$fdur)^(-0.14141))-1)/-0.14141) #boxcox-derived transformed fdur

  #print(paste("clfdur: ",clfdur))

  #########################
  #
  # Run BoxCox to determine normalizing transform
  #
  #########################
  #bform <- fdur ~ (c.(sentpos) + c.(nrchar) + c.(previsfix) + c.(nextisfix) + c.(uprob) + c.(fwprob) +
  #                          c.(bwprob) + c.(totsurp)+c.(cumwdelta) + c.(cumtotsurp) + c.(cumentred))^2 +
  #                          (c.(lagsentpos) + c.(lagnrchar) + c.(lagprevisfix) + c.(laguprob) + c.(lagfwprob) +
  #                          c.(lagbwprob) + c.(lagtotsurp)+c.(lagcumwdelta) + c.(lagcumtotsurp) + c.(lagcumentred))^2 + (subject) + (word)
  #base <- glm(bform,data=test)
  #
  #print("BoxCox")
  #base.bc <- MASS:::boxcox(base)
  #print(base.bc$x[which.max(base.bc$y)])
  #return(0)
  ######################
  # Lambda for BoxCox = -0.14141
  ######################

  if (SIMPLIFY || !RERESID) {
    bc <- ~ (c.(sentpos) + c.(nrchar) + c.(previsfix) + c.(nextisfix) + c.(uprob) + 
                           c.(fwprob) + c.(bwprob))^2
    bl <- ~ (c.(lagsentpos) + c.(lagnrchar) + c.(lagprevisfix) + c.(lagnextisfix) + c.(laguprob) +
                            c.(lagfwprob) + c.(lagbwprob))^2

    if (CMCL2013) {
      r.cw <- residuals(lm(update.formula(bc,c.(cumwdelta)~.), test))
      r.lcw <- residuals(lm(update.formula(bl,c.(lagcumwdelta)~.) , test))

      r.csurp <- residuals(lm(c.(cumtotsurp) ~ (c.(sentpos) + c.(nrchar) + c.(previsfix) + c.(nextisfix) + c.(uprob) + 
                             c.(fwprob) + c.(bwprob) + c.(r.cw))^2, test))
      r.lcsurp <- residuals(lm(c.(lagcumtotsurp) ~ (c.(lagsentpos) + c.(lagnrchar) + c.(lagprevisfix) + c.(lagnextisfix) + c.(laguprob) +
                              c.(lagfwprob) + c.(lagbwprob) + c.(r.lcw))^2, test))
      r.tsurp <- residuals(lm(c.(totsurp) ~ (c.(sentpos) + c.(nrchar) + c.(previsfix) + c.(nextisfix) + c.(uprob) + 
                             c.(fwprob) + c.(bwprob) + c.(r.cw) + c.(r.csurp))^2, test))
      r.ltsurp <- residuals(lm(c.(lagtotsurp) ~ (c.(lagsentpos) + c.(lagnrchar) + c.(lagprevisfix) + c.(lagnextisfix) + c.(laguprob) +
                              c.(lagfwprob) + c.(lagbwprob) + c.(r.lcw) + c.(r.lcsurp))^2, test))

      r.ent <- residuals(lm(c.(cumentred) ~ (c.(sentpos) + c.(nrchar) + c.(previsfix) + c.(nextisfix) + c.(uprob) + 
                             c.(fwprob) + c.(bwprob) + c.(r.cw) + c.(r.csurp)+c.(r.tsurp))^2 , test))
      r.lent <- residuals(lm(c.(lagcumentred) ~ (c.(lagsentpos) + c.(lagnrchar) + c.(lagprevisfix) + c.(lagnextisfix) + c.(laguprob) +
                              c.(lagfwprob) + c.(lagbwprob) + c.(r.lcw) + c.(r.lcsurp)+c.(r.ltsurp))^2, test))

      rform <- ~(c.(sentpos) + c.(nrchar) + c.(previsfix) + c.(nextisfix) + c.(uprob) +
                               c.(fwprob) + c.(bwprob) + c.(r.cw) + c.(r.csurp)+c.(r.tsurp)+c.(r.ent))^2 
      lrform <- ~(c.(lagsentpos) + c.(lagnrchar) + c.(lagprevisfix) + c.(lagnextisfix) + c.(laguprob) +
                              c.(lagfwprob) + c.(lagbwprob) + c.(r.lcw) +c.(r.lcsurp)+c.(r.ltsurp)+c.(r.lent))^2

      bform <- clfdur ~ (c.(sentpos) + c.(nrchar) + c.(previsfix) + c.(nextisfix) + c.(uprob) + c.(fwprob) +
                                c.(bwprob) + c.(r.cw)+ c.(r.csurp)+c.(r.tsurp)+c.(r.ent))^2 +
                                (c.(lagsentpos) + c.(lagnrchar) + c.(lagprevisfix) + c.(laguprob) + c.(lagfwprob) +
                                c.(lagbwprob) + c.(r.lcw)+c.(r.lcsurp)+c.(r.ltsurp)+c.(r.lent))^2 + (1|word) #+ (1|subject)
    }
    else {
      rform <- ~ (c.(sentpos) + c.(nrchar) + c.(previsfix) + c.(nextisfix) + c.(uprob) + c.(fwprob) +
                                c.(bwprob) + c.(cumwdelta)+ c.(cumtotsurp)+c.(totsurp)+c.(cumentred))^2

      lrform <- ~ (c.(lagsentpos) + c.(lagnrchar) + c.(lagprevisfix) + c.(laguprob) + c.(lagfwprob) +
                                c.(lagbwprob) + c.(lagcumwdelta)+c.(lagcumtotsurp)+c.(lagtotsurp)+c.(lagcumentred))^2

      bform <- clfdur ~ (c.(sentpos) + c.(nrchar) + c.(previsfix) + c.(nextisfix) + c.(uprob) + c.(fwprob) +
                                c.(bwprob) + c.(cumwdelta)+ c.(cumtotsurp)+c.(totsurp)+c.(cumentred))^2 +
                                (c.(lagsentpos) + c.(lagnrchar) + c.(lagprevisfix) + c.(laguprob) + c.(lagfwprob) +
                                c.(lagbwprob) + c.(lagcumwdelta)+c.(lagcumtotsurp)+c.(lagtotsurp)+c.(lagcumentred))^2 + (1|word)
    }
  }
  ######################
  #
  # Find simplest model fit
  # WARNING: Takes a /long/ time!
  # Make SIMPLIFY=F if unnecessary!
  #
  ######################
  if (SIMPLIFY) {
    print("Finding simplest model fit")
    bform <- fitfix(bform,test,alpha=PVAL,verbose=VERBOSE)
    rform <- bform
    lrform <- bform
  }
  ######################
  #
  # Re-residualize according to simplest model fit results
  #
  ######################
  if (RERESID) {
    if (length(grep('badd',colnames(test),fixed=T)) == 0) {
      bc <- ~ c.(sentpos) + c.(nrchar) + c.(previsfix) + c.(nextisfix) + 
        c.(uprob) + c.(fwprob) + c.(bwprob) + c.(sentpos):c.(fwprob) + 
        c.(nrchar):c.(previsfix) + c.(nrchar):c.(uprob) + c.(nrchar):c.(fwprob) + 
        c.(previsfix):c.(nextisfix) + c.(nextisfix):c.(fwprob) + 
        c.(nextisfix):c.(bwprob)

      bl <- ~ c.(lagsentpos) + c.(lagnrchar) + c.(lagprevisfix) + 
        c.(laguprob) + c.(lagfwprob) + c.(lagbwprob) + 
        c.(lagsentpos):c.(lagnrchar) + c.(lagsentpos):c.(lagbwprob) + 
        c.(lagnrchar):c.(laguprob) + c.(lagnrchar):c.(lagfwprob) + 
        c.(lagnrchar):c.(lagbwprob) + c.(laguprob):c.(lagfwprob)

      r.cw <- residuals(lm(update.formula(bc,c.(cumwdelta)~.), test))
      r.lcw <- residuals(lm(update.formula(bl,c.(lagcumwdelta)~.) , test))

      r.csurp <- residuals(lm(c.(cumtotsurp) ~ c.(sentpos) + c.(nrchar) + c.(previsfix) + c.(nextisfix) + 
        c.(uprob) + c.(fwprob) + c.(bwprob) + c.(sentpos):c.(fwprob) + 
        c.(nrchar):c.(previsfix) + c.(nrchar):c.(uprob) + c.(nrchar):c.(fwprob) + 
        c.(previsfix):c.(nextisfix) + c.(nextisfix):c.(fwprob) + 
        c.(r.cw) + c.(previsfix):c.(r.cw) +
        c.(nextisfix):c.(bwprob), test))

      r.lcsurp <- residuals(lm(c.(lagcumtotsurp) ~ c.(lagsentpos) + c.(lagnrchar) + c.(lagprevisfix) + 
        c.(laguprob) + c.(lagfwprob) + c.(lagbwprob) + 
        c.(lagsentpos):c.(lagnrchar) + c.(lagsentpos):c.(lagbwprob) + 
        c.(lagnrchar):c.(laguprob) + c.(lagnrchar):c.(lagfwprob) + 
        c.(lagnrchar):c.(lagbwprob) + c.(laguprob):c.(lagfwprob) + 
        c.(r.lcw) + c.(lagfwprob):c.(r.lcw), test))

      r.tsurp <- residuals(lm(c.(totsurp) ~ c.(sentpos) + c.(nrchar) + c.(previsfix) + c.(nextisfix) + 
        c.(uprob) + c.(fwprob) + c.(bwprob) + c.(sentpos):c.(fwprob) + 
        c.(nrchar):c.(previsfix) + c.(nrchar):c.(uprob) + c.(nrchar):c.(fwprob) + 
        c.(previsfix):c.(nextisfix) + c.(nextisfix):c.(fwprob) + 
        c.(r.cw) + c.(previsfix):c.(r.cw) + c.(r.csurp) + c.(sentpos):c.(r.csurp) +
        c.(previsfix):c.(r.csurp) + c.(fwprob):c.(r.csurp) + c.(r.cw):c.(r.csurp) +
        c.(nextisfix):c.(bwprob) , test))

      r.ltsurp <- residuals(lm(c.(lagtotsurp) ~ c.(lagsentpos) + c.(lagnrchar) + c.(lagprevisfix) + 
        c.(laguprob) + c.(lagfwprob) + c.(lagbwprob) + 
        c.(lagsentpos):c.(lagnrchar) + c.(lagsentpos):c.(lagbwprob) + 
        c.(lagnrchar):c.(laguprob) + c.(lagnrchar):c.(lagfwprob) + 
        c.(lagnrchar):c.(lagbwprob) + c.(laguprob):c.(lagfwprob) + 
        c.(r.lcw) + c.(lagfwprob):c.(r.lcw) + c.(r.lcsurp) + c.(lagbwprob):c.(r.lcsurp), test))

      r.ent <- residuals(lm(c.(cumentred) ~ c.(sentpos) + c.(nrchar) + c.(previsfix) + c.(nextisfix) + 
        c.(uprob) + c.(fwprob) + c.(bwprob) + c.(sentpos):c.(fwprob) + 
        c.(nrchar):c.(previsfix) + c.(nrchar):c.(uprob) + c.(nrchar):c.(fwprob) + 
        c.(previsfix):c.(nextisfix) + c.(nextisfix):c.(fwprob) + 
        c.(r.cw) + c.(previsfix):c.(r.cw) + c.(r.csurp) + c.(sentpos):c.(r.csurp) +
        c.(previsfix):c.(r.csurp) + c.(fwprob):c.(r.csurp) + c.(r.cw):c.(r.csurp) +
        c.(r.tsurp) + c.(sentpos):c.(r.tsurp) + c.(fwprob):c.(r.tsurp) +
        c.(r.tsurp):c.(r.cw) + c.(r.tsurp):c.(r.csurp) +
        c.(nextisfix):c.(bwprob) , test))

      r.lent <- residuals(lm(c.(lagcumentred) ~ c.(lagsentpos) + c.(lagnrchar) + c.(lagprevisfix) + 
        c.(laguprob) + c.(lagfwprob) + c.(lagbwprob) + 
        c.(lagsentpos):c.(lagnrchar) + c.(lagsentpos):c.(lagbwprob) + 
        c.(lagnrchar):c.(laguprob) + c.(lagnrchar):c.(lagfwprob) + 
        c.(lagnrchar):c.(lagbwprob) + c.(laguprob):c.(lagfwprob) + 
        c.(r.lcw) + c.(lagfwprob):c.(r.lcw) + c.(r.lcsurp) + c.(lagbwprob):c.(r.lcsurp) +
        c.(r.ltsurp) + c.(lagsentpos):c.(r.ltsurp) + c.(lagbwprob):c.(r.ltsurp), test))

      bform <- clfdur ~ c.(sentpos) + c.(nrchar) + c.(previsfix) + c.(nextisfix) + 
        c.(uprob) + c.(fwprob) + c.(bwprob) + c.(r.tsurp) + 
        c.(r.cw) + c.(r.csurp) + c.(r.ent) + c.(lagsentpos) + c.(lagnrchar) + 
        c.(lagprevisfix) + c.(laguprob) + c.(lagfwprob) + 
        c.(lagbwprob) + c.(r.ltsurp) + c.(r.lcw) + c.(r.lcsurp) + 
        c.(r.lent) + (1 | subject) + (1 | word) + c.(sentpos):c.(fwprob) + 
        c.(sentpos):c.(r.tsurp) + c.(sentpos):c.(r.csurp) + c.(sentpos):c.(r.ent) + 
        c.(nrchar):c.(previsfix) + c.(nrchar):c.(uprob) + c.(nrchar):c.(fwprob) + 
        c.(previsfix):c.(nextisfix) + c.(previsfix):c.(r.cw) + c.(previsfix):c.(r.csurp) + 
        c.(previsfix):c.(r.ent) + c.(nextisfix):c.(fwprob) + 
        c.(nextisfix):c.(bwprob) + c.(nextisfix):c.(r.ent) + 
        c.(fwprob):c.(r.tsurp) + c.(fwprob):c.(r.csurp) + 
        c.(fwprob):c.(r.ent) + c.(r.tsurp):c.(r.cw) + c.(r.tsurp):c.(r.csurp) + 
        c.(r.tsurp):c.(r.ent) + c.(r.cw):c.(r.csurp) + c.(r.cw):c.(r.ent) + 
        c.(r.csurp):c.(r.ent) + c.(lagsentpos):c.(lagnrchar) + c.(lagsentpos):c.(lagbwprob) + 
        c.(lagsentpos):c.(r.ltsurp) + c.(lagnrchar):c.(laguprob) + 
        c.(lagnrchar):c.(lagfwprob) + c.(lagnrchar):c.(lagbwprob) + 
        c.(lagprevisfix):c.(r.lent) + c.(laguprob):c.(lagfwprob) + 
        c.(lagfwprob):c.(r.lcw) + c.(lagfwprob):c.(r.lent) + 
        c.(lagbwprob):c.(r.ltsurp) + c.(lagbwprob):c.(r.lcsurp)

      rform <- ~ c.(sentpos) + c.(nrchar) + c.(previsfix) + c.(nextisfix) + 
        c.(uprob) + c.(fwprob) + c.(bwprob) + c.(r.tsurp) + 
        c.(r.cw) + c.(r.csurp) + c.(r.ent) + c.(sentpos):c.(fwprob) + 
        c.(sentpos):c.(r.tsurp) + c.(sentpos):c.(r.csurp) + c.(sentpos):c.(r.ent) + 
        c.(nrchar):c.(previsfix) + c.(nrchar):c.(uprob) + c.(nrchar):c.(fwprob) + 
        c.(previsfix):c.(nextisfix) + c.(previsfix):c.(r.cw) + c.(previsfix):c.(r.csurp) + 
        c.(previsfix):c.(r.ent) + c.(nextisfix):c.(fwprob) + 
        c.(nextisfix):c.(bwprob) + c.(nextisfix):c.(r.ent) + 
        c.(fwprob):c.(r.tsurp) + c.(fwprob):c.(r.csurp) + 
        c.(fwprob):c.(r.ent) + c.(r.tsurp):c.(r.cw) + c.(r.tsurp):c.(r.csurp) + 
        c.(r.tsurp):c.(r.ent) + c.(r.cw):c.(r.csurp) + c.(r.cw):c.(r.ent) + 
        c.(r.csurp):c.(r.ent) 

      lrform <- ~ c.(lagsentpos) + c.(lagnrchar) + c.(lagprevisfix) + 
        c.(laguprob) + c.(lagfwprob) + c.(lagbwprob) + 
        c.(r.ltsurp) + c.(r.lcw) + c.(r.lcsurp) + c.(r.lent) + 
        c.(lagsentpos):c.(lagnrchar) + c.(lagsentpos):c.(lagbwprob) + 
        c.(lagsentpos):c.(r.ltsurp) + c.(lagnrchar):c.(laguprob) + 
        c.(lagnrchar):c.(lagfwprob) + c.(lagnrchar):c.(lagbwprob) + 
        c.(lagprevisfix):c.(r.lent) + c.(laguprob):c.(lagfwprob) + 
        c.(lagfwprob):c.(r.lcw) + c.(lagfwprob):c.(r.lent) + 
        c.(lagbwprob):c.(r.ltsurp) + c.(lagbwprob):c.(r.lcsurp)
    }
    else {
      bc <- ~ c.(sentpos) + c.(nrchar) + c.(previsfix) + c.(nextisfix) +
        c.(uprob) + c.(fwprob) + c.(bwprob) + c.(sentpos):c.(previsfix) +
        c.(sentpos):c.(nextisfix) + c.(sentpos):c.(uprob) +
        c.(sentpos):c.(fwprob) + c.(nrchar):c.(uprob) +
        c.(nrchar):c.(fwprob) + c.(previsfix):c.(nextisfix) +
        c.(previsfix):c.(uprob) + c.(previsfix):c.(fwprob) +
        c.(nextisfix):c.(fwprob) + c.(nextisfix):c.(bwprob)

      bl <- ~ c.(lagsentpos) + c.(lagnrchar) +
        c.(lagprevisfix) + c.(laguprob) + c.(lagfwprob) + c.(lagbwprob) + 
        c.(lagsentpos):c.(lagnrchar) + c.(lagsentpos):c.(lagbwprob) +
        c.(lagnrchar):c.(laguprob) 

      r.cw <- residuals(lm(update.formula(bc,c.(cumwdelta)~.), test))
      r.lcw <- residuals(lm(update.formula(bl,c.(lagcumwdelta)~.) , test))

      r.csurp <- residuals(lm(c.(cumtotsurp) ~ c.(sentpos) + c.(nrchar) + c.(previsfix) + c.(nextisfix) +
        c.(r.cw) + c.(sentpos):c.(r.cw) + c.(previsfix):c.(r.cw) + c.(nextisfix):c.(r.cw) +
        c.(uprob):c.(r.cw) + c.(fwprob):c.(r.cw) +
        c.(uprob) + c.(fwprob) + c.(bwprob) + c.(sentpos):c.(previsfix) +
        c.(sentpos):c.(nextisfix) + c.(sentpos):c.(uprob) +
        c.(sentpos):c.(fwprob) + c.(nrchar):c.(uprob) +
        c.(nrchar):c.(fwprob) + c.(previsfix):c.(nextisfix) +
        c.(previsfix):c.(uprob) + c.(previsfix):c.(fwprob) +
        c.(nextisfix):c.(fwprob) + c.(nextisfix):c.(bwprob),test))

      r.lcsurp <- residuals(lm(c.(lagcumtotsurp) ~ c.(lagsentpos) + c.(lagnrchar) +
        c.(lagprevisfix) + c.(laguprob) + c.(lagfwprob) + c.(lagbwprob) + 
        c.(r.lcw) + c.(lagfwprob):c.(r.lcw) +
        c.(lagsentpos):c.(lagnrchar) + c.(lagsentpos):c.(lagbwprob) +
        c.(lagnrchar):c.(laguprob) , test))

      r.tsurp <- residuals(lm(c.(totsurp) ~ c.(sentpos) + c.(nrchar) + c.(previsfix) + c.(nextisfix) +
        c.(r.cw) + c.(r.csurp) + c.(sentpos):c.(r.cw) + c.(previsfix):c.(r.cw) + c.(nextisfix):c.(r.cw) +
        c.(uprob):c.(r.cw) + c.(fwprob):c.(r.cw) + c.(sentpos):c.(r.csurp) +
        c.(uprob) + c.(fwprob) + c.(bwprob) + c.(sentpos):c.(previsfix) +
        c.(sentpos):c.(nextisfix) + c.(sentpos):c.(uprob) + c.(r.cw):c.(r.csurp) +
        c.(sentpos):c.(fwprob) + c.(nrchar):c.(uprob) +
        c.(nrchar):c.(fwprob) + c.(previsfix):c.(nextisfix) +
        c.(previsfix):c.(uprob) + c.(previsfix):c.(fwprob) +
        c.(nextisfix):c.(fwprob) + c.(nextisfix):c.(bwprob) + 
        c.(previsfix):c.(r.csurp) + c.(uprob):c.(r.csurp),test))

      r.ltsurp <- residuals(lm(c.(lagtotsurp) ~ c.(lagsentpos) + c.(lagnrchar) +
        c.(lagprevisfix) + c.(laguprob) + c.(lagfwprob) + c.(lagbwprob) + 
        c.(r.lcw) + c.(r.lcsurp) + c.(lagfwprob):c.(r.lcw) +
        c.(lagsentpos):c.(lagnrchar) + c.(lagsentpos):c.(lagbwprob) +
        c.(lagnrchar):c.(r.lcsurp) + c.(r.lcw):c.(r.lcsurp) +
        c.(lagnrchar):c.(laguprob) , test))

      r.ent <- residuals(lm(c.(cumentred) ~ c.(sentpos) + c.(nrchar) + c.(previsfix) + c.(nextisfix) +
        c.(r.cw) + c.(r.csurp) + c.(r.tsurp) + c.(sentpos):c.(r.tsurp) + c.(r.tsurp):c.(r.cw) +
        c.(r.tsurp):c.(r.csurp) + c.(sentpos):c.(r.cw) + c.(previsfix):c.(r.cw) + c.(nextisfix):c.(r.cw) +
        c.(uprob):c.(r.cw) + c.(fwprob):c.(r.cw) + c.(sentpos):c.(r.csurp) +
        c.(uprob) + c.(fwprob) + c.(bwprob) + c.(sentpos):c.(previsfix) +
        c.(sentpos):c.(nextisfix) + c.(sentpos):c.(uprob) + c.(r.cw):c.(r.csurp) +
        c.(sentpos):c.(fwprob) + c.(nrchar):c.(uprob) +
        c.(nrchar):c.(fwprob) + c.(previsfix):c.(nextisfix) +
        c.(previsfix):c.(uprob) + c.(previsfix):c.(fwprob) +
        c.(nextisfix):c.(fwprob) + c.(nextisfix):c.(bwprob) + 
        c.(previsfix):c.(r.csurp) + c.(uprob):c.(r.csurp),test))

      r.lent <- residuals(lm(c.(lagcumentred) ~ c.(lagsentpos) + c.(lagnrchar) +
        c.(lagprevisfix) + c.(laguprob) + c.(lagfwprob) + c.(lagbwprob) + 
        c.(r.lcw) + c.(r.lcsurp) + c.(r.ltsurp) + c.(lagfwprob):c.(r.lcw) +
        c.(lagsentpos):c.(r.ltsurp) + c.(lagnrchar):c.(r.ltsurp) + c.(lagbwprob):c.(r.ltsurp) +
        c.(lagsentpos):c.(lagnrchar) + c.(lagsentpos):c.(lagbwprob) + c.(r.ltsurp):c.(r.lcsurp) +
        c.(lagnrchar):c.(r.lcsurp) + c.(r.lcw):c.(r.lcsurp) + c.(r.ltsurp):c.(r.lcw) +
        c.(lagnrchar):c.(laguprob) , test))

      bform <- clfdur ~ c.(sentpos) + c.(nrchar) + c.(previsfix) + c.(nextisfix) +
        c.(uprob) + c.(fwprob) + c.(bwprob) + c.(r.tsurp) +
        c.(r.cw) + c.(r.csurp) + c.(r.ent) + c.(lagsentpos) + c.(lagnrchar) +
        c.(lagprevisfix) + c.(laguprob) + c.(lagfwprob) +
        c.(lagbwprob) + c.(r.ltsurp) + c.(r.lcw) + c.(r.lcsurp) +
        c.(r.lent) + (1 | subject) + (1 | word) + c.(sentpos):c.(previsfix) +
        c.(sentpos):c.(nextisfix) + c.(sentpos):c.(uprob) +
        c.(sentpos):c.(fwprob) + c.(sentpos):c.(r.tsurp) + c.(sentpos):c.(r.cw) +
        c.(sentpos):c.(r.csurp) + c.(sentpos):c.(r.ent) + c.(nrchar):c.(uprob) +
        c.(nrchar):c.(fwprob) + c.(previsfix):c.(nextisfix) +
        c.(previsfix):c.(uprob) + c.(previsfix):c.(fwprob) +
        c.(previsfix):c.(r.cw) + c.(previsfix):c.(r.csurp) + c.(nextisfix):c.(fwprob) +
        c.(nextisfix):c.(bwprob) + c.(nextisfix):c.(r.cw) +
        c.(nextisfix):c.(r.ent) + c.(uprob):c.(r.cw) + c.(uprob):c.(r.csurp) +
        c.(fwprob):c.(r.cw) + c.(r.tsurp):c.(r.cw) + c.(r.tsurp):c.(r.csurp) +
        c.(r.tsurp):c.(r.ent) + c.(r.cw):c.(r.csurp) + c.(r.cw):c.(r.ent) +
        c.(r.csurp):c.(r.ent) + c.(lagsentpos):c.(lagnrchar) + c.(lagsentpos):c.(lagbwprob) +
        c.(lagsentpos):c.(r.ltsurp) + c.(lagsentpos):c.(r.lent) +
        c.(lagnrchar):c.(laguprob) + c.(lagnrchar):c.(r.ltsurp) +
        c.(lagnrchar):c.(r.lcsurp) + c.(lagprevisfix):c.(r.lent) +
        c.(lagfwprob):c.(r.lcw) + c.(lagfwprob):c.(r.lent) +
        c.(lagbwprob):c.(r.ltsurp) + c.(r.ltsurp):c.(r.lcw) +
        c.(r.ltsurp):c.(r.lcsurp) + c.(r.ltsurp):c.(r.lent) + c.(r.lcw):c.(r.lcsurp)

      rform <- ~ c.(sentpos) + c.(nrchar) + c.(previsfix) + c.(nextisfix) +
        c.(uprob) + c.(fwprob) + c.(bwprob) + c.(r.tsurp) +
        c.(r.cw) + c.(r.csurp) + c.(r.ent) + c.(sentpos):c.(previsfix) +
        c.(sentpos):c.(nextisfix) + c.(sentpos):c.(uprob) +
        c.(sentpos):c.(fwprob) + c.(sentpos):c.(r.tsurp) + c.(sentpos):c.(r.cw) +
        c.(sentpos):c.(r.csurp) + c.(sentpos):c.(r.ent) + c.(nrchar):c.(uprob) +
        c.(nrchar):c.(fwprob) + c.(previsfix):c.(nextisfix) +
        c.(previsfix):c.(uprob) + c.(previsfix):c.(fwprob) +
        c.(previsfix):c.(r.cw) + c.(previsfix):c.(r.csurp) + c.(nextisfix):c.(fwprob) +
        c.(nextisfix):c.(bwprob) + c.(nextisfix):c.(r.cw) +
        c.(nextisfix):c.(r.ent) + c.(uprob):c.(r.cw) + c.(uprob):c.(r.csurp) +
        c.(fwprob):c.(r.cw) + c.(r.tsurp):c.(r.cw) + c.(r.tsurp):c.(r.csurp) +
        c.(r.tsurp):c.(r.ent) + c.(r.cw):c.(r.csurp) + c.(r.cw):c.(r.ent) +
        c.(r.csurp):c.(r.ent) 

      lrform <- ~ c.(lagsentpos) + c.(lagnrchar) +
        c.(lagprevisfix) + c.(laguprob) + c.(lagfwprob) +
        c.(lagbwprob) + c.(r.ltsurp) + c.(r.lcw) + c.(r.lcsurp) +
        c.(r.lent) + c.(lagsentpos):c.(lagnrchar) + c.(lagsentpos):c.(lagbwprob) +
        c.(lagsentpos):c.(r.ltsurp) + c.(lagsentpos):c.(r.lent) +
        c.(lagnrchar):c.(laguprob) + c.(lagnrchar):c.(r.ltsurp) +
        c.(lagnrchar):c.(r.lcsurp) + c.(lagprevisfix):c.(r.lent) +
        c.(lagfwprob):c.(r.lcw) + c.(lagfwprob):c.(r.lent) +
        c.(lagbwprob):c.(r.ltsurp) + c.(r.ltsurp):c.(r.lcw) +
        c.(r.ltsurp):c.(r.lcsurp) + c.(r.ltsurp):c.(r.lent) + c.(r.lcw):c.(r.lcsurp)
    }
  }
  ######################
  #
  # Create a baseline for main F/L effects
  #
  ######################

#  print("Build Base")
#  base <- lmer(bform, test, REML=F)

  ######################
  #
  # Test F/L/B Metrics
  #
  ######################

  rforma <- rform
  lrforma <- lrform
  bforma <- bform

  if (ALL) {
    print("Build F-L- Base")
    write("Build F-L- Base",stderr())
    base <- lmer(update.formula(bform,.~.+(1+c.(cumfmlm)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)), test, REML=F)

    print("Testing F-L-")
#    write("Residing F-L-",stderr())
#    r.fmlm <- residuals(lm(update.formula(rform,c.(cumfmlm) ~ .), test))
#    r.lfmlm <- residuals(lm(update.formula(lrform,c.(lagcumfmlm) ~ .), test))

    write("Testing F-L-",stderr())
#    baseofmlm <- lmer(update.formula(bform,.~.+ c.(r.fmlm) + (1+c.(cumfmlm)|subject)), test, REML=F)
    baseofmlm <- lmer(update.formula(bform,.~.+ c.(cumfmlm) + (1+c.(cumfmlm)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)), test, REML=F)
#    baselfmlm <- lmer(update.formula(bform,.~.+ c.(r.lfmlm)), test, REML=F)
#    basebfmlm <- lmer(update.formula(bform,.~.+ c.(r.fmlm) + c.(r.lfmlm)), test, REML=F)

    print(vif.mer(baseofmlm))
#    print(anova(base,basebfmlm))
    print(anova(base,baseofmlm))
#    print(anova(base,baselfmlm))
#    print(anova(baselfmlm,basebfmlm))
#    print(anova(baseofmlm,basebfmlm))
    print(baseofmlm)
#    print(baselfmlm)
#    print(basebfmlm)
    #basebfmlm.p <- pvals.fnc(basebfmlm)
    #print(basebfmlm.p)
    #basebfmlm.p <- NULL

    if (DEPTH) {
      print("Testing Dep1F-L-")
      write("Testing Dep1F-L-",stderr())
#      rform <- update.formula(rforma,. ~ . + c.(r.fmlm))
#      lrform <- update.formula(lrforma,. ~ . + c.(r.lfmlm))
#      bform <- update.formula(bforma,. ~ . + c.(r.fmlm) + c.(r.lfmlm))

      r.d1fmlm <- residuals(lm(update.formula(rform,c.(cumd1fmlm) ~ .), test))
      r.ld1fmlm <- residuals(lm(update.formula(lrform,c.(lagcumd1fmlm) ~ .), test))

      baseod1fmlm <- lmer(update.formula(bform,.~.+ c.(r.d1fmlm)+(cumd1fmlm|subject)), test, REML=F)
      baseld1fmlm <- lmer(update.formula(bform,.~.+ c.(r.ld1fmlm)+(lagcumd1fmlm|subject)), test, REML=F)
#      basebd1fmlm <- lmer(update.formula(bform,.~.+ c.(r.d1fmlm) + c.(r.ld1fmlm)), test, REML=F)

#      print(anova(base,basebd1fmlm))
      print(anova(base,baseod1fmlm))
      print(anova(base,baseld1fmlm))
      print(baseod1fmlm)
      print(baseld1fmlm)
#      print(basebd1fmlm)
      #basebd1fmlm.p <- pvals.fnc(basebd1fmlm)
      #print(basebd1fmlm.p)
      #basebd1fmlm.p <- NULL
      baseod1fmlm <- NULL
      baseld1fmlm <- NULL
#      basebd1fmlm <- NULL
      #baseofmlm <- NULL
      #baselfmlm <- NULL
      #basebfmlm <- NULL
      rform <- rforma
      lrform <- lrforma
      bform <- bforma

      print("Testing Dep2F-L-")
      write("Testing Dep2F-L-",stderr())
#      rform <- update.formula(rforma,. ~ . + c.(r.fmlm))
#      lrform <- update.formula(lrforma,. ~ . + c.(r.lfmlm))
#      bform <- update.formula(bforma,. ~ . + c.(r.fmlm) + c.(r.lfmlm))

      r.d2fmlm <- residuals(lm(update.formula(rform,c.(cumd2fmlm) ~ .), test))
      r.ld2fmlm <- residuals(lm(update.formula(lrform,c.(lagcumd2fmlm) ~ .), test))

      baseod2fmlm <- lmer(update.formula(bform,.~.+ c.(r.d2fmlm)+(cumd2fmlm|subject)), test, REML=F)
      baseld2fmlm <- lmer(update.formula(bform,.~.+ c.(r.ld2fmlm)+(lagcumd2fmlm|subject)), test, REML=F)
#      basebd2fmlm <- lmer(update.formula(bform,.~.+ c.(r.d2fmlm) + c.(r.ld2fmlm)), test, REML=F)

#      print(anova(base,basebd2fmlm))
      print(anova(base,baseod2fmlm))
      print(anova(base,baseld2fmlm))
      print(baseod2fmlm)
      print(baseld2fmlm)
#      print(basebd2fmlm)
      #basebd2fmlm.p <- pvals.fnc(basebd2fmlm)
      #print(basebd2fmlm.p)
      #basebd2fmlm.p <- NULL
      baseod2fmlm <- NULL
      baseld2fmlm <- NULL
#      basebd2fmlm <- NULL
      #baseofmlm <- NULL
      #baselfmlm <- NULL
      #basebfmlm <- NULL
      rform <- rforma
      lrform <- lrforma
      bform <- bforma

      print("Testing Dep3F-L-")
      write("Testing Dep3F-L-",stderr())
#      rform <- update.formula(rforma,. ~ . + c.(r.fmlm))
#      lrform <- update.formula(lrforma,. ~ . + c.(r.lfmlm))
#      bform <- update.formula(bforma,. ~ . + c.(r.fmlm) + c.(r.lfmlm))

      r.d3fmlm <- residuals(lm(update.formula(rform,c.(cumd3fmlm) ~ .), test))
      r.ld3fmlm <- residuals(lm(update.formula(lrform,c.(lagcumd3fmlm) ~ .), test))

      baseod3fmlm <- lmer(update.formula(bform,.~.+ c.(r.d3fmlm)+(cumd3fmlm|subject)), test, REML=F)
      baseld3fmlm <- lmer(update.formula(bform,.~.+ c.(r.ld3fmlm)+(cumd3fmlm|subject)), test, REML=F)
#      basebd3fmlm <- lmer(update.formula(bform,.~.+ c.(r.d3fmlm) + c.(r.ld3fmlm)), test, REML=F)

#      print(anova(base,basebd3fmlm))
      print(anova(base,baseod3fmlm))
      print(anova(base,baseld3fmlm))
      print(baseod3fmlm)
      print(baseld3fmlm)
#      print(basebd3fmlm)
      #basebd3fmlm.p <- pvals.fnc(basebd3fmlm)
      #print(basebd3fmlm.p)
      #basebd3fmlm.p <- NULL
      baseod3fmlm <- NULL
      baseld3fmlm <- NULL
#      basebd3fmlm <- NULL
      #baseofmlm <- NULL
      #baselfmlm <- NULL
      #basebfmlm <- NULL
      rform <- rforma
      lrform <- lrforma
      bform <- bforma

      print("Testing Dep4F-L-")
      write("Testing Dep4F-L-",stderr())
#      rform <- update.formula(rforma,. ~ . + c.(r.fmlm))
#      lrform <- update.formula(lrforma,. ~ . + c.(r.lfmlm))
#      bform <- update.formula(bforma,. ~ . + c.(r.fmlm) + c.(r.lfmlm))

      r.d4fmlm <- residuals(lm(update.formula(rform,c.(cumd4fmlm) ~ .), test))
      r.ld4fmlm <- residuals(lm(update.formula(lrform,c.(lagcumd4fmlm) ~ .), test))

      baseod4fmlm <- lmer(update.formula(bform,.~.+ c.(r.d4fmlm)+(cumd4fmlm|subject)), test, REML=F)
      baseld4fmlm <- lmer(update.formula(bform,.~.+ c.(r.ld4fmlm)+(cumd4fmlm|subject)), test, REML=F)
#      basebd4fmlm <- lmer(update.formula(bform,.~.+ c.(r.d4fmlm) + c.(r.ld4fmlm)), test, REML=F)

#      print(anova(base,basebd4fmlm))
      print(anova(base,baseod4fmlm))
      print(anova(base,baseld4fmlm))
      print(baseod4fmlm)
      print(baseld4fmlm)
#      print(basebd4fmlm)
      #basebd4fmlm.p <- pvals.fnc(basebd4fmlm)
      #print(basebd4fmlm.p)
      #basebd4fmlm.p <- NULL
      baseod4fmlm <- NULL
      baseld4fmlm <- NULL
#      basebd4fmlm <- NULL
    }
    baseofmlm <- NULL
  #  baselfmlm <- NULL
  #  basebfmlm <- NULL
  }
  rform <- rforma
  lrform <- lrforma
  bform <- bform

  print("Build F+L- Base")
  write("Build F+L- Base",stderr())
  base <- lmer(update.formula(bform,.~.+(1+c.(cumfplm)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)), test, REML=F)

  print("Testing F+L-")
  if (CMCL2013) {
    write("Residing F+L-",stderr())
    r.fplm <- residuals(lm(update.formula(rform,c.(cumfplm) ~ .), test))
    #r.lfplm <- residuals(lm(update.formula(lrform,c.(lagcumfplm) ~ .), test))
  }

  write("Testing F+L-",stderr())
  if (CMCL2013) {
    baseofplm <- lmer(update.formula(bform,.~.+ c.(r.fplm) + (1+c.(cumfplm)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)), test, REML=F)
  }
  else {
    baseofplm <- lmer(update.formula(bform,.~.+ c.(cumfplm) + (1+c.(cumfplm)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)), test, REML=F)
  }
#  baselfplm <- lmer(update.formula(bform,.~.+ c.(r.lfplm)), test, REML=F)
#  basebfplm <- lmer(update.formula(bform,.~.+ c.(r.fplm) + c.(r.lfplm)), test, REML=F)

  print(vif.mer(baseofplm))
#  print(anova(base,basebfplm))
  print(anova(base,baseofplm))
#  print(anova(base,baselfplm))
#  print(anova(baselfplm,basebfplm))
#  print(anova(baseofplm,basebfplm))
  print(baseofplm)
#  print(baselfplm)
#  print(basebfplm)
  #basebfplm.p <- pvals.fnc(basebfplm)
  #print(basebfplm.p)
  #basebfplm.p <- NULL

  if (DEPTH) {
    print("Testing Dep1F+L-")
    write("Testing Dep1F+L-",stderr())
#    rform <- update.formula(rforma,. ~ . + c.(r.fplm))
#    lrform <- update.formula(lrforma,. ~ . + c.(r.lfplm))
#    bform <- update.formula(bforma,. ~ . + c.(r.fplm) + c.(r.lfplm))

    r.d1fplm <- residuals(lm(update.formula(rform,c.(cumd1fplm) ~ .), test))
    r.ld1fplm <- residuals(lm(update.formula(lrform,c.(lagcumd1fplm) ~ .), test))

    baseod1fplm <- lmer(update.formula(bform,.~.+ c.(r.d1fplm) + (cumd1fplm|subject)), test, REML=F)
    baseld1fplm <- lmer(update.formula(bform,.~.+ c.(r.ld1fplm)+ (lagcumd1fplm|subject)), test, REML=F)
#    basebd1fplm <- lmer(update.formula(bform,.~.+ c.(r.d1fplm) + c.(r.ld1fplm)), test, REML=F)

#    print(anova(base,basebd1fplm))
    print(anova(base,baseod1fplm))
    print(anova(base,baseld1fplm))
    print(baseod1fplm)
    print(baseld1fplm)
#    print(basebd1fplm)
    #basebd1fplm.p <- pvals.fnc(basebd1fplm)
    #print(basebd1fplm.p)
    #basebd1fplm.p <- NULL
    baseod1fplm <- NULL
    baseld1fplm <- NULL
#    basebd1fplm <- NULL
    #baseofplm <- NULL
    #baselfplm <- NULL
    #basebfplm <- NULL
    rform <- rforma
    lrform <- lrforma
    bform <- bforma

    print("Testing Dep2F+L-")
    write("Testing Dep2F+L-",stderr())
#    rform <- update.formula(rforma,. ~ . + c.(r.fplm))
#    lrform <- update.formula(lrforma,. ~ . + c.(r.lfplm))
#    bform <- update.formula(bforma,. ~ . + c.(r.fplm) + c.(r.lfplm))

    r.d2fplm <- residuals(lm(update.formula(rform,c.(cumd2fplm) ~ .), test))
    r.ld2fplm <- residuals(lm(update.formula(lrform,c.(lagcumd2fplm) ~ .), test))

    baseod2fplm <- lmer(update.formula(bform,.~.+ c.(r.d2fplm)+(cumd2fplm|subject)), test, REML=F)
    baseld2fplm <- lmer(update.formula(bform,.~.+ c.(r.ld2fplm)+(lagcumd2fplm|subject)), test, REML=F)
#    basebd2fplm <- lmer(update.formula(bform,.~.+ c.(r.d2fplm) + c.(r.ld2fplm)), test, REML=F)

#    print(anova(base,basebd2fplm))
    print(anova(base,baseod2fplm))
    print(anova(base,baseld2fplm))
    print(baseod2fplm)
    print(baseld2fplm)
#    print(basebd2fplm)
    #basebd2fplm.p <- pvals.fnc(basebd2fplm)
    #print(basebd2fplm.p)
    #basebd2fplm.p <- NULL
    baseod2fplm <- NULL
    baseld2fplm <- NULL
#    basebd2fplm <- NULL
    #baseofplm <- NULL
    #baselfplm <- NULL
    #basebfplm <- NULL
    rform <- rforma
    lrform <- lrforma
    bform <- bforma

    print("Testing Dep3F+L-")
    write("Testing Dep3F+L-",stderr())
#    rform <- update.formula(rforma,. ~ . + c.(r.fplm))
#    lrform <- update.formula(lrforma,. ~ . + c.(r.lfplm))
#    bform <- update.formula(bforma,. ~ . + c.(r.fplm) + c.(r.lfplm))

    r.d3fplm <- residuals(lm(update.formula(rform,c.(cumd3fplm) ~ .), test))
    r.ld3fplm <- residuals(lm(update.formula(lrform,c.(lagcumd3fplm) ~ .), test))

    baseod3fplm <- lmer(update.formula(bform,.~.+ c.(r.d3fplm)+(cumd3fplm|subject)), test, REML=F)
    baseld3fplm <- lmer(update.formula(bform,.~.+ c.(r.ld3fplm)+(cumd3fplm|subject)), test, REML=F)
#    basebd3fplm <- lmer(update.formula(bform,.~.+ c.(r.d3fplm) + c.(r.ld3fplm)), test, REML=F)

#    print(anova(base,basebd3fplm))
    print(anova(base,baseod3fplm))
    print(anova(base,baseld3fplm))
    print(baseod3fplm)
    print(baseld3fplm)
#    print(basebd3fplm)
    #basebd3fplm.p <- pvals.fnc(basebd3fplm)
    #print(basebd3fplm.p)
    #basebd3fplm.p <- NULL
    baseod3fplm <- NULL
    baseld3fplm <- NULL
#    basebd3fplm <- NULL
    #baseofplm <- NULL
    #baselfplm <- NULL
    #basebfplm <- NULL
    rform <- rforma
    lrform <- lrforma
    bform <- bforma

    print("Testing Dep4F+L-")
    write("Testing Dep4F+L-",stderr())
#    rform <- update.formula(rforma,. ~ . + c.(r.fplm))
#    lrform <- update.formula(lrforma,. ~ . + c.(r.lfplm))
#    bform <- update.formula(bforma,. ~ . + c.(r.fplm) + c.(r.lfplm))

    r.d4fplm <- residuals(lm(update.formula(rform,c.(cumd4fplm) ~ .), test))
    r.ld4fplm <- residuals(lm(update.formula(lrform,c.(lagcumd4fplm) ~ .), test))

    baseod4fplm <- lmer(update.formula(bform,.~.+ c.(r.d4fplm)+(cumd4fplm|subject)), test, REML=F)
    baseld4fplm <- lmer(update.formula(bform,.~.+ c.(r.ld4fplm)+(cumd4fplm|subject)), test, REML=F)
#    basebd4fplm <- lmer(update.formula(bform,.~.+ c.(r.d4fplm) + c.(r.ld4fplm)), test, REML=F)

#    print(anova(base,basebd4fplm))
    print(anova(base,baseod4fplm))
    print(anova(base,baseld4fplm))
    print(baseod4fplm)
    print(baseld4fplm)
#    print(basebd4fplm)
    #basebd4fplm.p <- pvals.fnc(basebd4fplm)
    #print(basebd4fplm.p)
    #basebd4fplm.p <- NULL
    baseod4fplm <- NULL
    baseld4fplm <- NULL
#    basebd4fplm <- NULL
  }
  baseofplm <- NULL
  baselfplm <- NULL
  basebfplm <- NULL
  rform <- rforma
  lrform <- lrforma
  bform <- bforma

  print("Build F-L+ Base")
  write("Build F-L+ Base",stderr())
  base <- lmer(update.formula(bform,.~.+(1+c.(cumfmlp)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)), test, REML=F)

  print("Testing F-L+")
  if (CMCL2013) {
    write("Residing F-L+",stderr())
    r.fmlp <- residuals(lm(update.formula(rform,c.(cumfmlp) ~ .), test))
    #r.lfmlp <- residuals(lm(update.formula(lrform,c.(lagcumfmlp) ~ .), test))
  }

  write("Testing F-L+",stderr())
  if (CMCL2013) {
    baseofmlp <- lmer(update.formula(bform,.~.+ c.(r.fmlp) + (1+c.(cumfmlp)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)), test, REML=F)
  }
  else {
    baseofmlp <- lmer(update.formula(bform,.~.+ c.(cumfmlp) + (1+c.(cumfmlp)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)), test, REML=F)
  }
#  baselfmlp <- lmer(update.formula(bform,.~.+ c.(r.lfmlp)), test, REML=F)
#  basebfmlp <- lmer(update.formula(bform,.~.+ c.(r.fmlp) + c.(r.lfmlp)), test, REML=F)

  print(vif.mer(baseofmlp))
#  print(anova(base,basebfmlp))
  print(anova(base,baseofmlp))
#  print(anova(base,baselfmlp))
#  print(anova(baselfmlp,basebfmlp))
#  print(anova(baseofmlp,basebfmlp))
  print(baseofmlp)
#  print(baselfmlp)
#  print(basebfmlp)
  #basebfmlp.p <- pvals.fnc(basebfmlp)
  #print(basebfmlp.p)
  #basebfmlp.p <- NULL

  if (DEPTH) {
    print("Testing Dep1F-L+")
    write("Testing Dep1F-L+",stderr())
#    rform <- update.formula(rforma,. ~ . + c.(r.fmlp))
#    lrform <- update.formula(lrforma,. ~ . + c.(r.lfmlp))
#    bform <- update.formula(bforma,. ~ . + c.(r.fmlp) + c.(r.lfmlp))

    r.d1fmlp <- residuals(lm(update.formula(rform,c.(cumd1fmlp) ~ .), test))
    r.ld1fmlp <- residuals(lm(update.formula(lrform,c.(lagcumd1fmlp) ~ .), test))

    baseod1fmlp <- lmer(update.formula(bform,.~.+ c.(r.d1fmlp)+(cumd1fmlp|subject)), test, REML=F)
    baseld1fmlp <- lmer(update.formula(bform,.~.+ c.(r.ld1fmlp)+(lagcumd1fmlp|subject)), test, REML=F)
#    basebd1fmlp <- lmer(update.formula(bform,.~.+ c.(r.d1fmlp) + c.(r.ld1fmlp)), test, REML=F)

#    print(anova(base,basebd1fmlp))
    print(anova(base,baseod1fmlp))
    print(anova(base,baseld1fmlp))
    print(baseod1fmlp)
    print(baseld1fmlp)
#    print(basebd1fmlp)
    #basebd1fmlp.p <- pvals.fnc(basebd1fmlp)
    #print(basebd1fmlp.p)
    #basebd1fmlp.p <- NULL
    baseod1fmlp <- NULL
    baseld1fmlp <- NULL
#    basebd1fmlp <- NULL
#    baseofmlp <- NULL
#    baselfmlp <- NULL
#    basebfmlp <- NULL
    rform <- rforma
    lrform <- lrforma
    bform <- bforma

    print("Testing Dep2F-L+")
    write("Testing Dep2F-L+",stderr())
#    rform <- update.formula(rforma,. ~ . + c.(r.fmlp))
#    lrform <- update.formula(lrforma,. ~ . + c.(r.lfmlp))
#    bform <- update.formula(bforma,. ~ . + c.(r.fmlp) + c.(r.lfmlp))

    r.d2fmlp <- residuals(lm(update.formula(rform,c.(cumd2fmlp) ~ .), test))
    r.ld2fmlp <- residuals(lm(update.formula(lrform,c.(lagcumd2fmlp) ~ .), test))

    baseod2fmlp <- lmer(update.formula(bform,.~.+ c.(r.d2fmlp)+(cumd2fmlp|subject)), test, REML=F)
    baseld2fmlp <- lmer(update.formula(bform,.~.+ c.(r.ld2fmlp)+(lagcumd2fmlp|subject)), test, REML=F)
#    basebd2fmlp <- lmer(update.formula(bform,.~.+ c.(r.d2fmlp) + c.(r.ld2fmlp)), test, REML=F)

#    print(anova(base,basebd2fmlp))
    print(anova(base,baseod2fmlp))
    print(anova(base,baseld2fmlp))
    print(baseod2fmlp)
    print(baseld2fmlp)
#    print(basebd2fmlp)
    #basebd2fmlp.p <- pvals.fnc(basebd2fmlp)
    #print(basebd2fmlp.p)
    #basebd2fmlp.p <- NULL
    baseod2fmlp <- NULL
    baseld2fmlp <- NULL
#    basebd2fmlp <- NULL
#    baseofmlp <- NULL
#    baselfmlp <- NULL
#    basebfmlp <- NULL
    rform <- rforma
    lrform <- lrforma
    bform <- bforma

    print("Testing Dep3F-L+")
    write("Testing Dep3F-L+",stderr())
#    rform <- update.formula(rforma,. ~ . + c.(r.fmlp))
#    lrform <- update.formula(lrforma,. ~ . + c.(r.lfmlp))
#    bform <- update.formula(bforma,. ~ . + c.(r.fmlp) + c.(r.lfmlp))

    r.d3fmlp <- residuals(lm(update.formula(rform,c.(cumd3fmlp) ~ .), test))
    r.ld3fmlp <- residuals(lm(update.formula(lrform,c.(lagcumd3fmlp) ~ .), test))

    baseod3fmlp <- lmer(update.formula(bform,.~.+ c.(r.d3fmlp)+(cumd3fmlp|subject)), test, REML=F)
    baseld3fmlp <- lmer(update.formula(bform,.~.+ c.(r.ld3fmlp)+(lagcumd3fmlp|subject)), test, REML=F)
#    basebd3fmlp <- lmer(update.formula(bform,.~.+ c.(r.d3fmlp) + c.(r.ld3fmlp)), test, REML=F)

#    print(anova(base,basebd3fmlp))
    print(anova(base,baseod3fmlp))
    print(anova(base,baseld3fmlp))
    print(baseod3fmlp)
    print(baseld3fmlp)
#    print(basebd3fmlp)
    #basebd3fmlp.p <- pvals.fnc(basebd3fmlp)
    #print(basebd3fmlp.p)
    #basebd3fmlp.p <- NULL
    baseod3fmlp <- NULL
    baseld3fmlp <- NULL
#    basebd3fmlp <- NULL
#    baseofmlp <- NULL
#    baselfmlp <- NULL
#    basebfmlp <- NULL
    rform <- rforma
    lrform <- lrforma
    bform <- bforma

    print("Testing Dep4F-L+")
    write("Testing Dep4F-L+",stderr())
#    rform <- update.formula(rforma,. ~ . + c.(r.fmlp))
#    lrform <- update.formula(lrforma,. ~ . + c.(r.lfmlp))
#    bform <- update.formula(bforma,. ~ . + c.(r.fmlp) + c.(r.lfmlp))

    r.d4fmlp <- residuals(lm(update.formula(rform,c.(cumd4fmlp) ~ .), test))
    r.ld4fmlp <- residuals(lm(update.formula(lrform,c.(lagcumd4fmlp) ~ .), test))

    baseod4fmlp <- lmer(update.formula(bform,.~.+ c.(r.d4fmlp)+(cumd4fmlp|subject)), test, REML=F)
    baseld4fmlp <- lmer(update.formula(bform,.~.+ c.(r.ld4fmlp)+(cumd4fmlp|subject)), test, REML=F)
#    basebd4fmlp <- lmer(update.formula(bform,.~.+ c.(r.d4fmlp) + c.(r.ld4fmlp)), test, REML=F)

#    print(anova(base,basebd4fmlp))
    print(anova(base,baseod4fmlp))
    print(anova(base,baseld4fmlp))
    print(baseod4fmlp)
    print(baseld4fmlp)
#    print(basebd4fmlp)
    #basebd4fmlp.p <- pvals.fnc(basebd4fmlp)
    #print(basebd4fmlp.p)
    #basebd4fmlp.p <- NULL
    baseod4fmlp <- NULL
    baseld4fmlp <- NULL
#    basebd4fmlp <- NULL
  }
  baseofmlp <- NULL
  baselfmlp <- NULL
  basebfmlp <- NULL
  rform <- rforma
  lrform <- lrforma
  bform <- bforma

  if (ALL) {
    print("Build F+L+ Base")
    write("Build F+L+ Base",stderr())
    base <- lmer(update.formula(bform,.~.+(1+c.(cumfplp)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)), test, REML=F)

    print("Testing F+L+")
#    write("Residing F+L+",stderr())
#    r.fplp <- residuals(lm(update.formula(rform,c.(cumfplp) ~ .), test))
#    r.lfplp <- residuals(lm(update.formula(lrform,c.(lagcumfplp) ~ .), test))

    write("Testing F+L+",stderr())
#    baseofplp <- lmer(update.formula(bform,.~.+ c.(r.fplp) + (1+c.(cumfplp)|subject)), test, REML=F)
    baseofplp <- lmer(update.formula(bform,.~.+ c.(cumfplp) + (1+c.(cumfplp)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)), test, REML=F)
#    baselfplp <- lmer(update.formula(bform,.~.+ c.(r.lfplp)), test, REML=F)
#    basebfplp <- lmer(update.formula(bform,.~.+ c.(r.fplp) + c.(r.lfplp)), test, REML=F)

#    print(anova(base,basebfplp))
    print(anova(base,baseofplp))
#    print(anova(base,baselfplp))
#    print(anova(baselfplp,basebfplp))
#    print(anova(baseofplp,basebfplp))
    print(baseofplp)
#    print(baselfplp)
#    print(basebfplp)
    #basebfplp.p <- pvals.fnc(basebfplp)
    #print(basebfplp.p)
    #basebfplp.p <- NULL
  
    if (DEPTH) {
      print("Testing Dep1F+L+")
      write("Testing Dep1F+L+",stderr())
#      rform <- update.formula(rforma,. ~ . + c.(r.fplp))
#      lrform <- update.formula(lrforma,. ~ . + c.(r.lfplp))
#      bform <- update.formula(bforma,. ~ . + c.(r.fplp) + c.(r.lfplp))

      r.d1fplp <- residuals(lm(update.formula(rform,c.(cumd1fplp) ~ .), test))
      r.ld1fplp <- residuals(lm(update.formula(lrform,c.(lagcumd1fplp) ~ .), test))

      baseod1fplp <- lmer(update.formula(bform,.~.+ c.(r.d1fplp)+(cumd1fplp|subject)), test, REML=F)
      baseld1fplp <- lmer(update.formula(bform,.~.+ c.(r.ld1fplp)+(lagcumd1fplp|subject)), test, REML=F)
#      basebd1fplp <- lmer(update.formula(bform,.~.+ c.(r.d1fplp) + c.(r.ld1fplp)), test, REML=F)

#      print(anova(base,basebd1fplp))
      print(anova(base,baseod1fplp))
      print(anova(base,baseld1fplp))
      print(baseod1fplp)
      print(baseld1fplp)
#      print(basebd1fplp)
      #basebd1fplp.p <- pvals.fnc(basebd1fplp)
      #print(basebd1fplp.p)
      #basebd1fplp.p <- NULL
      baseod1fplp <- NULL
      baseld1fplp <- NULL
#      basebd1fplp <- NULL
#      baseofplp <- NULL
#      baselfplp <- NULL
#      basebfplp <- NULL
      rform <- rforma
      lrform <- lrforma
      bform <- bforma

      print("Testing Dep2F+L+")
      write("Testing Dep2F+L+",stderr())
#      rform <- update.formula(rforma,. ~ . + c.(r.fplp))
#      lrform <- update.formula(lrforma,. ~ . + c.(r.lfplp))
#      bform <- update.formula(bforma,. ~ . + c.(r.fplp) + c.(r.lfplp))

      r.d2fplp <- residuals(lm(update.formula(rform,c.(cumd2fplp) ~ .), test))
      r.ld2fplp <- residuals(lm(update.formula(lrform,c.(lagcumd2fplp) ~ .), test))

      baseod2fplp <- lmer(update.formula(bform,.~.+ c.(r.d2fplp)+(cumd2fplp|subject)), test, REML=F)
      baseld2fplp <- lmer(update.formula(bform,.~.+ c.(r.ld2fplp)+(lagcumd2fplp|subject)), test, REML=F)
#      basebd2fplp <- lmer(update.formula(bform,.~.+ c.(r.d2fplp) + c.(r.ld2fplp)), test, REML=F)

#      print(anova(base,basebd2fplp))
      print(anova(base,baseod2fplp))
      print(anova(base,baseld2fplp))
      print(baseod2fplp)
      print(baseld2fplp)
#      print(basebd2fplp)
      #basebd2fplp.p <- pvals.fnc(basebd2fplp)
      #print(basebd2fplp.p)
      #basebd2fplp.p <- NULL
      baseod2fplp <- NULL
      baseld2fplp <- NULL
#      basebd2fplp <- NULL
#      baseofplp <- NULL
#      baselfplp <- NULL
#      basebfplp <- NULL
      rform <- rforma
      lrform <- lrforma
      bform <- bforma

      print("Testing Dep3F+L+")
      write("Testing Dep3F+L+",stderr())
#      rform <- update.formula(rforma,. ~ . + c.(r.fplp))
#      lrform <- update.formula(lrforma,. ~ . + c.(r.lfplp))
#      bform <- update.formula(bforma,. ~ . + c.(r.fplp) + c.(r.lfplp))

      r.d3fplp <- residuals(lm(update.formula(rform,c.(cumd3fplp) ~ .), test))
      r.ld3fplp <- residuals(lm(update.formula(lrform,c.(lagcumd3fplp) ~ .), test))

      baseod3fplp <- lmer(update.formula(bform,.~.+ c.(r.d3fplp)+(cumd3fplp|subject)), test, REML=F)
      baseld3fplp <- lmer(update.formula(bform,.~.+ c.(r.ld3fplp)+(lagcumd3fplp|subject)), test, REML=F)
#      basebd3fplp <- lmer(update.formula(bform,.~.+ c.(r.d3fplp) + c.(r.ld3fplp)), test, REML=F)

#      print(anova(base,basebd3fplp))
      print(anova(base,baseod3fplp))
      print(anova(base,baseld3fplp))
      print(baseod3fplp)
      print(baseld3fplp)
#      print(basebd3fplp)
      #basebd3fplp.p <- pvals.fnc(basebd3fplp)
      #print(basebd3fplp.p)
      #basebd3fplp.p <- NULL
      baseod3fplp <- NULL
      baseld3fplp <- NULL
#      basebd3fplp <- NULL
#      baseofplp <- NULL
#      baselfplp <- NULL
#      basebfplp <- NULL
      rform <- rforma
      lrform <- lrforma
      bform <- bforma

      print("Testing Dep4F+L+")
      write("Testing Dep4F+L+",stderr())
#      rform <- update.formula(rforma,. ~ . + c.(r.fplp))
#      lrform <- update.formula(lrforma,. ~ . + c.(r.lfplp))
#      bform <- update.formula(bforma,. ~ . + c.(r.fplp) + c.(r.lfplp))
#
      r.d4fplp <- residuals(lm(update.formula(rform,c.(cumd4fplp) ~ .), test))
      r.ld4fplp <- residuals(lm(update.formula(lrform,c.(lagcumd4fplp) ~ .), test))

      baseod4fplp <- lmer(update.formula(bform,.~.+ c.(r.d4fplp)+(cumd4fplp|subject)), test, REML=F)
      baseld4fplp <- lmer(update.formula(bform,.~.+ c.(r.ld4fplp)+(lagcumd4fplp|subject)), test, REML=F)
#      basebd4fplp <- lmer(update.formula(bform,.~.+ c.(r.d4fplp) + c.(r.ld4fplp)), test, REML=F)

#      print(anova(base,basebd4fplp))
      print(anova(base,baseod4fplp))
      print(anova(base,baseld4fplp))
      print(baseod4fplp)
      print(baseld4fplp)
#      print(basebd4fplp)
#      #basebd4fplp.p <- pvals.fnc(basebd4fplp)
#      #print(basebd4fplp.p)
#      #basebd4fplp.p <- NULL
      baseod4fplp <- NULL
      baseld4fplp <- NULL
#      basebd4fplp <- NULL
    }
    baseofplp <- NULL
    baselfplp <- NULL
    basebfplp <- NULL
  }
  rform <- rforma
  lrform <- lrforma
  bform <- bforma

  ######################
  #
  # Non-efabp will crash here due to lack of dist output
  # Stems from inefficiency of dist calculation without referent binding
  # so abort:

  if (length( grep('cumDfmlp',colnames(test),fixed=T) ) == 0) return(0)

  #
  ######################

  if (DIST){
    print("Testing DistF-L+")
    write("Testing DistF-L+",stderr())
    rform <- update.formula(rforma,. ~ . + c.(r.fmlp))
    lrform <- update.formula(lrforma,. ~ . + c.(r.lfmlp))
    bform <- update.formula(bforma,. ~ . + c.(r.fmlp) + c.(r.lfmlp))

    r.Dfmlp <- residuals(lm(update.formula(rform,c.(cumDfmlp) ~ .), test))
    r.lDfmlp <- residuals(lm(update.formula(lrform,c.(lagcumDfmlp) ~ .), test))

    baseoDfmlp <- lmer(update.formula(bform,.~.+ c.(r.Dfmlp)), test, REML=F)
    baselDfmlp <- lmer(update.formula(bform,.~.+ c.(r.lDfmlp)), test, REML=F)
    basebDfmlp <- lmer(update.formula(bform,.~.+ c.(r.Dfmlp) + c.(r.lDfmlp)), test, REML=F)

    print(anova(basebfmlp,basebDfmlp))
    print(anova(basebfmlp,baseoDfmlp))
    print(anova(basebfmlp,baselDfmlp))
    print(anova(baselDfmlp,basebDfmlp))
    print(anova(baseoDfmlp,basebDfmlp))
    print(baseoDfmlp)
    print(baselDfmlp)
    print(basebDfmlp)
    #basebDfmlp.p <- pvals.fnc(basebDfmlp)
    #print(basebDfmlp.p)
    #basebDfmlp.p <- NULL
    baseoDfmlp <- NULL
    baselDfmlp <- NULL
    basebDfmlp <- NULL
#    baseofmlp <- NULL
#    baselfmlp <- NULL
#    basebfmlp <- NULL

    print("Testing DepDistF-L+")
    write("Testing DepDistF-L+",stderr())
    r.dDfmlp <- residuals(lm(update.formula(rform,c.(cumdDfmlp) ~ .), test))
    r.ldDfmlp <- residuals(lm(update.formula(lrform,c.(lagcumdDfmlp) ~ .), test))

    baseodDfmlp <- lmer(update.formula(bform,.~.+ c.(r.dDfmlp)), test, REML=F)
    baseldDfmlp <- lmer(update.formula(bform,.~.+ c.(r.ldDfmlp)), test, REML=F)
    basebdDfmlp <- lmer(update.formula(bform,.~.+ c.(r.dDfmlp) + c.(r.ldDfmlp)), test, REML=F)

    print(anova(basebfmlp,basebdDfmlp))
    print(anova(basebfmlp,baseodDfmlp))
    print(anova(basebfmlp,baseldDfmlp))
    print(anova(baseldDfmlp,basebdDfmlp))
    print(anova(baseodDfmlp,basebdDfmlp))
    print(baseodDfmlp)
    print(baseldDfmlp)
    print(basebdDfmlp)
    #basebdDfmlp.p <- pvals.fnc(basebdDfmlp)
    #print(basebdDfmlp.p)
    #basebdDfmlp.p <- NULL
    baseodDfmlp <- NULL
    baseldDfmlp <- NULL
    basebdDfmlp <- NULL
    baseofmlp <- NULL
    baselfmlp <- NULL
    basebfmlp <- NULL
  }

  print("Preparing to test B-metrics")

  rform <- rforma
  lrform <- lrforma
  bform <- bforma

#  r.fplm <- residuals(lm(update.formula(rform,c.(cumfplm) ~ .+c.(r.fmlp)), test))
#  r.lfplm <- residuals(lm(update.formula(lrform,c.(lagcumfplm) ~ .+c.(r.lfmlp)), test))
#  r.fmlm <- residuals(lm(update.formula(rform,c.(cumfmlm) ~ .+c.(r.fmlp)+c.(r.fplm)), test))
#  r.lfmlm <- residuals(lm(update.formula(lrform,c.(lagcumfmlm) ~ .+c.(r.lfmlp)+c.(r.lfplm)), test))
#  r.fplp <- residuals(lm(update.formula(rform,c.(cumfplp) ~ .+c.(r.fmlp)+c.(r.fplm)+c.(r.fmlm)), test))
#  r.lfplp <- residuals(lm(update.formula(lrform,c.(lagcumfplp) ~ .+c.(r.lfmlp)+c.(r.lfplm)+c.(r.lfmlm)), test))

#  rformb <- update.formula(rform,.~c.(r.fmlp)+c.(r.fmlm)+c.(r.fplm)+c.(r.fplp)+.)
#  lrformb <- update.formula(lrform,.~c.(r.lfmlp)+c.(r.lfmlm)+c.(r.lfplm)+c.(r.lfplp)+.)
#  bformb <- update.formula(bform,.~c.(r.fmlp)+c.(r.fmlm)+c.(r.fplm)+c.(r.fplp)+c.(r.lfmlp)+c.(r.lfmlm)+c.(r.lfplm)+c.(r.lfplp)+.)
##  rformb <- update.formula(rform,.~c.(cumfmlp)+c.(cumfmlm)+c.(cumfplm)+c.(cumfplp)+.)
##  lrformb <- update.formula(lrform,.~c.(lagcumfmlp)+c.(lagcumfmlm)+c.(lagcumfplm)+c.(lagcumfplp)+.)
##  bformb <- update.formula(bform,.~c.(cumfmlp)+c.(cumfmlm)+c.(cumfplm)+c.(cumfplp)+c.(lagcumfmlp)+c.(lagcumfmlm)+c.(lagcumfplm)+c.(lagcumfplp)+.)
rformb <- rform
lrformb <- lrform
bformb <- bform
#  baseb <- lmer(bformb,test,REML=F)

  print("Testing Cum Integration")
  write("Testing Cum Integration",stderr())
  write("Building Base: Cum Integration",stderr())
  if (CMCL2013){
    baseb <- lmer(update.formula(bformb,.~.+(1+c.(cumfmlmbp+cumfmlpbp+cumfplmbp+cumfplpbp+cumfmlpbo+cumfmlpba+cumfmlpbc)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)),test,REML=F)
  }
  else {
    baseb <- lmer(update.formula(bformb,.~.+(1+c.(cumfmlmbp+cumfmlpbp+cumfplmbp+cumfplpbp+cumfmlpbo+cumfmlpba+cumfmlpbc)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)),test,REML=F)
  }

  if (CMCL2013) {
    write("Residing Cum Integration",stderr())
    r.cint <- residuals(lm(update.formula(rformb,c.(cumfmlmbp+cumfmlpbp+cumfplmbp+cumfplpbp+cumfmlpbo+cumfmlpba+cumfmlpbc) ~ .), test))
    #r.lint <- residuals(lm(update.formula(lrformb,c.(lagcumfmlmbo+lagcumfmlpbo+lagcumfplmbo+lagcumfplpbo) ~ .), test))
  }

  write("Testing",stderr())
  if (CMCL2013) {
    baseocint <- lmer(update.formula(bformb,.~.+ c.(r.cint) + (1+c.(cumfmlmbp+cumfmlpbp+cumfplmbp+cumfplpbp+cumfmlpbo+cumfmlpba+cumfmlpbc)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)), test, REML=F)
  }
  else {
    baseocint <- lmer(update.formula(bformb,.~.+ c.(cumfmlmbp+cumfmlpbp+cumfplmbp+cumfplpbp+cumfmlpbo+cumfmlpba+cumfmlpbc) + (1+c.(cumfmlmbp+cumfmlpbp+cumfplmbp+cumfplpbp+cumfmlpbo+cumfmlpba+cumfmlpbc)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)), test, REML=F)
  }
#  baselint <- lmer(update.formula(bformb,.~.+ c.(r.lint)), test, REML=F)
#  basebcint <- lmer(update.formula(bformb,.~.+ c.(r.cint) + c.(r.lint)), test, REML=F)

#  print(anova(baseb,basebcint))
  print(anova(baseb,baseocint))
#  print(anova(baseb,baselint))
#  print(anova(baselint,basebcint))
#  print(anova(baseocint,basebcint))
  print(baseocint)
#  print(baselint)
#  print(basebcint)
  #basebcint.p <- pvals.fnc(basebcint)
  #print(bcint.p)
  #basebcint.p <- NULL
  baseocint <- NULL
#  baselint <- NULL
#  basebcint <- NULL

  if (CMCL2013){
    return(0)
  }

#  print("Testing B Addition")
#  write("Testing B Addition",stderr())

#  r.ba <- residuals(lm(update.formula(rformb,c.(badd) ~ .), test))

#  baseoba <- lmer(update.formula(bformb,.~.+ c.(r.ba)), test, REML=F)

#  print(anova(baseb,baseoba))
#  print(baseoba)
  #baseoba.p <- pvals.fnc(baseoba)
  #print(baseoba.p)
  #baseoba.p <- NULL
#  baseoba <- NULL

  print("Testing Cum B Addition")
  write("Testing Cum B Addition",stderr())
  write("Building Base: Cum B Addition",stderr())
  baseb <- lmer(update.formula(bformb,.~.+(1+c.(cumbadd)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)),test,REML=F)

#  write("Residing Cum B Addition",stderr())
#  r.cba <- residuals(lm(update.formula(rformb,c.(cumbadd) ~ .), test))
#  r.lba <- residuals(lm(update.formula(lrformb,c.(lagcumbadd) ~ .), test))

  write("Testing",stderr())
#  baseocba <- lmer(update.formula(bformb,.~.+ c.(r.cba) + (1+c.(cumbadd)|subject)), test, REML=F)
  baseocba <- lmer(update.formula(bformb,.~.+ c.(cumbadd) + (1+c.(cumbadd)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)), test, REML=F)
#  baselba <- lmer(update.formula(bformb,.~.+ c.(r.lba)), test, REML=F)
#  basebcba <- lmer(update.formula(bformb,.~.+ c.(r.cba) + c.(r.lba)), test, REML=F)

#  print(anova(baseb,basebcba))
  print(anova(baseb,baseocba))
#  print(anova(baseb,baselba))
#  print(anova(baselba,basebcba))
#  print(anova(baseocba,basebcba))
  print(baseocba)
#  print(baselba)
#  print(basebcba)
  #basebcba.p <- pvals.fnc(basebcba)
  #print(bcba.p)
  #basebcba.p <- NULL
  baseocba <- NULL
#  baselba <- NULL
#  basebcba <- NULL

#  print("Testing B+")
#  write("Testing B+",stderr())

#  r.bp <- residuals(lm(update.formula(rformb,c.(bp) ~ .), test))

#  baseobp <- lmer(update.formula(bformb,.~.+ c.(r.bp)), test, REML=F)

#  print(anova(baseb,baseobp))
#  print(baseobp)
  #baseobp.p <- pvals.fnc(baseobp)
  #print(baseobp.p)
  #baseobp.p <- NULL
#  baseobp <- NULL

  print("Testing Cum B+")
  write("Testing Cum B+",stderr())
  write("Building Base: Cum B+",stderr())
  baseb <- lmer(update.formula(bformb,.~.+(1+c.(cumbp)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)),test,REML=F)

#  write("Residing Cum B+",stderr())
#  r.cbp <- residuals(lm(update.formula(rformb,c.(cumbp) ~ .), test))
#  r.lbp <- residuals(lm(update.formula(lrformb,c.(lagcumbp) ~ .), test))

  write("Testing",stderr())
#  baseocbp <- lmer(update.formula(bformb,.~.+ c.(r.cbp) + (1+c.(cumbp)|subject)), test, REML=F)
  baseocbp <- lmer(update.formula(bformb,.~.+ c.(cumbp) + (1+c.(cumbp)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)), test, REML=F)
#  baselbp <- lmer(update.formula(bformb,.~.+ c.(r.lbp)), test, REML=F)
#  basebcbp <- lmer(update.formula(bformb,.~.+ c.(r.cbp) + c.(r.lbp)), test, REML=F)

#  print(anova(baseb,basebcbp))
  print(anova(baseb,baseocbp))
#  print(anova(baseb,baselbp))
#  print(anova(baselbp,basebcbp))
#  print(anova(baseocbp,basebcbp))
  print(baseocbp)
#  print(baselbp)
#  print(basebcbp)
  #basebcbp.p <- pvals.fnc(basebcbp)
  #print(bcbp.p)
  #basebcbp.p <- NULL
  baseocbp <- NULL
#  baselbp <- NULL
#  basebcbp <- NULL

  print("Testing Cum Dep B+")
  write("Testing Cum Dep B+",stderr())
  write("Building Base: Cum Dep B+",stderr())
  baseb <- lmer(update.formula(bformb,.~.+(1+c.(cumdbp)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)),test,REML=F)

#  write("Residing Cum Dep B+",stderr())
#  r.cdbp <- residuals(lm(update.formula(rformb,c.(cumdbp) ~ .), test))
#  r.ldbp <- residuals(lm(update.formula(lrformb,c.(lagcumdbp) ~ .), test))

  write("Testing",stderr())
#  baserocdbp <- lmer(update.formula(bformb,.~.+ c.(r.cdbp) + (1+c.(cumdbp)|subject)), test, REML=F)
  baseocdbp <- lmer(update.formula(bformb,.~.+ c.(cumdbp) + (1+c.(cumdbp)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)), test, REML=F)
#  baseldbp <- lmer(update.formula(bformb,.~.+ c.(r.ldbp)), test, REML=F)
#  basebcdbp <- lmer(update.formula(bformb,.~.+ c.(r.cdbp) + c.(r.ldbp)), test, REML=F)

#  print(anova(baseb,basebcdbp))
#  print(anova(baseb,baserocdbp))
  print(anova(baseb,baseocdbp))
#  print(anova(baseb,baseldbp))
#  print(anova(baseldbp,basebcdbp))
#  print(anova(baseocdbp,basebcdbp))
#  print(baserocdbp)
  print(baseocdbp)
#  print(baseldbp)
#  print(basebcdbp)
  #basebcdbp.p <- pvals.fnc(basebcdbp)
  #print(basebcdbp.p)
  #basebcdbp.p <- NULL
#  baserocdbp <- NULL
  baseocdbp <- NULL
#  baseldbp <- NULL
#  basebcdbp <- NULL

#  print("Testing Cum Dist B+")
#  write("Testing Cum Dist B+",stderr())
#  r.cDbp <- residuals(lm(update.formula(rformb,c.(cumDbp) ~ .), test))
#  r.lDbp <- residuals(lm(update.formula(lrformb,c.(lagcumDbp) ~ .), test))

#  baseocDbp <- lmer(update.formula(bformb,.~.+ c.(r.cDbp)), test, REML=F)
#  baselDbp <- lmer(update.formula(bformb,.~.+ c.(r.lDbp)), test, REML=F)
#  basebcDbp <- lmer(update.formula(bformb,.~.+ c.(r.cDbp) + c.(r.lDbp)), test, REML=F)

#  print(anova(baseb,basebcDbp))
#  print(anova(baseb,baseocDbp))
#  print(anova(baseb,baselDbp))
#  print(anova(baselDbp,basebcDbp))
#  print(anova(baseocDbp,basebcDbp))
#  print(baseocDbp)
#  print(baselDbp)
#  print(basebcDbp)
  #basebcDbp.p <- pvals.fnc(basebcDbp)
  #print(basebcDbp.p)
  #basebcDbp.p <- NULL
#  baseocDbp <- NULL
#  baselDbp <- NULL
#  basebcDbp <- NULL

#  print("Testing Cum BSto")
#  write("Testing Cum BSto",stderr())
#  r.cbs <- residuals(lm(update.formula(rformb,c.(cumbsto) ~ .), test))
#  r.lbs <- residuals(lm(update.formula(lrformb,c.(lagcumbsto) ~ .), test))

#  baseocbs <- lmer(update.formula(bformb,.~.+ c.(r.cbs)), test, REML=F)
#  baselbs <- lmer(update.formula(bformb,.~.+ c.(r.lbs)), test, REML=F)
#  basebcbs <- lmer(update.formula(bformb,.~.+ c.(r.cbs) + c.(r.lbs)), test, REML=F)

#  print(anova(baseb,basebcbs))
#  print(anova(baseb,baseocbs))
#  print(anova(baseb,baselbs))
#  print(anova(baselbs,basebcbs))
#  print(anova(baseocbs,basebcbs))
#  print(baseocbs)
#  print(baselbs)
#  print(basebcbs)
  #basebcbs.p <- pvals.fnc(basebcbs)
  #print(bcbs.p)
  #basebcbs.p <- NULL
#  baseocbs <- NULL
#  baselbs <- NULL
#  basebcbs <- NULL

  print("Testing Cum BCDR")
  write("Testing Cum BCDR",stderr())
  write("Building Base: Cum BCDR",stderr())
  baseb <- lmer(update.formula(bformb,.~.+(1+c.(cumbcdr)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)),test,REML=F)

#  write("Residing Cum BCDR",stderr())
#  r.cbcdr <- residuals(lm(update.formula(rformb,c.(cumbcdr) ~ .), test))
#  r.lbcdr <- residuals(lm(update.formula(lrformb,c.(lagcumbcdr) ~ .), test))

  write("Testing",stderr())
#  baseocbcdr <- lmer(update.formula(bformb,.~.+ c.(r.cbcdr) + (1+c.(cumbcdr)|subject)), test, REML=F)
  baseocbcdr <- lmer(update.formula(bformb,.~.+ c.(cumbcdr) + (1+c.(cumbcdr)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)), test, REML=F)
#  baselbcdr <- lmer(update.formula(bformb,.~.+ c.(r.lbcdr)), test, REML=F)
#  basebcbcdr <- lmer(update.formula(bformb,.~.+ c.(r.cbcdr) + c.(r.lbcdr)), test, REML=F)

#  print(anova(baseb,basebcbcdr))
  print(anova(baseb,baseocbcdr))
#  print(anova(baseb,baselbcdr))
#  print(anova(baselbcdr,basebcbcdr))
#  print(anova(baseocbcdr,basebcbcdr))
  print(baseocbcdr)
#  print(baselbcdr)
#  print(basebcbcdr)
  #basebcbcdr.p <- pvals.fnc(basebcbcdr)
  #print(bcbcdr.p)
  #basebcbcdr.p <- NULL
  baseocbcdr <- NULL
#  baselbcdr <- NULL
#  basebcbcdr <- NULL

  print("Testing Cum BNil")
  write("Testing Cum BNil",stderr())
  write("Building Base: Cum BNil",stderr())
  baseb <- lmer(update.formula(bformb,.~.+(1+c.(cumfmlmbo+cumfmlpbo+cumfplmbo+cumfplpbo)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)),test,REML=F)

#  write("Residing Cum BNil",stderr())
#  r.cbnil <- residuals(lm(update.formula(rformb,c.(cumfmlmbo+cumfmlpbo+cumfplmbo+cumfplpbo) ~ .), test))
#  r.lbnil <- residuals(lm(update.formula(lrformb,c.(lagcumfmlmbo+lagcumfmlpbo+lagcumfplmbo+lagcumfplpbo) ~ .), test))

  write("Testing",stderr())
#  baseocbnil <- lmer(update.formula(bformb,.~.+ c.(r.cbnil) + (1+c.(cumfmlmbo+cumfmlpbo+cumfplmbo+cumfplpbo)|subject)), test, REML=F)
  baseocbnil <- lmer(update.formula(bformb,.~.+ c.(cumfmlmbo+cumfmlpbo+cumfplmbo+cumfplpbo) + (1+c.(cumfmlmbo+cumfmlpbo+cumfplmbo+cumfplpbo)+c.(cumtotsurp)+c.(previsfix)+c.(cumwdelta)|subject)), test, REML=F)
#  baselbnil <- lmer(update.formula(bformb,.~.+ c.(r.lbnil)), test, REML=F)
#  basebcbnil <- lmer(update.formula(bformb,.~.+ c.(r.cbnil) + c.(r.lbnil)), test, REML=F)

#  print(anova(baseb,basebcbnil))
  print(anova(baseb,baseocbnil))
#  print(anova(baseb,baselbnil))
#  print(anova(baselbnil,basebcbnil))
#  print(anova(baseocbnil,basebcbnil))
  print(baseocbnil)
#  print(baselbnil)
#  print(basebcbnil)
  #basebcbnil.p <- pvals.fnc(basebcbnil)
  #print(bcbnil.p)
  #basebcbnil.p <- NULL
  baseocbnil <- NULL
#  baselbnil <- NULL
#  basebcbnil <- NULL

  return(0) #END-OF-THE-LINE

  print("Testing Cum B-")
  write("Testing Cum B-",stderr())
  r.cbm <- residuals(lm(update.formula(rformb,c.(cumbm) ~ .), test))
  r.lbm <- residuals(lm(update.formula(lrformb,c.(lagcumbm) ~ .), test))

  baseocbm <- lmer(update.formula(bformb,.~.+ c.(r.cbm)), test, REML=F)
  baselbm <- lmer(update.formula(bformb,.~.+ c.(r.lbm)), test, REML=F)
  basebcbm <- lmer(update.formula(bformb,.~.+ c.(r.cbm) + c.(r.lbm)), test, REML=F)

  print(anova(baseb,basebcbm))
  print(anova(baseb,baseocbm))
  print(anova(baseb,baselbm))
  print(anova(baselbm,basebcbm))
  print(anova(baseocbm,basebcbm))
  print(baseocbm)
  print(baselbm)
  print(basebcbm)
  #basebcbm.p <- pvals.fnc(basebcbm)
  #print(bcbm.p)
  #basebcbm.p <- NULL
  baseocbm <- NULL
  baselbm <- NULL
  basebcbm <- NULL

  print("Testing Dep B-")
  write("Testing Dep B-",stderr())
  print("Note that Non-Cummetrics do not include lag metrics into their baseline")
  r.dbm <- residuals(lm(update.formula(rformb,c.(dbm) ~ .), test))

  baseodbm <- lmer(update.formula(bformb,.~.+ c.(r.dbm)), test, REML=F)

  print(anova(baseb,baseodbm))
  print(baseodbm)
  #baseodbm.p <- pvals.fnc(baseodbm)
  #print(baseodbm.p)
  #baseodbm.p <- NULL
  baseodbm <- NULL

  print("Testing Cum Dep B-")
  write("Testing Cum Dep B-",stderr())
  r.cdbm <- residuals(lm(update.formula(rformb,c.(cumdbm) ~ .), test))
  r.ldbm <- residuals(lm(update.formula(lrformb,c.(lagcumdbm) ~ .), test))

  baseocdbm <- lmer(update.formula(bformb,.~.+ c.(r.cdbm)), test, REML=F)
  baseldbm <- lmer(update.formula(bformb,.~.+ c.(r.ldbm)), test, REML=F)
  basebcdbm <- lmer(update.formula(bformb,.~.+ c.(r.cdbm) + c.(r.ldbm)), test, REML=F)

  print(anova(baseb,basebcdbm))
  print(anova(baseb,baseocdbm))
  print(anova(baseb,baseldbm))
  print(anova(baseldbm,basebcdbm))
  print(anova(baseocdbm,basebcdbm))
  print(baseocdbm)
  print(baseldbm)
  print(basebcdbm)
  #basebcdbm.p <- pvals.fnc(basebcdbm)
  #print(basebcdbm.p)
  #basebcdbm.p <- NULL
  baseocdbm <- NULL
  baseldbm <- NULL
  basebcdbm <- NULL

  #reset regression forms prior to elemental tests
  bformb <- bform
  rformb <- rform
  lrformb <- lrform
  baseb <- base

  print("16 elemental measures")
  print("Testing F-L-Ba")
  write("Testing F-L-Ba",stderr())
  print("Note that Non-Cummetrics do not include lag metrics into their baseline")
  r.fmlmba <- residuals(lm(update.formula(rformb,c.(fmlmba) ~ .), test))

  baseofmlmba <- lmer(update.formula(bformb,.~.+ c.(r.fmlmba)), test, REML=F)

  print(anova(baseb,baseofmlmba))
  print(baseofmlmba)
  #baseofmlmba.p <- pvals.fnc(baseofmlmba)
  #print(baseofmlmba.p)
  #baseofmlmba.p <- NULL
  baseofmlmba <- NULL

  print("Testing Cum F-L-Ba")
  write("Testing Cum F-L-Ba",stderr())
  r.cfmlmba <- residuals(lm(update.formula(rformb,c.(cumfmlmba) ~ .), test))
  r.lfmlmba <- residuals(lm(update.formula(lrformb,c.(lagcumfmlmba) ~ .), test))

  baseocfmlmba <- lmer(update.formula(bformb,.~.+ c.(r.cfmlmba)), test, REML=F)
  baselfmlmba <- lmer(update.formula(bformb,.~.+ c.(r.lfmlmba)), test, REML=F)
  basebcfmlmba <- lmer(update.formula(bformb,.~.+ c.(r.cfmlmba) + c.(r.lfmlmba)), test, REML=F)

  print(anova(baseb,basebcfmlmba))
  print(anova(baseb,baseocfmlmba))
  print(anova(baseb,baselfmlmba))
  print(anova(baselfmlmba,basebcfmlmba))
  print(anova(baseocfmlmba,basebcfmlmba))
  print(baseocfmlmba)
  print(baselfmlmba)
  print(basebcfmlmba)
  #basebcfmlmba.p <- pvals.fnc(basebcfmlmba)
  #print(basebcfmlmba.p)
  #basebcfmlmba.p <- NULL
  baseocfmlmba <- NULL
  baselfmlmba <- NULL
  basebcfmlmba <- NULL

  print("Testing F-L-Bc")
  write("Testing F-L-Bc",stderr())
  print("Note that Non-Cummetrics do not include lag metrics into their baseline")
  r.fmlmbc <- residuals(lm(update.formula(rformb,c.(fmlmbc) ~ .), test))

  baseofmlmbc <- lmer(update.formula(bformb,.~.+ c.(r.fmlmbc)), test, REML=F)

  print(anova(baseb,baseofmlmbc))
  print(baseofmlmbc)
  #baseofmlmbc.p <- pvals.fnc(baseofmlmbc)
  #print(baseofmlmbc.p)
  #baseofmlmbc.p <- NULL
  baseofmlmbc <- NULL

  print("Testing Cum F-L-Bc")
  write("Testing Cum F-L-Bc",stderr())
  r.cfmlmbc <- residuals(lm(update.formula(rformb,c.(cumfmlmbc) ~ .), test))
  r.lfmlmbc <- residuals(lm(update.formula(lrformb,c.(lagcumfmlmbc) ~ .), test))

  baseocfmlmbc <- lmer(update.formula(bformb,.~.+ c.(r.cfmlmbc)), test, REML=F)
  baselfmlmbc <- lmer(update.formula(bformb,.~.+ c.(r.lfmlmbc)), test, REML=F)
  basebcfmlmbc <- lmer(update.formula(bformb,.~.+ c.(r.cfmlmbc) + c.(r.lfmlmbc)), test, REML=F)

  print(anova(baseb,basebcfmlmbc))
  print(anova(baseb,baseocfmlmbc))
  print(anova(baseb,baselfmlmbc))
  print(anova(baselfmlmbc,basebcfmlmbc))
  print(anova(baseocfmlmbc,basebcfmlmbc))
  print(baseocfmlmbc)
  print(baselfmlmbc)
  print(basebcfmlmbc)
  #basebcfmlmbc.p <- pvals.fnc(basebcfmlmbc)
  #print(basebcfmlmbc.p)
  #basebcfmlmbc.p <- NULL
  baseocfmlmbc <- NULL
  baselfmlmbc <- NULL
  basebcfmlmbc <- NULL

  print("Testing F-L-Bo")
  write("Testing F-L-Bo",stderr())
  print("Note that Non-Cummetrics do not include lag metrics into their baseline")
  r.fmlmbo <- residuals(lm(update.formula(rformb,c.(fmlmbo) ~ .), test))

  baseofmlmbo <- lmer(update.formula(bformb,.~.+ c.(r.fmlmbo)), test, REML=F)

  print(anova(baseb,baseofmlmbo))
  print(baseofmlmbo)
  #baseofmlmbo.p <- pvals.fnc(baseofmlmbo)
  #print(baseofmlmbo.p)
  #baseofmlmbo.p <- NULL
  baseofmlmbo <- NULL

  print("Testing Cum F-L-Bo")
  write("Testing Cum F-L-Bo",stderr())
  r.cfmlmbo <- residuals(lm(update.formula(rformb,c.(cumfmlmbo) ~ .), test))
  r.lfmlmbo <- residuals(lm(update.formula(lrformb,c.(lagcumfmlmbo) ~ .), test))

  baseocfmlmbo <- lmer(update.formula(bformb,.~.+ c.(r.cfmlmbo)), test, REML=F)
  baselfmlmbo <- lmer(update.formula(bformb,.~.+ c.(r.lfmlmbo)), test, REML=F)
  basebcfmlmbo <- lmer(update.formula(bformb,.~.+ c.(r.cfmlmbo) + c.(r.lfmlmbo)), test, REML=F)

  print(anova(baseb,basebcfmlmbo))
  print(anova(baseb,baseocfmlmbo))
  print(anova(baseb,baselfmlmbo))
  print(anova(baselfmlmbo,basebcfmlmbo))
  print(anova(baseocfmlmbo,basebcfmlmbo))
  print(baseocfmlmbo)
  print(baselfmlmbo)
  print(basebcfmlmbo)
  #basebcfmlmbo.p <- pvals.fnc(basebcfmlmbo)
  #print(basebcfmlmbo.p)
  #basebcfmlmbo.p <- NULL
  baseocfmlmbo <- NULL
  baselfmlmbo <- NULL
  basebcfmlmbo <- NULL

  print("Testing F-L-B+")
  write("Testing F-L-B+",stderr())
  print("Note that Non-Cummetrics do not include lag metrics into their baseline")
  r.fmlmbp <- residuals(lm(update.formula(rformb,c.(fmlmbp) ~ .), test))

  baseofmlmbp <- lmer(update.formula(bformb,.~.+ c.(r.fmlmbp)), test, REML=F)

  print(anova(baseb,baseofmlmbp))
  print(baseofmlmbp)
  #baseofmlmbp.p <- pvals.fnc(baseofmlmbp)
  #print(baseofmlmbp.p)
  #baseofmlmbp.p <- NULL
  baseofmlmbp <- NULL

  print("Testing Cum F-L-B+")
  write("Testing Cum F-L-B+",stderr())
  r.cfmlmbp <- residuals(lm(update.formula(rformb,c.(cumfmlmbp) ~ .), test))
  r.lfmlmbp <- residuals(lm(update.formula(lrformb,c.(lagcumfmlmbp) ~ .), test))

  baseocfmlmbp <- lmer(update.formula(bformb,.~.+ c.(r.cfmlmbp)), test, REML=F)
  baselfmlmbp <- lmer(update.formula(bformb,.~.+ c.(r.lfmlmbp)), test, REML=F)
  basebcfmlmbp <- lmer(update.formula(bformb,.~.+ c.(r.cfmlmbp) + c.(r.lfmlmbp)), test, REML=F)

  print(anova(baseb,basebcfmlmbp))
  print(anova(baseb,baseocfmlmbp))
  print(anova(baseb,baselfmlmbp))
  print(anova(baselfmlmbp,basebcfmlmbp))
  print(anova(baseocfmlmbp,basebcfmlmbp))
  print(baseocfmlmbp)
  print(baselfmlmbp)
  print(basebcfmlmbp)
  #basebcfmlmbp.p <- pvals.fnc(basebcfmlmbp)
  #print(basebcfmlmbp.p)
  #basebcfmlmbp.p <- NULL
  baseocfmlmbp <- NULL
  baselfmlmbp <- NULL
  basebcfmlmbp <- NULL

  print("Testing F-L+Ba")
  write("Testing F-L+Ba",stderr())
  print("Note that Non-Cummetrics do not include lag metrics into their baseline")
  r.fmlpba <- residuals(lm(update.formula(rformb,c.(fmlpba) ~ .), test))

  baseofmlpba <- lmer(update.formula(bformb,.~.+ c.(r.fmlpba)), test, REML=F)

  print(anova(baseb,baseofmlpba))
  print(baseofmlpba)
  #baseofmlpba.p <- pvals.fnc(baseofmlpba)
  #print(baseofmlpba.p)
  #baseofmlpba.p <- NULL
  baseofmlpba <- NULL

  print("Testing Cum F-L+Ba")
  write("Testing Cum F-L+Ba",stderr())
  r.cfmlpba <- residuals(lm(update.formula(rformb,c.(cumfmlpba) ~ .), test))
  r.lfmlpba <- residuals(lm(update.formula(lrformb,c.(lagcumfmlpba) ~ .), test))

  baseocfmlpba <- lmer(update.formula(bformb,.~.+ c.(r.cfmlpba)), test, REML=F)
  baselfmlpba <- lmer(update.formula(bformb,.~.+ c.(r.lfmlpba)), test, REML=F)
  basebcfmlpba <- lmer(update.formula(bformb,.~.+ c.(r.cfmlpba) + c.(r.lfmlpba)), test, REML=F)

  print(anova(baseb,basebcfmlpba))
  print(anova(baseb,baseocfmlpba))
  print(anova(baseb,baselfmlpba))
  print(anova(baselfmlpba,basebcfmlpba))
  print(anova(baseocfmlpba,basebcfmlpba))
  print(baseocfmlpba)
  print(baselfmlpba)
  print(basebcfmlpba)
  #basebcfmlpba.p <- pvals.fnc(basebcfmlpba)
  #print(basebcfmlpba.p)
  #basebcfmlpba.p <- NULL
  baseocfmlpba <- NULL
  baselfmlpba <- NULL
  basebcfmlpba <- NULL

  print("Testing F-L+Bc")
  write("Testing F-L+Bc",stderr())
  print("Note that Non-Cummetrics do not include lag metrics into their baseline")
  r.fmlpbc <- residuals(lm(update.formula(rformb,c.(fmlpbc) ~ .), test))

  baseofmlpbc <- lmer(update.formula(bformb,.~.+ c.(r.fmlpbc)), test, REML=F)

  print(anova(baseb,baseofmlpbc))
  print(baseofmlpbc)
  #baseofmlpbc.p <- pvals.fnc(baseofmlpbc)
  #print(baseofmlpbc.p)
  #baseofmlpbc.p <- NULL
  baseofmlpbc <- NULL

  print("Testing Cum F-L+Bc")
  write("Testing Cum F-L+Bc",stderr())
  r.cfmlpbc <- residuals(lm(update.formula(rformb,c.(cumfmlpbc) ~ .), test))
  r.lfmlpbc <- residuals(lm(update.formula(lrformb,c.(lagcumfmlpbc) ~ .), test))

  baseocfmlpbc <- lmer(update.formula(bformb,.~.+ c.(r.cfmlpbc)), test, REML=F)
  baselfmlpbc <- lmer(update.formula(bformb,.~.+ c.(r.lfmlpbc)), test, REML=F)
  basebcfmlpbc <- lmer(update.formula(bformb,.~.+ c.(r.cfmlpbc) + c.(r.lfmlpbc)), test, REML=F)

  print(anova(baseb,basebcfmlpbc))
  print(anova(baseb,baseocfmlpbc))
  print(anova(baseb,baselfmlpbc))
  print(anova(baselfmlpbc,basebcfmlpbc))
  print(anova(baseocfmlpbc,basebcfmlpbc))
  print(baseocfmlpbc)
  print(baselfmlpbc)
  print(basebcfmlpbc)
  #basebcfmlpbc.p <- pvals.fnc(basebcfmlpbc)
  #print(basebcfmlpbc.p)
  #basebcfmlpbc.p <- NULL
  baseocfmlpbc <- NULL
  baselfmlpbc <- NULL
  basebcfmlpbc <- NULL

  print("Testing F-L+Bo")
  write("Testing F-L+Bo",stderr())
  print("Note that Non-Cummetrics do not include lag metrics into their baseline")
  r.fmlpbo <- residuals(lm(update.formula(rformb,c.(fmlpbo) ~ .), test))

  baseofmlpbo <- lmer(update.formula(bformb,.~.+ c.(r.fmlpbo)), test, REML=F)

  print(anova(baseb,baseofmlpbo))
  print(baseofmlpbo)
  #baseofmlpbo.p <- pvals.fnc(baseofmlpbo)
  #print(baseofmlpbo.p)
  #baseofmlpbo.p <- NULL
  baseofmlpbo <- NULL

  print("Testing Cum F-L+Bo")
  write("Testing Cum F-L+Bo",stderr())
  r.cfmlpbo <- residuals(lm(update.formula(rformb,c.(cumfmlpbo) ~ .), test))
  r.lfmlpbo <- residuals(lm(update.formula(lrformb,c.(lagcumfmlpbo) ~ .), test))

  baseocfmlpbo <- lmer(update.formula(bformb,.~.+ c.(r.cfmlpbo)), test, REML=F)
  baselfmlpbo <- lmer(update.formula(bformb,.~.+ c.(r.lfmlpbo)), test, REML=F)
  basebcfmlpbo <- lmer(update.formula(bformb,.~.+ c.(r.cfmlpbo) + c.(r.lfmlpbo)), test, REML=F)

  print(anova(baseb,basebcfmlpbo))
  print(anova(baseb,baseocfmlpbo))
  print(anova(baseb,baselfmlpbo))
  print(anova(baselfmlpbo,basebcfmlpbo))
  print(anova(baseocfmlpbo,basebcfmlpbo))
  print(baseocfmlpbo)
  print(baselfmlpbo)
  print(basebcfmlpbo)
  #basebcfmlpbo.p <- pvals.fnc(basebcfmlpbo)
  #print(basebcfmlpbo.p)
  #basebcfmlpbo.p <- NULL
  baseocfmlpbo <- NULL
  baselfmlpbo <- NULL
  basebcfmlpbo <- NULL

  print("Testing Dep F-L+B+")
  write("Testing Dep F-L+B+",stderr())
  print("Note that Non-Cummetrics do not include lag metrics into their baseline")
  r.fmlpbp <- residuals(lm(update.formula(rformb,c.(fmlpbp) ~ .), test))

  baseofmlpbp <- lmer(update.formula(bformb,.~.+ c.(r.fmlpbp)), test, REML=F)

  print(anova(baseb,baseofmlpbp))
  print(baseofmlpbp)
  #baseofmlpbp.p <- pvals.fnc(baseofmlpbp)
  #print(baseofmlpbp.p)
  #baseofmlpbp.p <- NULL
  baseofmlpbp <- NULL

  print("Testing Cum F-L+B+")
  write("Testing Cum F-L+B+",stderr())
  r.cfmlpbp <- residuals(lm(update.formula(rformb,c.(cumfmlpbp) ~ .), test))
  r.lfmlpbp <- residuals(lm(update.formula(lrformb,c.(lagcumfmlpbp) ~ .), test))

  baseocfmlpbp <- lmer(update.formula(bformb,.~.+ c.(r.cfmlpbp)), test, REML=F)
  baselfmlpbp <- lmer(update.formula(bformb,.~.+ c.(r.lfmlpbp)), test, REML=F)
  basebcfmlpbp <- lmer(update.formula(bformb,.~.+ c.(r.cfmlpbp) + c.(r.lfmlpbp)), test, REML=F)

  print(anova(baseb,basebcfmlpbp))
  print(anova(baseb,baseocfmlpbp))
  print(anova(baseb,baselfmlpbp))
  print(anova(baselfmlpbp,basebcfmlpbp))
  print(anova(baseocfmlpbp,basebcfmlpbp))
  print(baseocfmlpbp)
  print(baselfmlpbp)
  print(basebcfmlpbp)
  #basebcfmlpbp.p <- pvals.fnc(basebcfmlpbp)
  #print(basebcfmlpbp.p)
  #basebcfmlpbp.p <- NULL
  baseocfmlpbp <- NULL
  baselfmlpbp <- NULL
  basebcfmlpbp <- NULL

  print("Testing F+L-Ba")
  write("Testing F+L-Ba",stderr())
  print("Note that Non-Cummetrics do not include lag metrics into their baseline")
  r.fplmba <- residuals(lm(update.formula(rformb,c.(fplmba) ~ .), test))

  baseofplmba <- lmer(update.formula(bformb,.~.+ c.(r.fplmba)), test, REML=F)

  print(anova(baseb,baseofplmba))
  print(baseofplmba)
  #baseofplmba.p <- pvals.fnc(baseofplmba)
  #print(baseofplmba.p)
  #baseofplmba.p <- NULL
  baseofplmba <- NULL

  print("Testing Cum F+L-Ba")
  write("Testing Cum F+L-Ba",stderr())
  r.cfplmba <- residuals(lm(update.formula(rformb,c.(cumfplmba) ~ .), test))
  r.lfplmba <- residuals(lm(update.formula(lrformb,c.(lagcumfplmba) ~ .), test))

  baseocfplmba <- lmer(update.formula(bformb,.~.+ c.(r.cfplmba)), test, REML=F)
  baselfplmba <- lmer(update.formula(bformb,.~.+ c.(r.lfplmba)), test, REML=F)
  basebcfplmba <- lmer(update.formula(bformb,.~.+ c.(r.cfplmba) + c.(r.lfplmba)), test, REML=F)

  print(anova(baseb,basebcfplmba))
  print(anova(baseb,baseocfplmba))
  print(anova(baseb,baselfplmba))
  print(anova(baselfplmba,basebcfplmba))
  print(anova(baseocfplmba,basebcfplmba))
  print(baseocfplmba)
  print(baselfplmba)
  print(basebcfplmba)
  #basebcfplmba.p <- pvals.fnc(basebcfplmba)
  #print(basebcfplmba.p)
  #basebcfplmba.p <- NULL
  baseocfplmba <- NULL
  baselfplmba <- NULL
  basebcfplmba <- NULL

  print("Testing F+L-Bc")
  write("Testing F+L-Bc",stderr())
  print("Note that Non-Cummetrics do not include lag metrics into their baseline")
  r.fplmbc <- residuals(lm(update.formula(rformb,c.(fplmbc) ~ .), test))

  baseofplmbc <- lmer(update.formula(bformb,.~.+ c.(r.fplmbc)), test, REML=F)

  print(anova(baseb,baseofplmbc))
  print(baseofplmbc)
  #baseofplmbc.p <- pvals.fnc(baseofplmbc)
  #print(baseofplmbc.p)
  #baseofplmbc.p <- NULL
  baseofplmbc <- NULL

  print("Testing Cum F+L-Bc")
  write("Testing Cum F+L-Bc",stderr())
  r.cfplmbc <- residuals(lm(update.formula(rformb,c.(cumfplmbc) ~ .), test))
  r.lfplmbc <- residuals(lm(update.formula(lrformb,c.(lagcumfplmbc) ~ .), test))

  baseocfplmbc <- lmer(update.formula(bformb,.~.+ c.(r.cfplmbc)), test, REML=F)
  baselfplmbc <- lmer(update.formula(bformb,.~.+ c.(r.lfplmbc)), test, REML=F)
  basebcfplmbc <- lmer(update.formula(bformb,.~.+ c.(r.cfplmbc) + c.(r.lfplmbc)), test, REML=F)

  print(anova(baseb,basebcfplmbc))
  print(anova(baseb,baseocfplmbc))
  print(anova(baseb,baselfplmbc))
  print(anova(baselfplmbc,basebcfplmbc))
  print(anova(baseocfplmbc,basebcfplmbc))
  print(baseocfplmbc)
  print(baselfplmbc)
  print(basebcfplmbc)
  #basebcfplmbc.p <- pvals.fnc(basebcfplmbc)
  #print(basebcfplmbc.p)
  #basebcfplmbc.p <- NULL
  baseocfplmbc <- NULL
  baselfplmbc <- NULL
  basebcfplmbc <- NULL

  print("Testing F+L-Bo")
  write("Testing F+L-Bo",stderr())
  print("Note that Non-Cummetrics do not include lag metrics into their baseline")
  r.fplmbo <- residuals(lm(update.formula(rformb,c.(fplmbo) ~ .), test))

  baseofplmbo <- lmer(update.formula(bformb,.~.+ c.(r.fplmbo)), test, REML=F)

  print(anova(baseb,baseofplmbo))
  print(baseofplmbo)
  #baseofplmbo.p <- pvals.fnc(baseofplmbo)
  #print(baseofplmbo.p)
  #baseofplmbo.p <- NULL
  baseofplmbo <- NULL

  print("Testing Cum F+L-Bo")
  write("Testing Cum F+L-Bo",stderr())
  r.cfplmbo <- residuals(lm(update.formula(rformb,c.(cumfplmbo) ~ .), test))
  r.lfplmbo <- residuals(lm(update.formula(lrformb,c.(lagcumfplmbo) ~ .), test))

  baseocfplmbo <- lmer(update.formula(bformb,.~.+ c.(r.cfplmbo)), test, REML=F)
  baselfplmbo <- lmer(update.formula(bformb,.~.+ c.(r.lfplmbo)), test, REML=F)
  basebcfplmbo <- lmer(update.formula(bformb,.~.+ c.(r.cfplmbo) + c.(r.lfplmbo)), test, REML=F)

  print(anova(baseb,basebcfplmbo))
  print(anova(baseb,baseocfplmbo))
  print(anova(baseb,baselfplmbo))
  print(anova(baselfplmbo,basebcfplmbo))
  print(anova(baseocfplmbo,basebcfplmbo))
  print(baseocfplmbo)
  print(baselfplmbo)
  print(basebcfplmbo)
  #basebcfplmbo.p <- pvals.fnc(basebcfplmbo)
  #print(basebcfplmbo.p)
  #basebcfplmbo.p <- NULL
  baseocfplmbo <- NULL
  baselfplmbo <- NULL
  basebcfplmbo <- NULL

  print("Testing F+L-B+")
  write("Testing F+L-B+",stderr())
  print("Note that Non-Cummetrics do not include lag metrics into their baseline")
  r.fplmbp <- residuals(lm(update.formula(rformb,c.(fplmbp) ~ .), test))

  baseofplmbp <- lmer(update.formula(bformb,.~.+ c.(r.fplmbp)), test, REML=F)

  print(anova(baseb,baseofplmbp))
  print(baseofplmbp)
  #baseofplmbp.p <- pvals.fnc(baseofplmbp)
  #print(baseofplmbp.p)
  #baseofplmbp.p <- NULL
  baseofplmbp <- NULL

  print("Testing Cum F+L-B+")
  write("Testing Cum F+L-B+",stderr())
  r.cfplmbp <- residuals(lm(update.formula(rformb,c.(cumfplmbp) ~ .), test))
  r.lfplmbp <- residuals(lm(update.formula(lrformb,c.(lagcumfplmbp) ~ .), test))

  baseocfplmbp <- lmer(update.formula(bformb,.~.+ c.(r.cfplmbp)), test, REML=F)
  baselfplmbp <- lmer(update.formula(bformb,.~.+ c.(r.lfplmbp)), test, REML=F)
  basebcfplmbp <- lmer(update.formula(bformb,.~.+ c.(r.cfplmbp) + c.(r.lfplmbp)), test, REML=F)

  print(anova(baseb,basebcfplmbp))
  print(anova(baseb,baseocfplmbp))
  print(anova(baseb,baselfplmbp))
  print(anova(baselfplmbp,basebcfplmbp))
  print(anova(baseocfplmbp,basebcfplmbp))
  print(baseocfplmbp)
  print(baselfplmbp)
  print(basebcfplmbp)
  #basebcfplmbp.p <- pvals.fnc(basebcfplmbp)
  #print(basebcfplmbp.p)
  #basebcfplmbp.p <- NULL
  baseocfplmbp <- NULL
  baselfplmbp <- NULL
  basebcfplmbp <- NULL

  print("Testing F+L+Ba")
  write("Testing F+L+Ba",stderr())
  print("Note that Non-Cummetrics do not include lag metrics into their baseline")
  r.fplpba <- residuals(lm(update.formula(rformb,c.(fplpba) ~ .), test))

  baseofplpba <- lmer(update.formula(bformb,.~.+ c.(r.fplpba)), test, REML=F)

  print(anova(baseb,baseofplpba))
  print(baseofplpba)
  #baseofplpba.p <- pvals.fnc(baseofplpba)
  #print(baseofplpba.p)
  #baseofplpba.p <- NULL
  baseofplpba <- NULL

  print("Testing Cum F+L+Ba")
  write("Testing Cum F+L+Ba",stderr())
  r.cfplpba <- residuals(lm(update.formula(rformb,c.(cumfplpba) ~ .), test))
  r.lfplpba <- residuals(lm(update.formula(lrformb,c.(lagcumfplpba) ~ .), test))

  baseocfplpba <- lmer(update.formula(bformb,.~.+ c.(r.cfplpba)), test, REML=F)
  baselfplpba <- lmer(update.formula(bformb,.~.+ c.(r.lfplpba)), test, REML=F)
  basebcfplpba <- lmer(update.formula(bformb,.~.+ c.(r.cfplpba) + c.(r.lfplpba)), test, REML=F)

  print(anova(baseb,basebcfplpba))
  print(anova(baseb,baseocfplpba))
  print(anova(baseb,baselfplpba))
  print(anova(baselfplpba,basebcfplpba))
  print(anova(baseocfplpba,basebcfplpba))
  print(baseocfplpba)
  print(baselfplpba)
  print(basebcfplpba)
  #basebcfplpba.p <- pvals.fnc(basebcfplpba)
  #print(basebcfplpba.p)
  #basebcfplpba.p <- NULL
  baseocfplpba <- NULL
  baselfplpba <- NULL
  basebcfplpba <- NULL

  print("Testing F+L+Bc")
  write("Testing F+L+Bc",stderr())
  print("Note that Non-Cummetrics do not include lag metrics into their baseline")
  r.fplpbc <- residuals(lm(update.formula(rformb,c.(fplpbc) ~ .), test))

  baseofplpbc <- lmer(update.formula(bformb,.~.+ c.(r.fplpbc)), test, REML=F)

  print(anova(baseb,baseofplpbc))
  print(baseofplpbc)
  #baseofplpbc.p <- pvals.fnc(baseofplpbc)
  #print(baseofplpbc.p)
  #baseofplpbc.p <- NULL
  baseofplpbc <- NULL

  print("Testing Cum F+L+Bc")
  write("Testing Cum F+L+Bc",stderr())
  r.cfplpbc <- residuals(lm(update.formula(rformb,c.(cumfplpbc) ~ .), test))
  r.lfplpbc <- residuals(lm(update.formula(lrformb,c.(lagcumfplpbc) ~ .), test))

  baseocfplpbc <- lmer(update.formula(bformb,.~.+ c.(r.cfplpbc)), test, REML=F)
  baselfplpbc <- lmer(update.formula(bformb,.~.+ c.(r.lfplpbc)), test, REML=F)
  basebcfplpbc <- lmer(update.formula(bformb,.~.+ c.(r.cfplpbc) + c.(r.lfplpbc)), test, REML=F)

  print(anova(baseb,basebcfplpbc))
  print(anova(baseb,baseocfplpbc))
  print(anova(baseb,baselfplpbc))
  print(anova(baselfplpbc,basebcfplpbc))
  print(anova(baseocfplpbc,basebcfplpbc))
  print(baseocfplpbc)
  print(baselfplpbc)
  print(basebcfplpbc)
  #basebcfplpbc.p <- pvals.fnc(basebcfplpbc)
  #print(basebcfplpbc.p)
  #basebcfplpbc.p <- NULL
  baseocfplpbc <- NULL
  baselfplpbc <- NULL
  basebcfplpbc <- NULL

  print("Testing F+L+Bo")
  write("Testing F+L+Bo",stderr())
  print("Note that Non-Cummetrics do not include lag metrics into their baseline")
  r.fplpbo <- residuals(lm(update.formula(rformb,c.(fplpbo) ~ .), test))

  baseofplpbo <- lmer(update.formula(bformb,.~.+ c.(r.fplpbo)), test, REML=F)

  print(anova(baseb,baseofplpbo))
  print(baseofplpbo)
  #baseofplpbo.p <- pvals.fnc(baseofplpbo)
  #print(baseofplpbo.p)
  #baseofplpbo.p <- NULL
  baseofplpbo <- NULL

  print("Testing Cum F+L+Bo")
  write("Testing Cum F+L+Bo",stderr())
  r.cfplpbo <- residuals(lm(update.formula(rformb,c.(cumfplpbo) ~ .), test))
  r.lfplpbo <- residuals(lm(update.formula(lrformb,c.(lagcumfplpbo) ~ .), test))

  baseocfplpbo <- lmer(update.formula(bformb,.~.+ c.(r.cfplpbo)), test, REML=F)
  baselfplpbo <- lmer(update.formula(bformb,.~.+ c.(r.lfplpbo)), test, REML=F)
  basebcfplpbo <- lmer(update.formula(bformb,.~.+ c.(r.cfplpbo) + c.(r.lfplpbo)), test, REML=F)

  print(anova(baseb,basebcfplpbo))
  print(anova(baseb,baseocfplpbo))
  print(anova(baseb,baselfplpbo))
  print(anova(baselfplpbo,basebcfplpbo))
  print(anova(baseocfplpbo,basebcfplpbo))
  print(baseocfplpbo)
  print(baselfplpbo)
  print(basebcfplpbo)
  #basebcfplpbo.p <- pvals.fnc(basebcfplpbo)
  #print(basebcfplpbo.p)
  #basebcfplpbo.p <- NULL
  baseocfplpbo <- NULL
  baselfplpbo <- NULL
  basebcfplpbo <- NULL

  print("Testing F+L+B+")
  write("Testing F+L+B+",stderr())
  print("Note that Non-Cummetrics do not include lag metrics into their baseline")
  r.fplpbp <- residuals(lm(update.formula(rformb,c.(fplpbp) ~ .), test))

  baseofplpbp <- lmer(update.formula(bformb,.~.+ c.(r.fplpbp)), test, REML=F)

  print(anova(baseb,baseofplpbp))
  print(baseofplpbp)
  #baseofplpbp.p <- pvals.fnc(baseofplpbp)
  #print(baseofplpbp.p)
  #baseofplpbp.p <- NULL
  baseofplpbp <- NULL

  print("Testing Cum F+L+B+")
  write("Testing Cum F+L+B+",stderr())
  r.cfplpbp <- residuals(lm(update.formula(rformb,c.(cumfplpbp) ~ .), test))
  r.lfplpbp <- residuals(lm(update.formula(lrformb,c.(lagcumfplpbp) ~ .), test))

  baseocfplpbp <- lmer(update.formula(bformb,.~.+ c.(r.cfplpbp)), test, REML=F)
  baselfplpbp <- lmer(update.formula(bformb,.~.+ c.(r.lfplpbp)), test, REML=F)
  basebcfplpbp <- lmer(update.formula(bformb,.~.+ c.(r.cfplpbp) + c.(r.lfplpbp)), test, REML=F)

  print(anova(baseb,basebcfplpbp))
  print(anova(baseb,baseocfplpbp))
  print(anova(baseb,baselfplpbp))
  print(anova(baselfplpbp,basebcfplpbp))
  print(anova(baseocfplpbp,basebcfplpbp))
  print(baseocfplpbp)
  print(baselfplpbp)
  print(basebcfplpbp)
  #basebcfplpbp.p <- pvals.fnc(basebcfplpbp)
  #print(basebcfplpbp.p)
  #basebcfplpbp.p <- NULL
  baseocfplpbp <- NULL
  baselfplpbp <- NULL
  basebcfplpbp <- NULL

  #NB: Todo
  #collect effect of interest pvals and coefficients from MEM into a single dataframe
  #output that dataframe

  return(0) #NB: !!!?
}

run.holdout <- function(data.nooutliers) {
  results <- data.frame()
#  for (s in unique(nooutliers[,"subject"])) {
  s <- "all"
    cat("Testing ",s,"\n")
    # Create dev set
    #data.dev <- create.dev(data.nooutliers,"subject",s)
    data.dev <- data.nooutliers
    # Create test set
    #data.test <- create.test(data.nooutliers,data.dev)
    data.test <- data.nooutliers
    run.test(data.dev,data.test)
    # Run the test
    #results <- rbind(results,run.test(data.dev,data.test))
#  }
  return(results)
}

#########################
#
# Main Program
#
#########################

print('With strongest (|t| > 10) fixed effects included as random slopes-by-subject')

# Tag and remove outliers
print(paste('data dimensions: ',dim(data)))
data.nooutliers <- data

#data.nooutliers$outliers[outliers(log(data$fdur),data$subject)] <- 'Error'
#data.nooutliers$outliers[outliers(data$cumwdelta,data$subject)] <- 'Error'
data.nooutliers$outliers[data$cumwdelta > 4] <- 'Error'
data.outliers <- subset(data.nooutliers, outliers == 'Error')
data.nooutliers <- data.nooutliers[setdiff(rownames(data.nooutliers),rownames(data.outliers)),]
data.nooutliers$outliers <- NULL
print(paste('data dimensions (no outliers): ',dim(data.nooutliers)))

print(paste('maximum cumwdelta: ',max(data.nooutliers$cumwdelta)))

#remove any incomplete rows
data.nooutliers <- data.nooutliers[complete.cases(data.nooutliers),]

print(paste('data dimensions (complete.cases): ',dim(data.nooutliers)))

outcome <- run.holdout(data.nooutliers)
## outcome consists of all the pvalues for each subject for each factor of interest
## Need to print() all factors and significance for those with significance > .1
warnings()
