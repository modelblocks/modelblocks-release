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

args <- commandArgs(trailingOnly=TRUE)
cliargs <- args[-1]
data <- read.table(args[1],header=TRUE,quote='',comment.char='')
options('warn'=1) #report non-convergences, etc

library(lme4)
library(languageR)
library(optimx)
source("scripts/rtools.R")
#The below scripts cannot be distributed with Modelblocks
source("extraScripts/mer-utils.R") #obtained from https://github.com/aufrank
source("extraScripts/mtoolbox.R")  
source("extraScripts/regression-utils.R") #obtained from https://github.com/aufrank

#########################
#
# Data Columns
#
#########################

#subject word fdur wordid startoffile lineid nextisfix fileid previsfix screenid startofline endofscreen wlen olen 
# endoffile startofscreen endofline uprob bwprob fwprob endofsentence startofsentence sentpos totsurp lexsurp synsurp 
# entred embdep embdif F-L- F+L- F-L+ F+L+ d1F-L- d1F+L- d1F-L+ d1F+L+ d2F-L- d2F+L- d2F-L+ d2F+L+ d3F-L- d3F+L- d3F-L+ d3F+L+ 
# d4F-L- d4F+L- d4F-L+ d4F+L+ distF-L+ Badd B+ Bsto Bcdr B- dB+ dB- DB+ 
# F-L-Badd F-L-Bcdr F-L-BNil F-L-B+ F+L-Badd F+L-Bcdr F+L-BNil F+L-B+ F-L+Badd F-L+Bcdr F-L+BNil F-L+B+ F+L+Badd F+L+Bcdr F+L+BNil F+L+B+ parsed 
# cumfdur cumwlen cumolen cumuprob cumbwprob cumfwprob cumsentpos cumtotsurp cumlexsurp cumsynsurp 
# cumentred cumembdep cumembdif cumF-L- cumF+L- cumF-L+ cumF+L+ cumd1F-L- cumd1F+L- cumd1F-L+ cumd1F+L+ cumd2F-L- cumd2F+L- cumd2F-L+ cumd2F+L+ 
# cumd3F-L- cumd3F+L- cumd3F-L+ cumd3F+L+ cumd4F-L- cumd4F+L- cumd4F-L+ cumd4F+L+ cumdistF-L+ 
# cumBadd cumB+ cumBsto cumBcdr cumB- cumdB+ cumdB- cumDB+ cumF-L-Badd cumF-L-Bcdr cumF-L-BNil cumF-L-B+ cumF+L-Badd cumF+L-Bcdr cumF+L-BNil cumF+L-B+
# cumF-L+Badd cumF-L+Bcdr cumF-L+BNil cumF-L+B+ cumF+L+Badd cumF+L+Bcdr cumF+L+BNil cumF+L+B+  lagfdur lagwordid lagstartoffile laglineid lagnextisfix 
# lagfileid lagprevisfix lagscreenid lagstartofline lagendofscreen lagwlen lagolen lagendoffile lagstartofscreen lagendofline laguprob lagbwprob lagfwprob
# lagendofsentence lagstartofsentence lagsentpos lagtotsurp laglexsurp lagsynsurp lagentred lagembdep lagembdif lagF-L- lagF+L- lagF-L+ lagF+L+ 
# lagd1F-L- lagd1F+L- lagd1F-L+ lagd1F+L+ lagd2F-L- lagd2F+L- lagd2F-L+ lagd2F+L+ lagd3F-L- lagd3F+L- lagd3F-L+ lagd3F+L+ 
# lagd4F-L- lagd4F+L- lagd4F-L+ lagd4F+L+ lagdistF-L+ lagBadd lagB+ lagBsto lagBcdr lagB- lagdB+ lagdB- lagDB+ 
# lagF-L-Badd lagF-L-Bcdr lagF-L-BNil lagF-L-B+ lagF+L-Badd lagF+L-Bcdr lagF+L-BNil lagF+L-B+ lagF-L+Badd lagF-L+Bcdr lagF-L+BNil lagF-L+B+ lagF+L+Badd
# lagF+L+Bcdr lagF+L+BNil lagF+L+B+ lagcumfdur lagcumwlen lagcumolen 
# lagcumuprob lagcumbwprob lagcumfwprob lagcumsentpos lagcumtotsurp lagcumlexsurp lagcumsynsurp lagcumentred lagcumembdep lagcumembdif 
# lagcumF-L- lagcumF+L- lagcumF-L+ lagcumF+L+ lagcumd1F-L- lagcumd1F+L- lagcumd1F-L+ lagcumd1F+L+ 
# lagcumd2F-L- lagcumd2F+L- lagcumd2F-L+ lagcumd2F+L+ lagcumd3F-L- lagcumd3F+L- lagcumd3F-L+ lagcumd3F+L+ 
# lagcumd4F-L- lagcumd4F+L- lagcumd4F-L+ lagcumd4F+L+ lagcumdistF-L+ 
# lagcumBadd lagcumB+ lagcumBsto lagcumBcdr lagcumB- lagcumdB+ lagcumdB- lagcumDB+
# lagcumF-L-Badd lagcumF-L-Bcdr lagcumF-L-BNil lagcumF-L-B+ lagcumF+L-Badd lagcumF+L-Bcdr lagcumF+L-BNil lagcumF+L-B+ 
# lagcumF-L+Badd lagcumF-L+Bcdr lagcumF-L+BNil lagcumF-L+B+ lagcumF+L+Badd lagcumF+L+Bcdr lagcumF+L+BNil lagcumF+L+B+

#subject word sentpos wlen previsfix nextisfix locfreq bakfreq uprob prevlogprob fwprob bwprob laundist landpos \
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

#  lagsentpos lagwlen lagprevisfix lagnextisfix laglaundist laglandpos laglocfreq lagbakfreq \
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

  if (length(grep('S',cliargs,fixed=T)) > 0 ) {
    SIMPLIFY <- T
    print("Simplifying")
  }
  else SIMPLIFY <- F
  if (length(grep('A',cliargs,fixed=F)) > 0 ) {
    ALL <- T
    print("All evals")
  }
  else ALL <- F
  if (length(grep('R',cliargs,fixed=T)) > 0 ) {
    RERESID <- T
    print("Reresid")
  }
  else RERESID <- F
  if (length(grep('V',cliargs,fixed=T)) > 0 ) {
    VERBOSE <- T
    print("verbose")
  }
  else VERBOSE <- F
  if (length(grep('d',cliargs,fixed=T)) > 0 ) {
    DEPTH <- T
    print("Depth")
  }
  else DEPTH <- F
  if (length(grep('L',cliargs,fixed=T)) > 0 ) {
    LAG <- T
    print("Lag")
  }
  else LAG <- F
  if (length(grep('cmcl2013',cliargs,fixed=T)) > 0 ) {
    CMCL2013 <- T
    print("cmcl2013")
  }
  else CMCL2013 <- F
  if (length(grep('D',cliargs,fixed=T)) > 0 ) {
    DIST <- T
    print("Distance eval")
  }
  else DIST <- F
  if (length(grep('P',cliargs,fixed=T)) > 0 ) {
    PVAL <- as.integer(substr(a,regexpr('P',cliargs,fixed=T)[1]+1,regexpr('P',cliargs,fixed=T)[1]+3))/100
    print("Changing the p-value")
  }
  else PVAL <- .05
  if (length(grep('fl',cliargs,fixed=T)) > 0 ) {
    FILTERLINES <- T
    print("Filtering line boundaries")
  }
  else FILTERLINES <- F
  if (length(grep('fse',cliargs,fixed=T)) > 0 ) {
    FILTERSENTS <- T
    print("Filtering sentence boundaries")
  }
  else FILTERSENTS <- F
  if (length(grep('fsc',cliargs,fixed=T)) > 0 ) {
    FILTERSCREENS <- T
    print("Filtering screen boundaries")
  }
  else FILTERSCREENS <- F
  if (length(grep('ff',cliargs,fixed=T)) > 0 ) {
    FILTERFILES <- T
    print("Filtering file boundaries")
  }
  else FILTERFILES <- F
  if (length(grep('nogram',cliargs,fixed=T)) > 0 ) {
    NOGRAMMAR <- T
    print("Omitting grammar metrics")
  }
  else NOGRAMMAR <- F
  if (length(grep('nlminb',cliargs,fixed=T)) > 0 ) {
    OPTIM <- "nlminb"
    print("Using Optimizer: nlminb")
  }
  else {
    if (length(grep('neldermead',cliargs,fixed=T)) > 0 ) {
      OPTIM <- "neldermead"
      print("Using Optimizer: Nelder Mead")
    }
    else {
      OPTIM <- "bobyqa"
      print("Using Optimizer: bobyqa")
    }
  }

  test$removal <- test$sentpos #just create a vector to fill
  if (CMCL2013 || FILTERFILES) {
    test$removal[test$startoffile == 1] <- 'Error'
    test$removal[test$endoffile == 1] <- 'Error'
  }
  if (CMCL2013 || FILTERLINES) {
    test$removal[test$startofline == 1] <- 'Error'
    test$removal[test$endofline == 1] <- 'Error'
  }
  if (CMCL2013 || FILTERSENTS) {
    test$removal[test$startofsent == 1] <- 'Error'
    test$removal[test$endofsent == 1] <- 'Error'
  }
  if (CMCL2013 || FILTERSCREENS) {
    test$removal[test$startofscreen == 1] <- 'Error'
    test$removal[test$endofscreen == 1] <- 'Error'
  }
  test.removed <- subset(test, removal == 'Error')
  test <- test[setdiff(rownames(test),rownames(test.removed)),]
  test$removal <- NULL
  
  print(paste('testdata dimensions: ',dim(test)))
  print("Building Baseline")

  test$cumfdur <- as.numeric(as.character(test$cumfdur))
  test$sentpos <- as.integer(test$sentpos)
  test$wlen <- as.integer(test$wlen)
  test$prevwasfix <- as.integer(test$prevwasfix)
  test$uprob <- as.numeric(as.character(test$uprob))
  test$fwprob <- as.numeric(as.character(test$fwprob))
  test$bwprob <- as.numeric(as.character(test$bwprob))
  test$cumwdelta <- as.integer(as.character(test$cumwdelta))

  if (length(grep('cumtotsurp',cliargs,fixed=T)) > 0 ) {
    test$totsurp <- as.numeric(as.character(test$cumtotsurp))
    print("Using cumulative surprisal")
  }
  else test$totsurp <- as.numeric(as.character(test$totsurp))
  #punctuation <- z.(test$lpar + test$rpar)
  if (length(grep('logfdur',cliargs,fixed=T)) > 0 ) {
    clfdur <- c.(log(test$cumfdur)) #log-transformed fdur
    print("Log transforming fdur")
  }
  else clfdur <- c.(test$cumfdur) #standard fdur
  #clfdur <- c.((((test$cumfdur)^(-0.14141))-1)/-0.14141) #boxcox-derived transformed fdur
  #cuminteg <- c.(cumfmlmbp+cumfmlpbp+cumfplmbp+cumfplpbp+cumfmlpbo+cumfmlpba+cumfmlpbc)
  #cuminteg <- z.(test$cumfmlmbp+test$cumfmlpbp+test$cumfplmbp+test$cumfplpbp+test$cumfmlpbo+test$cumfmlpba+test$cumfmlpbc)

  #print(paste("clfdur: ",clfdur))

  #########################
  #
  # Run BoxCox to determine normalizing transform
  #
  #########################
  #bform <- fdur ~ (z.(sentpos) + z.(wlen) + z.(previsfix) + z.(nextisfix) + z.(uprob) + z.(fwprob) +
  #                          z.(bwprob) + z.(totsurp)+z.(cumwdelta) + z.(cumtotsurp) + z.(cumentred))^2 +
  #                          (z.(lagsentpos) + z.(lagwlen) + z.(lagprevisfix) + z.(laguprob) + z.(lagfwprob) +
  #                          z.(lagbwprob) + z.(lagtotsurp)+z.(lagcumwdelta) + z.(lagcumtotsurp) + z.(lagcumentred))^2 + (subject) + (word)
  #base <- glm(bform,data=test)
  #
  #print("BoxCox")
  #base.bc <- MASS:::boxcox(base)
  #print(base.bc$x[which.max(base.bc$y)])
  #return(0)
  ######################
  # Lambda for BoxCox = -0.14141
  ######################

  rform <- ~ (z.(sentpos) + z.(wlen) + z.(previsfix) + z.(nextisfix) + z.(uprob) + z.(fwprob) +
                            z.(bwprob) + z.(cumwdelta)+ z.(cumtotsurp))^2

  lrform <- ~ (z.(lagsentpos) + z.(lagwlen) + z.(lagprevisfix) + z.(lagnextisfix) + z.(laguprob) + z.(lagfwprob) +
                            z.(lagbwprob) + z.(lagcumwdelta)+z.(lagcumtotsurp))^2

##### FORK
#      bform <- clfdur ~ (z.(sentpos) + z.(wlen) + z.(prevwasfix) + z.(nextwillfix) + z.(uprob) + z.(fwprob) +
#                                z.(bwprob) + z.(cumwdelta))^2 +
#                                (z.(lagsentpos) + z.(lagwlen) + z.(lagprevwasfix) + z.(lagnextwillfix) + z.(laguprob) + z.(lagfwprob) +
#                                z.(lagbwprob) + z.(lagcumwdelta))^2 + (1|word)
  bform <- clfdur ~ z.(sentpos) + z.(wlen) + z.(prevwasfix) + z.(fwprob) +
                            z.(cumwdelta) + (1|word)
	

#                                (z.(lagsentpos) + z.(lagwlen) + z.(lagprevwasfix) + z.(lagnextwillfix) + z.(laguprob) + z.(lagfwprob) +
#                                z.(lagbwprob) + z.(lagcumwdelta) + z.(lagcumtotsurp))^2 + (1|word)
#      bform <- clfdur ~ (z.(sentpos) + z.(wlen) + z.(uprob) + z.(fwprob) + z.(bwprob) + z.(cumtotsurp) + z.(previsfix))^2 + (1|word)
#     bform <- clfdur ~ (z.(sentpos) + z.(wlen) + z.(uprob) + z.(fwprob) + z.(bwprob))^2 + (1|word)

  ######################
  #
  # Create a baseline for main F/L effects
  #
  ######################

#### FORK
  print("Build Base")
#  baseinteg <- lmer(update.formula(bform,.~.+(1+z.(sentpos)+z.(wlen)+z.(prevwasfix)+z.(cumwdelta)|subject)), test, REML=F)

####
# These optimizers are derivative-free and are thought to be 'better' but they have more ways to fail
####
if (OPTIM == "bobyqa"){
  print("Regressing for Base")
  write("Regressing for Base",stderr())
  baseinteg <- lmer(update.formula(bform,.~.+(1+z.(sentpos)+z.(wlen)+z.(prevwasfix)+z.(fwprob)+z.(cumwdelta) + z.(totsurp)|subject)), test, REML=F, control = lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=50000)))
  print("Regressing for Surp")
  write("Regressing for Surp",stderr())
  surpinteg <- lmer(update.formula(bform,.~.+z.(totsurp)+(1+z.(sentpos)+z.(wlen)+z.(prevwasfix)+z.(fwprob)+z.(cumwdelta) + z.(totsurp)|subject)), test, REML=F, control = lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=50000)))
  }
if (OPTIM == "neldermead"){
  print("Regressing for Base")
  write("Regressing for Base",stderr())
  baseinteg <- lmer(update.formula(bform,.~.+(1+z.(sentpos)+z.(wlen)+z.(prevwasfix)+z.(fwprob)+z.(cumwdelta) + z.(totsurp)|subject)), test, REML=F, control = lmerControl(optimizer="Nelder_Mead",optCtrl=list(maxfun=50000)))
  print("Regressing for Surp")
  write("Regressing for Surp",stderr())
  surpinteg <- lmer(update.formula(bform,.~.+z.(totsurp)+(1+z.(sentpos)+z.(wlen)+z.(prevwasfix)+z.(fwprob)+z.(cumwdelta) + z.(totsurp)|subject)), test, REML=F, control = lmerControl(optimizer="Nelder_Mead",optCtrl=list(maxfun=50000)))
   }

####
# These optimizers are approximate-derivative-based; if the above fail, you can try these (but they can get stuck in local minima easier)
####
if (OPTIM == "nlminb"){
  print("Regressing for Base")
  write("Regressing for Base",stderr())
  baseinteg <- lmer(update.formula(bform,.~.+(1+z.(sentpos)+z.(wlen)+z.(prevwasfix)+z.(fwprob)+z.(cumwdelta) + z.(totsurp)|subject)), test, REML=F, control = lmerControl(optimizer="optimx",optCtrl=list(method=c("nlminb"),maxit=50000)))
  print("Regressing for Surp")
  write("Regressing for Surp",stderr())
  surpinteg <- lmer(update.formula(bform,.~.+z.(totsurp)+(1+z.(sentpos)+z.(wlen)+z.(prevwasfix)+z.(fwprob)+z.(cumwdelta) + z.(totsurp)|subject)), test, REML=F, control = lmerControl(optimizer="optimx",optCtrl=list(method=c("nlminb"),maxit=50000)))
    }
#    baseinteg <- lmer(update.formula(bform,.~.+(1+z.(sentpos)+z.(wlen)+z.(prevwasfix)+z.(uprob)+z.(cumwdelta) + z.(cumtotsurp)|subject)), test, REML=F, control = lmerControl(optimizer="optimx",optCtrl=list(method=c("L-BFGS-B"),maxit=50000)))

#     (((               z.(uprob)+z.(fwprob)+z.(bwprob)|subject)), test, REML=F)
#  baseinteg <- lmer(update.formula(bform,.~.+z.(cuminteg)+(1|subject)), test, REML=F)
  print("Base Summary")
  print(summary(baseinteg))
  print("Base Log Likelihood")
  print(logLik(baseinteg))
  relgrad <- with(baseinteg@optinfo$derivs,solve(Hessian,gradient))
  print("Relative Gradient (<0.001?)") #check for convergence even if warned that convergence failed
  print(max(abs(relgrad)))
  print("Base AIC")
  print(AIC(logLik(baseinteg)))

  print("Surp Summary")
  print(summary(surpinteg))
  print("Surp Log Likelihood")
  print(logLik(surpinteg))
  relgrad <- with(surpinteg@optinfo$derivs,solve(Hessian,gradient))
  print("Relative Gradient (<0.001?)") #check for convergence even if warned that convergence failed
  print(max(abs(relgrad)))
  print("Surp AIC")
  print(AIC(logLik(surpinteg)))

  print("ANOVA")
  print(anova(baseinteg,surpinteg))

#  baseinteg <- lmer(update.formula(bform,.~.+(1+z.(sentpos)+z.(wlen)+z.(prevwasfix)+z.(nextwillfix)+z.(cumwdelta)+z.(cumtotsurp)|subject)), test, REML=F)
#  cuminteg <- lmer(update.formula(cbform,.~.+(1+z.(sentpos)+z.(wlen)+z.(prevwasfix)+z.(nextwillfix)+z.(cumwdelta)+z.(cumtotsurp)|subject)), test, REML=F)
#  base <- lmer(update.formula(bform,.~.+(1+z.(cuminteg)+z.(sentpos)+z.(wlen)+z.(previsfix)+z.(cumtotsurp)|subject)), test, REML=F) #CUNY 2013
#  base <- lmer(update.formula(bform,.~.+(1+z.(cuminteg)+z.(sentpos)+z.(wlen)+z.(bwprob)|subject)), test, REML=F)
#  base <- lmer(update.formula(bform,.~.+(1|subject)), test, REML=F)
# print(baseinteg)
#  print(anova(baseinteg,cuminteg)) #CUNY 2013

  return(0)
}

run.holdout <- function(data.nooutliers) {
  results <- data.frame()
#  for (s in unique(nooutliers[,"subject"])) {
  s <- "all"
#    cat("Testing ",s,"\n")
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
data.nooutliers$previsfix <- data.nooutliers$prevwasfix
data.nooutliers$nextisfix <- data.nooutliers$nextwasfix

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
