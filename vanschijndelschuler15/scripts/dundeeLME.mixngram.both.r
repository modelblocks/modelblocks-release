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
    OPTIM <- "bobyqa"
    print("Using Optimizer: bobyqa")
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
  test$cumfwprob <- as.numeric(as.character(test$cumfwprob))
  test$bwprob <- as.numeric(as.character(test$bwprob))
  test$cumwdelta <- as.integer(as.character(test$cumwdelta))

  if (length(grep('cumtotsurp',cliargs,fixed=T)) > 0 ) {
    test$gcgtotsurp <- as.numeric(as.character(test$gcgcumtotsurp))
    test$ptbtotsurp <- as.numeric(as.character(test$ptbcumtotsurp))
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


##### FORK
#      bform <- clfdur ~ (z.(sentpos) + z.(wlen) + z.(prevwasfix) + z.(nextwillfix) + z.(uprob) + z.(fwprob) +
#                                z.(bwprob) + z.(cumwdelta))^2 +
#                                (z.(lagsentpos) + z.(lagwlen) + z.(lagprevwasfix) + z.(lagnextwillfix) + z.(laguprob) + z.(lagfwprob) +
#                                z.(lagbwprob) + z.(lagcumwdelta))^2 + (1|word)

             bform <- clfdur ~ z.(sentpos) + z.(wlen) + z.(prevwasfix) + z.(fwprob) + z.(cumfwprob) +
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
    print("Regressing Base")
    write("Regressing Base",stderr())
    baseinteg <- lmer(update.formula(bform,.~.+(1+z.(sentpos)+z.(wlen)+z.(prevwasfix)+z.(fwprob)+z.(cumfwprob)+z.(cumwdelta) + z.(gcgtotsurp) + z.(ptbtotsurp) |subject)), test, REML=F, control = lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=50000)))
    print("Regressing for GCG")
    write("Regressing for GCG",stderr())
    gcginteg <- lmer(update.formula(bform,.~.+z.(gcgtotsurp) + (1+z.(sentpos)+z.(wlen)+z.(prevwasfix)+z.(fwprob)+z.(cumfwprob)+z.(cumwdelta) + z.(gcgtotsurp) + z.(ptbtotsurp) |subject)), test, REML=F, control = lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=50000)))
    print("Regressing for PTB")
    write("Regressing for PTB",stderr())
    ptbinteg <- lmer(update.formula(bform,.~.+z.(ptbtotsurp) + (1+z.(sentpos)+z.(wlen)+z.(prevwasfix)+z.(fwprob)+z.(cumfwprob)+z.(cumwdelta) + z.(gcgtotsurp) + z.(ptbtotsurp)|subject)), test, REML=F, control = lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=50000)))
    print("Regressing Both")
    write("Regressing Both",stderr())
    bothinteg <- lmer(update.formula(bform,.~.+z.(gcgtotsurp) + z.(ptbtotsurp) + (1+z.(sentpos)+z.(wlen)+z.(prevwasfix)+z.(fwprob)+z.(cumfwprob)+z.(cumwdelta) + z.(gcgtotsurp) + z.(ptbtotsurp) |subject)), test, REML=F, control = lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=50000)))

}
#    baseinteg <- lmer(update.formula(bform,.~.+(1+z.(sentpos)+z.(wlen)+z.(prevwasfix)+z.(fwprob)+z.(cumfwprob)+z.(cumwdelta) + z.(cumtotsurp)|subject)), test, REML=F, control = lmerControl(optimizer="Nelder_Mead",optCtrl=list(maxfun=50000)))

####
# These optimizers are approximate-derivative-based; if the above fail, you can try these (but they can get stuck in local minima easier)
####
if (OPTIM == "nlminb"){
     print("Regressing Base")
     write("Regressing Base",stderr())
     baseinteg <- lmer(update.formula(bform,.~.+(1+z.(sentpos)+z.(wlen)+z.(prevwasfix)+z.(fwprob)+z.(cumfwprob)+z.(cumwdelta) + z.(gcgtotsurp) + z.(ptbtotsurp) |subject)), test, REML=F, control = lmerControl(optimizer="optimx",optCtrl=list(method=c("nlminb"),maxit=50000)))
     print("Regressing GCG")
     write("Regressing GCG",stderr())
     gcginteg <- lmer(update.formula(bform,.~.+z.(gcgtotsurp) + (1+z.(sentpos)+z.(wlen)+z.(prevwasfix)+z.(fwprob)+z.(cumfwprob)+z.(cumwdelta) + z.(gcgtotsurp) + z.(ptbtotsurp) |subject)), test, REML=F, control = lmerControl(optimizer="optimx",optCtrl=list(method=c("nlminb"),maxit=50000)))
     print("Regressing PTB")
     write("Regressing PTB",stderr())
     ptbinteg <- lmer(update.formula(bform,.~.+z.(ptbtotsurp) + (1+z.(sentpos)+z.(wlen)+z.(prevwasfix)+z.(fwprob)+z.(cumfwprob)+z.(cumwdelta) + z.(gcgtotsurp) + z.(ptbtotsurp) |subject)), test, REML=F, control = lmerControl(optimizer="optimx",optCtrl=list(method=c("nlminb"),maxit=50000)))
     print("Regressing Both")
     write("Regressing Both",stderr())
     bothinteg <- lmer(update.formula(bform,.~.+z.(gcgtotsurp) + z.(ptbtotsurp) + (1+z.(sentpos)+z.(wlen)+z.(prevwasfix)+z.(fwprob)+z.(cumfwprob)+z.(cumwdelta) + z.(gcgtotsurp) + z.(ptbtotsurp) |subject)), test, REML=F, control = lmerControl(optimizer="optimx",optCtrl=list(method=c("nlminb"),maxit=50000)))
  }
#    baseinteg <- lmer(update.formula(bform,.~.+(1+z.(sentpos)+z.(wlen)+z.(prevwasfix)+z.(fwprob)+z.(cumwdelta) + z.(cumtotsurp)|subject)), test, REML=F, control = lmerControl(optimizer="optimx",optCtrl=list(method=c("L-BFGS-B"),maxit=50000)))

#     (((               z.(fwprob)+z.(fwprob)+z.(bwprob)|subject)), test, REML=F)
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
    
    print("PTB Summary")
    print(summary(ptbinteg))
    print("PTB Log Likelihood")
    print(logLik(ptbinteg))
    relgrad <- with(ptbinteg@optinfo$derivs,solve(Hessian,gradient))
    print("Relative Gradient (<0.001?)") #check for convergence even if warned that convergence failed
    print(max(abs(relgrad)))
    print("PTB AIC")
    print(AIC(logLik(ptbinteg)))

    print("GCG Summary")
    print(summary(gcginteg))
    print("GCG Log Likelihood")
    print(logLik(gcginteg))
    relgrad <- with(gcginteg@optinfo$derivs,solve(Hessian,gradient))
    print("Relative Gradient (<0.001?)") #check for convergence even if warned that convergence failed
    print(max(abs(relgrad)))
    print("GCG AIC")
    print(AIC(logLik(gcginteg)))

    print("Both Summary")
    print(summary(bothinteg))
    print("Both Log Likelihood")
    print(logLik(bothinteg))
    relgrad <- with(bothinteg@optinfo$derivs,solve(Hessian,gradient))
    print("Relative Gradient (<0.001?)") #check for convergence even if warned that convergence failed
    print(max(abs(relgrad)))
    print("Both AIC")
    print(AIC(logLik(bothinteg)))

    print("Base vs PTB")
    print(anova(baseinteg,ptbinteg))
    print("Base vs GCG")
    print(anova(baseinteg,gcginteg))
    print("PTB vs Both")
    print(anova(ptbinteg,bothinteg))
    print("GCG vs Both")
    print(anova(gcginteg,bothinteg))

#  baseinteg <- lmer(update.formula(bform,.~.+(1+z.(sentpos)+z.(wlen)+z.(prevwasfix)+z.(nextwillfix)+z.(cumwdelta)+z.(cumtotsurp)|subject)), test, REML=F)
#  cuminteg <- lmer(update.formula(cbform,.~.+(1+z.(sentpos)+z.(wlen)+z.(prevwasfix)+z.(nextwillfix)+z.(cumwdelta)+z.(cumtotsurp)|subject)), test, REML=F)
#  base <- lmer(update.formula(bform,.~.+(1+z.(cuminteg)+z.(sentpos)+z.(wlen)+z.(previsfix)+z.(cumtotsurp)|subject)), test, REML=F) #CUNY 2013
#  base <- lmer(update.formula(bform,.~.+(1+z.(cuminteg)+z.(sentpos)+z.(wlen)+z.(bwprob)|subject)), test, REML=F)
#  base <- lmer(update.formula(bform,.~.+(1|subject)), test, REML=F)
# print(baseinteg)
#  print(anova(baseinteg,cuminteg)) #CUNY 2013

  return(0) #don't do the rest of the eval

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
    base <- lmer(update.formula(bform,.~.+(1+z.(cumfmlm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)

    print("Testing F-L-")
#    write("Residing F-L-",stderr())
#    r.fmlm <- residuals(lm(update.formula(rform,z.(cumfmlm) ~ .), test))
#    r.lfmlm <- residuals(lm(update.formula(lrform,z.(lagcumfmlm) ~ .), test))

    write("Testing F-L-",stderr())
#    baseofmlm <- lmer(update.formula(bform,.~.+ z.(r.fmlm) + (1+z.(cumfmlm)|subject)), test, REML=F)
    baseofmlm <- lmer(update.formula(bform,.~.+ z.(cumfmlm) + (1+z.(cumfmlm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#    baselfmlm <- lmer(update.formula(bform,.~.+ z.(r.lfmlm)), test, REML=F)
#    basebfmlm <- lmer(update.formula(bform,.~.+ z.(r.fmlm) + z.(r.lfmlm)), test, REML=F)

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
#      rform <- update.formula(rforma,. ~ . + z.(r.fmlm))
#      lrform <- update.formula(lrforma,. ~ . + z.(r.lfmlm))
#      bform <- update.formula(bforma,. ~ . + z.(r.fmlm) + z.(r.lfmlm))
      base <- lmer(update.formula(bform,.~. +(1+z.(cumd1fmlm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)

#      r.d1fmlm <- residuals(lm(update.formula(rform,z.(cumd1fmlm) ~ .), test))
#      r.ld1fmlm <- residuals(lm(update.formula(lrform,z.(lagcumd1fmlm) ~ .), test))

      write("Testing",stderr())
      baseod1fmlm <- lmer(update.formula(bform,.~.+ z.(cumd1fmlm)+(1+z.(cumd1fmlm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#      baseld1fmlm <- lmer(update.formula(bform,.~.+ z.(cumld1fmlm)+(lagcumd1fmlm|subject)), test, REML=F)
#      baseod1fmlm <- lmer(update.formula(bform,.~.+ z.(r.d1fmlm)+(cumd1fmlm|subject)), test, REML=F)
#      baseld1fmlm <- lmer(update.formula(bform,.~.+ z.(r.ld1fmlm)+(lagcumd1fmlm|subject)), test, REML=F)
#      basebd1fmlm <- lmer(update.formula(bform,.~.+ z.(r.d1fmlm) + z.(r.ld1fmlm)), test, REML=F)

      print(vif.mer(baseod1fmlm))
#      print(anova(base,basebd1fmlm))
      print(anova(base,baseod1fmlm))
#      print(anova(base,baseld1fmlm))
      print(baseod1fmlm)
#      print(baseld1fmlm)
#      print(basebd1fmlm)
      #basebd1fmlm.p <- pvals.fnc(basebd1fmlm)
      #print(basebd1fmlm.p)
      #basebd1fmlm.p <- NULL
      baseod1fmlm <- NULL
#      baseld1fmlm <- NULL
#      basebd1fmlm <- NULL
      #baseofmlm <- NULL
      #baselfmlm <- NULL
      #basebfmlm <- NULL
      rform <- rforma
      lrform <- lrforma
      bform <- bforma

      print("Testing Dep2F-L-")
      write("Testing Dep2F-L-",stderr())
#      rform <- update.formula(rforma,. ~ . + z.(r.fmlm))
#      lrform <- update.formula(lrforma,. ~ . + z.(r.lfmlm))
#      bform <- update.formula(bforma,. ~ . + z.(r.fmlm) + z.(r.lfmlm))
      base <- lmer(update.formula(bform,.~. +(1+z.(cumd2fmlm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)

#      r.d2fmlm <- residuals(lm(update.formula(rform,z.(cumd2fmlm) ~ .), test))
#      r.ld2fmlm <- residuals(lm(update.formula(lrform,z.(lagcumd2fmlm) ~ .), test))

      write("Testing",stderr())
      baseod2fmlm <- lmer(update.formula(bform,.~.+ z.(cumd2fmlm)+(1+z.(cumd2fmlm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#      baseod2fmlm <- lmer(update.formula(bform,.~.+ z.(r.d2fmlm)+(cumd2fmlm|subject)), test, REML=F)
#      baseld2fmlm <- lmer(update.formula(bform,.~.+ z.(r.ld2fmlm)+(lagcumd2fmlm|subject)), test, REML=F)
#      basebd2fmlm <- lmer(update.formula(bform,.~.+ z.(r.d2fmlm) + z.(r.ld2fmlm)), test, REML=F)

      print(vif.mer(baseod2fmlm))
#      print(anova(base,basebd2fmlm))
      print(anova(base,baseod2fmlm))
#      print(anova(base,baseld2fmlm))
      print(baseod2fmlm)
#      print(baseld2fmlm)
#      print(basebd2fmlm)
      #basebd2fmlm.p <- pvals.fnc(basebd2fmlm)
      #print(basebd2fmlm.p)
      #basebd2fmlm.p <- NULL
      baseod2fmlm <- NULL
#      baseld2fmlm <- NULL
#      basebd2fmlm <- NULL
      #baseofmlm <- NULL
      #baselfmlm <- NULL
      #basebfmlm <- NULL
      rform <- rforma
      lrform <- lrforma
      bform <- bforma

      print("Testing Dep3F-L-")
      write("Testing Dep3F-L-",stderr())
#      rform <- update.formula(rforma,. ~ . + z.(r.fmlm))
#      lrform <- update.formula(lrforma,. ~ . + z.(r.lfmlm))
#      bform <- update.formula(bforma,. ~ . + z.(r.fmlm) + z.(r.lfmlm))
      base <- lmer(update.formula(bform,.~. +(1+z.(cumd3fmlm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)

#      r.d3fmlm <- residuals(lm(update.formula(rform,z.(cumd3fmlm) ~ .), test))
#      r.ld3fmlm <- residuals(lm(update.formula(lrform,z.(lagcumd3fmlm) ~ .), test))

      write("Testing",stderr())
      baseod3fmlm <- lmer(update.formula(bform,.~.+ z.(cumd3fmlm)+(1+z.(cumd3fmlm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#      baseod3fmlm <- lmer(update.formula(bform,.~.+ z.(r.d3fmlm)+(cumd3fmlm|subject)), test, REML=F)
#      baseld3fmlm <- lmer(update.formula(bform,.~.+ z.(r.ld3fmlm)+(cumd3fmlm|subject)), test, REML=F)
#      basebd3fmlm <- lmer(update.formula(bform,.~.+ z.(r.d3fmlm) + z.(r.ld3fmlm)), test, REML=F)

      print(vif.mer(baseod3fmlm))
#      print(anova(base,basebd3fmlm))
      print(anova(base,baseod3fmlm))
#      print(anova(base,baseld3fmlm))
      print(baseod3fmlm)
#      print(baseld3fmlm)
#      print(basebd3fmlm)
      #basebd3fmlm.p <- pvals.fnc(basebd3fmlm)
      #print(basebd3fmlm.p)
      #basebd3fmlm.p <- NULL
      baseod3fmlm <- NULL
#      baseld3fmlm <- NULL
#      basebd3fmlm <- NULL
      #baseofmlm <- NULL
      #baselfmlm <- NULL
      #basebfmlm <- NULL
      rform <- rforma
      lrform <- lrforma
      bform <- bforma

      print("Testing Dep4F-L-")
      write("Testing Dep4F-L-",stderr())
#      rform <- update.formula(rforma,. ~ . + z.(r.fmlm))
#      lrform <- update.formula(lrforma,. ~ . + z.(r.lfmlm))
#      bform <- update.formula(bforma,. ~ . + z.(r.fmlm) + z.(r.lfmlm))
      base <- lmer(update.formula(bform,.~. +(1+z.(cumd4fmlm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)

#      r.d4fmlm <- residuals(lm(update.formula(rform,z.(cumd4fmlm) ~ .), test))
#      r.ld4fmlm <- residuals(lm(update.formula(lrform,z.(lagcumd4fmlm) ~ .), test))

      write("Testing",stderr())
      baseod4fmlm <- lmer(update.formula(bform,.~.+ z.(cumd4fmlm)+(1+z.(cumd4fmlm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#      baseod4fmlm <- lmer(update.formula(bform,.~.+ z.(r.d4fmlm)+(cumd4fmlm|subject)), test, REML=F)
#      baseld4fmlm <- lmer(update.formula(bform,.~.+ z.(r.ld4fmlm)+(cumd4fmlm|subject)), test, REML=F)
#      basebd4fmlm <- lmer(update.formula(bform,.~.+ z.(r.d4fmlm) + z.(r.ld4fmlm)), test, REML=F)

      print(vif.mer(baseod4fmlm))
#      print(anova(base,basebd4fmlm))
      print(anova(base,baseod4fmlm))
#      print(anova(base,baseld4fmlm))
      print(baseod4fmlm)
#      print(baseld4fmlm)
#      print(basebd4fmlm)
      #basebd4fmlm.p <- pvals.fnc(basebd4fmlm)
      #print(basebd4fmlm.p)
      #basebd4fmlm.p <- NULL
      baseod4fmlm <- NULL
#      baseld4fmlm <- NULL
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
  base <- lmer(update.formula(bform,.~.+(1+z.(olen)+z.(cumfplm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)+z.(nextisfix)|subject)), test, REML=F)
#  base <- lmer(update.formula(bform,.~.+(1+z.(cumfplm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)

  print("Testing F+L-")
  if (CMCL2013) {
    write("Residing F+L-",stderr())
    r.fplm <- residuals(lm(update.formula(rform,z.(cumfplm) ~ .), test))
    #r.lfplm <- residuals(lm(update.formula(lrform,z.(lagcumfplm) ~ .), test))
  }

  write("Testing F+L-",stderr())
  if (CMCL2013) {
    baseofplm <- lmer(update.formula(bform,.~.+ z.(r.fplm) + (1+z.(cumfplm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
  }
  else {
    baseofplm <- lmer(update.formula(bform,.~.+ z.(cumfplm) + (1+z.(olen)+z.(cumfplm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)+z.(nextisfix)|subject)), test, REML=F)
#    baseofplm <- lmer(update.formula(bform,.~.+ z.(cumfplm) + (1+z.(cumfplm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
  }
#  baselfplm <- lmer(update.formula(bform,.~.+ z.(r.lfplm)), test, REML=F)
#  basebfplm <- lmer(update.formula(bform,.~.+ z.(r.fplm) + z.(r.lfplm)), test, REML=F)

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
#    rform <- update.formula(rforma,. ~ . + z.(r.fplm))
#    lrform <- update.formula(lrforma,. ~ . + z.(r.lfplm))
#    bform <- update.formula(bforma,. ~ . + z.(r.fplm) + z.(r.lfplm))
    base <- lmer(update.formula(bform,.~. +(1+z.(cumd1fplm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)

#    r.d1fplm <- residuals(lm(update.formula(rform,z.(cumd1fplm) ~ .), test))
#    r.ld1fplm <- residuals(lm(update.formula(lrform,z.(lagcumd1fplm) ~ .), test))

    write("Testing",stderr())
    baseod1fplm <- lmer(update.formula(bform,.~.+ z.(cumd1fplm) + (1+z.(cumd1fplm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#    baseod1fplm <- lmer(update.formula(bform,.~.+ z.(r.d1fplm) + (cumd1fplm|subject)), test, REML=F)
#    baseld1fplm <- lmer(update.formula(bform,.~.+ z.(r.ld1fplm)+ (lagcumd1fplm|subject)), test, REML=F)
#    basebd1fplm <- lmer(update.formula(bform,.~.+ z.(r.d1fplm) + z.(r.ld1fplm)), test, REML=F)

    print(vif.mer(baseod1fplm))
#    print(anova(base,basebd1fplm))
    print(anova(base,baseod1fplm))
#    print(anova(base,baseld1fplm))
    print(baseod1fplm)
#    print(baseld1fplm)
#    print(basebd1fplm)
    #basebd1fplm.p <- pvals.fnc(basebd1fplm)
    #print(basebd1fplm.p)
    #basebd1fplm.p <- NULL
    baseod1fplm <- NULL
#    baseld1fplm <- NULL
#    basebd1fplm <- NULL
    #baseofplm <- NULL
    #baselfplm <- NULL
    #basebfplm <- NULL
    rform <- rforma
    lrform <- lrforma
    bform <- bforma

    print("Testing Dep2F+L-")
    write("Testing Dep2F+L-",stderr())
#    rform <- update.formula(rforma,. ~ . + z.(r.fplm))
#    lrform <- update.formula(lrforma,. ~ . + z.(r.lfplm))
#    bform <- update.formula(bforma,. ~ . + z.(r.fplm) + z.(r.lfplm))
    base <- lmer(update.formula(bform,.~. +(1+z.(cumd2fplm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)

#    r.d2fplm <- residuals(lm(update.formula(rform,z.(cumd2fplm) ~ .), test))
#    r.ld2fplm <- residuals(lm(update.formula(lrform,z.(lagcumd2fplm) ~ .), test))

    write("Testing",stderr())
    baseod2fplm <- lmer(update.formula(bform,.~.+ z.(cumd2fplm)+(1+z.(cumd2fplm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#    baseod2fplm <- lmer(update.formula(bform,.~.+ z.(r.d2fplm)+(cumd2fplm|subject)), test, REML=F)
#    baseld2fplm <- lmer(update.formula(bform,.~.+ z.(r.ld2fplm)+(lagcumd2fplm|subject)), test, REML=F)
#    basebd2fplm <- lmer(update.formula(bform,.~.+ z.(r.d2fplm) + z.(r.ld2fplm)), test, REML=F)

    print(vif.mer(baseod2fplm))
#    print(anova(base,basebd2fplm))
    print(anova(base,baseod2fplm))
#    print(anova(base,baseld2fplm))
    print(baseod2fplm)
#    print(baseld2fplm)
#    print(basebd2fplm)
    #basebd2fplm.p <- pvals.fnc(basebd2fplm)
    #print(basebd2fplm.p)
    #basebd2fplm.p <- NULL
    baseod2fplm <- NULL
#    baseld2fplm <- NULL
#    basebd2fplm <- NULL
    #baseofplm <- NULL
    #baselfplm <- NULL
    #basebfplm <- NULL
    rform <- rforma
    lrform <- lrforma
    bform <- bforma

    print("Testing Dep3F+L-")
    write("Testing Dep3F+L-",stderr())
#    rform <- update.formula(rforma,. ~ . + z.(r.fplm))
#    lrform <- update.formula(lrforma,. ~ . + z.(r.lfplm))
#    bform <- update.formula(bforma,. ~ . + z.(r.fplm) + z.(r.lfplm))
    base <- lmer(update.formula(bform,.~. +(1+z.(cumd3fplm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)

#    r.d3fplm <- residuals(lm(update.formula(rform,z.(cumd3fplm) ~ .), test))
#    r.ld3fplm <- residuals(lm(update.formula(lrform,z.(lagcumd3fplm) ~ .), test))

    write("Testing",stderr())
    baseod3fplm <- lmer(update.formula(bform,.~.+ z.(cumd3fplm)+(1+z.(cumd3fplm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#    baseod3fplm <- lmer(update.formula(bform,.~.+ z.(r.d3fplm)+(cumd3fplm|subject)), test, REML=F)
#    baseld3fplm <- lmer(update.formula(bform,.~.+ z.(r.ld3fplm)+(cumd3fplm|subject)), test, REML=F)
#    basebd3fplm <- lmer(update.formula(bform,.~.+ z.(r.d3fplm) + z.(r.ld3fplm)), test, REML=F)

    print(vif.mer(baseod3fplm))
#    print(anova(base,basebd3fplm))
    print(anova(base,baseod3fplm))
#    print(anova(base,baseld3fplm))
    print(baseod3fplm)
#    print(baseld3fplm)
#    print(basebd3fplm)
    #basebd3fplm.p <- pvals.fnc(basebd3fplm)
    #print(basebd3fplm.p)
    #basebd3fplm.p <- NULL
    baseod3fplm <- NULL
#    baseld3fplm <- NULL
#    basebd3fplm <- NULL
    #baseofplm <- NULL
    #baselfplm <- NULL
    #basebfplm <- NULL
    rform <- rforma
    lrform <- lrforma
    bform <- bforma

    print("Testing Dep4F+L-")
    write("Testing Dep4F+L-",stderr())
#    rform <- update.formula(rforma,. ~ . + z.(r.fplm))
#    lrform <- update.formula(lrforma,. ~ . + z.(r.lfplm))
#    bform <- update.formula(bforma,. ~ . + z.(r.fplm) + z.(r.lfplm))
    base <- lmer(update.formula(bform,.~. +(1+z.(cumd4fplm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)

#    r.d4fplm <- residuals(lm(update.formula(rform,z.(cumd4fplm) ~ .), test))
#    r.ld4fplm <- residuals(lm(update.formula(lrform,z.(lagcumd4fplm) ~ .), test))

    write("Testing",stderr())
    baseod4fplm <- lmer(update.formula(bform,.~.+ z.(cumd4fplm)+(1+z.(cumd4fplm)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#    baseod4fplm <- lmer(update.formula(bform,.~.+ z.(r.d4fplm)+(cumd4fplm|subject)), test, REML=F)
#    baseld4fplm <- lmer(update.formula(bform,.~.+ z.(r.ld4fplm)+(cumd4fplm|subject)), test, REML=F)
#    basebd4fplm <- lmer(update.formula(bform,.~.+ z.(r.d4fplm) + z.(r.ld4fplm)), test, REML=F)

    print(vif.mer(baseod4fplm))
#    print(anova(base,basebd4fplm))
    print(anova(base,baseod4fplm))
#    print(anova(base,baseld4fplm))
    print(baseod4fplm)
#    print(baseld4fplm)
#    print(basebd4fplm)
    #basebd4fplm.p <- pvals.fnc(basebd4fplm)
    #print(basebd4fplm.p)
    #basebd4fplm.p <- NULL
    baseod4fplm <- NULL
#    baseld4fplm <- NULL
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
  base <- lmer(update.formula(bform,.~.+(1+z.(olen)+z.(cumfmlp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)+z.(nextisfix)|subject)), test, REML=F)
#  base <- lmer(update.formula(bform,.~.+(1+z.(cumfmlp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)

  print("Testing F-L+")
  if (CMCL2013) {
    write("Residing F-L+",stderr())
    r.fmlp <- residuals(lm(update.formula(rform,z.(cumfmlp) ~ .), test))
    #r.lfmlp <- residuals(lm(update.formula(lrform,z.(lagcumfmlp) ~ .), test))
  }

  write("Testing F-L+",stderr())
  if (CMCL2013) {
    baseofmlp <- lmer(update.formula(bform,.~.+ z.(r.fmlp) + (1+z.(cumfmlp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
  }
  else {
    baseofmlp <- lmer(update.formula(bform,.~.+ z.(cumfmlp) + (1+z.(olen)+z.(cumfmlp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)+z.(nextisfix)|subject)), test, REML=F)
#    baseofmlp <- lmer(update.formula(bform,.~.+ z.(cumfmlp) + (1+z.(cumfmlp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
  }
#  baselfmlp <- lmer(update.formula(bform,.~.+ z.(r.lfmlp)), test, REML=F)
#  basebfmlp <- lmer(update.formula(bform,.~.+ z.(r.fmlp) + z.(r.lfmlp)), test, REML=F)

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
#    rform <- update.formula(rforma,. ~ . + z.(r.fmlp))
#    lrform <- update.formula(lrforma,. ~ . + z.(r.lfmlp))
#    bform <- update.formula(bforma,. ~ . + z.(r.fmlp) + z.(r.lfmlp))
    base <- lmer(update.formula(bform,.~. +(1+z.(cumd1fmlp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)

#    r.d1fmlp <- residuals(lm(update.formula(rform,z.(cumd1fmlp) ~ .), test))
#    r.ld1fmlp <- residuals(lm(update.formula(lrform,z.(lagcumd1fmlp) ~ .), test))

    write("Testing",stderr())
    baseod1fmlp <- lmer(update.formula(bform,.~.+ z.(cumd1fmlp)+(1+z.(cumd1fmlp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#    baseod1fmlp <- lmer(update.formula(bform,.~.+ z.(r.d1fmlp)+(cumd1fmlp|subject)), test, REML=F)
#    baseld1fmlp <- lmer(update.formula(bform,.~.+ z.(r.ld1fmlp)+(lagcumd1fmlp|subject)), test, REML=F)
#    basebd1fmlp <- lmer(update.formula(bform,.~.+ z.(r.d1fmlp) + z.(r.ld1fmlp)), test, REML=F)

    print(vif.mer(baseod1fmlp))
#    print(anova(base,basebd1fmlp))
    print(anova(base,baseod1fmlp))
#    print(anova(base,baseld1fmlp))
    print(baseod1fmlp)
#    print(baseld1fmlp)
#    print(basebd1fmlp)
    #basebd1fmlp.p <- pvals.fnc(basebd1fmlp)
    #print(basebd1fmlp.p)
    #basebd1fmlp.p <- NULL
    baseod1fmlp <- NULL
#    baseld1fmlp <- NULL
#    basebd1fmlp <- NULL
#    baseofmlp <- NULL
#    baselfmlp <- NULL
#    basebfmlp <- NULL
    rform <- rforma
    lrform <- lrforma
    bform <- bforma

    print("Testing Dep2F-L+")
    write("Testing Dep2F-L+",stderr())
#    rform <- update.formula(rforma,. ~ . + z.(r.fmlp))
#    lrform <- update.formula(lrforma,. ~ . + z.(r.lfmlp))
#    bform <- update.formula(bforma,. ~ . + z.(r.fmlp) + z.(r.lfmlp))
    base <- lmer(update.formula(bform,.~. +(1+z.(cumd2fmlp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)

#    r.d2fmlp <- residuals(lm(update.formula(rform,z.(cumd2fmlp) ~ .), test))
#    r.ld2fmlp <- residuals(lm(update.formula(lrform,z.(lagcumd2fmlp) ~ .), test))

    write("Testing",stderr())
    baseod2fmlp <- lmer(update.formula(bform,.~.+ z.(cumd2fmlp)+(1+z.(cumd2fmlp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#    baseod2fmlp <- lmer(update.formula(bform,.~.+ z.(r.d2fmlp)+(cumd2fmlp|subject)), test, REML=F)
#    baseld2fmlp <- lmer(update.formula(bform,.~.+ z.(r.ld2fmlp)+(lagcumd2fmlp|subject)), test, REML=F)
#    basebd2fmlp <- lmer(update.formula(bform,.~.+ z.(r.d2fmlp) + z.(r.ld2fmlp)), test, REML=F)

    print(vif.mer(baseod2fmlp))
#    print(anova(base,basebd2fmlp))
    print(anova(base,baseod2fmlp))
#    print(anova(base,baseld2fmlp))
    print(baseod2fmlp)
#    print(baseld2fmlp)
#    print(basebd2fmlp)
    #basebd2fmlp.p <- pvals.fnc(basebd2fmlp)
    #print(basebd2fmlp.p)
    #basebd2fmlp.p <- NULL
    baseod2fmlp <- NULL
#    baseld2fmlp <- NULL
#    basebd2fmlp <- NULL
#    baseofmlp <- NULL
#    baselfmlp <- NULL
#    basebfmlp <- NULL
    rform <- rforma
    lrform <- lrforma
    bform <- bforma

    print("Testing Dep3F-L+")
    write("Testing Dep3F-L+",stderr())
#    rform <- update.formula(rforma,. ~ . + z.(r.fmlp))
#    lrform <- update.formula(lrforma,. ~ . + z.(r.lfmlp))
#    bform <- update.formula(bforma,. ~ . + z.(r.fmlp) + z.(r.lfmlp))
    base <- lmer(update.formula(bform,.~. +(1+z.(cumd3fmlp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)

#    r.d3fmlp <- residuals(lm(update.formula(rform,z.(cumd3fmlp) ~ .), test))
#    r.ld3fmlp <- residuals(lm(update.formula(lrform,z.(lagcumd3fmlp) ~ .), test))

    write("Testing",stderr())
    baseod3fmlp <- lmer(update.formula(bform,.~.+ z.(cumd3fmlp)+(1+z.(cumd3fmlp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#    baseld3fmlp <- lmer(update.formula(bform,.~.+ z.(r.ld3fmlp)+(lagcumd3fmlp|subject)), test, REML=F)
#    basebd3fmlp <- lmer(update.formula(bform,.~.+ z.(r.d3fmlp) + z.(r.ld3fmlp)), test, REML=F)

    print(vif.mer(baseod3fmlp))
#    print(anova(base,basebd3fmlp))
    print(anova(base,baseod3fmlp))
#    print(anova(base,baseld3fmlp))
    print(baseod3fmlp)
#    print(baseld3fmlp)
#    print(basebd3fmlp)
    #basebd3fmlp.p <- pvals.fnc(basebd3fmlp)
    #print(basebd3fmlp.p)
    #basebd3fmlp.p <- NULL
    baseod3fmlp <- NULL
#    baseld3fmlp <- NULL
#    basebd3fmlp <- NULL
#    baseofmlp <- NULL
#    baselfmlp <- NULL
#    basebfmlp <- NULL
    rform <- rforma
    lrform <- lrforma
    bform <- bforma

    print("Testing Dep4F-L+")
    write("Testing Dep4F-L+",stderr())
#    rform <- update.formula(rforma,. ~ . + z.(r.fmlp))
#    lrform <- update.formula(lrforma,. ~ . + z.(r.lfmlp))
#    bform <- update.formula(bforma,. ~ . + z.(r.fmlp) + z.(r.lfmlp))
    base <- lmer(update.formula(bform,.~. +(1+z.(cumd4fmlp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)

#    r.d4fmlp <- residuals(lm(update.formula(rform,z.(cumd4fmlp) ~ .), test))
#    r.ld4fmlp <- residuals(lm(update.formula(lrform,z.(lagcumd4fmlp) ~ .), test))

    write("Testing",stderr())
    baseod4fmlp <- lmer(update.formula(bform,.~.+ z.(cumd4fmlp)+(1+z.(cumd4fmlp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#    baseod4fmlp <- lmer(update.formula(bform,.~.+ z.(r.d4fmlp)+(cumd4fmlp|subject)), test, REML=F)
#    baseld4fmlp <- lmer(update.formula(bform,.~.+ z.(r.ld4fmlp)+(cumd4fmlp|subject)), test, REML=F)
#    basebd4fmlp <- lmer(update.formula(bform,.~.+ z.(r.d4fmlp) + z.(r.ld4fmlp)), test, REML=F)

    print(vif.mer(baseod4fmlp))
#    print(anova(base,basebd4fmlp))
    print(anova(base,baseod4fmlp))
#    print(anova(base,baseld4fmlp))
    print(baseod4fmlp)
#    print(baseld4fmlp)
#    print(basebd4fmlp)
    #basebd4fmlp.p <- pvals.fnc(basebd4fmlp)
    #print(basebd4fmlp.p)
    #basebd4fmlp.p <- NULL
    baseod4fmlp <- NULL
#    baseld4fmlp <- NULL
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
    base <- lmer(update.formula(bform,.~.+(1+z.(cumfplp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)

    print("Testing F+L+")
#    write("Residing F+L+",stderr())
#    r.fplp <- residuals(lm(update.formula(rform,z.(cumfplp) ~ .), test))
#    r.lfplp <- residuals(lm(update.formula(lrform,z.(lagcumfplp) ~ .), test))

    write("Testing F+L+",stderr())
#    baseofplp <- lmer(update.formula(bform,.~.+ z.(r.fplp) + (1+z.(cumfplp)|subject)), test, REML=F)
    baseofplp <- lmer(update.formula(bform,.~.+ z.(cumfplp) + (1+z.(cumfplp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#    baselfplp <- lmer(update.formula(bform,.~.+ z.(r.lfplp)), test, REML=F)
#    basebfplp <- lmer(update.formula(bform,.~.+ z.(r.fplp) + z.(r.lfplp)), test, REML=F)

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
#      rform <- update.formula(rforma,. ~ . + z.(r.fplp))
#      lrform <- update.formula(lrforma,. ~ . + z.(r.lfplp))
#      bform <- update.formula(bforma,. ~ . + z.(r.fplp) + z.(r.lfplp))
      base <- lmer(update.formula(bform,.~. +(1+z.(cumd1fplp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)

#      r.d1fplp <- residuals(lm(update.formula(rform,z.(cumd1fplp) ~ .), test))
#      r.ld1fplp <- residuals(lm(update.formula(lrform,z.(lagcumd1fplp) ~ .), test))

      write("Testing",stderr())
      baseod1fplp <- lmer(update.formula(bform,.~.+ z.(cumd1fplp)+(1+z.(cumd1fplp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#      baseod1fplp <- lmer(update.formula(bform,.~.+ z.(r.d1fplp)+(cumd1fplp|subject)), test, REML=F)
#      baseld1fplp <- lmer(update.formula(bform,.~.+ z.(r.ld1fplp)+(lagcumd1fplp|subject)), test, REML=F)
#      basebd1fplp <- lmer(update.formula(bform,.~.+ z.(r.d1fplp) + z.(r.ld1fplp)), test, REML=F)

      print(vif.mer(baseod1fplp))
#      print(anova(base,basebd1fplp))
      print(anova(base,baseod1fplp))
#      print(anova(base,baseld1fplp))
      print(baseod1fplp)
#      print(baseld1fplp)
#      print(basebd1fplp)
      #basebd1fplp.p <- pvals.fnc(basebd1fplp)
      #print(basebd1fplp.p)
      #basebd1fplp.p <- NULL
      baseod1fplp <- NULL
#      baseld1fplp <- NULL
#      basebd1fplp <- NULL
#      baseofplp <- NULL
#      baselfplp <- NULL
#      basebfplp <- NULL
      rform <- rforma
      lrform <- lrforma
      bform <- bforma

      print("Testing Dep2F+L+")
      write("Testing Dep2F+L+",stderr())
#      rform <- update.formula(rforma,. ~ . + z.(r.fplp))
#      lrform <- update.formula(lrforma,. ~ . + z.(r.lfplp))
#      bform <- update.formula(bforma,. ~ . + z.(r.fplp) + z.(r.lfplp))
      base <- lmer(update.formula(bform,.~. +(1+z.(cumd2fplp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)

#      r.d2fplp <- residuals(lm(update.formula(rform,z.(cumd2fplp) ~ .), test))
#      r.ld2fplp <- residuals(lm(update.formula(lrform,z.(lagcumd2fplp) ~ .), test))

      write("Testing",stderr())
      baseod2fplp <- lmer(update.formula(bform,.~.+ z.(cumd2fplp)+(1+z.(cumd2fplp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#      baseod2fplp <- lmer(update.formula(bform,.~.+ z.(r.d2fplp)+(cumd2fplp|subject)), test, REML=F)
#      baseld2fplp <- lmer(update.formula(bform,.~.+ z.(r.ld2fplp)+(lagcumd2fplp|subject)), test, REML=F)
#      basebd2fplp <- lmer(update.formula(bform,.~.+ z.(r.d2fplp) + z.(r.ld2fplp)), test, REML=F)

      print(vif.mer(baseod2fplp))
#      print(anova(base,basebd2fplp))
      print(anova(base,baseod2fplp))
#      print(anova(base,baseld2fplp))
      print(baseod2fplp)
#      print(baseld2fplp)
#      print(basebd2fplp)
      #basebd2fplp.p <- pvals.fnc(basebd2fplp)
      #print(basebd2fplp.p)
      #basebd2fplp.p <- NULL
      baseod2fplp <- NULL
#      baseld2fplp <- NULL
#      basebd2fplp <- NULL
#      baseofplp <- NULL
#      baselfplp <- NULL
#      basebfplp <- NULL
      rform <- rforma
      lrform <- lrforma
      bform <- bforma

      print("Testing Dep3F+L+")
      write("Testing Dep3F+L+",stderr())
#      rform <- update.formula(rforma,. ~ . + z.(r.fplp))
#      lrform <- update.formula(lrforma,. ~ . + z.(r.lfplp))
#      bform <- update.formula(bforma,. ~ . + z.(r.fplp) + z.(r.lfplp))
      base <- lmer(update.formula(bform,.~. +(1+z.(cumd3fplp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)

#      r.d3fplp <- residuals(lm(update.formula(rform,z.(cumd3fplp) ~ .), test))
#      r.ld3fplp <- residuals(lm(update.formula(lrform,z.(lagcumd3fplp) ~ .), test))

      write("Testing",stderr())
      baseod3fplp <- lmer(update.formula(bform,.~.+ z.(cumd3fplp)+(1+z.(cumd3fplp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#      baseod3fplp <- lmer(update.formula(bform,.~.+ z.(r.d3fplp)+(cumd3fplp|subject)), test, REML=F)
#      baseld3fplp <- lmer(update.formula(bform,.~.+ z.(r.ld3fplp)+(lagcumd3fplp|subject)), test, REML=F)
#      basebd3fplp <- lmer(update.formula(bform,.~.+ z.(r.d3fplp) + z.(r.ld3fplp)), test, REML=F)

      print(vif.mer(baseod3fplp))
#      print(anova(base,basebd3fplp))
      print(anova(base,baseod3fplp))
#      print(anova(base,baseld3fplp))
      print(baseod3fplp)
#      print(baseld3fplp)
#      print(basebd3fplp)
      #basebd3fplp.p <- pvals.fnc(basebd3fplp)
      #print(basebd3fplp.p)
      #basebd3fplp.p <- NULL
      baseod3fplp <- NULL
#      baseld3fplp <- NULL
#      basebd3fplp <- NULL
#      baseofplp <- NULL
#      baselfplp <- NULL
#      basebfplp <- NULL
      rform <- rforma
      lrform <- lrforma
      bform <- bforma

      print("Testing Dep4F+L+")
      write("Testing Dep4F+L+",stderr())
#      rform <- update.formula(rforma,. ~ . + z.(r.fplp))
#      lrform <- update.formula(lrforma,. ~ . + z.(r.lfplp))
#      bform <- update.formula(bforma,. ~ . + z.(r.fplp) + z.(r.lfplp))
      base <- lmer(update.formula(bform,.~. +(1+z.(cumd4fplp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)

#      r.d4fplp <- residuals(lm(update.formula(rform,z.(cumd4fplp) ~ .), test))
#      r.ld4fplp <- residuals(lm(update.formula(lrform,z.(lagcumd4fplp) ~ .), test))

      write("Testing",stderr())
      baseod4fplp <- lmer(update.formula(bform,.~.+ z.(cumd4fplp)+(1+z.(cumd4fplp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#      baseod4fplp <- lmer(update.formula(bform,.~.+ z.(r.d4fplp)+(cumd4fplp|subject)), test, REML=F)
#      baseld4fplp <- lmer(update.formula(bform,.~.+ z.(r.ld4fplp)+(lagcumd4fplp|subject)), test, REML=F)
#      basebd4fplp <- lmer(update.formula(bform,.~.+ z.(r.d4fplp) + z.(r.ld4fplp)), test, REML=F)

      print(vif.mer(baseod4fplp))
#      print(anova(base,basebd4fplp))
      print(anova(base,baseod4fplp))
#      print(anova(base,baseld4fplp))
      print(baseod4fplp)
#      print(baseld4fplp)
#      print(basebd4fplp)
#      #basebd4fplp.p <- pvals.fnc(basebd4fplp)
#      #print(basebd4fplp.p)
#      #basebd4fplp.p <- NULL
      baseod4fplp <- NULL
#      baseld4fplp <- NULL
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
    rform <- update.formula(rforma,. ~ . + z.(r.fmlp))
    lrform <- update.formula(lrforma,. ~ . + z.(r.lfmlp))
    bform <- update.formula(bforma,. ~ . + z.(r.fmlp) + z.(r.lfmlp))

    r.Dfmlp <- residuals(lm(update.formula(rform,z.(cumDfmlp) ~ .), test))
    r.lDfmlp <- residuals(lm(update.formula(lrform,z.(lagcumDfmlp) ~ .), test))

    baseoDfmlp <- lmer(update.formula(bform,.~.+ z.(r.Dfmlp)), test, REML=F)
    baselDfmlp <- lmer(update.formula(bform,.~.+ z.(r.lDfmlp)), test, REML=F)
    basebDfmlp <- lmer(update.formula(bform,.~.+ z.(r.Dfmlp) + z.(r.lDfmlp)), test, REML=F)

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
    r.dDfmlp <- residuals(lm(update.formula(rform,z.(cumdDfmlp) ~ .), test))
    r.ldDfmlp <- residuals(lm(update.formula(lrform,z.(lagcumdDfmlp) ~ .), test))

    baseodDfmlp <- lmer(update.formula(bform,.~.+ z.(r.dDfmlp)), test, REML=F)
    baseldDfmlp <- lmer(update.formula(bform,.~.+ z.(r.ldDfmlp)), test, REML=F)
    basebdDfmlp <- lmer(update.formula(bform,.~.+ z.(r.dDfmlp) + z.(r.ldDfmlp)), test, REML=F)

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

#  r.fplm <- residuals(lm(update.formula(rform,z.(cumfplm) ~ .+z.(r.fmlp)), test))
#  r.lfplm <- residuals(lm(update.formula(lrform,z.(lagcumfplm) ~ .+z.(r.lfmlp)), test))
#  r.fmlm <- residuals(lm(update.formula(rform,z.(cumfmlm) ~ .+z.(r.fmlp)+z.(r.fplm)), test))
#  r.lfmlm <- residuals(lm(update.formula(lrform,z.(lagcumfmlm) ~ .+z.(r.lfmlp)+z.(r.lfplm)), test))
#  r.fplp <- residuals(lm(update.formula(rform,z.(cumfplp) ~ .+z.(r.fmlp)+z.(r.fplm)+z.(r.fmlm)), test))
#  r.lfplp <- residuals(lm(update.formula(lrform,z.(lagcumfplp) ~ .+z.(r.lfmlp)+z.(r.lfplm)+z.(r.lfmlm)), test))

#  rformb <- update.formula(rform,.~z.(r.fmlp)+z.(r.fmlm)+z.(r.fplm)+z.(r.fplp)+.)
#  lrformb <- update.formula(lrform,.~z.(r.lfmlp)+z.(r.lfmlm)+z.(r.lfplm)+z.(r.lfplp)+.)
#  bformb <- update.formula(bform,.~z.(r.fmlp)+z.(r.fmlm)+z.(r.fplm)+z.(r.fplp)+z.(r.lfmlp)+z.(r.lfmlm)+z.(r.lfplm)+z.(r.lfplp)+.)
##  rformb <- update.formula(rform,.~z.(cumfmlp)+z.(cumfmlm)+z.(cumfplm)+z.(cumfplp)+.)
##  lrformb <- update.formula(lrform,.~z.(lagcumfmlp)+z.(lagcumfmlm)+z.(lagcumfplm)+z.(lagcumfplp)+.)
##  bformb <- update.formula(bform,.~z.(cumfmlp)+z.(cumfmlm)+z.(cumfplm)+z.(cumfplp)+z.(lagcumfmlp)+z.(lagcumfmlm)+z.(lagcumfplm)+z.(lagcumfplp)+.)
rformb <- rform
lrformb <- lrform
bformb <- bform
#  baseb <- lmer(bformb,test,REML=F)

  print("Testing Cum Integration")
  write("Testing Cum Integration",stderr())
  write("Building Base: Cum Integration",stderr())
  if (CMCL2013){
    baseb <- lmer(update.formula(bformb,.~.+(1+z.(cumfmlmbp+cumfmlpbp+cumfplmbp+cumfplpbp+cumfmlpbo+cumfmlpba+cumfmlpbc)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)),test,REML=F)
  }
  else {
    baseb <- lmer(update.formula(bformb,.~.+(1+z.(cumfmlmbp+cumfmlpbp+cumfplmbp+cumfplpbp+cumfmlpbo+cumfmlpba+cumfmlpbc)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)),test,REML=F)
  }

  if (CMCL2013) {
    write("Residing Cum Integration",stderr())
    r.cint <- residuals(lm(update.formula(rformb,z.(cumfmlmbp+cumfmlpbp+cumfplmbp+cumfplpbp+cumfmlpbo+cumfmlpba+cumfmlpbc) ~ .), test))
    #r.lint <- residuals(lm(update.formula(lrformb,z.(lagcumfmlmbo+lagcumfmlpbo+lagcumfplmbo+lagcumfplpbo) ~ .), test))
  }

  write("Testing",stderr())
  if (CMCL2013) {
    baseocint <- lmer(update.formula(bformb,.~.+ z.(r.cint) + (1+z.(cumfmlmbp+cumfmlpbp+cumfplmbp+cumfplpbp+cumfmlpbo+cumfmlpba+cumfmlpbc)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
  }
  else {
    baseocint <- lmer(update.formula(bformb,.~.+ z.(cumfmlmbp+cumfmlpbp+cumfplmbp+cumfplpbp+cumfmlpbo+cumfmlpba+cumfmlpbc) + (1+z.(cumfmlmbp+cumfmlpbp+cumfplmbp+cumfplpbp+cumfmlpbo+cumfmlpba+cumfmlpbc)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
  }
#  baselint <- lmer(update.formula(bformb,.~.+ z.(r.lint)), test, REML=F)
#  basebcint <- lmer(update.formula(bformb,.~.+ z.(r.cint) + z.(r.lint)), test, REML=F)

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

  return(0)
  if (CMCL2013){
    return(0)
  }

#  print("Testing B Addition")
#  write("Testing B Addition",stderr())

#  r.ba <- residuals(lm(update.formula(rformb,z.(badd) ~ .), test))

#  baseoba <- lmer(update.formula(bformb,.~.+ z.(r.ba)), test, REML=F)

#  print(anova(baseb,baseoba))
#  print(baseoba)
  #baseoba.p <- pvals.fnc(baseoba)
  #print(baseoba.p)
  #baseoba.p <- NULL
#  baseoba <- NULL

  print("Testing Cum B Addition")
  write("Testing Cum B Addition",stderr())
  write("Building Base: Cum B Addition",stderr())
  baseb <- lmer(update.formula(bformb,.~.+(1+z.(cumbadd)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)),test,REML=F)

#  write("Residing Cum B Addition",stderr())
#  r.cba <- residuals(lm(update.formula(rformb,z.(cumbadd) ~ .), test))
#  r.lba <- residuals(lm(update.formula(lrformb,z.(lagcumbadd) ~ .), test))

  write("Testing",stderr())
#  baseocba <- lmer(update.formula(bformb,.~.+ z.(r.cba) + (1+z.(cumbadd)|subject)), test, REML=F)
  baseocba <- lmer(update.formula(bformb,.~.+ z.(cumbadd) + (1+z.(cumbadd)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#  baselba <- lmer(update.formula(bformb,.~.+ z.(r.lba)), test, REML=F)
#  basebcba <- lmer(update.formula(bformb,.~.+ z.(r.cba) + z.(r.lba)), test, REML=F)

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

#  r.bp <- residuals(lm(update.formula(rformb,z.(bp) ~ .), test))

#  baseobp <- lmer(update.formula(bformb,.~.+ z.(r.bp)), test, REML=F)

#  print(anova(baseb,baseobp))
#  print(baseobp)
  #baseobp.p <- pvals.fnc(baseobp)
  #print(baseobp.p)
  #baseobp.p <- NULL
#  baseobp <- NULL

  print("Testing Cum B+")
  write("Testing Cum B+",stderr())
  write("Building Base: Cum B+",stderr())
  baseb <- lmer(update.formula(bformb,.~.+(1+z.(cumbp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)),test,REML=F)

#  write("Residing Cum B+",stderr())
#  r.cbp <- residuals(lm(update.formula(rformb,z.(cumbp) ~ .), test))
#  r.lbp <- residuals(lm(update.formula(lrformb,z.(lagcumbp) ~ .), test))

  write("Testing",stderr())
#  baseocbp <- lmer(update.formula(bformb,.~.+ z.(r.cbp) + (1+z.(cumbp)|subject)), test, REML=F)
  baseocbp <- lmer(update.formula(bformb,.~.+ z.(cumbp) + (1+z.(cumbp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#  baselbp <- lmer(update.formula(bformb,.~.+ z.(r.lbp)), test, REML=F)
#  basebcbp <- lmer(update.formula(bformb,.~.+ z.(r.cbp) + z.(r.lbp)), test, REML=F)

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
  baseb <- lmer(update.formula(bformb,.~.+(1+z.(cumdbp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)),test,REML=F)

#  write("Residing Cum Dep B+",stderr())
#  r.cdbp <- residuals(lm(update.formula(rformb,z.(cumdbp) ~ .), test))
#  r.ldbp <- residuals(lm(update.formula(lrformb,z.(lagcumdbp) ~ .), test))

  write("Testing",stderr())
#  baserocdbp <- lmer(update.formula(bformb,.~.+ z.(r.cdbp) + (1+z.(cumdbp)|subject)), test, REML=F)
  baseocdbp <- lmer(update.formula(bformb,.~.+ z.(cumdbp) + (1+z.(cumdbp)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#  baseldbp <- lmer(update.formula(bformb,.~.+ z.(r.ldbp)), test, REML=F)
#  basebcdbp <- lmer(update.formula(bformb,.~.+ z.(r.cdbp) + z.(r.ldbp)), test, REML=F)

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
#  r.cDbp <- residuals(lm(update.formula(rformb,z.(cumDbp) ~ .), test))
#  r.lDbp <- residuals(lm(update.formula(lrformb,z.(lagcumDbp) ~ .), test))

#  baseocDbp <- lmer(update.formula(bformb,.~.+ z.(r.cDbp)), test, REML=F)
#  baselDbp <- lmer(update.formula(bformb,.~.+ z.(r.lDbp)), test, REML=F)
#  basebcDbp <- lmer(update.formula(bformb,.~.+ z.(r.cDbp) + z.(r.lDbp)), test, REML=F)

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
#  r.cbs <- residuals(lm(update.formula(rformb,z.(cumbsto) ~ .), test))
#  r.lbs <- residuals(lm(update.formula(lrformb,z.(lagcumbsto) ~ .), test))

#  baseocbs <- lmer(update.formula(bformb,.~.+ z.(r.cbs)), test, REML=F)
#  baselbs <- lmer(update.formula(bformb,.~.+ z.(r.lbs)), test, REML=F)
#  basebcbs <- lmer(update.formula(bformb,.~.+ z.(r.cbs) + z.(r.lbs)), test, REML=F)

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
  baseb <- lmer(update.formula(bformb,.~.+(1+z.(cumbcdr)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)),test,REML=F)

#  write("Residing Cum BCDR",stderr())
#  r.cbcdr <- residuals(lm(update.formula(rformb,z.(cumbcdr) ~ .), test))
#  r.lbcdr <- residuals(lm(update.formula(lrformb,z.(lagcumbcdr) ~ .), test))

  write("Testing",stderr())
#  baseocbcdr <- lmer(update.formula(bformb,.~.+ z.(r.cbcdr) + (1+z.(cumbcdr)|subject)), test, REML=F)
  baseocbcdr <- lmer(update.formula(bformb,.~.+ z.(cumbcdr) + (1+z.(cumbcdr)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#  baselbcdr <- lmer(update.formula(bformb,.~.+ z.(r.lbcdr)), test, REML=F)
#  basebcbcdr <- lmer(update.formula(bformb,.~.+ z.(r.cbcdr) + z.(r.lbcdr)), test, REML=F)

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
  baseb <- lmer(update.formula(bformb,.~.+(1+z.(cumfmlmbo+cumfmlpbo+cumfplmbo+cumfplpbo)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)),test,REML=F)

#  write("Residing Cum BNil",stderr())
#  r.cbnil <- residuals(lm(update.formula(rformb,z.(cumfmlmbo+cumfmlpbo+cumfplmbo+cumfplpbo) ~ .), test))
#  r.lbnil <- residuals(lm(update.formula(lrformb,z.(lagcumfmlmbo+lagcumfmlpbo+lagcumfplmbo+lagcumfplpbo) ~ .), test))

  write("Testing",stderr())
#  baseocbnil <- lmer(update.formula(bformb,.~.+ z.(r.cbnil) + (1+z.(cumfmlmbo+cumfmlpbo+cumfplmbo+cumfplpbo)|subject)), test, REML=F)
  baseocbnil <- lmer(update.formula(bformb,.~.+ z.(cumfmlmbo+cumfmlpbo+cumfplmbo+cumfplpbo) + (1+z.(cumfmlmbo+cumfmlpbo+cumfplmbo+cumfplpbo)+z.(cumtotsurp)+z.(previsfix)+z.(cumwdelta)|subject)), test, REML=F)
#  baselbnil <- lmer(update.formula(bformb,.~.+ z.(r.lbnil)), test, REML=F)
#  basebcbnil <- lmer(update.formula(bformb,.~.+ z.(r.cbnil) + z.(r.lbnil)), test, REML=F)

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
  r.cbm <- residuals(lm(update.formula(rformb,z.(cumbm) ~ .), test))
  r.lbm <- residuals(lm(update.formula(lrformb,z.(lagcumbm) ~ .), test))

  baseocbm <- lmer(update.formula(bformb,.~.+ z.(r.cbm)), test, REML=F)
  baselbm <- lmer(update.formula(bformb,.~.+ z.(r.lbm)), test, REML=F)
  basebcbm <- lmer(update.formula(bformb,.~.+ z.(r.cbm) + z.(r.lbm)), test, REML=F)

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
  r.dbm <- residuals(lm(update.formula(rformb,z.(dbm) ~ .), test))

  baseodbm <- lmer(update.formula(bformb,.~.+ z.(r.dbm)), test, REML=F)

  print(anova(baseb,baseodbm))
  print(baseodbm)
  #baseodbm.p <- pvals.fnc(baseodbm)
  #print(baseodbm.p)
  #baseodbm.p <- NULL
  baseodbm <- NULL

  print("Testing Cum Dep B-")
  write("Testing Cum Dep B-",stderr())
  r.cdbm <- residuals(lm(update.formula(rformb,z.(cumdbm) ~ .), test))
  r.ldbm <- residuals(lm(update.formula(lrformb,z.(lagcumdbm) ~ .), test))

  baseocdbm <- lmer(update.formula(bformb,.~.+ z.(r.cdbm)), test, REML=F)
  baseldbm <- lmer(update.formula(bformb,.~.+ z.(r.ldbm)), test, REML=F)
  basebcdbm <- lmer(update.formula(bformb,.~.+ z.(r.cdbm) + z.(r.ldbm)), test, REML=F)

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
  r.fmlmba <- residuals(lm(update.formula(rformb,z.(fmlmba) ~ .), test))

  baseofmlmba <- lmer(update.formula(bformb,.~.+ z.(r.fmlmba)), test, REML=F)

  print(anova(baseb,baseofmlmba))
  print(baseofmlmba)
  #baseofmlmba.p <- pvals.fnc(baseofmlmba)
  #print(baseofmlmba.p)
  #baseofmlmba.p <- NULL
  baseofmlmba <- NULL

  print("Testing Cum F-L-Ba")
  write("Testing Cum F-L-Ba",stderr())
  r.cfmlmba <- residuals(lm(update.formula(rformb,z.(cumfmlmba) ~ .), test))
  r.lfmlmba <- residuals(lm(update.formula(lrformb,z.(lagcumfmlmba) ~ .), test))

  baseocfmlmba <- lmer(update.formula(bformb,.~.+ z.(r.cfmlmba)), test, REML=F)
  baselfmlmba <- lmer(update.formula(bformb,.~.+ z.(r.lfmlmba)), test, REML=F)
  basebcfmlmba <- lmer(update.formula(bformb,.~.+ z.(r.cfmlmba) + z.(r.lfmlmba)), test, REML=F)

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
  r.fmlmbc <- residuals(lm(update.formula(rformb,z.(fmlmbc) ~ .), test))

  baseofmlmbc <- lmer(update.formula(bformb,.~.+ z.(r.fmlmbc)), test, REML=F)

  print(anova(baseb,baseofmlmbc))
  print(baseofmlmbc)
  #baseofmlmbc.p <- pvals.fnc(baseofmlmbc)
  #print(baseofmlmbc.p)
  #baseofmlmbc.p <- NULL
  baseofmlmbc <- NULL

  print("Testing Cum F-L-Bc")
  write("Testing Cum F-L-Bc",stderr())
  r.cfmlmbc <- residuals(lm(update.formula(rformb,z.(cumfmlmbc) ~ .), test))
  r.lfmlmbc <- residuals(lm(update.formula(lrformb,z.(lagcumfmlmbc) ~ .), test))

  baseocfmlmbc <- lmer(update.formula(bformb,.~.+ z.(r.cfmlmbc)), test, REML=F)
  baselfmlmbc <- lmer(update.formula(bformb,.~.+ z.(r.lfmlmbc)), test, REML=F)
  basebcfmlmbc <- lmer(update.formula(bformb,.~.+ z.(r.cfmlmbc) + z.(r.lfmlmbc)), test, REML=F)

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
  r.fmlmbo <- residuals(lm(update.formula(rformb,z.(fmlmbo) ~ .), test))

  baseofmlmbo <- lmer(update.formula(bformb,.~.+ z.(r.fmlmbo)), test, REML=F)

  print(anova(baseb,baseofmlmbo))
  print(baseofmlmbo)
  #baseofmlmbo.p <- pvals.fnc(baseofmlmbo)
  #print(baseofmlmbo.p)
  #baseofmlmbo.p <- NULL
  baseofmlmbo <- NULL

  print("Testing Cum F-L-Bo")
  write("Testing Cum F-L-Bo",stderr())
  r.cfmlmbo <- residuals(lm(update.formula(rformb,z.(cumfmlmbo) ~ .), test))
  r.lfmlmbo <- residuals(lm(update.formula(lrformb,z.(lagcumfmlmbo) ~ .), test))

  baseocfmlmbo <- lmer(update.formula(bformb,.~.+ z.(r.cfmlmbo)), test, REML=F)
  baselfmlmbo <- lmer(update.formula(bformb,.~.+ z.(r.lfmlmbo)), test, REML=F)
  basebcfmlmbo <- lmer(update.formula(bformb,.~.+ z.(r.cfmlmbo) + z.(r.lfmlmbo)), test, REML=F)

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
  r.fmlmbp <- residuals(lm(update.formula(rformb,z.(fmlmbp) ~ .), test))

  baseofmlmbp <- lmer(update.formula(bformb,.~.+ z.(r.fmlmbp)), test, REML=F)

  print(anova(baseb,baseofmlmbp))
  print(baseofmlmbp)
  #baseofmlmbp.p <- pvals.fnc(baseofmlmbp)
  #print(baseofmlmbp.p)
  #baseofmlmbp.p <- NULL
  baseofmlmbp <- NULL

  print("Testing Cum F-L-B+")
  write("Testing Cum F-L-B+",stderr())
  r.cfmlmbp <- residuals(lm(update.formula(rformb,z.(cumfmlmbp) ~ .), test))
  r.lfmlmbp <- residuals(lm(update.formula(lrformb,z.(lagcumfmlmbp) ~ .), test))

  baseocfmlmbp <- lmer(update.formula(bformb,.~.+ z.(r.cfmlmbp)), test, REML=F)
  baselfmlmbp <- lmer(update.formula(bformb,.~.+ z.(r.lfmlmbp)), test, REML=F)
  basebcfmlmbp <- lmer(update.formula(bformb,.~.+ z.(r.cfmlmbp) + z.(r.lfmlmbp)), test, REML=F)

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
  r.fmlpba <- residuals(lm(update.formula(rformb,z.(fmlpba) ~ .), test))

  baseofmlpba <- lmer(update.formula(bformb,.~.+ z.(r.fmlpba)), test, REML=F)

  print(anova(baseb,baseofmlpba))
  print(baseofmlpba)
  #baseofmlpba.p <- pvals.fnc(baseofmlpba)
  #print(baseofmlpba.p)
  #baseofmlpba.p <- NULL
  baseofmlpba <- NULL

  print("Testing Cum F-L+Ba")
  write("Testing Cum F-L+Ba",stderr())
  r.cfmlpba <- residuals(lm(update.formula(rformb,z.(cumfmlpba) ~ .), test))
  r.lfmlpba <- residuals(lm(update.formula(lrformb,z.(lagcumfmlpba) ~ .), test))

  baseocfmlpba <- lmer(update.formula(bformb,.~.+ z.(r.cfmlpba)), test, REML=F)
  baselfmlpba <- lmer(update.formula(bformb,.~.+ z.(r.lfmlpba)), test, REML=F)
  basebcfmlpba <- lmer(update.formula(bformb,.~.+ z.(r.cfmlpba) + z.(r.lfmlpba)), test, REML=F)

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
  r.fmlpbc <- residuals(lm(update.formula(rformb,z.(fmlpbc) ~ .), test))

  baseofmlpbc <- lmer(update.formula(bformb,.~.+ z.(r.fmlpbc)), test, REML=F)

  print(anova(baseb,baseofmlpbc))
  print(baseofmlpbc)
  #baseofmlpbc.p <- pvals.fnc(baseofmlpbc)
  #print(baseofmlpbc.p)
  #baseofmlpbc.p <- NULL
  baseofmlpbc <- NULL

  print("Testing Cum F-L+Bc")
  write("Testing Cum F-L+Bc",stderr())
  r.cfmlpbc <- residuals(lm(update.formula(rformb,z.(cumfmlpbc) ~ .), test))
  r.lfmlpbc <- residuals(lm(update.formula(lrformb,z.(lagcumfmlpbc) ~ .), test))

  baseocfmlpbc <- lmer(update.formula(bformb,.~.+ z.(r.cfmlpbc)), test, REML=F)
  baselfmlpbc <- lmer(update.formula(bformb,.~.+ z.(r.lfmlpbc)), test, REML=F)
  basebcfmlpbc <- lmer(update.formula(bformb,.~.+ z.(r.cfmlpbc) + z.(r.lfmlpbc)), test, REML=F)

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
  r.fmlpbo <- residuals(lm(update.formula(rformb,z.(fmlpbo) ~ .), test))

  baseofmlpbo <- lmer(update.formula(bformb,.~.+ z.(r.fmlpbo)), test, REML=F)

  print(anova(baseb,baseofmlpbo))
  print(baseofmlpbo)
  #baseofmlpbo.p <- pvals.fnc(baseofmlpbo)
  #print(baseofmlpbo.p)
  #baseofmlpbo.p <- NULL
  baseofmlpbo <- NULL

  print("Testing Cum F-L+Bo")
  write("Testing Cum F-L+Bo",stderr())
  r.cfmlpbo <- residuals(lm(update.formula(rformb,z.(cumfmlpbo) ~ .), test))
  r.lfmlpbo <- residuals(lm(update.formula(lrformb,z.(lagcumfmlpbo) ~ .), test))

  baseocfmlpbo <- lmer(update.formula(bformb,.~.+ z.(r.cfmlpbo)), test, REML=F)
  baselfmlpbo <- lmer(update.formula(bformb,.~.+ z.(r.lfmlpbo)), test, REML=F)
  basebcfmlpbo <- lmer(update.formula(bformb,.~.+ z.(r.cfmlpbo) + z.(r.lfmlpbo)), test, REML=F)

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
  r.fmlpbp <- residuals(lm(update.formula(rformb,z.(fmlpbp) ~ .), test))

  baseofmlpbp <- lmer(update.formula(bformb,.~.+ z.(r.fmlpbp)), test, REML=F)

  print(anova(baseb,baseofmlpbp))
  print(baseofmlpbp)
  #baseofmlpbp.p <- pvals.fnc(baseofmlpbp)
  #print(baseofmlpbp.p)
  #baseofmlpbp.p <- NULL
  baseofmlpbp <- NULL

  print("Testing Cum F-L+B+")
  write("Testing Cum F-L+B+",stderr())
  r.cfmlpbp <- residuals(lm(update.formula(rformb,z.(cumfmlpbp) ~ .), test))
  r.lfmlpbp <- residuals(lm(update.formula(lrformb,z.(lagcumfmlpbp) ~ .), test))

  baseocfmlpbp <- lmer(update.formula(bformb,.~.+ z.(r.cfmlpbp)), test, REML=F)
  baselfmlpbp <- lmer(update.formula(bformb,.~.+ z.(r.lfmlpbp)), test, REML=F)
  basebcfmlpbp <- lmer(update.formula(bformb,.~.+ z.(r.cfmlpbp) + z.(r.lfmlpbp)), test, REML=F)

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
  r.fplmba <- residuals(lm(update.formula(rformb,z.(fplmba) ~ .), test))

  baseofplmba <- lmer(update.formula(bformb,.~.+ z.(r.fplmba)), test, REML=F)

  print(anova(baseb,baseofplmba))
  print(baseofplmba)
  #baseofplmba.p <- pvals.fnc(baseofplmba)
  #print(baseofplmba.p)
  #baseofplmba.p <- NULL
  baseofplmba <- NULL

  print("Testing Cum F+L-Ba")
  write("Testing Cum F+L-Ba",stderr())
  r.cfplmba <- residuals(lm(update.formula(rformb,z.(cumfplmba) ~ .), test))
  r.lfplmba <- residuals(lm(update.formula(lrformb,z.(lagcumfplmba) ~ .), test))

  baseocfplmba <- lmer(update.formula(bformb,.~.+ z.(r.cfplmba)), test, REML=F)
  baselfplmba <- lmer(update.formula(bformb,.~.+ z.(r.lfplmba)), test, REML=F)
  basebcfplmba <- lmer(update.formula(bformb,.~.+ z.(r.cfplmba) + z.(r.lfplmba)), test, REML=F)

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
  r.fplmbc <- residuals(lm(update.formula(rformb,z.(fplmbc) ~ .), test))

  baseofplmbc <- lmer(update.formula(bformb,.~.+ z.(r.fplmbc)), test, REML=F)

  print(anova(baseb,baseofplmbc))
  print(baseofplmbc)
  #baseofplmbc.p <- pvals.fnc(baseofplmbc)
  #print(baseofplmbc.p)
  #baseofplmbc.p <- NULL
  baseofplmbc <- NULL

  print("Testing Cum F+L-Bc")
  write("Testing Cum F+L-Bc",stderr())
  r.cfplmbc <- residuals(lm(update.formula(rformb,z.(cumfplmbc) ~ .), test))
  r.lfplmbc <- residuals(lm(update.formula(lrformb,z.(lagcumfplmbc) ~ .), test))

  baseocfplmbc <- lmer(update.formula(bformb,.~.+ z.(r.cfplmbc)), test, REML=F)
  baselfplmbc <- lmer(update.formula(bformb,.~.+ z.(r.lfplmbc)), test, REML=F)
  basebcfplmbc <- lmer(update.formula(bformb,.~.+ z.(r.cfplmbc) + z.(r.lfplmbc)), test, REML=F)

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
  r.fplmbo <- residuals(lm(update.formula(rformb,z.(fplmbo) ~ .), test))

  baseofplmbo <- lmer(update.formula(bformb,.~.+ z.(r.fplmbo)), test, REML=F)

  print(anova(baseb,baseofplmbo))
  print(baseofplmbo)
  #baseofplmbo.p <- pvals.fnc(baseofplmbo)
  #print(baseofplmbo.p)
  #baseofplmbo.p <- NULL
  baseofplmbo <- NULL

  print("Testing Cum F+L-Bo")
  write("Testing Cum F+L-Bo",stderr())
  r.cfplmbo <- residuals(lm(update.formula(rformb,z.(cumfplmbo) ~ .), test))
  r.lfplmbo <- residuals(lm(update.formula(lrformb,z.(lagcumfplmbo) ~ .), test))

  baseocfplmbo <- lmer(update.formula(bformb,.~.+ z.(r.cfplmbo)), test, REML=F)
  baselfplmbo <- lmer(update.formula(bformb,.~.+ z.(r.lfplmbo)), test, REML=F)
  basebcfplmbo <- lmer(update.formula(bformb,.~.+ z.(r.cfplmbo) + z.(r.lfplmbo)), test, REML=F)

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
  r.fplmbp <- residuals(lm(update.formula(rformb,z.(fplmbp) ~ .), test))

  baseofplmbp <- lmer(update.formula(bformb,.~.+ z.(r.fplmbp)), test, REML=F)

  print(anova(baseb,baseofplmbp))
  print(baseofplmbp)
  #baseofplmbp.p <- pvals.fnc(baseofplmbp)
  #print(baseofplmbp.p)
  #baseofplmbp.p <- NULL
  baseofplmbp <- NULL

  print("Testing Cum F+L-B+")
  write("Testing Cum F+L-B+",stderr())
  r.cfplmbp <- residuals(lm(update.formula(rformb,z.(cumfplmbp) ~ .), test))
  r.lfplmbp <- residuals(lm(update.formula(lrformb,z.(lagcumfplmbp) ~ .), test))

  baseocfplmbp <- lmer(update.formula(bformb,.~.+ z.(r.cfplmbp)), test, REML=F)
  baselfplmbp <- lmer(update.formula(bformb,.~.+ z.(r.lfplmbp)), test, REML=F)
  basebcfplmbp <- lmer(update.formula(bformb,.~.+ z.(r.cfplmbp) + z.(r.lfplmbp)), test, REML=F)

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
  r.fplpba <- residuals(lm(update.formula(rformb,z.(fplpba) ~ .), test))

  baseofplpba <- lmer(update.formula(bformb,.~.+ z.(r.fplpba)), test, REML=F)

  print(anova(baseb,baseofplpba))
  print(baseofplpba)
  #baseofplpba.p <- pvals.fnc(baseofplpba)
  #print(baseofplpba.p)
  #baseofplpba.p <- NULL
  baseofplpba <- NULL

  print("Testing Cum F+L+Ba")
  write("Testing Cum F+L+Ba",stderr())
  r.cfplpba <- residuals(lm(update.formula(rformb,z.(cumfplpba) ~ .), test))
  r.lfplpba <- residuals(lm(update.formula(lrformb,z.(lagcumfplpba) ~ .), test))

  baseocfplpba <- lmer(update.formula(bformb,.~.+ z.(r.cfplpba)), test, REML=F)
  baselfplpba <- lmer(update.formula(bformb,.~.+ z.(r.lfplpba)), test, REML=F)
  basebcfplpba <- lmer(update.formula(bformb,.~.+ z.(r.cfplpba) + z.(r.lfplpba)), test, REML=F)

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
  r.fplpbc <- residuals(lm(update.formula(rformb,z.(fplpbc) ~ .), test))

  baseofplpbc <- lmer(update.formula(bformb,.~.+ z.(r.fplpbc)), test, REML=F)

  print(anova(baseb,baseofplpbc))
  print(baseofplpbc)
  #baseofplpbc.p <- pvals.fnc(baseofplpbc)
  #print(baseofplpbc.p)
  #baseofplpbc.p <- NULL
  baseofplpbc <- NULL

  print("Testing Cum F+L+Bc")
  write("Testing Cum F+L+Bc",stderr())
  r.cfplpbc <- residuals(lm(update.formula(rformb,z.(cumfplpbc) ~ .), test))
  r.lfplpbc <- residuals(lm(update.formula(lrformb,z.(lagcumfplpbc) ~ .), test))

  baseocfplpbc <- lmer(update.formula(bformb,.~.+ z.(r.cfplpbc)), test, REML=F)
  baselfplpbc <- lmer(update.formula(bformb,.~.+ z.(r.lfplpbc)), test, REML=F)
  basebcfplpbc <- lmer(update.formula(bformb,.~.+ z.(r.cfplpbc) + z.(r.lfplpbc)), test, REML=F)

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
  r.fplpbo <- residuals(lm(update.formula(rformb,z.(fplpbo) ~ .), test))

  baseofplpbo <- lmer(update.formula(bformb,.~.+ z.(r.fplpbo)), test, REML=F)

  print(anova(baseb,baseofplpbo))
  print(baseofplpbo)
  #baseofplpbo.p <- pvals.fnc(baseofplpbo)
  #print(baseofplpbo.p)
  #baseofplpbo.p <- NULL
  baseofplpbo <- NULL

  print("Testing Cum F+L+Bo")
  write("Testing Cum F+L+Bo",stderr())
  r.cfplpbo <- residuals(lm(update.formula(rformb,z.(cumfplpbo) ~ .), test))
  r.lfplpbo <- residuals(lm(update.formula(lrformb,z.(lagcumfplpbo) ~ .), test))

  baseocfplpbo <- lmer(update.formula(bformb,.~.+ z.(r.cfplpbo)), test, REML=F)
  baselfplpbo <- lmer(update.formula(bformb,.~.+ z.(r.lfplpbo)), test, REML=F)
  basebcfplpbo <- lmer(update.formula(bformb,.~.+ z.(r.cfplpbo) + z.(r.lfplpbo)), test, REML=F)

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
  r.fplpbp <- residuals(lm(update.formula(rformb,z.(fplpbp) ~ .), test))

  baseofplpbp <- lmer(update.formula(bformb,.~.+ z.(r.fplpbp)), test, REML=F)

  print(anova(baseb,baseofplpbp))
  print(baseofplpbp)
  #baseofplpbp.p <- pvals.fnc(baseofplpbp)
  #print(baseofplpbp.p)
  #baseofplpbp.p <- NULL
  baseofplpbp <- NULL

  print("Testing Cum F+L+B+")
  write("Testing Cum F+L+B+",stderr())
  r.cfplpbp <- residuals(lm(update.formula(rformb,z.(cumfplpbp) ~ .), test))
  r.lfplpbp <- residuals(lm(update.formula(lrformb,z.(lagcumfplpbp) ~ .), test))

  baseocfplpbp <- lmer(update.formula(bformb,.~.+ z.(r.cfplpbp)), test, REML=F)
  baselfplpbp <- lmer(update.formula(bformb,.~.+ z.(r.lfplpbp)), test, REML=F)
  basebcfplpbp <- lmer(update.formula(bformb,.~.+ z.(r.cfplpbp) + z.(r.lfplpbp)), test, REML=F)

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
