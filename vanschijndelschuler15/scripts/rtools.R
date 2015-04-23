##########################
#
# Miscellaneous R Tools
#
# Unless otherwise specified, these tools were written by Marten van Schijndel
# They are freely available under the Gnu GPL
#
###########################

##########################
#
# Assign multiple variables at once
# Courtesy of Tommy from StackOverflow
#
# ...: variables to assign
# values: list of values to assign to those variables
#
###########################

vassign <- function(..., values, envir=parent.frame()) {
  vars <- as.character(substitute(...()))
  values <- rep(values, length.out=length(vars))
  for(i in seq_along(vars)) {
    assign(vars[[i]], values[[i]], envir)
  }
}

##########################
#
# Extract a sentence from a %.ccomplex dataframe
# ix: index of the sentence desired (sentence 1, sentence 2, etc)
#
# colname: name of the column that contains sentence position
#
###########################

grabsent <- function(data,ix,colname='sentpos'){
  sentnum <- 1
  spos <- 0
  rstart <- 0
  rend <- 0
  for (r in rownames(data)) {
    if (data[r,colname] < spos) {
      sentnum <- sentnum + 1
    }
    spos <- data[r,colname]
    if (sentnum == ix && spos == 1) {
      rstart <- r
    }
    if (sentnum > ix) {
      return(data[rstart:rend,])
    }
    rend <- r
  }
  return(data[rstart:rend,])
}

##########################
#
# Rescale a vector's range to be between 0 and 1
#
# x: a numeric vector
#
###########################

range01 <- function(x){ (x-min(x)) / (max(x)-min(x)) }

##########################
#
# Find the weakest fixed effect in an lmer model (according to t-value)
#
# model: a pre-fit lmer model
#
###########################

extractWeak <- function(model){
  summary(model)@coefs[summary(model)@coefs[,3]^2 == min(summary(model)@coefs[,3]^2),,drop=F]
}

##########################
#
# Sort fixed effects by t-value in ascending order
#
# model: a pre-fit lmer model
#
###########################

sortFixT <- function(model){
  summary(model)@coefs[sort.list(summary(model)@coefs[,3]^2),]
}

##########################
#
# Drop the weakest fixed effect from an lmer model (according to t-value)
#
# moda: a pre-fit model using formula form
# form: the formula to pare down
# rec: a vector that keeps a record of each dropped factor
# ix: the index of which factor to drop (in case multiple are equally weak)
#
# POST: Returns formula sans that factor
#
###########################

dropeff <- function(moda,form,rec,ix=1,quiet=F){
  if (!quiet)
    print("Finding weak factor")
  rn <- rownames(sortFixT(moda))
  rn <- rn[rn != "(Intercept)"] #Remove the intercept as a viable factor
  nix <- length(rn)

  #don't allow fixed effects to be removed if joint effects exist for them
  while (ix <= nix) {
    #NB: Could this be done without using a while loop?
    select <- rn[ix]
    if (length(grep(':',select,fixed=T))>0) break
    if (length(grep(select,rn[-ix],fixed=T)) > 0) {
      ix <- ix + 1
    }
    else break
  }
  if (!quiet)
    cat("Removing",select,"\n")

  return(list( update.formula(form, paste(".~.-",select)),rbind(rec,select),ix,nix-ix ))
}

##########################
#
# Wrapper for dropeff that makes the determination whether to drop the weakest factor or not
#
# form: the formula to use
# data: the dataset for the formula
# rec: a vector that keeps a record of each dropped factor
# ix: the index of which factor to drop (in case multiple are equally weak)
# alpha: p-criterion for weakness determination
#
# POST: Returns formula sans that factor (if not significant)
#
###########################

dropwrap <- function(moda,form,data,rec,ix=1,alpha=.05,quiet=F){
  vassign(formb,recb,ix,rem,values=dropeff(moda,form,rec,ix,quiet))
  if (!quiet)
    print("Fitting next model")
  modb <- lmer(formb,data,REML=F)
  ifelse (anova(moda,modb)$`Pr(>Chisq)` > alpha, {
    if (!quiet)
      print("Updating formula")
    return(list(formb,modb,recb,1,1))
  },{
    if (!quiet)
      print("Reverting formula")
    return(list(form,moda,rec,rem,ix + 1))
  })
}

##########################
#
# Drops fixed effects from an lmer formula to create the simplest fitted model
#
# form: the formula to use
# data: the dataset for the formula
# alpha: p-criterion for weakness determination
#
# POST: Returns simplest fitted version of given formula
#
###########################

fitfixguts <- function(form,data,alpha=.05,quiet=F){
  rec <- NULL
  if (!quiet)
    print("Fitting first model")
  moda <- lmer(form,data,REML=F)
  rem <- 1
  ix <- 1
  while (rem > 0) {
    vassign(form,moda,rec,rem,ix,values=dropwrap(moda,form,data,rec,ix,alpha,quiet))
  }
  return(list( form,moda,rec ))
}

##########################
#
# Wrapper for fitfixguts
#
# form: the formula to use
# data: the dataset for the formula
# alpha: p-criterion for weakness determination
#
# POST: Returns simplest fitting formula
#
###########################

fitfix <- function(form,data,alpha=.05,verbose=F){
  vassign(finalform,finalmod,fitrec,values=fitfixguts(form,data,alpha,!verbose))
  print("FitFix: Fitted Formula")
  print(finalform)
  print("FitFix: Fitted Model")
  print(finalmod)
  print("FitFix: Fitting History")
  print(fitrec)
  return(finalform)
}
