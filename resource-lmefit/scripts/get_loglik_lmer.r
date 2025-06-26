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
# library(languageR)
library(optimx)
library(ggplot2)
library(mvtnorm)
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

model_data <- get(load(args[1]))
model <- model_data$model
input <- file(args[2], open='rt', raw=TRUE)
df <- read.table(input, header=TRUE, sep=' ', quote='', comment.char='') 
close(input)
df <- cleanupData(df, stdout=FALSE)
df <- recastEffects(df, stdout=FALSE)
df <- df[order(df$subject),]
subj_table <- data.frame(table(df$subject))

f <- model_data$f
y <- model.frame(paste0(toString(f[2]), '~ 1'), data=df)

n_re <- ncol(ranef(model)$subject)

bar.f <- findbars(f)
mf <- model.frame(subbars(f), data=df)
rt <- mkReTrms(bar.f,mf)

# (n_subj * RE, num_points)
full_Zt <- rt$Zt

lt <- getME(model,"Lambdat")
var_e <- sigma(model)^2
# according to Eq. 4 of Bates et al.
#print(n_re)
#quit()

if (n_re > 1) {
  var_u <- var_e * crossprod(lt[1:n_re, 1:n_re])
} else {
  var_u <- as.vector(VarCorr(model)$subject)
}
#var_u <- var_e * crossprod(lt[1:n_re, 1:n_re])
#var_u <- as.vector(VarCorr(model)$subject)

start_idx <- 0
total_ll <- 0

for (row in seq_len(nrow(subj_table))){
    curr_subj_name <- as.character(subj_table[row, 1])
    curr_subj_count <- subj_table[row, 2]
    Zt <- full_Zt[((row-1)*n_re+1):(row*n_re), (start_idx+1):(start_idx+curr_subj_count)]
    curr_data <- df[(start_idx+1):(start_idx+curr_subj_count),]
    curr_y <- y[(start_idx+1):(start_idx+curr_subj_count),]
    if (n_re > 1) {
      var_y <- (t(Zt) %*% var_u %*% Zt)
    } else {
      var_y <- (t(t(c(Zt))) %*% var_u %*% c(Zt))
    }
    #var_y <- (t(Zt) %*% var_u %*% Zt)
    #var_y <- (t(t(c(Zt))) %*% var_u %*% c(Zt))
    var_y <- var_y + var_e * Diagonal(nrow(var_y))
    xb <- predict(model, re.form=NA, newdata=curr_data)
    curr_ll <- dmvnorm(curr_y, mean=xb, sigma=as.matrix(var_y), log=TRUE)
    curr_subj_output <- paste("Log likelihood from subject", curr_subj_name, ":", curr_ll)
    message(curr_subj_output)
    total_ll <- total_ll + curr_ll
    start_idx <- start_idx+curr_subj_count
}

print(total_ll)

# n <- nrow(df)
# ldl <- getME(model, "devcomp")$cmp["ldL2"]
# u <- getME(model, "devcomp")$cmp["ussq"]
#
# y_hat <- data.frame(list(y_hat=predict(model, newdata=df, type='response', allow.new.levels=TRUE)))
# err <- y-y_hat
# sse <- sum(err^2)
# psse <- sse + u
# bad_ll <- -(ldl + n*(1+log((2 * 3.14159265 * psse)/(n))))/2
# print(bad_ll)
# print(logLik(model))
