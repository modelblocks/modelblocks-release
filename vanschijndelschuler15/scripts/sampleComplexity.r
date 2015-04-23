#!/usr/bin/Rscript

#########################
#
# Samples a single complexity line from each (Subject , Sentence) pairing
#   Made for sampling %.eyemodels
#
#########################

args <- commandArgs(trailingOnly=TRUE)

library("plyr")

randomRows = function(df,n){
  return(df[sample(nrow(df),n),])
}

write(paste('Input File: ',args),stderr())
write(paste('Reading',args[1]),stderr())
data1 <- read.table(args[1],header=TRUE,quote='',comment.char='')

write('Filtering',stderr())
#remove any incomplete rows
data1 <- data1[complete.cases(data1),]

# generate a full dataset in order to randomly select a single obs from each sentence
data.all <- ddply(data1,.(subject,sentid),randomRows,1)

write('Saving data1.sampled',stderr())
write.table(data.all,file=paste(args[1],'.sampled',sep=""),row.names = FALSE, quote = FALSE)
