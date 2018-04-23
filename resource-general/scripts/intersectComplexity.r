#!/usr/bin/env Rscript

argoffset = 5 #this is the offset for sys.argv for running an rscript from cli

#########################
#
# Returns only the complexity lines that are shared between input files
#   Made for intersecting %.eyemodels
#
#########################

args <- commandArgs(trailingOnly=TRUE)

write(paste('Input File: ',args),stderr())
write(paste('Reading',args[1]),stderr())
data1 <- read.table(args[1],header=TRUE,quote='',comment.char='')
write(paste('Reading',args[2]),stderr())
data2 <- read.table(args[2],header=TRUE,quote='',comment.char='')
if (length(args) >= 3){
   write(paste('Reading',args[3]),stderr())
   data3 <- read.table(args[3],header=TRUE,quote='',comment.char='')
}
if (length(args) >= 4){
   write(paste('Reading',args[4]),stderr())
   data4 <- read.table(args[4],header=TRUE,quote='',comment.char='')
}
if (length(args) >= 5){
   write(paste('Reading',args[5]),stderr())
   data5 <- read.table(args[5],header=TRUE,quote='',comment.char='')
}
if (length(args) >= 6){
   write(paste('Reading',args[6]),stderr())
   data6 <- read.table(args[6],header=TRUE,quote='',comment.char='')
}

write('Filtering',stderr())
#remove any incomplete rows
data1 <- data1[complete.cases(data1),]
data2 <- data2[complete.cases(data2),]
if (length(args) >= 3){
   data3 <- data3[complete.cases(data3),]
}
if (length(args) >= 4){
   data4 <- data4[complete.cases(data4),]
}
if (length(args) >= 5){
   data5 <- data5[complete.cases(data5),]
}
if (length(args) >= 6){
   data6 <- data6[complete.cases(data6),]
}

write('Intersecting',stderr())
if (length(args) == 2){
   myvals <- intersect(unique(data2$corpusid),unique(data1$corpusid))
}
if (length(args) == 3){
   myvals <- intersect(unique(data3$corpusid),intersect(unique(data2$corpusid),unique(data1$corpusid)))
}
if (length(args) == 4){
   myvals <- intersect(unique(data4$corpusid),intersect(unique(data3$corpusid),intersect(unique(data2$corpusid),unique(data1$corpusid))))
}
if (length(args) == 5){
   myvals <- intersect(unique(data5$corpusid),intersect(unique(data4$corpusid),intersect(unique(data3$corpusid),intersect(unique(data2$corpusid),unique(data1$corpusid)))))
}
if (length(args) == 6){
   myvals <- intersect(unique(data6$corpusid),intersect(unique(data5$corpusid),intersect(unique(data4$corpusid),intersect(unique(data3$corpusid),intersect(unique(data2$corpusid),unique(data1$corpusid))))))
}

data1.new <- data1[data1$corpusid %in% myvals,]
data2.new <- data2[data2$corpusid %in% myvals,]
if (length(args) >= 3){
   data3.new <- data3[data3$corpusid %in% myvals,]
}
if (length(args) >= 4){
   data4.new <- data4[data4$corpusid %in% myvals,]
}
if (length(args) >= 5){
   data5.new <- data5[data5$corpusid %in% myvals,]
}
if (length(args) >= 6){
   data6.new <- data6[data6$corpusid %in% myvals,]
}

write(paste('data 1 dimensions: ',dim(data1.new),'/',dim(data1)),stderr())
write(paste('data 2 dimensions: ',dim(data2.new),'/',dim(data2)),stderr())
if (length(args) >= 3){
   write(paste('data 3 dimensions: ',dim(data3.new),'/',dim(data3)),stderr())
}
if (length(args) >= 4){
   write(paste('data 4 dimensions: ',dim(data4.new),'/',dim(data4)),stderr())
}
if (length(args) >= 5){
   write(paste('data 5 dimensions: ',dim(data5.new),'/',dim(data5)),stderr())
}
if (length(args) >= 6){
   write(paste('data 6 dimensions: ',dim(data6.new),'/',dim(data6)),stderr())
}
#write(paste('intersection dimensions: ',dim(data1.new)),stderr())

write('Saving data1.old',stderr())
write.table(data1,file=paste(args[1],'.old',sep=""),row.names = FALSE, quote = FALSE)
write('Saving data2.old',stderr())
write.table(data2,file=paste(args[2],'.old',sep=""),row.names = FALSE, quote = FALSE)
if (length(args) >= 3){
   write('Saving data3.old',stderr())
   write.table(data3,file=paste(args[3],'.old',sep=""),row.names = FALSE, quote = FALSE)
}
if (length(args) >= 4){
   write('Saving data4.old',stderr())
   write.table(data4,file=paste(args[4],'.old',sep=""),row.names = FALSE, quote = FALSE)
}
if (length(args) >= 5){
   write('Saving data5.old',stderr())
   write.table(data5,file=paste(args[5],'.old',sep=""),row.names = FALSE, quote = FALSE)
}
if (length(args) >= 6){
   write('Saving data6.old',stderr())
   write.table(data6,file=paste(args[6],'.old',sep=""),row.names = FALSE, quote = FALSE)
}

write('Writing data1',stderr())
write.table(data1.new,file=args[1],row.names = FALSE, quote = FALSE)
write('Writing data2',stderr())
write.table(data2.new,file=args[2],row.names = FALSE, quote = FALSE)
if (length(args) >= 3){
   write('Writing data3',stderr())
   write.table(data3.new,file=args[3],row.names = FALSE, quote = FALSE)
}
if (length(args) >= 4){
   write('Writing data4',stderr())
   write.table(data4.new,file=args[4],row.names = FALSE, quote = FALSE)
}
if (length(args) >= 5){
   write('Writing data5',stderr())
   write.table(data5.new,file=args[5],row.names = FALSE, quote = FALSE)
}
if (length(args) >= 6){
   write('Writing data6',stderr())
   write.table(data6.new,file=args[6],row.names = FALSE, quote = FALSE)
}