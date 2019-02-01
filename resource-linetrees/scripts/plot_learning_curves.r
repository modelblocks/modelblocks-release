#!/usr/bin/env Rscript

warnings()
library(ggplot2)

cliargs = commandArgs(TRUE)

# Code for relative path from Suppressingfire on SO
initial.options = commandArgs(FALSE)
file.arg.name <- "--file="
script.name <- sub(file.arg.name, "", initial.options[grep(file.arg.name, initial.options)])
script.basename <- dirname(script.name)
setwd(script.basename)

df = read.csv(cliargs[1], header=TRUE, quote='',comment.char='', sep=' ')
df.corr = as.data.frame(matrix(nrow=nrow(df),ncol=0))

if (length(cliargs) > 1) {
  jobs_file = file(cliargs[2], open='r')
} else jobs_file = file('plot_jobs.txt', open='r')

jobs = readLines(jobs_file)

dir = gsub(' *([^ ]+)constitevaltable.txt', '\\1', cliargs[1])

skip = TRUE
for (job in jobs) {
  if (skip) {
    skip = FALSE
    next
  }
  if (substr(job, 1, 1) == '#') {
    next
  }
  write(paste0('Job: ', job), stderr())
  job = strsplit(job, '\t')[[1]]
  iv = job[1]
  ivName = job[2]
  dv = job[3]
  dvName = job[4]
  title = job[5]
  if (!(iv %in% colnames(df))) {
    write(paste0('Column ', iv, ' not in data table. Skipping plot ', title, '.'), stderr())
    next
  }
  if (!(dv %in% colnames(df))) {
    write(paste0('Column ', dv, ' not in data table. Skipping plot ', title, '.'), stderr())
    next
  }
  df.corr[[dv]] = df[[dv]]
  newplot = ggplot(df, aes(x=df[[iv]],y=df[[dv]])) + 
	    geom_line() + theme_classic() + xlab(ivName) + ylab(dvName) # + ggtitle(title)
  ggsave(filename=paste0(dir, dv, '-by-', iv, '_curve.jpg'), plot=newplot, width=5, height=3)
}

write.table(round(cor(df.corr), 2), file=paste0(dir, 'corr_matr.txt'), sep=' ', quote=FALSE)

#corr_matr_dim = ncol(df.corr)
#jpeg(paste0(dir, 'corr_matr.jpg'))
#pairs(as.formula(paste0('~',paste(colnames(df.corr), sep='', collapse='+'))), data = df.corr, main='Correlation Matrix', width=corr_matr_dim, height=corr_matr_dim)
#dev.off()

close(jobs_file)
