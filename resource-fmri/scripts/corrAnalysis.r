#!/usr/bin/Rscript

options(width=500)

df = read.csv(file('stdin'), header=TRUE, comment.char = '', quote = '', sep = ' ')
dfw = reshape(df, timevar='subject', idvar='sampleid', direction='wide')
subjs = levels(df$subject)
rois = colnames(df)[grep('^bold', colnames(df))]

cat('====================================================\n')
cat('Between-subject BOLD response correlations by region\n')
cat('====================================================\n\n')

for (r in rois) {
colnames = paste0(r, '.', subjs)
print(cor(dfw[colnames]))
cat('\n')
}

cat('===============================================================\n')
cat('Between-subject BOLD response correlations averaged over region\n')
cat('===============================================================\n\n')

responseVars = paste0(rois, '.')
for (s in subjs) {
dfw[paste0('boldMean.', s)] = rowMeans(dfw[paste0(rois, '.', s)])
}
meanResponseVars = paste0('boldMean.', subjs)
print(cor(dfw[meanResponseVars]))
cat('\n')

cat('==============================================================================\n')
cat('By-subject BOLD response correlation with mean of remaining subjects by region\n')
cat('==============================================================================\n\n')

grandx = list()
for (s in subjs) {
    grandx[[s]] = 0
}
grandmeanrho = 0
nregions = length(rois)

for (r in rois) {
meanrho = 0
samplesize = 0
colnames = paste0(r, '.', subjs)
for (i in 1:length(colnames)) {
x = as.numeric(cor(dfw[colnames[i]], rowMeans(dfw[colnames[-i]])))
grandx[[subjs[i]]] = grandx[[subjs[i]]] + x
meanrho = meanrho + x
samplesize = samplesize + 1
cat(paste0(colnames[i], ': ', x, '\n'))
}
meanrho = meanrho / samplesize
grandmeanrho = grandmeanrho + meanrho
SEM = meanrho / sqrt(samplesize)
cat(paste0('Mean: ', meanrho, '\n'))
cat(paste0('SEM: ', SEM, '\n'))
cat('\n')
}
for (s in subjs) {
grandx[[s]] = grandx[[s]] / nregions 
}

cat('=========================================================================================\n')
cat('By-subject BOLD response correlation with mean of remaining subjects averaged over region\n')
cat('=========================================================================================\n\n')

for (s in subjs) {
    cat(paste0('boldMean.', s, ': ', grandx[[s]], '\n'))
}
grandmeanrho = grandmeanrho / nregions
grandSEM = grandmeanrho / sqrt(nregions)
cat(paste0('Mean: ', grandmeanrho, '\n'))
cat(paste0('SEM: ', grandSEM, '\n'))
cat('\n')

cat('====================================================\n')
cat('Between-region BOLD response correlations by subject\n')
cat('====================================================\n\n')

for (s in subjs) {
    print(cor(dfw[paste0(rois,'.',s)]))
    cat('\n')
}

cat('=========================================\n')
cat('Between-region BOLD response correlations\n')
cat('=========================================\n\n')

print(cor(df[paste0(rois)]))
cat('\n')

cat('==================================================================\n')
cat('By-region BOLD response correlation with mean of remaining regions\n')
cat('==================================================================\n\n')

for (i in 1:length(rois)) {
    x = as.numeric(cor(df[paste0(rois[i])], rowMeans(df[paste0(rois[-i])])))
    cat(paste0(rois[i], ': ', x, '\n'))
}
cat('\n')

