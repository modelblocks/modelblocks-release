# Rscript preprocess_meco.R /fs/project/schuler.77/meco/joint_fix_trimmed.rda /fs/project/schuler.77/meco/joint_l1_acc_breakdown.rda en
# Language codes (wave 1): "du" "ee" "en" "fi" "ge" "gr" "he" "it" "ko" "no" "ru" "sp" "tr"
# Language codes (wave 2): "ba" "bp" "da" "en_uk" "ge_po" "ge_zu" "hi_iiith" "hi_iitk" "ic" "no" "ru_mo" "se" "sp_ch" "tr"
# library(hash)
args <- commandArgs(trailingOnly=TRUE)
df <- get(load(args[1]))
acc_df <- get(load(args[2]))

df <- df[df$lang==args[3],]
# df <- df[df$type=="in",]
df <- df[!is.na(df$trialid) & !is.na(df$sentnum) & !is.na(df$wordnum) & !is.na(df$word) & (df$word!=""),]
acc_df <- acc_df[acc_df$lang==args[3],]
acc_df <- acc_df[c("number", "ACCURACY", "uniform_id")]

# df <- df[df$type=="in",]
# df <- df[2:100,]
# print(unique(df$blink))
# print(df[df$blink==1,])
# quit()
# print(df[1:10,])
# print(df[df$type=="out",])

# MECO needs some manual clean-up...
if (args[3] == "ee") {  # Estonian
  df[(df$uniform_id == "ee_22" & df$trialid == 6),]$trialid <- 7
  df[(df$uniform_id == "ee_22" & df$trialid == 4),]$trialid <- 5
  df[(df$uniform_id == "ee_22" & df$trialid == 3),]$trialid <- 4
  df[(df$uniform_id == "ee_22" & df$trialid == 2),]$trialid <- 3
  df[(df$uniform_id == "ee_22" & df$trialid == 1),]$trialid <- 2

  df[(df$uniform_id == "ee_9" & df$trialid == 10),]$trialid <- 11
  df[(df$uniform_id == "ee_9" & df$trialid == 9),]$trialid <- 10
  df[(df$uniform_id == "ee_9" & df$trialid == 8),]$trialid <- 9
  df[(df$uniform_id == "ee_9" & df$trialid == 7),]$trialid <- 8
  df[(df$uniform_id == "ee_9" & df$trialid == 6),]$trialid <- 7
  df[(df$uniform_id == "ee_9" & df$trialid == 5),]$trialid <- 6
  df[(df$uniform_id == "ee_9" & df$trialid == 4),]$trialid <- 5

  for (id in c(10:28)) {
    df[(df$trialid == 8 & df$wordnum == id),]$wordnum <- id-1
  }

  for (id in c(30:106)) {
    df[(df$trialid == 8 & df$wordnum == id),]$wordnum <- id-2
  }

} else if (args[3] == "fi") {  # Finnish
  for (id in c(42:135)) {
    df[(df$trialid == 7 & df$wordnum == id),]$wordnum <- id-1
  }

  for (id in c(96:120)) {
    df[(df$trialid == 11 & df$wordnum == id),]$wordnum <- id-1
  }

} else if (args[3] == "gr") {  # Greek
  for (id in c(8:202)) {
    df[(df$trialid == 7 & df$wordnum == id),]$wordnum <- id-1
  }

  for (id in c(135:181)) {
    df[(df$trialid == 11 & df$wordnum == id),]$wordnum <- id-1
  }

} else if (args[3] == "it") {  # Italian
  for (id in c(148:164)) {
    df[(df$trialid == 9 & df$wordnum == id),]$wordnum <- id-1
  }

} else if (args[3] == "ru") {  # Russian
  df[(df$uniform_id == "ru_8" & df$trialid == 11),]$trialid <- 12
  df[(df$uniform_id == "ru_8" & df$trialid == 10),]$trialid <- 11
  df[(df$uniform_id == "ru_8" & df$trialid == 9),]$trialid <- 10
  df[(df$uniform_id == "ru_8" & df$trialid == 8),]$trialid <- 9
  df[(df$uniform_id == "ru_8" & df$trialid == 7),]$trialid <- 8
  df[(df$uniform_id == "ru_8" & df$trialid == 6),]$trialid <- 7
  df[(df$uniform_id == "ru_8" & df$trialid == 5),]$trialid <- 6
  df[(df$uniform_id == "ru_8" & df$trialid == 4),]$trialid <- 5

} else if (args[3] == "tr") {  # Turkish
  for (id in c(30:126)) {
    df[(df$trialid == 8 & df$wordnum == id),]$wordnum <- id-1
  }

} else if (args[3] == "ba") {  # Basque
  for (id in c(20:161)) {
    df[(df$trialid == 5 & df$wordnum == id),]$wordnum <- id-1
  }

  for (id in c(58:150)) {
    df[(df$trialid == 7 & df$wordnum == id),]$wordnum <- id-1
  }

} else if (args[3] == "bp") {  # Brazilian Portugese
  for (id in c(69:194)) {
    df[(df$trialid == 7 & df$wordnum == id),]$wordnum <- id-1
  }

  for (id in c(196:216)) {
    df[(df$trialid == 7 & df$wordnum == id),]$wordnum <- id-2
  }

} else if (args[3] == "ge_po" | args[3] == "ge_zu") {  # German
  for (id in c(23:152)) {
    df[(df$trialid == 5 & df$wordnum == id),]$wordnum <- id-1
  }
  for (id in c(131:198)) {
    df[(df$trialid == 7 & df$wordnum == id),]$wordnum <- id-1
  }

} else if (args[3] == "ic") {  # Icelandic
  df[(df$uniform_id == "ic_6" & df$trialid == 10),]$trialid <- 11
  df[(df$uniform_id == "ic_6" & df$trialid == 9),]$trialid <- 10
  df[(df$uniform_id == "ic_6" & df$trialid == 7),]$trialid <- 8
  for (id in c(23:168)) {
    df[(df$trialid == 11 & df$wordnum == id),]$wordnum <- id-1
  }

} else if (args[3] == "en_uk") {  # English
  df[(df$uniform_id == "en_uk_46" & df$trialid == 11),]$trialid <- 12
  
  df[(df$uniform_id == "en_uk_57" & df$trialid == 11),]$trialid <- 12
  df[(df$uniform_id == "en_uk_57" & df$trialid == 10),]$trialid <- 11
  df[(df$uniform_id == "en_uk_57" & df$trialid == 9),]$trialid <- 10
}

names(acc_df)[names(acc_df) == "number"] <- "trialid"
names(acc_df)[names(acc_df) == "ACCURACY"] <- "correct"
df <- merge(df, acc_df, by=c("uniform_id", "trialid"))

# print(names(df))
#  [1] "subid"            "trialid"          "itemid"           "cond"
#  [5] "fixid"            "start"            "stop"             "xs"
#  [9] "ys"               "xn"               "yn"               "ym"
# [13] "dur"              "sac.in"           "sac.out"          "type"
# [17] "blink"            "line"             "line.change"      "line.let"
# [21] "line.word"        "letternum"        "letter"           "wordnum"
# [25] "word"             "ianum"            "ia"               "sentnum"
# [29] "sent"             "sent.nwords"      "trial.nwords"     "word.fix"
# [33] "word.run"         "word.runid"       "word.run.fix"     "word.firstskip"
# [37] "word.refix"       "word.launch"      "word.land"        "word.cland"
# [41] "word.reg.out"     "word.reg.in"      "word.reg.out.to"  "word.reg.in.from"
# [45] "ia.fix"           "ia.run"           "ia.runid"         "ia.run.fix"
# [49] "ia.firstskip"     "ia.refix"         "ia.launch"        "ia.land"
# [53] "ia.cland"         "ia.reg.out"       "ia.reg.in"        "ia.reg.out.to"
# [57] "ia.reg.in.from"   "sent.word"        "sent.fix"         "sent.run"
# [61] "sent.runid"       "sent.run.fix"     "sent.firstskip"   "sent.refix"
# [65] "sent.reg.out"     "sent.reg.in"      "sent.reg.out.to"  "sent.reg.in.from"
# [69] "lang"             "renamed_trial"    "trial"            "supplementary_id"
# [73] "uniform_id"

#          subid trialid itemid cond fixid start stop  xs  ys  xn  yn  ym dur
# 211819 macmo10       1      1    1     1     5  216 681 322 681 322 346 212
#        sac.in sac.out type blink line line.change line.let line.word letternum
# 211819     NA      35   in     0    6           0       39         7       573
#        letter wordnum word ianum  ia sentnum                 sent sent.nwords
# 211819      n     104  and   104 and       6 Janus frequently sym          28
#        trial.nwords word.fix word.run word.runid word.run.fix word.firstskip
# 211819          182        1        1          1            1              0
#        word.refix word.launch word.land word.cland word.reg.out word.reg.in
# 211819          0          NA         2          0            1           0
#        word.reg.out.to word.reg.in.from ia.fix ia.run ia.runid ia.run.fix
# 211819               2               NA      1      1        1          1
#        ia.firstskip ia.refix ia.launch ia.land ia.cland ia.reg.out ia.reg.in
# 211819            0        0        NA       2        0          1         0
#        ia.reg.out.to ia.reg.in.from sent.word sent.fix sent.run sent.runid
# 211819             2             NA        21        1        1          1
#        sent.run.fix sent.firstskip sent.refix sent.reg.out sent.reg.in
# 211819            1              0          0            1           0
#        sent.reg.out.to sent.reg.in.from lang                 renamed_trial
# 211819               1               NA   en In ancient Roman religion and
#        trial supplementary_id uniform_id
# 211819    NA          macmo10      en_10

# Step 1: text_data -- from sentitems, word-level indices
# Step 2: iteration over ET data 1, annotate startofsentence/line/file
# Step 3: iteration over text_data: update starts and ends
# Step 4: iteration over ET data 2, RT aggregated to first fixations only
# Step 5: print output

# need: word subject docid discpos discid sentpos sentid resid time wdelta prevwasfix nextwasfix startoffile endoffile
# startofscreen endofscreen startofline endofline startofsentence endofsentence blinkbeforefix blinkafterfix
# offscreenbeforefix offscreenafterfix inregression fdurSP fdurSPsummed blinkdurSPsummed blinkduringSPsummed
# fdurFP blinkdurFP blinkduringFP fdurGP blinkdurGP blinkduringGP fdurTT

# sentid starts at 0
# sentpos starts at 1
# df <- df[c("uniform_id", "trialid", "correct", "fixid", "word", "wordnum",  "sentnum", "sent.word", "line.word", "dur", "blink", "type")]
df <- df[c("uniform_id", "trialid", "correct", "word", "wordnum",  "sentnum", "firstrun.dur", "firstrun.gopast", "dur", "blink")]
df <- df[with(df, order(uniform_id, trialid, sentnum, wordnum)),]
write.table(df, file = "", append=FALSE, quote=FALSE, sep=" ", eol = "\n", na="NA", dec = ".", row.names=FALSE,
            col.names=TRUE, qmethod = c("escape", "double"), fileEncoding = "")
