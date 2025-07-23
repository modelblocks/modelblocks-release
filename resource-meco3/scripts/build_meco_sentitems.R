# Rscript build_meco_sentitems.R /fs/project/schuler.77/meco/joint_fix_trimmed.rda ee
# Language codes (wave 1): "du" "ee" "en" "fi" "ge" "gr" "he" "it" "ko" "no" "ru" "sp" "tr"
# Language codes (wave 2): "ba" "bp" "da" "en_uk" "ge_po" "ge_zu" "hi_iiith" "hi_iitk" "ic" "no" "ru_mo" "se" "sp_ch" "tr"
args <- commandArgs(trailingOnly=TRUE)
df <- get(load(args[1]))
df <- df[df$lang==args[2],]
df <- df[!is.na(df$trialid) & !is.na(df$sentnum) & !is.na(df$wordnum) & !is.na(df$word),]
# print(table(df$renamed_trial))
# quit()

# MECO needs some manual clean-up...
if (args[2] == "ee") {  # Estonian
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

} else if (args[2] == "ru") {  # Russian
  df[(df$uniform_id == "ru_8" & df$trialid == 11),]$trialid <- 12
  df[(df$uniform_id == "ru_8" & df$trialid == 10),]$trialid <- 11
  df[(df$uniform_id == "ru_8" & df$trialid == 9),]$trialid <- 10
  df[(df$uniform_id == "ru_8" & df$trialid == 8),]$trialid <- 9
  df[(df$uniform_id == "ru_8" & df$trialid == 7),]$trialid <- 8
  df[(df$uniform_id == "ru_8" & df$trialid == 6),]$trialid <- 7
  df[(df$uniform_id == "ru_8" & df$trialid == 5),]$trialid <- 6
  df[(df$uniform_id == "ru_8" & df$trialid == 4),]$trialid <- 5

} else if (args[2] == "en_uk") {  # English
  df[(df$uniform_id == "en_uk_46" & df$trialid == 11),]$trialid <- 12
  
  df[(df$uniform_id == "en_uk_57" & df$trialid == 11),]$trialid <- 12
  df[(df$uniform_id == "en_uk_57" & df$trialid == 10),]$trialid <- 11
  df[(df$uniform_id == "en_uk_57" & df$trialid == 9),]$trialid <- 10

} else if (args[2] == "ic") {  # Icelandic
  df[(df$uniform_id == "ic_6" & df$trialid == 10),]$trialid <- 11
  df[(df$uniform_id == "ic_6" & df$trialid == 9),]$trialid <- 10
  df[(df$uniform_id == "ic_6" & df$trialid == 7),]$trialid <- 8
}

idx <- unique(df[,c("trialid", "sentnum", "wordnum", "word")])

# adding words unattested in the eye-tracking data
# if (args[2] == "du") {  # Dutch
#   idx[nrow(idx) + 1,] <- list(3, 5, 14, "–")
# } else if (args[2] == "it") {  # Italian
#   idx[nrow(idx) + 1,] <- list(3, 8, 29, "da")
# } else if (args[2] == "ko") {  # Korean
#   idx[nrow(idx) + 1,] <- list(6, 7, 19, "한다.")
#   idx[nrow(idx) + 1,] <- list(7, 5, 13, "더")
#   idx[nrow(idx) + 1,] <- list(8, 4, 5, "의")
#   idx[nrow(idx) + 1,] <- list(11, 5, 12, "로")
# } else if (args[2] == "tr" & !grepl("wave2", args[1])) {  # Turkish (wave 1)
#   idx[nrow(idx) + 1,] <- list(9, 3, 13, "de")
# } else if (args[2] == "no" & grepl("wave2", args[1])) {  # Norwegian (wave 2)
#   idx[nrow(idx) + 1,] <- list(7, 8, 10, "-")
# } else if (args[2] == "tr" & grepl("wave2", args[1])) {  # Turkish (wave 2)
#   idx[nrow(idx) + 1,] <- list(4, 8, 6, "ve")
#   idx[nrow(idx) + 1,] <- list(5, 8, 4, "da")
# }

idx <- idx[with(idx, order(trialid, sentnum, wordnum)),]
# print(idx)

trialid <- 1
sentid <- 1
wordid <- 0
words <- list()
cat("!ARTICLE\n")

for (i in seq_len(nrow(idx))) {
  row <- idx[i,]
  # print(row)
  if (anyNA(row) | row$word=="") {
  # if (anyNA(row)) {
    next
  }
  if (row$sentnum != sentid) {
    sentid <- row$sentnum
    x <- paste(words, collapse=" ")
    cat(x)
    cat("\n")
    words <- list()
    wordid <- 0
    # words[[length(words)+1]] <- paste(row$trialid, row$sentnum, row$sent.word, row$word, sep=" ")
    words[[length(words)+1]] <- row$word
    if (row$trialid != trialid) {
      cat("!ARTICLE\n")
      trialid <- row$trialid
    }
  } else {
    # print(row$sentnum)
    # print(row$sent.word)
    # print(row$word)
    # print(wordid)
    # words[[length(words)+1]] <- paste(row$trialid, row$sentnum, row$sent.word, row$word, sep=" ")
    words[[length(words)+1]] <- row$word
  }
  stopifnot("Unattested word" = (row$sent.word == wordid + 1))
  wordid <- row$sent.word
}

x <- paste(words, collapse=" ")
cat(x)
cat("\n")
