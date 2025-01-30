# Rscript preprocess_meco.R /fs/project/schuler.77/meco/joint_fix_trimmed.rda /fs/project/schuler.77/meco/joint_l1_acc_breakdown.rda en
# Language codes: "du" "ee" "en" "fi" "ge" "gr" "he" "it" "ko" "no" "ru" "sp" "tr"
# library(hash)
args <- commandArgs(trailingOnly=TRUE)
df <- get(load(args[1]))
acc_df <- get(load(args[2]))

df <- df[df$lang==args[3],]
df <- df[df$type=="in",]
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
  df[(df$uniform_id == "ee_22" & df$renamed_trial == "Shaka märki seostatakse üldiselt Havai"),]$trialid <- 2
  df[(df$uniform_id == "ee_22" & df$renamed_trial == "Doping on spordis reeglitevastane aine,"),]$trialid <- 3
  df[(df$uniform_id == "ee_22" & df$renamed_trial == "Kukkurhunt on kiskeluviisiga kukkurloom, keda"),]$trialid <- 4
  df[(df$uniform_id == "ee_22" & df$renamed_trial == "Ülemaailmset keskkonnapäeva tähistatakse igal aastal"),]$trialid <- 5
  df[(df$uniform_id == "ee_9" & df$renamed_trial == "Ülemaailmset keskkonnapäeva tähistatakse igal aastal" & !is.na(df$renamed_trial)),]$trialid <- 5
  df[(df$uniform_id == "ee_9" & df$renamed_trial == "Monokkel on nägemise korrigeerimiseks mõeldud" & !is.na(df$renamed_trial)),]$trialid <- 6
  df[(df$uniform_id == "ee_22" & df$renamed_trial == "Veini degusteerimine on veini sensoorne"),]$trialid <- 7
  df[(df$uniform_id == "ee_9" & df$renamed_trial == "Veini degusteerimine on veini sensoorne" & !is.na(df$renamed_trial)),]$trialid <- 7
  df[(df$uniform_id == "ee_9" & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$trialid <- 8
  df[(df$uniform_id == "ee_9" & df$renamed_trial == "Mesindus on mesilaste pidamine mesindussaaduste" & !is.na(df$renamed_trial)),]$trialid <- 9
  df[(df$uniform_id == "ee_9" & df$renamed_trial == "Riigilipp on lipp, mis tähistab" & !is.na(df$renamed_trial)),]$trialid <- 10
  df[(df$uniform_id == "ee_9" & df$renamed_trial == "Rahvusvaheline Looduskaitseliit on rahvusvaheline organisatsioon," & !is.na(df$renamed_trial)),]$trialid <- 11
  df[(df$word == "vitamiini-" & df$sentnum == 5 & df$sent.word == 4 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 3
  df[(df$word == "ja" & df$sentnum == 5 & df$sent.word == 5 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 4
  df[(df$word == "mineraalainete" & df$sentnum == 5 & df$sent.word == 6 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 5
  df[(df$word == "sisaldusele" & df$sentnum == 5 & df$sent.word == 7 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 6
  df[(df$word == "on" & df$sentnum == 5 & df$sent.word == 8 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 7
  df[(df$word == "apelsinimahla" & df$sentnum == 5 & df$sent.word == 9 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 8
  df[(df$word == "tervislikkuse" & df$sentnum == 5 & df$sent.word == 10 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 9
  df[(df$word == "mõju" & df$sentnum == 5 & df$sent.word == 11 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 10
  df[(df$word == "vaieldav," & df$sentnum == 5 & df$sent.word == 12 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 11
  df[(df$word == "sest" & df$sentnum == 5 & df$sent.word == 13 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 12
  df[(df$word == "lisaks" & df$sentnum == 5 & df$sent.word == 14 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 13
  df[(df$word == "kasulikele" & df$sentnum == 5 & df$sent.word == 15 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 14
  df[(df$word == "ainetele" & df$sentnum == 5 & df$sent.word == 16 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 15
  df[(df$word == "on" & df$sentnum == 5 & df$sent.word == 17 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 16
  df[(df$word == "selles" & df$sentnum == 5 & df$sent.word == 18 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 17
  df[(df$word == "ka" & df$sentnum == 5 & df$sent.word == 19 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 18
  df[(df$word == "väga" & df$sentnum == 5 & df$sent.word == 20 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 19
  df[(df$word == "palju" & df$sentnum == 5 & df$sent.word == 21 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 20
  df[(df$word == "suhkruid," & df$sentnum == 5 & df$sent.word == 22 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 21
  df[(df$word == "mille" & df$sentnum == 5 & df$sent.word == 23 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 22
  df[(df$word == "hulk" & df$sentnum == 5 & df$sent.word == 24 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 23
  df[(df$word == "on" & df$sentnum == 5 & df$sent.word == 25 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 24
  df[(df$word == "võrreldav" & df$sentnum == 5 & df$sent.word == 26 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 25
  df[(df$word == "isegi" & df$sentnum == 5 & df$sent.word == 27 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 26
  df[(df$word == "karastusjookide" & df$sentnum == 5 & df$sent.word == 28 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 27
  df[(df$word == "omaga." & df$sentnum == 5 & df$sent.word == 29 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 28
  df[(df$word == "Apelsinimahl" & df$sentnum == 2 & df$sent.word == 2 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 1
  df[(df$word == "on" & df$sentnum == 2 & df$sent.word == 3 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 2
  df[(df$word == "pressitud" & df$sentnum == 2 & df$sent.word == 4 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 3
  df[(df$word == "kas" & df$sentnum == 2 & df$sent.word == 5 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 4
  df[(df$word == "koos" & df$sentnum == 2 & df$sent.word == 6 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 5
  df[(df$word == "koorega" & df$sentnum == 2 & df$sent.word == 7 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 6
  df[(df$word == "purustatud" & df$sentnum == 2 & df$sent.word == 8 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 7
  df[(df$word == "tervetest" & df$sentnum == 2 & df$sent.word == 9 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 8
  df[(df$word == "apelsinidest" & df$sentnum == 2 & df$sent.word == 10 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 9
  df[(df$word == "tööstusliku" & df$sentnum == 2 & df$sent.word == 11 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 10
  df[(df$word == "apelsinimahla" & df$sentnum == 2 & df$sent.word == 12 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 11
  df[(df$word == "kontsentraadi" & df$sentnum == 2 & df$sent.word == 13 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 12
  df[(df$word == "tarbeks" & df$sentnum == 2 & df$sent.word == 14 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 13
  df[(df$word == "või" & df$sentnum == 2 & df$sent.word == 15 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 14
  df[(df$word == "pooleks" & df$sentnum == 2 & df$sent.word == 16 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 15
  df[(df$word == "lõigatud" & df$sentnum == 2 & df$sent.word == 17 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 16
  df[(df$word == "apelsiniviljast" & df$sentnum == 2 & df$sent.word == 18 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 17
  df[(df$word == "koheseks" & df$sentnum == 2 & df$sent.word == 19 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 18
  df[(df$word == "tarbimiseks." & df$sentnum == 2 & df$sent.word == 20 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 19
  df[(df$word == "Apelsinimahl" & df$sentnum == 3 & df$sent.word == 2 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 1
  df[(df$word == "sisaldab" & df$sentnum == 3 & df$sent.word == 3 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 2
  df[(df$word == "C-" & df$sentnum == 3 & df$sent.word == 4 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 3
  df[(df$word == "vitamiini," & df$sentnum == 3 & df$sent.word == 5 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 4
  df[(df$word == "kaaliumi," & df$sentnum == 3 & df$sent.word == 6 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 5
  df[(df$word == "tiamiini," & df$sentnum == 3 & df$sent.word == 7 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 6
  df[(df$word == "fosforit," & df$sentnum == 3 & df$sent.word == 8 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 7
  df[(df$word == "foolhapet" & df$sentnum == 3 & df$sent.word == 9 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 8
  df[(df$word == "ja" & df$sentnum == 3 & df$sent.word == 10 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 9
  df[(df$word == "B-" & df$sentnum == 3 & df$sent.word == 11 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 10
  df[(df$word == "vitamiini." & df$sentnum == 3 & df$sent.word == 12 & df$renamed_trial == "Apelsinimahl on vedelik, mis on" & !is.na(df$renamed_trial)),]$sent.word <- 11
} else if (args[3] == "fi") {  # Finnish
  # Vaikka appelsiinimehussa onkin reilusti C-vitamiinia, teollisesti valmistettua mehua voi sokeripitoisuutensa puolesta verrata limsoihin.
  df[(df$word == "C-" & df$sentnum == 7 & df$sent.word == 6 & df$renamed_trial == "Appelsiinimehu on tuoreista appelsiineista puristamalla" & !is.na(df$renamed_trial)),]$sent.word <- 5
  df[(df$word == "vitamiinia," & df$sentnum == 7 & df$sent.word == 7 & df$renamed_trial == "Appelsiinimehu on tuoreista appelsiineista puristamalla" & !is.na(df$renamed_trial)),]$sent.word <- 6
  df[(df$word == "teollisesti" & df$sentnum == 7 & df$sent.word == 8 & df$renamed_trial == "Appelsiinimehu on tuoreista appelsiineista puristamalla" & !is.na(df$renamed_trial)),]$sent.word <- 7
  df[(df$word == "valmistettua" & df$sentnum == 7 & df$sent.word == 9 & df$renamed_trial == "Appelsiinimehu on tuoreista appelsiineista puristamalla" & !is.na(df$renamed_trial)),]$sent.word <- 8
  df[(df$word == "mehua" & df$sentnum == 7 & df$sent.word == 10 & df$renamed_trial == "Appelsiinimehu on tuoreista appelsiineista puristamalla" & !is.na(df$renamed_trial)),]$sent.word <- 9
  df[(df$word == "voi" & df$sentnum == 7 & df$sent.word == 11 & df$renamed_trial == "Appelsiinimehu on tuoreista appelsiineista puristamalla" & !is.na(df$renamed_trial)),]$sent.word <- 10
  df[(df$word == "sokeripitoisuutensa" & df$sentnum == 7 & df$sent.word == 12 & df$renamed_trial == "Appelsiinimehu on tuoreista appelsiineista puristamalla" & !is.na(df$renamed_trial)),]$sent.word <- 11
  df[(df$word == "puolesta" & df$sentnum == 7 & df$sent.word == 13 & df$renamed_trial == "Appelsiinimehu on tuoreista appelsiineista puristamalla" & !is.na(df$renamed_trial)),]$sent.word <- 12
  df[(df$word == "verrata" & df$sentnum == 7 & df$sent.word == 14 & df$renamed_trial == "Appelsiinimehu on tuoreista appelsiineista puristamalla" & !is.na(df$renamed_trial)),]$sent.word <- 13
  df[(df$word == "limsoihin." & df$sentnum == 7 & df$sent.word == 15 & df$renamed_trial == "Appelsiinimehu on tuoreista appelsiineista puristamalla" & !is.na(df$renamed_trial)),]$sent.word <- 14
  # Parhaiten organisaatio tunnetaan julkaisemastaan ”uhanalaisten lajien punaisesta listasta”, joka määrittää maailman lajien suojelutilanteen.
  df[(df$word == "Parhaiten" & df$sentnum == 7 & df$sent.word == 2 & df$renamed_trial == "Kansainvälien luonnonsuojeluliitto on luonnonsuojelun ja" & !is.na(df$renamed_trial)),]$sent.word <- 1
  df[(df$word == "organisaatio" & df$sentnum == 7 & df$sent.word == 3 & df$renamed_trial == "Kansainvälien luonnonsuojeluliitto on luonnonsuojelun ja" & !is.na(df$renamed_trial)),]$sent.word <- 2
  df[(df$word == "tunnetaan" & df$sentnum == 7 & df$sent.word == 4 & df$renamed_trial == "Kansainvälien luonnonsuojeluliitto on luonnonsuojelun ja" & !is.na(df$renamed_trial)),]$sent.word <- 3
  df[(df$word == "julkaisemastaan" & df$sentnum == 7 & df$sent.word == 5 & df$renamed_trial == "Kansainvälien luonnonsuojeluliitto on luonnonsuojelun ja" & !is.na(df$renamed_trial)),]$sent.word <- 4
  df[(df$word == "”uhanalaisten" & df$sentnum == 7 & df$sent.word == 6 & df$renamed_trial == "Kansainvälien luonnonsuojeluliitto on luonnonsuojelun ja" & !is.na(df$renamed_trial)),]$sent.word <- 5
  df[(df$word == "lajien" & df$sentnum == 7 & df$sent.word == 7 & df$renamed_trial == "Kansainvälien luonnonsuojeluliitto on luonnonsuojelun ja" & !is.na(df$renamed_trial)),]$sent.word <- 6
  df[(df$word == "punaisesta" & df$sentnum == 7 & df$sent.word == 8 & df$renamed_trial == "Kansainvälien luonnonsuojeluliitto on luonnonsuojelun ja" & !is.na(df$renamed_trial)),]$sent.word <- 7
  df[(df$word == "listasta”," & df$sentnum == 7 & df$sent.word == 9 & df$renamed_trial == "Kansainvälien luonnonsuojeluliitto on luonnonsuojelun ja" & !is.na(df$renamed_trial)),]$sent.word <- 8
  df[(df$word == "joka" & df$sentnum == 7 & df$sent.word == 10 & df$renamed_trial == "Kansainvälien luonnonsuojeluliitto on luonnonsuojelun ja" & !is.na(df$renamed_trial)),]$sent.word <- 9
  df[(df$word == "määrittää" & df$sentnum == 7 & df$sent.word == 11 & df$renamed_trial == "Kansainvälien luonnonsuojeluliitto on luonnonsuojelun ja" & !is.na(df$renamed_trial)),]$sent.word <- 10
  df[(df$word == "maailman" & df$sentnum == 7 & df$sent.word == 12 & df$renamed_trial == "Kansainvälien luonnonsuojeluliitto on luonnonsuojelun ja" & !is.na(df$renamed_trial)),]$sent.word <- 11
  df[(df$word == "lajien" & df$sentnum == 7 & df$sent.word == 13 & df$renamed_trial == "Kansainvälien luonnonsuojeluliitto on luonnonsuojelun ja" & !is.na(df$renamed_trial)),]$sent.word <- 12
  df[(df$word == "suojelutilanteen." & df$sentnum == 7 & df$sent.word == 14 & df$renamed_trial == "Kansainvälien luonnonsuojeluliitto on luonnonsuojelun ja" & !is.na(df$renamed_trial)),]$sent.word <- 13
  df[(df$sentnum == 7 & df$sent.word == 7 & df$trialid == 4 & !is.na(df$renamed_trial)),]$sent.word <- 6
  df[(df$sentnum == 7 & df$sent.word == 8 & df$trialid == 4 & !is.na(df$renamed_trial)),]$sent.word <- 7
  df[(df$sentnum == 7 & df$sent.word == 9 & df$trialid == 4 & !is.na(df$renamed_trial)),]$sent.word <- 8
  df[(df$sentnum == 7 & df$sent.word == 10 & df$trialid == 4 & !is.na(df$renamed_trial)),]$sent.word <- 9
  df[(df$sentnum == 7 & df$sent.word == 11 & df$trialid == 4 & !is.na(df$renamed_trial)),]$sent.word <- 10
  df[(df$sentnum == 7 & df$sent.word == 12 & df$trialid == 4 & !is.na(df$renamed_trial)),]$sent.word <- 11
  df[(df$sentnum == 7 & df$sent.word == 13 & df$trialid == 4 & !is.na(df$renamed_trial)),]$sent.word <- 12
  df[(df$sentnum == 7 & df$sent.word == 14 & df$trialid == 4 & !is.na(df$renamed_trial)),]$sent.word <- 13
  df[(df$sentnum == 7 & df$sent.word == 15 & df$trialid == 4 & !is.na(df$renamed_trial)),]$sent.word <- 14
  df[(df$sentnum == 7 & df$sent.word == 16 & df$trialid == 4 & !is.na(df$renamed_trial)),]$sent.word <- 15
  df[(df$sentnum == 7 & df$sent.word == 17 & df$trialid == 4 & !is.na(df$renamed_trial)),]$sent.word <- 16
  df[(df$sentnum == 7 & df$sent.word == 18 & df$trialid == 4 & !is.na(df$renamed_trial)),]$sent.word <- 17

  df[(df$sentnum == 3 & df$sent.word == 17 & df$trialid == 7 & !is.na(df$renamed_trial)),]$sent.word <- 16
  df[(df$sentnum == 4 & df$sent.word == 16 & df$trialid == 9 & !is.na(df$renamed_trial)),]$sent.word <- 15
  df[(df$sentnum == 4 & df$sent.word == 17 & df$trialid == 9 & !is.na(df$renamed_trial)),]$sent.word <- 16
} else if (args[3] == "gr") {  # Greek
  df[(df$sentnum == 1 & df$sent.word == 8 & df$trialid == 7 & !is.na(df$renamed_trial)),]$sent.word <- 7
  df[(df$sentnum == 1 & df$sent.word == 9 & df$trialid == 7 & !is.na(df$renamed_trial)),]$sent.word <- 8
  df[(df$sentnum == 7 & df$sent.word == 4 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 3
  df[(df$sentnum == 7 & df$sent.word == 5 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 4
  df[(df$sentnum == 7 & df$sent.word == 6 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 5
  df[(df$sentnum == 7 & df$sent.word == 7 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 6
  df[(df$sentnum == 7 & df$sent.word == 8 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 7
  df[(df$sentnum == 7 & df$sent.word == 9 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 8
  df[(df$sentnum == 7 & df$sent.word == 10 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 9
  df[(df$sentnum == 7 & df$sent.word == 11 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 10
  df[(df$sentnum == 7 & df$sent.word == 12 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 11
  df[(df$sentnum == 7 & df$sent.word == 13 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 12
  df[(df$sentnum == 7 & df$sent.word == 14 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 13
  df[(df$sentnum == 7 & df$sent.word == 15 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 14
  df[(df$sentnum == 7 & df$sent.word == 16 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 15
  df[(df$sentnum == 7 & df$sent.word == 17 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 16
  df[(df$sentnum == 7 & df$sent.word == 18 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 17
  df[(df$sentnum == 7 & df$sent.word == 19 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 18
  df[(df$sentnum == 7 & df$sent.word == 20 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 19
  df[(df$sentnum == 7 & df$sent.word == 21 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 20
  df[(df$sentnum == 7 & df$sent.word == 22 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 21
  df[(df$sentnum == 7 & df$sent.word == 23 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 22
  df[(df$sentnum == 7 & df$sent.word == 24 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 23
  df[(df$sentnum == 7 & df$sent.word == 25 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 24
  df[(df$sentnum == 7 & df$sent.word == 26 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 25
  df[(df$sentnum == 7 & df$sent.word == 27 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 26
  df[(df$sentnum == 7 & df$sent.word == 28 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 27
  df[(df$sentnum == 7 & df$sent.word == 29 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 28
  df[(df$sentnum == 7 & df$sent.word == 30 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 29
  df[(df$sentnum == 7 & df$sent.word == 31 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 30
  df[(df$sentnum == 7 & df$sent.word == 32 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 31
  df[(df$sentnum == 7 & df$sent.word == 33 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 32
  df[(df$sentnum == 7 & df$sent.word == 34 & df$trialid == 11 & !is.na(df$renamed_trial)),]$sent.word <- 33

} else if (args[3] == "it") {  # Italian
  df[(df$sentnum == 7 & df$sent.word == 16 & df$trialid == 9 & !is.na(df$renamed_trial)),]$sent.word <- 15
  df[(df$sentnum == 7 & df$sent.word == 17 & df$trialid == 9 & !is.na(df$renamed_trial)),]$sent.word <- 16
  df[(df$sentnum == 7 & df$sent.word == 18 & df$trialid == 9 & !is.na(df$renamed_trial)),]$sent.word <- 17
  df[(df$sentnum == 7 & df$sent.word == 19 & df$trialid == 9 & !is.na(df$renamed_trial)),]$sent.word <- 18
  df[(df$sentnum == 7 & df$sent.word == 20 & df$trialid == 9 & !is.na(df$renamed_trial)),]$sent.word <- 19
  df[(df$sentnum == 7 & df$sent.word == 21 & df$trialid == 9 & !is.na(df$renamed_trial)),]$sent.word <- 20
  df[(df$sentnum == 7 & df$sent.word == 22 & df$trialid == 9 & !is.na(df$renamed_trial)),]$sent.word <- 21
  df[(df$sentnum == 7 & df$sent.word == 23 & df$trialid == 9 & !is.na(df$renamed_trial)),]$sent.word <- 22
  df[(df$sentnum == 7 & df$sent.word == 24 & df$trialid == 9 & !is.na(df$renamed_trial)),]$sent.word <- 23
  df[(df$sentnum == 7 & df$sent.word == 25 & df$trialid == 9 & !is.na(df$renamed_trial)),]$sent.word <- 24
  df[(df$sentnum == 7 & df$sent.word == 26 & df$trialid == 9 & !is.na(df$renamed_trial)),]$sent.word <- 25
  df[(df$sentnum == 7 & df$sent.word == 27 & df$trialid == 9 & !is.na(df$renamed_trial)),]$sent.word <- 26
  df[(df$sentnum == 7 & df$sent.word == 28 & df$trialid == 9 & !is.na(df$renamed_trial)),]$sent.word <- 27
  df[(df$sentnum == 7 & df$sent.word == 29 & df$trialid == 9 & !is.na(df$renamed_trial)),]$sent.word <- 28
  df[(df$sentnum == 7 & df$sent.word == 30 & df$trialid == 9 & !is.na(df$renamed_trial)),]$sent.word <- 29
  df[(df$sentnum == 7 & df$sent.word == 31 & df$trialid == 9 & !is.na(df$renamed_trial)),]$sent.word <- 30
  df[(df$sentnum == 7 & df$sent.word == 32 & df$trialid == 9 & !is.na(df$renamed_trial)),]$sent.word <- 31

} else if (args[3] == "no") {  # Norwegian
  df[(df$sentnum == 5 & df$sent.word == 2 & df$trialid == 7 & !is.na(df$renamed_trial)),]$sent.word <- 1
  df[(df$sentnum == 5 & df$sent.word == 3 & df$trialid == 7 & !is.na(df$renamed_trial)),]$sent.word <- 2
  df[(df$sentnum == 5 & df$sent.word == 4 & df$trialid == 7 & !is.na(df$renamed_trial)),]$sent.word <- 3
  df[(df$sentnum == 5 & df$sent.word == 5 & df$trialid == 7 & !is.na(df$renamed_trial)),]$sent.word <- 4
  df[(df$sentnum == 5 & df$sent.word == 6 & df$trialid == 7 & !is.na(df$renamed_trial)),]$sent.word <- 5
  df[(df$sentnum == 5 & df$sent.word == 7 & df$trialid == 7 & !is.na(df$renamed_trial)),]$sent.word <- 6
  df[(df$sentnum == 5 & df$sent.word == 8 & df$trialid == 7 & !is.na(df$renamed_trial)),]$sent.word <- 7
  df[(df$sentnum == 5 & df$sent.word == 9 & df$trialid == 7 & !is.na(df$renamed_trial)),]$sent.word <- 8
  df[(df$sentnum == 5 & df$sent.word == 10 & df$trialid == 7 & !is.na(df$renamed_trial)),]$sent.word <- 9
  df[(df$sentnum == 5 & df$sent.word == 11 & df$trialid == 7 & !is.na(df$renamed_trial)),]$sent.word <- 10
  df[(df$sentnum == 5 & df$sent.word == 12 & df$trialid == 7 & !is.na(df$renamed_trial)),]$sent.word <- 11
  df[(df$sentnum == 5 & df$sent.word == 13 & df$trialid == 7 & !is.na(df$renamed_trial)),]$sent.word <- 12
  df[(df$sentnum == 5 & df$sent.word == 14 & df$trialid == 7 & !is.na(df$renamed_trial)),]$sent.word <- 13
  df[(df$sentnum == 5 & df$sent.word == 15 & df$trialid == 7 & !is.na(df$renamed_trial)),]$sent.word <- 14
  df[(df$sentnum == 5 & df$sent.word == 16 & df$trialid == 7 & !is.na(df$renamed_trial)),]$sent.word <- 15
  df[(df$sentnum == 5 & df$sent.word == 17 & df$trialid == 7 & !is.na(df$renamed_trial)),]$sent.word <- 16
  df[(df$sentnum == 5 & df$sent.word == 18 & df$trialid == 7 & !is.na(df$renamed_trial)),]$sent.word <- 17
  df[(df$sentnum == 5 & df$sent.word == 19 & df$trialid == 7 & !is.na(df$renamed_trial)),]$sent.word <- 18
  df[(df$sentnum == 5 & df$sent.word == 20 & df$trialid == 7 & !is.na(df$renamed_trial)),]$sent.word <- 19

} else if (args[3] == "ko") {  # Korean
  # 이 기구의 임무는 “전 세계의 사회를 대상으로 자연을 보호하고 천연자원을 공정하고 생태학적으로 지속가능하게 사용하도록 영향을 주고 장려하며 돕는 것”이다.
  df[(df$word == "“전" & df$sentnum == 3 & df$sent.word == 5 & df$renamed_trial == "국제자연보전연맹은 자연보호와 천연자원의 지속가능한 사용" & !is.na(df$renamed_trial)),]$sent.word <- 4
  df[(df$word == "세계의" & df$sentnum == 3 & df$sent.word == 6 & df$renamed_trial == "국제자연보전연맹은 자연보호와 천연자원의 지속가능한 사용" & !is.na(df$renamed_trial)),]$sent.word <- 5
  df[(df$word == "사회를" & df$sentnum == 3 & df$sent.word == 7 & df$renamed_trial == "국제자연보전연맹은 자연보호와 천연자원의 지속가능한 사용" & !is.na(df$renamed_trial)),]$sent.word <- 6
  df[(df$word == "대상으로" & df$sentnum == 3 & df$sent.word == 8 & df$renamed_trial == "국제자연보전연맹은 자연보호와 천연자원의 지속가능한 사용" & !is.na(df$renamed_trial)),]$sent.word <- 7
  df[(df$word == "자연을" & df$sentnum == 3 & df$sent.word == 9 & df$renamed_trial == "국제자연보전연맹은 자연보호와 천연자원의 지속가능한 사용" & !is.na(df$renamed_trial)),]$sent.word <- 8
  df[(df$word == "보호하고" & df$sentnum == 3 & df$sent.word == 10 & df$renamed_trial == "국제자연보전연맹은 자연보호와 천연자원의 지속가능한 사용" & !is.na(df$renamed_trial)),]$sent.word <- 9
  df[(df$word == "천연자원을" & df$sentnum == 3 & df$sent.word == 11 & df$renamed_trial == "국제자연보전연맹은 자연보호와 천연자원의 지속가능한 사용" & !is.na(df$renamed_trial)),]$sent.word <- 10
  df[(df$word == "공정하고" & df$sentnum == 3 & df$sent.word == 12 & df$renamed_trial == "국제자연보전연맹은 자연보호와 천연자원의 지속가능한 사용" & !is.na(df$renamed_trial)),]$sent.word <- 11
  df[(df$word == "생태학적으로" & df$sentnum == 3 & df$sent.word == 13 & df$renamed_trial == "국제자연보전연맹은 자연보호와 천연자원의 지속가능한 사용" & !is.na(df$renamed_trial)),]$sent.word <- 12
  df[(df$word == "지속가능하게" & df$sentnum == 3 & df$sent.word == 14 & df$renamed_trial == "국제자연보전연맹은 자연보호와 천연자원의 지속가능한 사용" & !is.na(df$renamed_trial)),]$sent.word <- 13
  df[(df$word == "사용" & df$sentnum == 3 & df$sent.word == 15 & df$renamed_trial == "국제자연보전연맹은 자연보호와 천연자원의 지속가능한 사용" & !is.na(df$renamed_trial)),]$sent.word <- 14
  df[(df$word == "하도록" & df$sentnum == 3 & df$sent.word == 16 & df$renamed_trial == "국제자연보전연맹은 자연보호와 천연자원의 지속가능한 사용" & !is.na(df$renamed_trial)),]$sent.word <- 15
  df[(df$word == "영향을" & df$sentnum == 3 & df$sent.word == 17 & df$renamed_trial == "국제자연보전연맹은 자연보호와 천연자원의 지속가능한 사용" & !is.na(df$renamed_trial)),]$sent.word <- 16
  df[(df$word == "주고" & df$sentnum == 3 & df$sent.word == 18 & df$renamed_trial == "국제자연보전연맹은 자연보호와 천연자원의 지속가능한 사용" & !is.na(df$renamed_trial)),]$sent.word <- 17
  df[(df$word == "장려하며" & df$sentnum == 3 & df$sent.word == 19 & df$renamed_trial == "국제자연보전연맹은 자연보호와 천연자원의 지속가능한 사용" & !is.na(df$renamed_trial)),]$sent.word <- 18
  df[(df$word == "돕는" & df$sentnum == 3 & df$sent.word == 20 & df$renamed_trial == "국제자연보전연맹은 자연보호와 천연자원의 지속가능한 사용" & !is.na(df$renamed_trial)),]$sent.word <- 19
  df[(df$word == "것”" & df$sentnum == 3 & df$sent.word == 21 & df$renamed_trial == "국제자연보전연맹은 자연보호와 천연자원의 지속가능한 사용" & !is.na(df$renamed_trial)),]$sent.word <- 20
  df[(df$word == "이다." & df$sentnum == 3 & df$sent.word == 22 & df$renamed_trial == "국제자연보전연맹은 자연보호와 천연자원의 지속가능한 사용" & !is.na(df$renamed_trial)),]$sent.word <- 21
} else if (args[3] == "ru") {  # Russian
  df[(df$renamed_trial == "Всемирный день окружающей среды отмечают"),]$trialid <- 5
  df[(df$renamed_trial == "Монокль — это оптический прибор"),]$trialid <- 6
  df[(df$renamed_trial == "Дегустация — это сенсорный анализ"),]$trialid <- 7
  df[(df$renamed_trial == "Апельсиновый сок — это напиток,"),]$trialid <- 8
  df[(df$renamed_trial == "Пчеловодство — отрасль сельского хозяйства,"),]$trialid <- 9
  df[(df$renamed_trial == "Национальный флаг — это флаг,"),]$trialid <- 10
  df[(df$renamed_trial == "Международный союз охраны природы —"),]$trialid <- 11
  df[(df$renamed_trial == "Регистрационный номер транспортного средства —"),]$trialid <- 12
} else if (args[3] == "tr") {  # Turkish
  # Nar ekşisi isminden de anlaşıldığı gibi mayhoş bir tada sahiptir.
  df[(df$word == "Nar" & df$sentnum == 3 & df$sent.word == 2 & df$renamed_trial == "Nar ekşisi, nar suyunun içindeki" & !is.na(df$renamed_trial)),]$sent.word <- 1
  df[(df$word == "ekşisi" & df$sentnum == 3 & df$sent.word == 3 & df$renamed_trial == "Nar ekşisi, nar suyunun içindeki" & !is.na(df$renamed_trial)),]$sent.word <- 2
  df[(df$word == "isminden" & df$sentnum == 3 & df$sent.word == 4 & df$renamed_trial == "Nar ekşisi, nar suyunun içindeki" & !is.na(df$renamed_trial)),]$sent.word <- 3
  df[(df$word == "de" & df$sentnum == 3 & df$sent.word == 5 & df$renamed_trial == "Nar ekşisi, nar suyunun içindeki" & !is.na(df$renamed_trial)),]$sent.word <- 4
  df[(df$word == "anlaşıldığı" & df$sentnum == 3 & df$sent.word == 6 & df$renamed_trial == "Nar ekşisi, nar suyunun içindeki" & !is.na(df$renamed_trial)),]$sent.word <- 5
  df[(df$word == "gibi" & df$sentnum == 3 & df$sent.word == 7 & df$renamed_trial == "Nar ekşisi, nar suyunun içindeki" & !is.na(df$renamed_trial)),]$sent.word <- 6
  df[(df$word == "mayhoş" & df$sentnum == 3 & df$sent.word == 8 & df$renamed_trial == "Nar ekşisi, nar suyunun içindeki" & !is.na(df$renamed_trial)),]$sent.word <- 7
  df[(df$word == "bir" & df$sentnum == 3 & df$sent.word == 9 & df$renamed_trial == "Nar ekşisi, nar suyunun içindeki" & !is.na(df$renamed_trial)),]$sent.word <- 8
  df[(df$word == "tada" & df$sentnum == 3 & df$sent.word == 10 & df$renamed_trial == "Nar ekşisi, nar suyunun içindeki" & !is.na(df$renamed_trial)),]$sent.word <- 9
  df[(df$word == "sahiptir." & df$sentnum == 3 & df$sent.word == 11 & df$renamed_trial == "Nar ekşisi, nar suyunun içindeki" & !is.na(df$renamed_trial)),]$sent.word <- 10
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
df <- df[c("uniform_id", "trialid", "correct", "fixid", "word", "wordnum",  "sentnum", "sent.word", "line.word", "dur", "blink", "type")]
write.table(df, file = "", append=FALSE, quote=FALSE, sep=" ", eol = "\n", na="NA", dec = ".", row.names=FALSE,
            col.names=TRUE, qmethod = c("escape", "double"), fileEncoding = "")
