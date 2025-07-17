# Rscript build_meco_sentitems.R /fs/project/schuler.77/meco/joint_fix_trimmed.rda ee
# Language codes: "du" "ee" "en" "fi" "ge" "gr" "he" "it" "ko" "no" "ru" "sp" "tr"
args <- commandArgs(trailingOnly=TRUE)
df <- get(load(args[1]))
df <- df[df$lang==args[2],]
# print(table(df$renamed_trial))
# quit()

# MECO needs some manual clean-up...
if (args[2] == "ee") {  # Estonian
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
} else if (args[2] == "fi") {  # Finnish
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

} else if (args[2] == "gr") {  # Greek
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

} else if (args[2] == "it") {  # Italian
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

} else if (args[2] == "no") {  # Norwegian
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

} else if (args[2] == "ko") {  # Korean
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
} else if (args[2] == "ru") {  # Russian
  df[(df$renamed_trial == "Всемирный день окружающей среды отмечают"),]$trialid <- 5
  df[(df$renamed_trial == "Монокль — это оптический прибор"),]$trialid <- 6
  df[(df$renamed_trial == "Дегустация — это сенсорный анализ"),]$trialid <- 7
  df[(df$renamed_trial == "Апельсиновый сок — это напиток,"),]$trialid <- 8
  df[(df$renamed_trial == "Пчеловодство — отрасль сельского хозяйства,"),]$trialid <- 9
  df[(df$renamed_trial == "Национальный флаг — это флаг,"),]$trialid <- 10
  df[(df$renamed_trial == "Международный союз охраны природы —"),]$trialid <- 11
  df[(df$renamed_trial == "Регистрационный номер транспортного средства —"),]$trialid <- 12
} else if (args[2] == "tr") {  # Turkish
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

idx <- unique(df[,c("trialid", "sentnum", "sent.word", "word")])

# adding words unattested in the eye-tracking data
if (args[2] == "en") {  # English
  idx[nrow(idx) + 1,] <- list(8, 6, 21, "is")
} else if (args[2] == "ge") {  # German
  idx[nrow(idx) + 1,] <- list(8, 5, 1, "Orangen-")
  idx[nrow(idx) + 1,] <- list(8, 6, 15, "und")
} else if (args[2] == "ko") {  # Korean
  idx[nrow(idx) + 1,] <- list(7, 5, 13, "더")
  idx[nrow(idx) + 1,] <- list(8, 2, 5, "는")
} else if (args[2] == "no") {  # Norwegian
  idx[nrow(idx) + 1,] <- list(2, 8, 18, "med")
  idx[nrow(idx) + 1,] <- list(2, 8, 20, "heraldiske")
  idx[nrow(idx) + 1,] <- list(2, 8, 21, "praksis.")
} else if (args[2] == "sp") {  # Spanish
  idx[nrow(idx) + 1,] <- list(8, 4, 3, "ya")
  idx[nrow(idx) + 1,] <- list(9, 4, 11, "y")
} else if (args[2] == "tr") {  # Turkish
  idx[nrow(idx) + 1,] <- list(9, 3, 13, "de")
}

idx <- idx[with(idx, order(trialid, sentnum, sent.word)),]
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
