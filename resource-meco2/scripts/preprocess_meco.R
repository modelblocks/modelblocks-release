# Rscript preprocess_meco.R /fs/project/schuler.77/meco/joint_fix_trimmed.rda /fs/project/schuler.77/meco/joint_l1_acc_breakdown.rda en
# Language codes (wave 1): "du" "ee" "en" "fi" "ge" "gr" "he" "it" "ko" "no" "ru" "sp" "tr"
# Language codes (wave 2): "ba" "bp" "da" "en_uk" "ge_po" "ge_zu" "hi_iiith" "hi_iitk" "ic" "no" "ru_mo" "se" "sp_ch" "tr"
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

  df[(df$word == "Apelsinimahl" & df$trialid == 8 & df$sentnum == 2 & df$sent.word == 2),]$sent.word <- 1
  df[(df$word == "on" & df$trialid == 8 & df$sentnum == 2 & df$sent.word == 3),]$sent.word <- 2
  df[(df$word == "pressitud" & df$trialid == 8 & df$sentnum == 2 & df$sent.word == 4),]$sent.word <- 3
  df[(df$word == "kas" & df$trialid == 8 & df$sentnum == 2 & df$sent.word == 5),]$sent.word <- 4
  df[(df$word == "koos" & df$trialid == 8 & df$sentnum == 2 & df$sent.word == 6),]$sent.word <- 5
  df[(df$word == "koorega" & df$trialid == 8 & df$sentnum == 2 & df$sent.word == 7),]$sent.word <- 6
  df[(df$word == "purustatud" & df$trialid == 8 & df$sentnum == 2 & df$sent.word == 8),]$sent.word <- 7
  df[(df$word == "tervetest" & df$trialid == 8 & df$sentnum == 2 & df$sent.word == 9),]$sent.word <- 8
  df[(df$word == "apelsinidest" & df$trialid == 8 & df$sentnum == 2 & df$sent.word == 10),]$sent.word <- 9
  df[(df$word == "tööstusliku" & df$trialid == 8 & df$sentnum == 2 & df$sent.word == 11),]$sent.word <- 10
  df[(df$word == "apelsinimahla" & df$trialid == 8 & df$sentnum == 2 & df$sent.word == 12),]$sent.word <- 11
  df[(df$word == "kontsentraadi" & df$trialid == 8 & df$sentnum == 2 & df$sent.word == 13),]$sent.word <- 12
  df[(df$word == "tarbeks" & df$trialid == 8 & df$sentnum == 2 & df$sent.word == 14),]$sent.word <- 13
  df[(df$word == "või" & df$trialid == 8 & df$sentnum == 2 & df$sent.word == 15),]$sent.word <- 14
  df[(df$word == "pooleks" & df$trialid == 8 & df$sentnum == 2 & df$sent.word == 16),]$sent.word <- 15
  df[(df$word == "lõigatud" & df$trialid == 8 & df$sentnum == 2 & df$sent.word == 17),]$sent.word <- 16
  df[(df$word == "apelsiniviljast" & df$trialid == 8 & df$sentnum == 2 & df$sent.word == 18),]$sent.word <- 17
  df[(df$word == "koheseks" & df$trialid == 8 & df$sentnum == 2 & df$sent.word == 19),]$sent.word <- 18
  df[(df$word == "tarbimiseks." & df$trialid == 8 & df$sentnum == 2 & df$sent.word == 20),]$sent.word <- 19

  df[(df$word == "Apelsinimahl" & df$trialid == 8 & df$sentnum == 3 & df$sent.word == 2),]$sent.word <- 1
  df[(df$word == "sisaldab" & df$trialid == 8 & df$sentnum == 3 & df$sent.word == 3),]$sent.word <- 2
  df[(df$word == "C-" & df$trialid == 8 & df$sentnum == 3 & df$sent.word == 4),]$sent.word <- 3
  df[(df$word == "vitamiini," & df$trialid == 8 & df$sentnum == 3 & df$sent.word == 5),]$sent.word <- 4
  df[(df$word == "kaaliumi," & df$trialid == 8 & df$sentnum == 3 & df$sent.word == 6),]$sent.word <- 5
  df[(df$word == "tiamiini," & df$trialid == 8 & df$sentnum == 3 & df$sent.word == 7),]$sent.word <- 6
  df[(df$word == "fosforit," & df$trialid == 8 & df$sentnum == 3 & df$sent.word == 8),]$sent.word <- 7
  df[(df$word == "foolhapet" & df$trialid == 8 & df$sentnum == 3 & df$sent.word == 9),]$sent.word <- 8
  df[(df$word == "ja" & df$trialid == 8 & df$sentnum == 3 & df$sent.word == 10),]$sent.word <- 9
  df[(df$word == "B-" & df$trialid == 8 & df$sentnum == 3 & df$sent.word == 11),]$sent.word <- 10
  df[(df$word == "vitamiini." & df$trialid == 8 & df$sentnum == 3 & df$sent.word == 12),]$sent.word <- 11

} else if (args[3] == "fi") {  # Finnish
  # Parhaiten organisaatio tunnetaan julkaisemastaan ”uhanalaisten lajien punaisesta listasta”, joka määrittää maailman lajien suojelutilanteen.
  df[(df$word == "Parhaiten" & df$trialid == 11 & df$sentnum == 7 & df$sent.word == 2),]$sent.word <- 1
  df[(df$word == "organisaatio" & df$trialid == 11 & df$sentnum == 7 & df$sent.word == 3),]$sent.word <- 2
  df[(df$word == "tunnetaan" & df$trialid == 11 & df$sentnum == 7 & df$sent.word == 4),]$sent.word <- 3
  df[(df$word == "julkaisemastaan" & df$trialid == 11 & df$sentnum == 7 & df$sent.word == 5),]$sent.word <- 4
  df[(df$word == "”uhanalaisten" & df$trialid == 11 & df$sentnum == 7 & df$sent.word == 6),]$sent.word <- 5
  df[(df$word == "lajien" & df$trialid == 11 & df$sentnum == 7 & df$sent.word == 7),]$sent.word <- 6
  df[(df$word == "punaisesta" & df$trialid == 11 & df$sentnum == 7 & df$sent.word == 8),]$sent.word <- 7
  df[(df$word == "listasta”," & df$trialid == 11 & df$sentnum == 7 & df$sent.word == 9),]$sent.word <- 8
  df[(df$word == "joka" & df$trialid == 11 & df$sentnum == 7 & df$sent.word == 10),]$sent.word <- 9
  df[(df$word == "määrittää" & df$trialid == 11 & df$sentnum == 7 & df$sent.word == 11),]$sent.word <- 10
  df[(df$word == "maailman" & df$trialid == 11 & df$sentnum == 7 & df$sent.word == 12),]$sent.word <- 11
  df[(df$word == "lajien" & df$trialid == 11 & df$sentnum == 7 & df$sent.word == 13),]$sent.word <- 12
  df[(df$word == "suojelutilanteen." & df$trialid == 11 & df$sentnum == 7 & df$sent.word == 14),]$sent.word <- 13

  df[(df$trialid == 7 & df$sentnum == 3 & df$sent.word == 17),]$sent.word <- 16

} else if (args[3] == "gr") {  # Greek
  df[(df$trialid == 7 & df$sentnum == 1 & df$sent.word == 8),]$sent.word <- 7
  df[(df$trialid == 7 & df$sentnum == 1 & df$sent.word == 9),]$sent.word <- 8

  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 4),]$sent.word <- 3
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 5),]$sent.word <- 4
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 6),]$sent.word <- 5
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 7),]$sent.word <- 6
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 8),]$sent.word <- 7
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 9),]$sent.word <- 8
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 10),]$sent.word <- 9
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 11),]$sent.word <- 10
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 12),]$sent.word <- 11
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 13),]$sent.word <- 12
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 14),]$sent.word <- 13
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 15),]$sent.word <- 14
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 16),]$sent.word <- 15
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 17),]$sent.word <- 16
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 18),]$sent.word <- 17
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 19),]$sent.word <- 18
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 20),]$sent.word <- 19
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 21),]$sent.word <- 20
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 22),]$sent.word <- 21
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 23),]$sent.word <- 22
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 24),]$sent.word <- 23
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 25),]$sent.word <- 24
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 26),]$sent.word <- 25
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 27),]$sent.word <- 26
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 28),]$sent.word <- 27
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 29),]$sent.word <- 28
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 30),]$sent.word <- 29
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 31),]$sent.word <- 30
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 32),]$sent.word <- 31
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 33),]$sent.word <- 32
  df[(df$trialid == 11 & df$sentnum == 7 & df$sent.word == 34),]$sent.word <- 33

} else if (args[3] == "it") {  # Italian
  df[(df$trialid == 9 & df$sentnum == 7 & df$sent.word == 16),]$sent.word <- 15
  df[(df$trialid == 9 & df$sentnum == 7 & df$sent.word == 17),]$sent.word <- 16
  df[(df$trialid == 9 & df$sentnum == 7 & df$sent.word == 18),]$sent.word <- 17
  df[(df$trialid == 9 & df$sentnum == 7 & df$sent.word == 19),]$sent.word <- 18
  df[(df$trialid == 9 & df$sentnum == 7 & df$sent.word == 20),]$sent.word <- 19
  df[(df$trialid == 9 & df$sentnum == 7 & df$sent.word == 21),]$sent.word <- 20
  df[(df$trialid == 9 & df$sentnum == 7 & df$sent.word == 22),]$sent.word <- 21
  df[(df$trialid == 9 & df$sentnum == 7 & df$sent.word == 23),]$sent.word <- 22
  df[(df$trialid == 9 & df$sentnum == 7 & df$sent.word == 24),]$sent.word <- 23
  df[(df$trialid == 9 & df$sentnum == 7 & df$sent.word == 25),]$sent.word <- 24
  df[(df$trialid == 9 & df$sentnum == 7 & df$sent.word == 26),]$sent.word <- 25
  df[(df$trialid == 9 & df$sentnum == 7 & df$sent.word == 27),]$sent.word <- 26
  df[(df$trialid == 9 & df$sentnum == 7 & df$sent.word == 28),]$sent.word <- 27
  df[(df$trialid == 9 & df$sentnum == 7 & df$sent.word == 29),]$sent.word <- 28
  df[(df$trialid == 9 & df$sentnum == 7 & df$sent.word == 30),]$sent.word <- 29
  df[(df$trialid == 9 & df$sentnum == 7 & df$sent.word == 31),]$sent.word <- 30
  df[(df$trialid == 9 & df$sentnum == 7 & df$sent.word == 32),]$sent.word <- 31

} else if (args[3] == "ru") {  # Russian
  for (id in c("Регистрационный номе",
               "Все страны требуют р",
               "Обязательны ли они д",
               "В некоторых странах ",
               "Франция была первой ",
               "Номерные знаки в нач",
               "Стандартизация знако")) {
    df[(df$trialid == 11 & df$sent == id),]$trialid <- 12
  }
  
  for (id in c("Международный союз о",
               "Она занимается сборо",
               "Ее миссия — «влиять,",
               "В последние десятиле",
               "В отличие от многих ",
               "Вместо этого организ",
               "Организация известна",
               "В настоящее время ор")) {
    df[(df$trialid == 10 & df$sent == id),]$trialid <- 11
  }

  for (id in c("Национальный флаг — ",
               "Национальный флаг об",
               "При разработке нацио",
               "Исторически флаги ис",
               "Только в начале семн",
               "Национальный флаг, к",
               "Все национальные фла",
               "Хотя пропорции флаго",
               "Флаги Швейцарии и Ва",
               "Наиболее популярные ",
               "Несмотря на то, что ")) {
    df[(df$trialid == 9 & df$sent == id),]$trialid <- 10
  }

  for (id in c("Пчеловодство — отрас",
               "Пчеловодство было из",
               "Более четырех тысяч ",
               "Принципы пчеловодств",
               "В наши дни считается",
               "Действительно, недав")) {
    df[(df$trialid == 8 & df$sent == id),]$trialid <- 9
  }

  for (id in c("Апельсиновый сок — э",
               "Апельсиновый сок с д",
               "Этот процесс влияет ",
               "Более того, некоторы",
               "Сложно сказать, наск",
               "Также необходимо учи",
               "Кроме того, высокая ")) {
    df[(df$trialid == 7 & df$sent == id),]$trialid <- 8
  }

  for (id in c("Дегустация — это сен",
               "Хотя дегустация вина",
               "Современные професси",
               "В последние годы поя",
               "Например, согласно р",
               "Когда дегустаторам п",
               "Другие исследования ",
               "Таким образом, для о",
               "Дегустация вслепую т")) {
    df[(df$trialid == 6 & df$sent == id),]$trialid <- 7
  }

  for (id in c("Монокль — это оптиче",
               "Он состоит из одиноч",
               "Оправа может быть ос",
               "Монокль вошёл в моду",
               "Монокль, как правило",
               "Несмотря на широкую ",
               "В значительной степе",
               "В результате совреме")) {
    df[(df$trialid == 5 & df$sent == id),]$trialid <- 6
  }

  for (id in c("Всемирный день окруж",
               "Организация Объедине",
               "Впервые эту дату отм",
               "Праздник поддерживаю",
               "В этом событии ежего",
               "К примеру, лозунг пр",
               "Задачей этой кампани",
               "Особое внимание удел",
               "Кампания принесла оп")) {
    df[(df$trialid == 4 & df$sent == id),]$trialid <- 5
  }

} else if (args[3] == "tr") {  # Turkish
  # Nar ekşisi isminden de anlaşıldığı gibi mayhoş bir tada sahiptir.
  df[(df$word == "Nar" & df$trialid == 8 & df$sentnum == 3 & df$sent.word == 2),]$sent.word <- 1
  df[(df$word == "ekşisi" & df$trialid == 8 & df$sentnum == 3 & df$sent.word == 3),]$sent.word <- 2
  df[(df$word == "isminden" & df$trialid == 8 & df$sentnum == 3 & df$sent.word == 4),]$sent.word <- 3
  df[(df$word == "de" & df$trialid == 8 & df$sentnum == 3 & df$sent.word == 5),]$sent.word <- 4
  df[(df$word == "anlaşıldığı" & df$trialid == 8 & df$sentnum == 3 & df$sent.word == 6),]$sent.word <- 5
  df[(df$word == "gibi" & df$trialid == 8 & df$sentnum == 3 & df$sent.word == 7),]$sent.word <- 6
  df[(df$word == "mayhoş" & df$trialid == 8 & df$sentnum == 3 & df$sent.word == 8),]$sent.word <- 7
  df[(df$word == "bir" & df$trialid == 8 & df$sentnum == 3 & df$sent.word == 9),]$sent.word <- 8
  df[(df$word == "tada" & df$trialid == 8 & df$sentnum == 3 & df$sent.word == 10),]$sent.word <- 9
  df[(df$word == "sahiptir." & df$trialid == 8 & df$sentnum == 3 & df$sent.word == 11),]$sent.word <- 10

} else if (args[3] == "en_uk") {  # English
  for (id in c("A vehicle registrati",
               "All countries requir",
               "Whether they are req",
               "The registration ide",
               "In some countries, t",
               "France was the first",
               "Early twentieth cent",
               "Standardization of p")) {
    df[(df$trialid == 11 & df$sent == id),]$trialid <- 12
  }

  for (id in c("The International Un",
               "It is involved in da",
               'Its mission is to "i',
               "Over the past decade",
               "Unlike many other in",
               "Instead, the organiz",
               "The organization is ",
               "Today, the organizat")) {
    df[(df$trialid == 10 & df$sent == id),]$trialid <- 11
  }

  for (id in c("A national flag is a",
               "The national flag is",
               "A national flag is d",
               "Historically, flags ",
               "The practice of flyi",
               "A country's constitu",
               "All national flags a",
               "The ratios of height",
               "The flags of Switzer",
               "The most popular col",
               "Although the nationa")) {
    df[(df$trialid == 9 & df$sent == id),]$trialid <- 10
  }

} else if (args[3] == "ge_po" | args[3] == "ge_zu") {  # German
  for (id in c(7:21)) {
    df[(df$trialid == 5 & df$sentnum == 3 & df$sent.word == id),]$sent.word <- id-1
  }
  for (id in c(2:26)) {
    df[(df$trialid == 7 & df$sentnum == 7 & df$sent.word == id),]$sent.word <- id-1
  }

} else if (args[3] == "ba") {  # Basque
  for (id in c(2:31)) {
    df[(df$trialid == 5 & df$sentnum == 2 & df$sent.word == id),]$sent.word <- id-1
  }
  for (id in c(2:37)) {
    df[(df$trialid == 7 & df$sentnum == 5 & df$sent.word == id),]$sent.word <- id-1
  }

  for (id in c("Ibilgailuen erregist",
               "Herrialde orok errep",
               "Beste ibilgailu batz",
               "Erregistroaren ident",
               "Herrialde batzuetan ",
               "Frantzia izan zen ma",
               "Hogeigarren mende ha",
               "Berrogeita hamargarr")) {
    df[(df$trialid == 11 & df$sent == id),]$trialid <- 12
  }

} else if (args[3] == "bp") {  # Brazilian Portugese
  for (id in c(2:17)) {
    df[(df$trialid == 7 & df$sentnum == 4 & df$sent.word == id),]$sent.word <- id-1
  }
  for (id in c(2:22)) {
    df[(df$trialid == 7 & df$sentnum == 8 & df$sent.word == id),]$sent.word <- id-1
  }

} else if (args[3] == "ic") {  # Icelandic
  for (id in c("Skráningarnúmeraplat",
               "Öll lönd krefjast þe",
               "Hvort slíkra spjalda",
               "Auðkennið er samansa",
               "Í sumum löndum er au",
               "Fyrstu númeraplöturn",
               "Númeraplötur við upp",
               "Staðlaðar númeraplöt")) {
    df[(df$trialid == 11 & df$sent == id),]$trialid <- 12
  }
  for (id in c("Alþjóðanáttúruvernda",
               " Í starfi samtakannn",
               "Markmið samtakanna e",
               "Á undanförnum áratug",
               "Ólíkt öðrum alþjóðle",
               "Samtökin eru best þe",
               "Í dag eru um þúsund ")) {
    df[(df$trialid == 10 & df$sent == id),]$trialid <- 11
  }
  for (id in c("Þjóðfáni er fáni sem",
               "Vanvirðing við þjóðf",
               "Engir tveir þjóðfána",
               "Dæmi um þetta eru þj",
               "Að grænlenska fánanu",
               "Íslenski, færeyski o",
               "Grænlenski fáninn er",
               "Efri hluti hringsins",
               "Þegar Jörundur hunda",
               "Fáninn var blár með ",
               "Þróun íslenska fánan",
               "Einna vinsælust var ",
               "Þessi fáni fékk heit",
               "Þessum fána var flag",
               "Að lokum samþykkti k",
               "Niðurstaðan varð núv")) {
    df[(df$trialid == 9 & df$sent == id),]$trialid <- 10
  }
  for (id in c("Býflugnarækt er huna",
               "Hér á landi á búskap",
               "Hérlendis eru um átt",
               "Í mörgum löndum er h",
               "Það eru til ótal ger",
               "Þetta skýrist af efn",
               "Þeim sem ekki þekkja")) {
    df[(df$trialid == 8 & df$sent == id),]$trialid <- 9
  }
  for (id in c("Ávaxtasafi er ávaxta",
               "Í hreinum ávaxtasafa",
               "Þegar um er að ræða ",
               "Einnig er oft bætt v",
               "Ávaxtasafi er ekki e",
               "Þar að auki er ávaxt",
               "Einnig eru til svoka",
               "Í honum er því helmi",
               "Þó eru nektarsafar h")) {
    df[(df$trialid == 7 & df$sent == id),]$trialid <- 8
  }
  for (id in c(2:16)) {
    df[(df$trialid == 11 & df$sentnum == 2 & df$sent.word == id),]$sent.word <- id-1
  }
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
