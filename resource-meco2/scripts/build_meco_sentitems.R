# Rscript build_meco_sentitems.R /fs/project/schuler.77/meco/joint_fix_trimmed.rda ee
# Language codes (wave 1): "du" "ee" "en" "fi" "ge" "gr" "he" "it" "ko" "no" "ru" "sp" "tr"
# Language codes (wave 2): "ba" "bp" "da" "en_uk" "ge_po" "ge_zu" "hi_iiith" "hi_iitk" "ic" "no" "ru_mo" "se" "sp_ch" "tr"
args <- commandArgs(trailingOnly=TRUE)
df <- get(load(args[1]))
df <- df[df$lang==args[2],]
df <- df[!is.na(df$trialid) & !is.na(df$sentnum) & !is.na(df$sent.word) & !is.na(df$word),]
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

} else if (args[2] == "fi") {  # Finnish
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

} else if (args[2] == "gr") {  # Greek
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

} else if (args[2] == "it") {  # Italian
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

} else if (args[2] == "ru") {  # Russian
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

} else if (args[2] == "tr") {  # Turkish
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

} else if (args[2] == "en_uk") {  # English
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

} else if (args[2] == "ge_po" | args[2] == "ge_zu") {  # German
  for (id in c(7:21)) {
    df[(df$trialid == 5 & df$sentnum == 3 & df$sent.word == id),]$sent.word <- id-1
  }
  for (id in c(2:26)) {
    df[(df$trialid == 7 & df$sentnum == 7 & df$sent.word == id),]$sent.word <- id-1
  }

} else if (args[2] == "ba") {  # Basque
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

} else if (args[2] == "bp") {  # Brazilian Portugese
  for (id in c(2:17)) {
    df[(df$trialid == 7 & df$sentnum == 4 & df$sent.word == id),]$sent.word <- id-1
  }
  for (id in c(2:22)) {
    df[(df$trialid == 7 & df$sentnum == 8 & df$sent.word == id),]$sent.word <- id-1
  }

} else if (args[2] == "ic") {  # Icelandic
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

idx <- unique(df[,c("trialid", "sentnum", "sent.word", "word")])

# adding words unattested in the eye-tracking data
if (args[2] == "du") {  # Dutch
  idx[nrow(idx) + 1,] <- list(3, 5, 14, "–")
} else if (args[2] == "it") {  # Italian
  idx[nrow(idx) + 1,] <- list(3, 8, 29, "da")
} else if (args[2] == "ko") {  # Korean
  idx[nrow(idx) + 1,] <- list(6, 7, 19, "한다.")
  idx[nrow(idx) + 1,] <- list(7, 5, 13, "더")
  idx[nrow(idx) + 1,] <- list(8, 4, 5, "의")
  idx[nrow(idx) + 1,] <- list(11, 5, 12, "로")
} else if (args[2] == "tr" & !grepl("wave2", args[1])) {  # Turkish (wave 1)
  idx[nrow(idx) + 1,] <- list(9, 3, 13, "de")
} else if (args[2] == "no" & grepl("wave2", args[1])) {  # Norwegian (wave 2)
  idx[nrow(idx) + 1,] <- list(7, 8, 10, "-")
} else if (args[2] == "tr" & grepl("wave2", args[1])) {  # Turkish (wave 2)
  idx[nrow(idx) + 1,] <- list(4, 8, 6, "ve")
  idx[nrow(idx) + 1,] <- list(5, 8, 4, "da")
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
