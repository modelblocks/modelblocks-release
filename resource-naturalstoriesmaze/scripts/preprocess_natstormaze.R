#!/usr/bin/env Rscript

library(plyr)
library(tidyverse)
library(readr)
library(here)

args = commandArgs(trailingOnly=TRUE)
data_path = args[[1]]

data <- read_rds(here(data_path))

data_filt <- data %>% filter(native %in% c("ENG", "English", "ENGLISH", "english")) #I peeked at what people put that semantically maps to english

data_error_summ <- data_filt %>% 
  mutate(correct.num=ifelse(correct=="yes", 1,0)) %>% 
  group_by(subject) %>%
  filter(type!="practice") %>% 
  filter(rt<5000) %>% 
  summarize(pct_correct=mean(correct.num)) %>% 
  ungroup() %>% 
  mutate(is.attentive=ifelse(pct_correct>.8, T,F)) %>% 
  select(subject, is.attentive)

data_out <- data_filt %>% 
  left_join(data_error_summ, by="subject") %>% 
  filter(type!="practice")

data_out <- data_out %>% 
  mutate(word_num_mistake=ifelse(correct=="no", word_num,NA)) %>% 
  mutate(docid=case_when(
    type == 'critical1' ~ 'Boar',
    type == 'critical2' ~ 'Aqua',
    type == 'critical3' ~ 'MatchstickSeller',
    type == 'critical4' ~ 'KingOfBirds',
    type == 'critical5' ~ 'Elvis',
    type == 'critical6' ~ 'MrSticky',
    type == 'critical7' ~ 'HighSchool',
    type == 'critical8' ~ 'Roswell',
    type == 'critical9' ~ 'Tulips',
    type == 'critical10' ~ 'Tourettes',
    TRUE ~ as.character(type)
  )) %>%
  mutate(item=as.numeric(substr(type, 9, nchar(type)))) %>%
  group_by(subject, item) %>%
  mutate(zone=row_number()) %>% ungroup() %>%
#  mutate(sentid=cumsum(replace_na(as.numeric(word_num != (lag(word_num) + 1)), 1)) - 1) %>%
#  mutate(sentpos=word_num + 1) %>%
  mutate(is.attentive=as.numeric(is.attentive)) %>%
  group_by(sentence, subject) %>% fill(word_num_mistake) %>% ungroup() %>% 
  mutate(sentposmistake=word_num_mistake + 1) %>%
  group_by(subject, docid) %>% mutate(time=cumsum(total_rt) / 1000) %>% ungroup() %>%
  mutate(after_mistake=word_num-word_num_mistake,
         after_mistake=ifelse(is.na(after_mistake),0,after_mistake)) %>%
  rename(onright=on_right, totalrt=total_rt, isattentive=is.attentive, aftermistake=after_mistake) %>%
  mutate(subject=paste0('s', as.character(sprintf('%03d', subject)))) %>%
  select(subject, docid, item, zone, time, word, distractor, onright, correct, rt, totalrt, isattentive, sentposmistake, aftermistake) %>%
  # select(!c(sentence, topic, strategy)) %>%
  write.table(stdout(), sep=' ', quote=FALSE, row.names=FALSE, na="NaN") 

