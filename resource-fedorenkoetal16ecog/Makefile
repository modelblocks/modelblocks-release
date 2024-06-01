################################################################################
##                                                                            ##
##  This file is part of ModelBlocks. Copyright 2009, ModelBlocks developers. ##
##                                                                            ##
##  ModelBlocks is free software: you can redistribute it and/or modify       ##
##  it under the terms of the GNU General Public License as published by      ##
##  the Free Software Foundation, either version 3 of the License, or         ##
##  (at your option) any later version.                                       ##
##                                                                            ##
##  ModelBlocks is distributed in the hope that it will be useful,            ##
##  but WITHOUT ANY WARRANTY; without even the implied warranty of            ##
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             ##
##  GNU General Public License for more details.                              ##
##                                                                            ##
##  You should have received a copy of the GNU General Public License         ##
##  along with ModelBlocks.  If not, see <http://www.gnu.org/licenses/>.      ##
##                                                                            ##
################################################################################

################################################################################
#
#  Includes
#
#  Include statements to this file should be preceded by the following includes
#
#  include $(dir $(CURDIR))resource-general/Makefile
#  include $(RESOURCE-TOKENIZER)/Makefile
#  include $(RESOURCE-LTREES)/Makefile
#  include $(RESOURCE-GCG)/Makefile
#  include $(RESOURCE-RT)/Makefile
#  include $(RESOURCE-LVPCFG)/Makefile
#  include $(RESOURCE-LCPARSE)/Makefile
#  include $(RESOURCE-BNC)/Makefile
#  include $(RESOURCE-TREEBANK)/Makefile
#  include $(RESOURCE-GIGAWORD)/Makefile
#  include $(RESOURCE-KENLM)/Makefile
#  include $(RESOURCE-SRILM)/Makefile
#  include $(RESOURCE-INCRSEM)/Makefile
#
################################################################################

################################################################################
#
#  Macros & variables
#
################################################################################

define ParamVal
Fedorenkoetal16ECoG:
  kernel: ECoGCorpus
  text: Fedorenko et al 16 (ECoG corpus)
  value: fedorenkoetal16
endef

FEDECOG-BASENAME := genmodel/fedorenkoetal16ecog_sents.wsj02to21-GRAMMAR-nol-prtrm-3sm-synproc-+c_+u_+b5000.gcg15_nodashtags.merged.syn.5-kenlm

.SUFFIXES:
.SECONDEXPANSION:

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-fedorenkoetal16ecog-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN,$(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
FEDECOGDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(FEDECOGDIR))))
RESOURCE-NAME := Fedorenko et al 2016 ECoG corpus
define RESOURCE-DESCR =

NAME: Fedorenko et al 2016 ECoG corpus
MB POINTER FILE: config/user-fedorenkoetal16ecog-directory.txt
AVAILABILITY: Unreleased
DESCRIPTION: A corpus of ecog measures of gamma power from subjects in RSVP experiment.
endef
FEDECOG-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-FEDECOG = $(error $(NATSTORIES-ERROR))
endif
endif



%/user-fedorenkoetal16ecog-directory.txt: | %   
	echo '/project/lin-dept/corpora/original/english/fedorenkoetal16_ecog' > $@

################################################################################
#
#  Data formatting items
#
################################################################################

%fedorenkoetal16ecogsents.itemmeasures: $$(DEP-FEDECOG) $$(DEP-TOKENIZER) $(CONFIG) $(FEDECOG-SCRIPTS)/build_itemmeasures.py $$(FEDECOGDIR)/materials_visual/1_sent/
	python $(word 2, $^) $(word 3, $^) > $@

%fedorenkoetal16ecogsents.lineitems: $(FEDECOG-SCRIPTS)/itemmeasures2lineitems.py %fedorenkoetal16ecogsents.itemmeasures
	cat $(word 2, $^) | python $< > $@

%fedorenkoetal16ecogsents.linetoks: $$(DEP-TOKENIZER) %fedorenkoetal16ecogsents.lineitems $$(TOKENIZERDIR)/ptb_tokenizer.sed
	cat $< | sed -f $(word 2, $^) | sed 's/ $$//g' > $@

%fedorenkoetal16ecogwords.itemmeasures: $$(DEP-FEDECOG) $$(DEP-TOKENIZER) $(CONFIG) $(FEDECOG-SCRIPTS)/build_itemmeasures.py $$(FEDECOGDIR)/materials_visual/2_words/
	python $(word 2, $^) $(word 3, $^) > $@

%fedorenkoetal16ecogwords.lineitems: $(FEDECOG-SCRIPTS)/itemmeasures2lineitems.py %fedorenkoetal16ecogwords.itemmeasures
	cat $(word 2, $^) | python $< > $@

%fedorenkoetal16ecogwords.linetoks: $$(DEP-TOKENIZER) %fedorenkoetal16ecogwords.lineitems $$(TOKENIZERDIR)/ptb_tokenizer.sed
	cat $< | sed -f $(word 2, $^) | sed 's/ $$//g' > $@

%fedorenkoetal16ecogjabsents.itemmeasures: $$(DEP-FEDECOG) $$(DEP-TOKENIZER) $(CONFIG) $(FEDECOG-SCRIPTS)/build_itemmeasures.py $$(FEDECOGDIR)/materials_visual/3_jabsent/
	python $(word 2, $^) $(word 3, $^) > $@

%fedorenkoetal16ecogjabsents.lineitems: $(FEDECOG-SCRIPTS)/itemmeasures2lineitems.py %fedorenkoetal16ecogjabsents.itemmeasures
	cat $(word 2, $^) | python $< > $@

%fedorenkoetal16ecogjabsents.linetoks: $$(DEP-TOKENIZER) %fedorenkoetal16ecogjabsents.lineitems $$(TOKENIZERDIR)/ptb_tokenizer.sed
	cat $< | sed -f $(word 2, $^) | sed 's/ $$//g' > $@

%fedorenkoetal16ecogjabwords.itemmeasures: $$(DEP-FEDECOG) $$(DEP-TOKENIZER) $(CONFIG) $(FEDECOG-SCRIPTS)/build_itemmeasures.py $$(FEDECOGDIR)/materials_visual/4_jabwords/
	python $(word 2, $^) $(word 3, $^) > $@

%fedorenkoetal16ecogjabwords.lineitems: $(FEDECOG-SCRIPTS)/itemmeasures2lineitems.py %fedorenkoetal16ecogjabwords.itemmeasures
	cat $(word 2, $^) | python $< > $@

%fedorenkoetal16ecogjabwords.linetoks: $$(DEP-TOKENIZER) %fedorenkoetal16ecogjabwords.lineitems $$(TOKENIZERDIR)/ptb_tokenizer.sed
	cat $< | sed -f $(word 2, $^) | sed 's/ $$//g' > $@

%fedorenkoetal16ecog.itemmeasures: %fedorenkoetal16ecogsents.itemmeasures %fedorenkoetal16ecogwords.itemmeasures %fedorenkoetal16ecogjabsents.itemmeasures %fedorenkoetal16ecogjabwords.itemmeasures
	cat $< <(tail -n+2 $(word 2, $^)) <(tail -n+2 $(word 3, $^)) <(tail -n+2 $(word 4, $^)) > $@

%fedorenkoetal16ecog.lineitems: %fedorenkoetal16ecogsents.lineitems %fedorenkoetal16ecogwords.lineitems %fedorenkoetal16ecogjabsents.lineitems %fedorenkoetal16ecogjabwords.lineitems
	cat $^ > $@

%fedorenkoetal16ecog.linetoks: %fedorenkoetal16ecogsents.linetoks %fedorenkoetal16ecogwords.linetoks %fedorenkoetal16ecogjabsents.linetoks %fedorenkoetal16ecogjabwords.linetoks
	cat $^ > $@

