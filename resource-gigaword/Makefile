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
#  include $(RESOURCE-LTREES)/Makefile
#
################################################################################

################################################################################
#
#  Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:

GIGA3SECTS = afp_eng apw_eng cna_eng ltw_eng nyt_eng xin_eng
GIGA4SECTS = afp_eng apw_eng cna_eng ltw_eng nyt_eng xin_eng
GIGA5SECTS = afp_eng apw_eng cna_eng ltw_eng nyt_eng wpb_eng xin_eng

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-gigaword3-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
GIGA3DIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(GIGA3DIR))))
RESOURCE-NAME := English Gigaword 3
define RESOURCE-DESCR =

NAME: English Gigaword 3
MB POINTER FILE: config/user-gigaword3-directory.txt
AVAILABILITY: Paid
URL: https://catalog.ldc.upenn.edu/LDC2007T07
DESCRIPTION: A comprehensive archive of newswire text data in English that
has been acquired over several years by the Linguistic Data Consortium.
endef
GIGAWORD3-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-GIGAWORD3 = $(error $(GIGAWORD3-ERROR))
else
DEP-GIGAWORD3 :=
endif

CONFIG := $(CONFIGDIR)/user-gigaword4-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
GIGA4DIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(GIGA4DIR))))
RESOURCE-NAME := English Gigaword 4
define RESOURCE-DESCR =

NAME: English Gigaword 4
MB POINTER FILE: config/user-gigaword4-directory.txt
AVAILABILITY: Paid
URL: https://catalog.ldc.upenn.edu/LDC2009T13
DESCRIPTION: A comprehensive archive of newswire text data in English that
has been acquired over several years by the Linguistic Data Consortium.
endef
GIGAWORD4-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-GIGAWORD4 = $(error $(GIGAWORD4-ERROR))
else
DEP-GIGAWORD4 :=
endif

CONFIG := $(CONFIGDIR)/user-gigaword5-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
GIGA5DIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(GIGA5DIR))))
RESOURCE-NAME := English Gigaword 5
define RESOURCE-DESCR =

NAME: English Gigaword 5
MB POINTER FILE: config/user-gigaword5-directory.txt
AVAILABILITY: Paid
URL: https://catalog.ldc.upenn.edu/ldc2011t07
DESCRIPTION: A comprehensive archive of newswire text data in English that
has been acquired over several years by the Linguistic Data Consortium.
endef
GIGAWORD5-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-GIGAWORD5 = $(error $(GIGAWORD5-ERROR))
else
DEP-GIGAWORD5 :=
endif
endif



%/user-gigaword3-directory.txt: | %
	echo '/home/corpora/original/english/gigaword-3' > $@
%/user-gigaword4-directory.txt: | %
	echo '/home/corpora/original/english/gigaword-4' > $@
%/user-gigaword5-directory.txt: | %
	echo '/home/corpora/original/english/gigaword-5' > $@

################################################################################
#
#  Recipes
#
################################################################################
    
gigaword3%.linetoks: $$(DEP-GIGAWORD3) $(GIGA3DIR)/data/$$*/*.gz $(GIGAWORD-SCRIPTS)/preprocess_gigaword.py \
$(CONFIGDIR)/user-gigaword3-directory.txt | genmodel
	mkdir tmp
	cp $(word 1, $^) tmp/
	gunzip tmp/*
	cat tmp/* > $@.tmp
	python $(word 2, $^) $@.tmp > $@
	rm -rf tmp $@.tmp

gigaword4%.linetoks: $$(DEP-GIGAWORD4) $(GIGA4DIR)/data/$$*/*.gz $(GIGAWORD-SCRIPTS)/preprocess_gigaword.py \
$(CONFIGDIR)/user-gigaword4-directory.txt | genmodel
	mkdir tmp
	cp $(word 1, $^) tmp/
	gunzip tmp/*
	cat tmp/* > $@.tmp
	python $(word 2, $^) $@.tmp > $@
	rm -rf tmp $@.tmp

gigaword5%.linetoks: $$(DEP-GIGAWORD5) $(GIGA5DIR)/data/$$*/*.gz $(GIGAWORD-SCRIPTS)/preprocess_gigaword.py \
$(CONFIGDIR)/user-gigaword5-directory.txt | genmodel
	mkdir tmp
	cp $(word 1, $^) tmp/
	gunzip tmp/*
	cat tmp/* > $@.tmp
	python $(word 2, $^) $@.tmp > $@
	rm -rf tmp $@.tmp

%gigaword3.linetoks: $(foreach sect,$(GIGA3SECTS),%gigaword3$(sect).linetoks) | genmodel
	cat $^ | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;s/^ //" > $@
	rm -f $^

%gigaword4.linetoks: $(foreach sect,$(GIGA4SECTS),%gigaword4$(sect).linetoks) | genmodel
	cat $^ | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;s/^ //" > $@
	rm -f $^

%gigaword5.linetoks: $(foreach sect,$(GIGA5SECTS),%gigaword5$(sect).linetoks) | genmodel
	cat $^ | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;s/^ //" > $@
	rm -f $^

%gigaword3lower.linetoks: $(foreach sect,$(GIGA3SECTS),%gigaword3$(sect).linetoks) | genmodel
	cat $^ | tr '[:upper:]' '[:lower:]' | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;s/^ //" > $@
	rm -f $^

%gigaword4lower.linetoks: $(foreach sect,$(GIGA4SECTS),%gigaword4$(sect).linetoks) | genmodel
	cat $^ | tr '[:upper:]' '[:lower:]' | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;s/^ //" > $@
	rm -f $^

%gigaword5lower.linetoks: $(foreach sect,$(GIGA5SECTS),%gigaword5$(sect).linetoks) | genmodel
	cat $^ | tr '[:upper:]' '[:lower:]' | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;s/^ //" > $@
	rm -f $^
