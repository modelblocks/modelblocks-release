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

GIGA1SECTS = afe apw nyt xie
GIGA2SECTS = afp_eng apw_eng cna_eng nyt_eng xin_eng
GIGA3SECTS = afp_eng apw_eng cna_eng nyt_eng ltw_eng xin_eng
GIGA4SECTS = afp_eng apw_eng cna_eng nyt_eng ltw_eng xin_eng
GIGA5SECTS = afp_eng apw_eng cna_eng nyt_eng ltw_eng wpb_eng xin_eng

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-gigaword-sects.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
GIGASECTS := $(shell cat $(CONFIG))

CONFIG := $(CONFIGDIR)/user-gigaword-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
GIGADIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(GIGADIR))))

RESOURCE-NAME := English Gigaword
define RESOURCE-DESCR =
NAME: English Gigaword 3
MB POINTER FILE: config/user-gigaword-directory.txt
AVAILABILITY: Paid
URL: https://catalog.ldc.upenn.edu/LDC2007T07
DESCRIPTION: A comprehensive archive of newswire text data in English that
has been acquired over several years by the Linguistic Data Consortium.
endef

define RESOURCE-DESCR =
NAME: English Gigaword 4
MB POINTER FILE: config/user-gigaword-directory.txt
AVAILABILITY: Paid
URL: https://catalog.ldc.upenn.edu/LDC2009T13
DESCRIPTION: A comprehensive archive of newswire text data in English that
has been acquired over several years by the Linguistic Data Consortium.
endef

define RESOURCE-DESCR =
NAME: English Gigaword 5
MB POINTER FILE: config/user-gigaword-directory.txt
AVAILABILITY: Paid
URL: https://catalog.ldc.upenn.edu/ldc2011t07
DESCRIPTION: A comprehensive archive of newswire text data in English that
has been acquired over several years by the Linguistic Data Consortium.
endef

GIGAWORD-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-GIGAWORD = $(error $(GIGAWORD-ERROR))
else
DEP-GIGAWORD :=
endif
endif

## Default version of Gigaword is Gigaword 4, FYI
%/user-gigaword-directory.txt: | %
	echo '/data/corpora/original/english/gigaword-4' > $@

%/user-gigaword-sects.txt: | %
	echo $(GIGA4SECTS) > $@

GIGASECTS = $(shell cat $(CONFIGDIR)/user-gigaword-sects.txt)

################################################################################
#
#  Recipes
#
################################################################################
    
gigaword%.linetoks: $$(DEP-GIGAWORD) $(GIGAWORD-SCRIPTS)/preprocess_gigaword.py $(GIGADIR)/data/$$*/*.gz \
$(CONFIGDIR)/user-gigaword-directory.txt | genmodel
	mkdir -p gigatmp
	cp $(wordlist 2,$(shell echo $$(($(words $^) - 1))),$^) gigatmp/
	gunzip gigatmp/*
	cat gigatmp/* > $@.gigatmp
	python $(word 1, $^) $@.gigatmp > $@
	rm -rf gigatmp $@.gigatmp

%gigaword.linetoks: $(foreach sect,$(GIGASECTS),%gigaword$(sect).linetoks) | genmodel
	cat $^ | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;s/^ //" > $@
	rm -f $^

%gigawordlower.linetoks: $(foreach sect,$(GIGASECTS),%gigaword$(sect).linetoks) | genmodel
	cat $^ | tr '[:upper:]' '[:lower:]' | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;s/^ //" > $@
	rm -f $^
