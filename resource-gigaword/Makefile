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

GIGASECTS = afp_eng apw_eng cna_eng ltw_eng nyt_eng xin_eng

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-gigaword4-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
GIGADIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(GIGADIR))))
RESOURCE-NAME := English Gigaword
define RESOURCE-DESCR =

NAME: English Gigaword
AVAILABILITY: Paid
URL: https://catalog.ldc.upenn.edu/ldc2003t05
DESCRIPTION: A comprehensive archive of newswire text data in English that
has been acquired over several years by the Linguistic Data Consortium.
endef
GIGAWORD-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-GIGAWORD = $(error $(GIGAWORD-ERROR))
else
DEP-GIGAWORD :=
endif
endif



%/user-gigaword4-directory.txt: | %
	echo '/home/corpora/original/english/gigaword-4' > $@

################################################################################
#
#  Recipes
#
################################################################################
    
gigaword4%.linetoks: $$(DEP-GIGAWORD) $(GIGADIR)/data/$$*/*.gz $(GIGAWORD-SCRIPTS)/preprocess_gigaword.py \
$(CONFIGDIR)/user-gigaword4-directory.txt | genmodel
	mkdir tmp
	cp $(word 2, $^) tmp/
	gunzip tmp/*
	cat tmp/* > $@.tmp
	python $(word 3, $^) $@.tmp > $@
	rm -rf tmp $@.tmp

%gigaword4.linetoks: $(foreach sect,$(GIGASECTS),%gigaword4$(sect).linetoks) | genmodel
	cat $^ | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;s/^ //" > $@
	rm -f $^
  
%gigaword4lower.linetoks: $(foreach sect,$(GIGASECTS),%gigaword4$(sect).linetoks) | genmodel
	cat $^ | tr '[:upper:]' '[:lower:]' | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;s/^ //" > $@
	rm -f $^
