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

.SUFFIXES:
.SECONDEXPANSION:
.DELETE_ON_ERROR:


################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-naturalstoriesmaze-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN,$(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
NATSTORMAZEDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(NATSTORMAZEDIR))))
RESOURCE-NAME := The Natural Stories Maze Corpus
define RESOURCE-DESCR =

NAME: The Natural Stories Maze Corpus
MB POINTER FILE: config/user-naturalstoriesmaze-directory.txt
AVAILABILITY: https://github.com/vboyce/natural-stories-maze/
DESCRIPTION: A corpus of crowd-sourced Maze task data using the Natural Stories materials
endef
NATSTORIESMAZE-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-NATSTORIESMAZE = $(error $(NATSTORIESMAZE-ERROR))
endif
endif

%/user-naturalstories-directory.txt: | %   
	echo '/data/corpora/original/english/natural-stories-maze' > $@

################################################################################
#
#  Reading time data formatting items
#
################################################################################

## copy partition ini into project genmodel directory
genmodel/%.ini: $(NATSTORIESMAZE-SCRIPTS)/%.ini  |  scripts
	cp $< $@

%naturalstoriesmaze.evmeasures: $$(DEP-NATSTORIESMAZE) $(NATSTORMAZEDIR)/Data/cleaned.rds \
$(NATSTORIESMAZE-SCRIPTS)/preprocess_natstormaze.R $(NATSTORIESMAZE-SCRIPTS)/merge_natstormaze.py %naturalstories.mfields.itemmeasures
	$(word 2, $^) $< | python3 $(word 3, $^) $(word 4, $^) > $@


################################################################################
#
#  Sents files
#
################################################################################    

naturalstoriesmaze%itemmeasures: naturalstories%itemmeasures
	cp $< $@

