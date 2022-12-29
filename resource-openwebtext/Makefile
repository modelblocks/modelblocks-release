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

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-openwebtext-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
OPENWEBTEXTDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(OPENWEBTEXTDIR))))
RESOURCE-NAME := OpenWebText
define RESOURCE-DESCR =
NAME: OpenWebText
MB POINTER FILE: config/user-openwebtext-directory.txt
AVAILABILITY: Free
URL: https://skylion007.github.io/OpenWebTextCorpus/
DESCRIPTION: An open source attempt to replicate the WebText corpus used to train GPT-3.
endef
OPENWEBTEXT-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-OPENWEBTEXT = $(error $(OPENWEBTEXT-ERROR))
else
DEP-OPENWEBTEXT :=
endif
endif

%/user-openwebtext-directory.txt: | %
	echo '/data/corpora/original/english/openwebtext' > $@

################################################################################
#
#  Recipes
#
################################################################################
    
%openwebtext.linetoks: $(OPENWEBTEXTDIR)/openwebtext.txt | %
	cat $< | awk NF > $@

%openwebtextlower.linetoks: %openwebtext.linetoks | %
	cat $^ | tr '[:upper:]' '[:lower:]' | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;s/^ //" > $@
