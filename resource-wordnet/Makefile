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
#  i. Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:

#The directory variables break when this makefile is called by another.
THISDIR := $(dir $(abspath $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))))
SRCMODEL := $(THISDIR)srcmodel
SCRIPTS  := $(THISDIR)scripts

################################################################################
#
#  ii. User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

.PRECIOUS: %config/
%config/:
	mkdir $*config/

#### Create user-*.txt file for resource at include time, prior to make-item dependency graph construction...
ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-wordnet-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG))) 
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
WORDNETDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(WORDNETDIR))))
RESOURCE-NAME := WordNet
define RESOURCE-DESCR =

NAME: WordNet
MB POINTER FILE: config/user-wordnet-directory.txt
AVAILABILITY: Free
URL: https://wordnet.princeton.edu/wordnet/download/
DESCRIPTION: WordNet is a large lexical database of English.
endef
WORDNET-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-WORDNET = $(error $(WORDNET-ERROR))
endif
endif



%/user-wordnet-directory.txt: | % 
	echo '/project/lin-dept/corpora/original/english/wordnet-3.1' > $@

################################################################################
#
#  1. Wordnet items
#
################################################################################

%/wordnet.hyps: $$(DEP-WORDNET) $(SCRIPTS)/wordnet2hyps.py  $(WORDNETDIR)  |  %  ##$$(wildcard $(WN_DIR)/dict/data.*)  |  %
	cat $(word 2,$^)/dict/data.*  |  grep '^[0-9]'  |  python $<  >  $@



