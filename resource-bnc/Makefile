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

BNCTRAINDIRS = A B C D E F G H J

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

MSG1 := The current config file, 
MSG2 := , points to a non-existent location (
MSG3 := ). Fix before re-running make, or (if not needed) set to ".".

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-bnc-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
BNCDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(BNCDIR))))
RESOURCE-NAME := The British National Corpus (BNC)
define RESOURCE-DESCR =

NAME: The British National Corpus (BNC)
AVAILABILITY: FREE
URL: http://www.natcorp.ox.ac.uk/
DESCRIPTION: The British National Corpus (BNC) is a 100 million word collection
of samples of written and spoken language from a wide range of sources, designed
to represent a wide cross-section of British English, both spoken and written,
from the late twentieth century.
endef
BNC-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-BNC = $(error $(BNC-ERROR))
else
DEP-BNC =
endif
endif



%/user-bnc-directory.txt: | %   
	echo '/home/corpora/original/english/bnc' > $@

################################################################################
#
#  1. Recipes
#
################################################################################
    
bnc%.linetoks: $$(DEP-BNC) $(BNCDIR) $(CONFIGDIR)/user-bnc-directory.txt | genmodel
	cat $(word 1, $^)/Texts/$*/*/* | sed 's|<teiHeader.*</teiHeader>||g;s|<[^<>]*>||g;s|^\s*\n||g;/^$$$$/d;s|  *| |g;' > $@
    
genmodel/bncTRAIN.linetoks: $(foreach sect,$(BNCTRAINDIRS),bnc$(sect).linetoks) | genmodel
	cat $^ > $@
