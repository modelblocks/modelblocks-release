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

define CONFIGWARN =

ATTENTION! I had to create $(CONFIG),
which contains an incorrect default value (".").
Targets with dependencies to this file will fail until it is fixed.

endef

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-bnc-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
BNCDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(BNCDIR))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(BNCDIR)$(MSG3))
endif
endif

%/user-bnc-directory.txt: | %   
	echo '.' > $@

%/user-bnc-directory.txt.cocomo: | %   
	echo '/home/corpora/original/english/bnc' > $(basename $@)

################################################################################
#
#  1. Recipes
#
################################################################################
    
bnc%.linetoks: $(BNCDIR) $(CONFIGDIR)/user-bnc-directory.txt | genmodel
	cat $</Texts/$*/*/* | sed 's|<teiHeader.*</teiHeader>||g;s|<[^<>]*>||g;s|^\s*\n||g;/^$$$$/d;s|  *| |g;' > $@
    
genmodel/bncTRAIN.linetoks: $(foreach sect,$(BNCTRAINDIRS),bnc$(sect).linetoks) | genmodel
	cat $^ > $@
