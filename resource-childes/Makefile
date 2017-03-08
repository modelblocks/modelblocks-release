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
#  $(RESOURCE-LTREES)/resource-linetrees/Makefile
#  include $(RESOURCE-GCG)/Makefile
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

MSG1 := The current config file, 
MSG2 := , points to a non-existent location (
MSG3 := ). Fix before re-running make, or (if not needed) set to ".".

define CONFIGWARN =

ATTENTION! I had to create $(CONFIG),
which contains an incorrect default value (".").
Targets with dependencies to this file will fail until it is fixed.

endef

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-childes-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
CHILDESDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(CHILDESDIR))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(CHILDESDIR)$(MSG3))
endif
endif



%/user-childes-directory.txt: | %   
	echo '/home/corpora/original/english/childes' > $@

################################################################################
#
#  Recipes
#
################################################################################

%adam.linetrees: $(CHILDES-SCRIPTS)/get_linetrees.py $(CHILDESDIR)/CHILDES-Treebank/brown-adam.parsed \
$(CONFIGDIR)/user-childes-directory.txt | genmodel
	cat $(word 2, $^) | sed 's/\r//g;'| python $< > $@

%eve.linetrees: $(CHILDES-SCRIPTS)/get_linetrees.py $(CHILDESDIR)/CHILDES-Treebank/brown-eve.parsed \
$(CONFIGDIR)/user-childes-directory.txt | genmodel
	cat $(word 2, $^) | sed 's/\r//g;'| python $< > $@

%sarah.linetrees: $(CHILDES-SCRIPTS)/get_linetrees.py $(CHILDESDIR)/CHILDES-Treebank/brown-sarah.parsed \
$(CONFIGDIR)/user-childes-directory.txt | genmodel
	cat $(word 2, $^) | sed 's/\r//g;'| python $< > $@

%childes_brown_all.linetrees: genmodel/adam.linetrees genmodel/eve.linetrees genmodel/sarah.linetrees | genmodel
	cat $^ > $@
