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

BROWNTRAINSECTS = cf cg ck cl cm cn cp cr

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
CONFIG := $(CONFIGDIR)/user-treebank-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
TREEBANKDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(TREEBANKDIR))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(TREEBANKDIR)$(MSG3))
endif
endif


    
%/user-treebank-directory.txt: | %   
	echo '/home/corpora/original/english/penn_treebank_3' > $@
    
################################################################################
#
#  Recipes
#
################################################################################
    
genmodel/brown%.linetrees: $(TREEBANKDIR) $(LTREES-SCRIPTS)/editabletrees2linetrees.pl \
$(CONFIGDIR)/user-treebank-directory.txt | genmodel
	cat $</parsed/mrg/brown/%/*.mrg | perl $(word 2,$^) > $@

genmodel/brownTRAIN.linetrees: $(foreach sect,$(BROWNTRAINSECTS),genmodel/brown$(sect).linetrees) | genmodel
	cat $^ > $@
  
genmodel/wsj%.linetrees: $(CONFIGDIR)/user-treebank-directory.txt $(TREEBANKDIR)/parsed/mrg/wsj/$$* \
$(LTREES-SCRIPTS)/editabletrees2linetrees.pl | genmodel
	cat $(word 2,$^)/*.mrg | perl $(word 3,$^) > $@
  
%/wsj02to21.linetrees:  $(foreach sect,02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21,%/wsj$(sect).linetrees)
	cat $^ > $@

genmodel/swbd%.linetrees: $(CONFIGDIR)/user-treebank-directory.txt $(TREEBANKDIR)/parsed/mrg/swbd/$$* \
$(LTREES-SCRIPTS)/editabletrees2linetrees.pl | genmodel
	cat $(word 2, $^)/*.mrg | perl $(word 3, $^) | awk '/^\s*\(/' > $@

%/swbd2to4.linetrees: $(foreach sect, 2 3 4, %/swbd$(sect).linetrees)
	cat $^ > $@
