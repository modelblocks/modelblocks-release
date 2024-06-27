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

MSG1 := The current config file, 
MSG2 := , points to a non-existent location (
MSG3 := ). Fix before re-running make, or (if not needed) set to ".".

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-tiger-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
TIGERDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(TIGERDIR))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(TIGERDIR)$(MSG3))
endif
endif


    
%/user-tiger-directory.txt: | %   
	echo '/fs/project/lin-dept/corpora/original/german/tigercorpus1' > $@
    
################################################################################
#
#  Recipes
#
################################################################################
    
genmodel/tiger.linetrees: $(TIGERDIR) $(LTREES-SCRIPTS)/editabletrees2linetrees.pl \
$(CONFIGDIR)/user-tiger-directory.txt | genmodel
	grep -v '^%' $</corpus/*.penn | sed 's/($$[^ ]* \([^)]*\))/(\1 \1)/g' | perl $(word 2,$^) > $@

genmodel/tiger%TRAIN.linetrees: genmodel/tiger%.linetrees | genmodel
	head -n 20894 $^ > $@

genmodel/tiger%TEST.linetrees: genmodel/tiger%.linetrees | genmodel
	tail -n 2611 $^ > $@
