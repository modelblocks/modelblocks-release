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
#  include $(dir $(CURDIR))/general/Makefile
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
CONFIG := $(CONFIGDIR)/user-ccl-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
CCLDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(CCLDIR))))
RESOURCE-NAME := CCL unsupervised parser (Seginer, 2007)
define RESOURCE-DESCR =

NAME: CCL unsupervised parser (Seginer, 2007)
MB POINTER FILE: config/user-ccl-directory.txt
AVAILABILITY: Free
URL: http://www.seggu.net/ccl/
DESCRIPTION: The CCL unsupervised constituency parser (Seginer, 2007).
endef
CCL-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-CCL = $(error $(CCL-ERROR))
endif
endif



%/user-ccl-directory.txt: | %
	echo '/project/lin-dept/compling/cclparser' > $@

################################################################################
#
#  Recipes
#
################################################################################

# Automatically runs CCL from scratch on a user-supplied linetoks file.
# The run name is a hyphen-delimited linetoks file basename.
#
# Example target:
#
#      <path>/eve-nt-lower-nounary-nolabel-nopunc.cclprs
#
# which runs the CCL system on eve.nt.lower.nounary.nolabel.nopunc.linetoks
# as input and saves the parser output to the target
#
%.cclprs: $$(DEP-CCL) genmodel/$$*.linetoks $(CCLDIR) $(CONFIGDIR)/user-ccl-directory.txt
	mkdir -p $(dir $*)
	echo $(abspath $<) line learn -o $(abspath $*) -s ccllog -R 100 > $*.ccl_exec_seq.txt
	echo $(abspath $<) line parse -o $(abspath $*) -s cclprs -R 100 >> $*.ccl_exec_seq.txt
	cd $(word 2, $^); ./main/UnknownOS/cclparser $(abspath $*.ccl_exec_seq.txt)
	rm $*.ccl_exec_seq.txt
  
# Converts CCL output to linetrees
%.ccl.linetrees: %.cclprs $(LTREES-SCRIPTS)/editabletrees2linetrees.pl \
$(LTREES-SCRIPTS)/make-trees-nounary.pl 
	cat $(abspath $*.cclprs) | sed 's/\([^ ()]\+\)/(\1)/g;s/(/(X /g;' | \
	perl $(word 2, $^) | sed 's/  \+/ /g;' | perl $(word 3, $^) > $@
