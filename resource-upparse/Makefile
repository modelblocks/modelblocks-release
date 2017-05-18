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
CONFIG := $(CONFIGDIR)/user-upparse-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
UPPARSEDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(UPPARSEDIR))))
RESOURCE-NAME := UPPARSE (Unsupervised parser, Ponvert et al, 2011)
define RESOURCE-DESCR =

NAME: UPPARSE (Unsupervised parser, Ponvert et al, 2011)
AVAILABILITY: Free
URL: https://github.com/eponvert/upparse
DESCRIPTION:i Efficient implementations of hidden Markov
models (HMMs) and probabilistic right linear grammars (PRLGs) for
unsupervised partial parsing (also known as: unsupervised chunking,
unsupervised NP identification, unsupervised phrasal segmentation).
endef
UPPARSE-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-UPPARSE = $(error $(UPPARSE-ERROR))
endif
endif



%/user-upparse-directory.txt: | %
	echo '/home/compling/upparse' > $@

################################################################################
#
#  Recipes
#
################################################################################

# Automatically runs upparse from scratch on a user-supplied linetoks file,
# the basename of which is the basename of this target. The target follows
# the following template:
#
#      <path>/<linetoks-basename>.upparse
#
# Example:
#
#      <path>/eve-nt-lower-nounary-nolabel-nopunc.upparse
#
# which runs the upparse system on eve.nt.lower.nounary.nolabel.nopunc.linetoks
# as input.
#
# Output is sent to a directory named identically to the target stem.
# If the target's directory does not exist, it will be created.
# Because the target is never made, this rule will always run.
#
%.upparse: $$(DEP-UPPARSE) genmodel/$$*.linetoks $(UPPARSEDIR) $(UPPARSEDIR)/scripts/chunk.py \
$(CONFIGDIR)/user-upparse-directory.txt | genmodel
	mkdir -p $(dir $*)
	cd $(word 2, $^); ./scripts/chunk.py -C -t $(abspath $<) -s $(abspath $<) -o $(abspath $*) -m prlg-uni

# Finds the parser output from an upparse run and promotes it to the project root
/%.upparseprs: $$(lastword $$(sort $$(wildcard /%/cascade*/test-eval)))
	cat $< > $@ 

# Because of oddities in how Make handles wildcard expansion in prereqs,
# this rule is necessary to allow relative-path targets
%.upparseprs: $$(abspath $$@);

# Generates linetrees files from upparse output.
# The stem should be the name of the upparse output directory
# from which to extract linetrees.
#
# Example target:
#      
#      <path>/eve-nt-lower-nounary-nolabel-nopunc.upparse.linetrees
#
# which will extract parser output contained in the directory
# <path>/eve-nt-lower-nounary-nolabel-nopunc.
#
# If no directory matches the stem, the rule will fail.
#
%.upparse.linetrees: %.upparseprs
	cat $< | sed 's/\([^ ()]\+\)/(\1)/g;s/(/(X /g;' > $@

