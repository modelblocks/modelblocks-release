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
###############################################################################

.SUFFIXES:
.SECONDEXPANSION:

nums = 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99

NUM_TAGS := 15

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-bmmm-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
BMMMDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(BMMMDIR))))
RESOURCE-NAME := The BMMM Unsupervised PoS tagger (Christodoulopoulos et al 2011)
define RESOURCE-DESCR =

NAME: The BMMM Unsupervised PoS tagger (Christodoulopoulos et al 2011)
MB POINTER FILE: config/user-bmmm-directory.txt
AVAILABILITY: Free
URL: https://github.com/christos-c/bmmm
DESCRIPTION: A Bayesian multinomial mixture model (BMMM) for unsupervised
part of speech tagging (Christodoulopoulos et al 2011).
endef
BMMM-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-BMMM = $(error $(BMMM-ERROR))
endif
endif



%/user-bmmm-directory.txt: | %
	echo '/home/compling/bmmm' > $@

################################################################################
#
#  Recipes
#
################################################################################

# Converts a raw linetoks file into CoNLL format for parsing
%.raw.conll: $$(DEP-BMMM) genmodel/$$*.linetoks $(BMMMDIR)/target/bmmm-2.0.??.jar $(CONFIGDIR)/user-bmmm-directory.txt | genmodel
	mkdir -p $(dir $*)
	cat $(abspath $<) | python $(BMMM-SCRIPTS)/raw2conll.py > $(abspath $@)


# First generation of the BMMM. Runs the BMMM POS tagger on raw
# corpus tokens without dependency information.
.PRECIOUS: %.bmmmbase.conll %.u.bmmmbase.conll
%.bmmmbase.conll %.u.bmmmbase.conll: $$(DEP-BMMM) $$(basename %).raw.conll $(BMMMDIR)/target/bmmm-2.0.??.jar $(CONFIGDIR)/user-bmmm-directory.txt | genmodel
	mkdir -p $(dir $*)
	cd $(BMMMDIR); java -jar $(abspath $(word 2, $^)) -in $(abspath $<) -out $(abspath $@) -classes $(subst .,,$(suffix $*))

# Recipe used to implement iterated POS/dependency parser per
# Christodoulopoulos et al (2012), using the following template:
#
#     <path>/<linetoks-basename>.<num-tags>.<iterations>.bmmm.conll
#
# Example:
#
#     <path>/wsj02to21.induc.45.10.bmmm.conll
#
# which runs produces runs the iterated BMMM/DMV learner for 10
# generations on genmodel/wsj02to21.induc.linetoks and produces a
# CoNLL-style table of POS and dependency data by token in
# the corpus, using 45 part-of-speech tags. A 'u' option can also
# be provided in order to use undirected rather than directed
# dependencies in the BMMM part-of-speech tagging system. With
# this option, the recipe stem uses the following template:
#
#     <path>/wsj02to21.induc.45.10.u.bmmm.conll
#
# 
# NB: In order to allow the recursion to bottom out, the first
# iteration is named differently, i.e.:
#
#     <path>/wsj02to21.induc.45.{u.}bmmmbase.conll
# 
# not:
# 
#     ## WRONG ## <path>/wsj02to21.induc.45.1.u.bmmm.conll
#
# (See preceding rule)
# 
.PRECIOUS: %.bmmm.conll
%.bmmm.conll: $$(DEP-BMMM) $(BMMMDIR)/target/bmmm-2.0.??.jar $(CONFIGDIR)/user-bmmm-directory.txt | genmodel
	mkdir -p $(dir $*)
	make $(basename $*).$(lastword $(wordlist 1, $(subst .,,$(suffix $*)), $(nums))).dmv.conll
	cd $(BMMMDIR); java -jar $(abspath $<) -in $(abspath $(basename $*).$(lastword $(wordlist 1, $(subst .,,$(suffix $*)), $(nums))).dmv.conll) -out $(abspath $@) -deps -classes $(subst .,,$(suffix $(basename $*)))

.PRECIOUS: %.u.bmmm.conll
%.u.bmmm.conll: $$(DEP-BMMM) $(BMMMDIR)/target/bmmm-2.0.??.jar $(CONFIGDIR)/user-bmmm-directory.txt | genmodel
	mkdir -p $(dir $*)
	make $(basename $*).$(lastword $(wordlist 1, $(subst .,,$(suffix $*)), $(nums))).u.dmv.conll
	cd $(BMMMDIR); java -jar $(abspath $<) -in $(abspath $(basename $*).$(lastword $(wordlist 1, $(subst .,,$(suffix $*)), $(nums))).u.dmv.conll) -out $(abspath $@) -deps undir -classes $(subst .,,$(suffix $(basename $*)))

# Custom BMMM run. Runs the BMMM on a user-specified CoNLL
# data table.
%.bmmm.conll: $$(DEP-BMMM) %.conll $(BMMMDIR)/target/bmmm-2.0.??.jar $(CONFIGDIR)/user-bmmm-directory.txt | genmodel
	mkdir -p $(dir $*)
	cd $(BMMMDIR); java -jar $(abspath $(word 2, $^)) -in $(abspath $<) -out $(abspath $@) -classes $(subst .,,$(suffix $(basename $*)))

