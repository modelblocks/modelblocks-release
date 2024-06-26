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
WINDOWLEN := 1

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
CONFIG := $(CONFIGDIR)/user-word2vec-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG))) 
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
WORD2VECDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(WORD2VECDIR))))
RESOURCE-NAME := Word2Vec
define RESOURCE-DESCR =

NAME: Word2Vec
MB POINTER FILE: config/user-word2vec-directory.txt
AVAILABILITY: Free
URL: https://github.com/dav/word2vec
DESCRIPTION: This tool provides an efficient implementation of the continuous bag-of-words
and skip-gram architectures for computing vector representations of words. These
representations can be subsequently used in many natural language processing applications
and for further research.

These recipes automate training of Word2Vec vectors on arbitrary text data.

They also support the use of embeddings pre-trained for computing semantic distance.
endef
WORD2VEC-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-WORD2VEC = $(error $(WORD2VEC-ERROR))
endif
endif



%/user-word2vec-directory.txt: | % 
	echo '/fs/project/schuler.77/compling/word2vec/' > $@

################################################################################
#
#  1. Wordnet items
#
################################################################################

%.word2vec.embeddings.tokmeasures: $$(basename $$(basename %)).linetoks $(GLOVE-SCRIPTS)/embedDocument.py $$(subst .,,$$(suffix $$(basename %)))$$(suffix $$*).word2vec.txt
	cat $(word 1, $^) | python $(word 2, $^) <(tail -n +2 $(word 3, $^)) -s > $@

%.word2vec.linemeasures: $$(basename $$(basename $$(basename %))).linetoks $(GLOVE-SCRIPTS)/embedDocument.py $$(subst .,,$$(suffix $$(basename $$(basename %))))$$(suffix $$(basename $$*)).word2vec.txt
	cat $(word 1, $^) | python $(word 2, $^) <(tail -n +2 $(word 3, $^)) -s -o linemeasures -w $(subst _, ,$(subst .,,$(suffix, $*))) > $@

%.word2vec.tokmeasures: $$(basename $$(basename $$(basename %))).linetoks $(GLOVE-SCRIPTS)/embedDocument.py $$(subst .,,$$(suffix $$(basename $$(basename %))))$$(suffix $$(basename $$*)).word2vec.txt
	cat $(word 1, $^) | python $(word 2, $^) <(tail -n +2 $(word 3, $^)) -s -o tokmeasures -w $(subst _, ,$(subst .,,$(suffix $*))) > $@

# OLD: dot delimited
%.word2vec.itemmeasures: $$(basename $$(basename $$(basename %))).lineitems $(GLOVE-SCRIPTS)/embedDocument.py $$(subst .,,$$(suffix $$(basename $$(basename %))))$$(suffix $$(basename $$*)).word2vec.txt
	cat $(word 1, $^) | python $(word 2, $^) <(tail -n +2 $(word 3, $^)) -s -o tokmeasures -w $(subst _, ,$(subst .,,$(suffix $*))) > $@

# CURRENT: hyphen delimited
%-word2vec.itemmeasures: $$(basename $$(basename $$(basename $$(subst -,.,%)))).lineitems $(GLOVE-SCRIPTS)/embedDocument.py $$(subst .,,$$(suffix $$(basename $$(basename $$(subst -,.,%)))))$$(suffix $$(basename $$(subst -,.,$$*))).word2vec.txt
	cat $(word 1, $^) | python $(word 2, $^) <(tail -n +2 $(word 3, $^)) -s -o tokmeasures -w $(subst _, ,$(subst .,,$(suffix $(subst -,.,$*)))) > $@

%.word2phrase1.txt: $$(DEP-WORD2VEC) %.linetoks $(WORD2VECDIR)word2phrase
	$(word 2, $^) -train $(word 1, $^) -output $@ -threshold 200 -debug 2

%.word2phrase2.txt: $$(DEP-WORD2VEC) %.word2phrase1.txt $(WORD2VECDIR)word2phrase
	$(word 2, $^) -train $(word 1, $^) -output $@ -threshold 100 -debug 2

.PRECIOUS: %.word2vec.txt
%.word2vec.txt: $$(DEP-WORD2VEC) $$(basename %).word2phrase2.txt $(WORD2VECDIR)word2vec
	$(word 2, $^) -train $(word 1, $^) -output $@ -cbow 1 -size $(subst .,,$(suffix $*)) -window 15 -negative 10 -hs 0 -sample 1e-5 -threads 40 -binary 0 -iter 3 -min-count 10

