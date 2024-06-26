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
#  Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:

# Only use the following directory variables in prerequisites.
# Otherwise, they may break when this makefile is called by another.
BIN      := bin
GENMODEL := genmodel

  
################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-epicjar-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
EPICDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(EPICDIR))))
RESOURCE-NAME := The Epic CRF Parser (jarfile)
define RESOURCE-DESCR =

NAME: The Epic CRF Parser (jarfile)
MB POINTER FILE: config/user-epic-directory.txt
AVAILABILITY: Free
URL: https://github.com/dlwh/epic
DESCRIPTION: The directory containing the jarfile for the Epic parser.
endef
EPIC-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
EPIC-ERROR:
	$(error $(EPIC-ERROR))
DEP-EPIC = $(error $(EPIC-ERROR))
DEP-EPIC-NOWILDCARD = EPIC-ERROR
endif

CONFIG := $(CONFIGDIR)/user-javaflags.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
JAVAFLAGS := $(shell cat $(CONFIG))
endif

%/user-epicjar-directory.txt: | %
	echo '/fs/project/schuler.77/compling/epic' > $@

EPIC-TRAINER = target/scala-2.11/epic-assembly-0.4.4.jar epic.parser.models.ParserTrainer

################################################################################
#
#  Code compilation items
#
################################################################################

%.epic.linetrees: %.linetrees
	cat $< | sed 's/\(.\+\)/(\1)/g' > $@

%.epicncrfmodel: $$(DEP-EPIC) $$(DEP-SKIPDEP) $(CONFIGDIR)/user-javaflags.txt $(EPICDIR)/target/scala-2.11/epic-assembly-0.4.4.jar $(SKIPDEPDIR)/skipdep_embeddings.txt %.epic.linetrees
	mkdir -p $@_output/trees
	cp $(word 4, $^) $@_output/trees/train.txt
	cp $(word 4, $^) $@_output/trees/dev.txt
	cp $(word 4, $^) $@_output/trees/test.txt
	cd $@_output; java -Xmx16g -cp $(word 2, $^) epic.parser.models.NeuralParserTrainer \
		--word2vecPath $(word 3, $^) \
		--treebankType simple \
		--treebank.path trees \
	       	--cache.path constraints.cache \
		--opt.useStochastic \
		--modelFactory epic.parser.models.PositionalNeuralModelFactory \
		--opt.batchSize 200 \
		--threads 8
	echo 'Epic training finished' > $@

%epic_parsed.output: $$(DEP-EPIC) $$(DEP-SKIPDEP) $(EPICDIR)/target/scala-2.11/ $(BIN)/parser-epic \
		$$(basename %).linetoks \
		$(GENMODEL)/$$(basename $$(subst +,-,$$(subst -,.,$$(subst .,,$$(suffix $$*))))).model_output \
		$(GENMODEL)/$$(basename $$(subst +,-,$$(subst -,.,$$(subst .,,$$(suffix $$*))))).model
	cat $(word 2,$^)  |  $(word 1,$^) $(subst _, ,$(subst +,-,$(subst .,,$(suffix $(subst -,.,$*))))) $(word 3,$^)  >  $@

