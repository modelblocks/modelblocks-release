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

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-gpt2-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
GPT2DIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(GPT2DIR))))
RESOURCE-NAME := GPT-2
define RESOURCE-DESCR =

NAME: GPT-2
MB POINTER FILE: config/user-gpt2-directory.txt
AVAILABILITY: Free
URL: https://github.com/openai/gpt-2
DESCRIPTION: 
endef
GPT2-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-GPT2 = $(error $(GPT2-ERROR))
else
DEP-GPT2 :=
endif
endif

%/user-gpt2-directory.txt: | %
	echo '/data/compling/gpt2' > $@

################################################################################

gpt2_env: $(RESOURCE-GPT2)/gpt2_conda_env.sh
	$(SHELL) $(word 1, $^)

# TODO get itemmeasures from all sizes of GPT-2

.PRECIOUS: %.gpt2.itemmeasures
%.gpt2.itemmeasures:  %.delim.lineitems  $(GPT2-SCRIPTS)/per_word_surprisal.py  |  $(CONFIGDIR)/user-gpt2-directory.txt
	##################################################################
	### This target assumes active (gpt2_env) conda environment #####
	### Please run "make gpt2_env" and "conda activate gpt2_env" ###
	##################################################################
	python3 $(word 2,$^) --models_dir $(shell cat $|)/models --model_name 124M > $@ 2> $@.log
