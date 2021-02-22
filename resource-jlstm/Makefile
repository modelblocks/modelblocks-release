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
CONFIG := $(CONFIGDIR)/user-glstm-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
JLSTMDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(JLSTMDIR))))
RESOURCE-NAME := Jozefowicz LSTM
define RESOURCE-DESCR =

NAME: Jozefowicz LSTM
MB POINTER FILE: config/user-glstm-directory.txt
AVAILABILITY: Free
URL: https://github.com/tensorflow/models/tree/archive/research/lm_1b
DESCRIPTION: LSTM estimates, filters, and queries language models. Estimation
is fast and scalable due to streaming algorithms.
endef
JLSTM-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-JLSTM = $(error $(JLSTM-ERROR))
else
DEP-JLSTM :=
endif
endif

%/user-jlstm-directory.txt: | %   
	echo '/data/compling/jlstm' > $@

################################################################################

jlstm_env: $(RESOURCE-JLSTM)/jlstm_conda_env.sh
	$(SHELL) $(word 1, $^)

jlstm: bin
	cp -r $(JLSTM-SCRIPTS)/* $<
	cd $< && bazel build -c opt lm_1b/... && cd ..

.PRECIOUS: %.jlstm.tokmeasures jlstm
%.jlstm.tokmeasures:  %.delim.linetoks jlstm  |  $(CONFIGDIR)/user-jlstm-directory.txt
	##################################################################
	### This target assumes active (jlstm_env) conda environment #####
	### Please run "make jlstm_env" and "conda activate jlstm_env" ###
	##################################################################
	bin/bazel-bin/lm_1b/lm_1b_eval --mode eval --pbtxt $(shell cat $|)/graph-2016-09-10.pbtxt \
	                               --vocab_file $(shell cat $|)/vocab-2016-09-10.txt  --input_data $(word 1,$^) \
                                   --ckpt '$(shell cat $|)/ckpt-*' > $@  2> $@.log
