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
#  include $(RESOURCE-TOKENIZER)/Makefile
#  include $(RESOURCE-LTREES)/Makefile
#  include $(RESOURCE-GCG)/Makefile
#  include $(RESOURCE-RT)/Makefile
#  include $(RESOURCE-LVPCFG)/Makefile
#  include $(RESOURCE-LCPARSE)/Makefile
#  include $(RESOURCE-BNC)/Makefile
#  include $(RESOURCE-TREEBANK)/Makefile
#  include $(RESOURCE-GIGAWORD)/Makefile
#  include $(RESOURCE-KENLM)/Makefile
#  include $(RESOURCE-SRILM)/Makefile
#  include $(RESOURCE-INCRSEM)/Makefile
#
################################################################################

################################################################################
#
#  Macros & variables
#
################################################################################

define ParamVal
UCL:
  kernel:
  - ETCorpus
  - SPRCorpus
  text: UCL (eye-tracking)
  value: ucl
endef

.SUFFIXES:
.SECONDEXPANSION:
.DELETE_ON_ERROR:

VPATH += $(RESOURCE-UCL)

UCL-BASENAME := genmodel/ucl.wsj02to21-GRAMMAR-nol-prtrm-3sm-synproc-+c_+u_+b5000.gcg15_nodashtags.merged.syn.5-kenlm.eyeGP
UCL-LMEDEFAULT := -NSCFl
$(eval $(call RT-EXPERIMENTS,ucl,$(UCL-BASENAME),$(UCL-LMEDEFAULT)))

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-ucl-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
UCLDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(UCLDIR))))
RESOURCE-NAME := The UCL Corpus (Frank et al, 2013)
define RESOURCE-DESCR =

NAME: The UCL corpus (Frank et al, 2013)
MB POINTER FILE: config/user-ucl-directory.txt
AVAILABILITY: Free
URL: http://www.stefanfrank.info/readingdata/Data.zip
DESCRIPTION: Eye-tracking and self-paced-reading data
from subjects reading isolated sentences from a corpus
of novels written by amateur authors.
endef
UCL-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-UCL = $(error $(UCL-ERROR))
endif

CONFIG := $(CONFIGDIR)/user-esn-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
ESNDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(ESNDIR))))
RESOURCE-NAME := Echo-state netork (ESN) directory
define RESOURCE-DESCR =

NAME: Echo-state netork (ESN) directory
MB POINTER FILE: config/user-esn-directory.txt
DESCRIPTION: A directory in which to store output of ESN
endef
ESN-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-ESN = $(error $(ESN-ERROR))
endif

CONFIG := $(CONFIGDIR)/user-geniatagger-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
GENIATAGDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(GENIATAGDIR))))
RESOURCE-NAME := GENIA Tagger
define RESOURCE-DESCR =

NAME: GENIA Tagger
MB POINTER FILE: config/user-geniatagger-directory.txt
AVAILABILITY: Free
URL: http://www.nactem.ac.uk/GENIA/tagger/
DESCRIPTION: Part-of-speech tagging, shallow parsing, and named entity recognition for biomedical text.
endef
GENIATAGGER-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-GENIATAGGER = $(error $(GENIATAGGER-ERROR))
endif
endif



%/user-ucl-directory.txt: | %
	echo '/home/corpora/original/english/ucl_novels' > $@

%/user-esn-directory.txt: | %
	echo '/home/corpora/original/english/dundee/echo_state' > $@

%/user-geniatagger-directory.txt: | %
	echo '/home/compling/geniatagger-2.0.2' > $@


################################################################################
#
#  Eye-tracking data formatting items
#
################################################################################

## copy partition ini into project genmodel directory
genmodel/%.ini: $(UCL-SCRIPTS)/%.ini
	cp $< $@

genmodel/ucl.merge_tables.params: | genmodel
	echo 'word sentid sentpos' > $@
  
genmodel/ucl.accumulateMetrics.params: | genmodel
	echo 'fwprob5 totsurp' $(ALL-MEM-SPILL) '-f fdurFP' > $@

genmodel/ucl.futureMetrics.params: | genmodel
	echo 'cumfwprob5 cumtotsurp' > $@

genmodel/ucl.spilloverMetrics.params: | genmodel
	echo '-n1 -p fwprob5 totsurp' > $@

genmodel/ucl.rm_unfix_items.params: | genmodel
	echo '-f fdurFP' > $@
  
%/ucl.linetoks: $$(DEP-UCL) $$(DEP-TOKENIZER) $(CONFIGDIR)/user-tokenizer-directory.txt $(CONFIGDIR)/user-ucl-directory.txt \
$(UCLDIR)/eye_movement/stimuli.txt $(TOKENIZERDIR)/ptb_tokenizer.sed | %
	cut -f 2 $(word 3,$^) | tail -n+2 | $(word 4,$^) | sed 's/ $$//g' > $@

%/ucl.lineitems: $$(DEP-UCL) $$(DEP-TOKENIZER) $(CONFIGDIR)/user-tokenizer-directory.txt $(CONFIGDIR)/user-ucl-directory.txt \
$(UCLDIR)/eye_movement/stimuli.txt | %
	cut -f 2 $(word 3,$^) | tail -n+2 > $@

%ucl.linetrees: $(GCG-SCRIPTS)/plugLeaves.py $(RESOURCE-UCL)/srcmodel/ucl.stripped.linetrees %ucl.linetoks
	python $^ > $@

%/ucl.evmeasures: $$(DEP-UCL) \
$(CONFIGDIR)/user-ucl-directory.txt \
$(UCL-SCRIPTS)/build_ucl_corpus.py \
$(UCLDIR)/eye_movement/eyetracking.RT.txt \
$(RT-SCRIPTS)/rename_cols.py \
$(RT-SCRIPTS)/filter_cols.awk \
$(DTSR-SCRIPTS)/compute_timestamps_from_RT.py | %
	cat $(word 3, $^) | python $(word 2,$^) > $@.tmp
	paste -d' ' $@.tmp <(awk -f $(word 5, $^) -v cols=sentid $@.tmp | python $(word 4, $^) sentid docid) | python $(word 6, $^) > $@
	rm $@.tmp

.PRECIOUS: $(THISDIR)/esns
$(THISDIR)/esns:
	mkdir $@
  
%.PRECIOUS: esns/ESN.mat
esns/ESN.mat: ESN.zip | esns
	#ESN.zip can be obtained upon request from Stefan Frank
	cp $(word 1,$^) esns/
	unzip esns/ESN.zip -d esns/

.PRECIOUS: genmodel/ucl_postags.mat
%/ucl_postags.mat: $$(DEP-GENIATAGGER) \
$(CONFIGDIR)/user-geniatagger-directory.txt \
$(GENIATAGDIR)/geniatagger genmodel/ucl.linetoks \
$(UCL-SCRIPTS)/build_tagseqs.py esns/ESN.mat | %
	#requires geniatagger
	ln -s $(shell cat user-geniatagger-directory.txt)/morphdic morphdic
	ln -s $(shell cat user-geniatagger-directory.txt)/models_medline models_medline
	ln -s $(shell cat user-geniatagger-directory.txt)/models_chunking models_chunking
	$(word 2,$^) < $(word 3,$^) | cut -f3 | python $(word 4,$^)
	rm -f morphdic models_medline models_chunking

.PRECIOUS: genmodel/ucl.esnsurps
genmodel/ucl.esnsurps: genmodel/ucl_postags.mat \
$(UCL-SCRIPTS)/do_testESN.m $(UCL-SCRIPTS)/pull_surp.py
	#uses ucl_postags.mat, though they aren't explicitly referenced in the recipe
	matlab -nosplash -nodesktop -r "run('$(basename $(word 2,$^))');quit;"
	python $(word 3,$^) > $@
