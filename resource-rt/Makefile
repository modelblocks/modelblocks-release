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
#  This directory contains reusable recipes for setting up reading time studies.
#  The recipes produce and then merge data from:
#  
#  1. %.pcfg.tokmeasures: Incremental pcfg parser output (parser-tokeniziation - 
#     separate punctuation toks)
#  2. %.syn.tokmeasures: PCFG parser output with center-embedding data 
#     calculated from gold trees (parser tokenization)
#  3. %.itemmeasures: All syntactic predictors with ngram probabilities 
#     (experiment tokenization, in which parser tokenization should be nested)
#  4. %.core.evmeasures: itemmeasures data merged with reading event data 
#     (*.evmeasures), i.e. the complete data set
#
#  To generate these files, the experiment must have (or have recipes to 
#  generate) the following intermediate files:
#  
#  1. %.linetoks: Parser-tokenized file of space separate sentences, one on each
#     line
#  2. %.lineitems: Experiment-tokenized file of space separated sentences,
#     one on each line
#  3. %.evmeasures: Space-delimited table of reading event data from source
#     experiment (IMPORTANT: this must include sentid and sentpos columns)
#  4. Params files:
#     a. <corpus_basename>.merge_tables.params: merge key fields (and other
#        params if needed) to pass to merge_tables.py. Typically
#        'sentid sentpos'
#     b. <corpus_basename>.accumulateMetrics.params: fields to accumulate
#        (and other params) to pass to accumulateMetrics.py
#     c. <corpus_basename>.rm_bad_toks.py: params to pass to rm_bad_toks.py
#
#  For working recipes to generate each of these kinds of files, refer to
#  resource-dundee/Makefile.
#
################################################################################

################################################################################
#
#  Includes
#
#  Include statements to all makefiles necessary to build experiment-specific
#  prerequisites to these recipes should precede the include statement for
#  this makefile.
#  
################################################################################

################################################################################
#
#  Macros & variables
#
################################################################################

.SUFFIXES:
SECONDEXPANSION:

EMPTY := 
SPACE :=$(EMPTY) $(EMPTY)
COMMA :=, 

## WS: VPATH MAKES IT IMPOSSIBLE TO CREATE LOCAL SCRIPTS DIRECTORY IN PROJECT DIRECTORY
#VPATH += $(RESOURCE-RT)

MERGESURP = paste -d' ' - <(cat $(subst GRAMMAR,$(grammar),$(basename $*))_parsed.tokmeasures | \
awk -f $(word 1, $^) -v cols=totsurp | \
sed 's/totsurp/$(subst -,,$(grammar))surp/g')

MERGEPARSERSURP = paste -d' ' - <(cat $(subst PARSER,$(parser),$(basename $*))_parsed.tokmeasures | \
awk -f $(word 1, $^) -v cols=totsurp | \
sed 's/totsurp/$(subst +,,$(subst -,,$(parser)))surp/g')

SPILLOVER := S1
SURP := fwprob5surp totsurp 
DLT := dlt dltc dltcv dltv dltm dltcm dltcvm dltvm
FJ := noF yesJ embddepthMin endembdMin startembdMin noFlen noFdr noFdrv embdlen embddr embddrv 
COREF := corefbin coreflenw coreflenr corefsize
SEMDIST := sdCosMeanW1 sdEuclMeanW1 sdNACosMeanW1 sdNAEuclMeanW1 sdCosMeanW2 sdEuclMeanW2 sdNACosMeanW2 sdNAEuclMeanW2 sdCosMeanW3 sdEuclMeanW3 sdNACosMeanW3 sdNAEuclMeanW3 sdCosMeanWinf sdEuclMeanWinf sdNACosMeanWinf sdNAEuclMeanWinf
GCG-CATS := A B C D F G
EMBDDEPTHS := 1 2 3 4 5

# The following macros automate generation of a lot of useful latency experiment
# configurations using the following template:
#
#     <experiment-name>-[cum]-[<spillover-shift>]_[<lmeargs>]
#
# Supported experiment names include all effect names in variables FJ and DLT above,
# as well as FJ (all effects stored in variable FJ), DLT (all effects stored in variable
# DLT), and MEM (all effects stored in variables FJ and DLT together). When included,
# the -cum- parameter runs the variant of the experiment with predictor(s) accumulated
# over saccade regions. When included, the -<spillover-shift>- parameter runs the
# experiment using the nth-degree spillover variant of the predictor(s) (currently S1,
# S2, and S3 supported). The optional _<lmeargs> parameter passes custom command line
# options to the main experiment script, resource-lmefit/scripts/evmeasures2lmefit.r.
# For documentation of these, run:
#
#     ../resource-lmefit/scripts/evmeasures2lmefit.r -h
#
# Examples:
#
#     1) MEM-NSCFl
#     2) DLT-S3-NSCFILl
#     3) fjprob fjprob-S1 fjprob-cum fjprob-cum-S1
#
# (1) runs all dlt and fj predictors using LME options "-NSCFl".
# (2) runs all third-degree spillover dlt predictors using LME options "-NSCFILl".
# (3) runs basic and 1st-degree spillover variants of accumulated and non-
#     accumulated variants of the fjprob predictor, using (locally-defined)
#     default LME options.
#
# To make these recipes available in a client makefile, simply add the line:
#
#     $(eval $(RT-EXPERIMENTS))
#
# Note: client makefiles must locally define the variables BASENAME and LMEDEFAULT,
# where BASENAME is the string preceding the main effect name in the *.lrtsignif
# target (see e.g. dundee/Makefile for example) and LMEDEFAULT are the default
# command line options to pass to resource-lmefit/scripts/evmeasures2lmefit.r
# (e.g. "NSCF").
#
# Also note: these recipes will not work unless the client makefile also includes
# $(RESOURCE-RT)/Makefile and all of its dependencies.
#
# Also also note: LME regressions are quite memory-intensive (usually requiring
# 2-3 times the size of your data table). Each pairwise anova created by the
# recipes below will require two such regressions. Keep this in mind when
# using parallel make (make -j), since it's easy to generate OOM errors.
# For example, KITCHENSINK below runs (|FJ|+|DLT|)*(|SPILLOVER|+1)*2*2 = 208
# total regressions, which will exceed the memory capacity of most systems.
# When running in parallel, make sure to hard-limit the number of parallel
# processes to a reasonable level (e.g. make -j10 KITCHENSINK).
#
define PRINTL
$(1)
endef

# ARGS:
#   1 = spillover position (S1, S2, etc.)
#   2 = corpus name
#   3 = basename string
#   4 = LME args
define SPILL
$(foreach var, $(DLT) $(FJ) $(COREF) $(SEMDIST),$(2)-$(var)-$(1):$(3).$(var)$(1).$(4)..lrt
)
$(foreach var, $(DLT) $(FJ) $(COREF) $(SEMDIST),$(2)-$(var)-$(1)_%:$(3).$(var)$(1).%..lrt;
)
$(2)-DLT-$(1): $(foreach var, $(DLT), $(2)-$(var)-$(1))
$(2)-DLT-$(1)_%: $(foreach var, $(DLT), $(2)-$(var)-$(1)_%);
$(2)-FJ-$(1): $(foreach var, $(FJ), $(2)-$(var)-$(1))
$(2)-FJ-$(1)_%: $(foreach var, $(FJ), $(2)-$(var)-$(1)_%);
$(2)-COREF-$(1): $(foreach var, $(COREF), $(2)-$(var)-$(1))
$(2)-COREF-$(1)_%: $(foreach var, $(COREF), $(2)-$(var)-$(1)_%);
$(2)-SEMDIST-$(1): $(foreach var, $(SEMDIST), $(2)-$(var)-$(1))
$(2)-SEMDIST-$(1)_%: $(foreach var, $(SEMDIST), $(2)-$(var)-$(1)_%);
$(2)-MEM-$(1): $(2)-DLT-$(1) $(2)-FJ-$(1) $(2)-COREF-$(1) $(2)-SEMDIST-$(1)
$(2)-MEM-$(1)_%: $(2)-DLT-$(1)_% $(2)-FJ-$(1)_% $(2)-COREF-$(1)_% $(2)-SEMDIST-$(1)_%;

$(foreach var, $(DLT) $(FJ) $(COREF) $(SEMDIST),$(2)-$(var)-cum-$(1):$(3).cum$(var)$(1).$(4)..lrt
)
$(foreach var, $(DLT) $(FJ) $(COREF) $(SEMDIST),$(2)-$(var)-cum-$(1)_%:$(3).cum$(var)$(1).%..lrt
)
$(2)-DLT-cum-$(1): $(foreach var, $(DLT), $(2)-$(var)-cum-$(1))
$(2)-DLT-cum-$(1)_%: $(foreach var, $(DLT), $(2)-$(var)-cum-$(1)_%);
$(2)-FJ-cum-$(1): $(foreach var, $(FJ), $(2)-$(var)-cum-$(1))
$(2)-FJ-cum-$(1)_%: $(foreach var, $(FJ), $(2)-$(var)-cum-$(1)_%);
$(2)-COREF-cum-$(1): $(foreach var, $(COREF), $(2)-$(var)-cum-$(1))
$(2)-COREF-cum-$(1)_%: $(foreach var, $(COREF), $(2)-$(var)-cum-$(1)_%);
$(2)-SEMDIST-cum-$(1): $(foreach var, $(SEMDIST), $(2)-$(var)-cum-$(1))
$(2)-SEMDIST-cum-$(1)_%: $(foreach var, $(SEMDIST), $(2)-$(var)-cum-$(1)_%);
$(2)-MEM-cum-$(1): $(2)-DLT-cum-$(1) $(2)-FJ-cum-$(1) $(2)-COREF-cum-$(1) $(2)-SEMDIST-cum-$(1)
$(2)-MEM-cum-$(1)_%: $(2)-DLT-cum-$(1)_% $(2)-FJ-cum-$(1)_% $(2)-COREF-cum-$(1) $(2)-SEMDIST-cum-$(1);
endef


# ARGS:
#   1 = corpus name
#   2 = basename string
#   3 = LME args
define RT-EXPERIMENTS
$(foreach var, $(DLT) $(FJ) $(COREF) $(SEMDIST),$(1)-$(var):$(2).$(var).$(3)..lrt
)
$(foreach var, $(DLT) $(FJ) $(COREF) $(SEMDIST),$(1)-$(var)_%:$(2).$(var).%..lrt;
)
$(1)-DLT: $(foreach var, $(DLT), $(1)-$(var))
$(1)-DLT_%: $(foreach var, $(DLT), $(1)-$(var)_%);
$(1)-FJ: $(foreach var, $(FJ), $(1)-$(var))
$(1)-FJ_%: $(foreach var, $(FJ), $(1)-$(var)_%);
$(1)-COREF: $(foreach var, $(COREF), $(1)-$(var))
$(1)-COREF_%: $(foreach var, $(COREF), $(1)-$(var)_%);
$(1)-SEMDIST: $(foreach var, $(SEMDIST), $(1)-$(var))
$(1)-SEMDIST_%: $(foreach var, $(SEMDIST), $(1)-$(var)_%);
$(1)-MEM: $(1)-DLT $(1)-FJ $(1)-COREF $(1)-SEMDIST
$(1)-MEM_%: $(1)-DLT_% $(1)-FJ_% $(1)-COREF_% $(1)-SEMDIST_%;

$(foreach var, $(DLT) $(FJ) $(COREF) $(SEMDIST),$(1)-$(var)-cum:$(2).cum$(var).$(3)..lrt
)
$(foreach var, $(DLT) $(FJ) $(COREF)i $(SEMDIST),$(1)-$(var)-cum_%:$(2).cum$(var).%..lrt;
)
$(1)-DLT-cum: $(foreach var, $(DLT), $(1)-$(var)-cum)
$(1)-DLT-cum_%: $(foreach var, $(DLT), $(1)-$(var)-cum_%);
$(1)-FJ-cum: $(foreach var, $(FJ), $(1)-$(var)-cum)
$(1)-FJ-cum_%: $(foreach var, $(FJ), $(1)-$(var)-cum_%);
$(1)-COREF-cum: $(foreach var, $(COREF), $(1)-$(var)-cum)
$(1)-COREF-cum_%: $(foreach var, $(COREF), $(1)-$(var)-cum_%);
$(1)-SEMDIST-cum: $(foreach var, $(SEMDIST), $(1)-$(var)-cum)
$(1)-SEMDIST-cum_%: $(foreach var, $(SEMDIST), $(1)-$(var)-cum_%);
$(1)-MEM-cum: $(1)-DLT-cum $(1)-FJ-cum $(1)-COREF-cum $(1)-SEMDIST-cum
$(1)-MEM-cum_%: $(1)-DLT-cum_% $(1)-FJ-cum_% $(1)-COREF-cum $(1)-SEMDIST-cum;

$(foreach spill, $(SPILLOVER),$(call SPILL,$(spill),$(1),$(2),$(3))
)

$(foreach var, $(DLT) $(FJ) $(COREF) $(SEMDIST),$(1)-$(var)-ALLSPILL: $(1)-$(var) $(foreach s,$(SPILLOVER),$(1)-$(var)-$(s))
)
$(foreach var, $(DLT) $(FJ) $(COREF) $(SEMDIST),$(1)-$(var)-ALLSPILL_%: $(1)-$(var)_% $(foreach s,$(SPILLOVER),$(1)-$(var)-$(s)_%);
)
$(1)-DLT-ALLSPILL: $(foreach var, $(DLT), $(1)-$(var)-ALLSPILL)
$(1)-DLT-ALLSPILL_%: $(foreach var, $(DLT), $(1)-$(var)-ALLSPILL_%);
$(1)-FJ-ALLSPILL: $(foreach var, $(FJ), $(1)-$(var)-ALLSPILL)
$(1)-FJ-ALLSPILL_%: $(foreach var, $(FJ), $(1)-$(var)-ALLSPILL_%);
$(1)-COREF-ALLSPILL: $(foreach var, $(COREF), $(1)-$(var)-ALLSPILL)
$(1)-COREF-ALLSPILL_%: $(foreach var, $(COREF), $(1)-$(var)-ALLSPILL_%);
$(1)-SEMDIST-ALLSPILL: $(foreach var, $(SEMDIST), $(1)-$(var)-ALLSPILL)
$(1)-SEMDIST-ALLSPILL_%: $(foreach var, $(SEMDIST), $(1)-$(var)-ALLSPILL_%);
$(1)-MEM-ALLSPILL: $(1)-DLT-ALLSPILL $(1)-FJ-ALLSPILL $(1)-COREF-ALLSPILL $(1)-SEMDIST-ALLSPILL
$(1)-MEM-ALLSPILL_%: $(1)-DLT-ALLSPILL_% $(1)-FJ-ALLSPILL_% $(1)-COREF-ALLSPILL $(1)-SEMDIST-ALLSPILL;

$(foreach var, $(DLT) $(FJ) $(COREF) $(SEMDIST),$(1)-$(var)-cum-ALLSPILL: $(1)-$(var)-cum $(foreach s,$(SPILLOVER),$(1)-$(var)-cum-$(s))
)
$(foreach var, $(DLT) $(FJ) $(COREF) $(SEMDIST),$(1)-$(var)-cum-ALLSPILL_%: $(1)-$(var)-cum_% $(foreach s,$(SPILLOVER),$(1)-$(var)-cum-$(s)_%);
)
$(1)-DLT-cum-ALLSPILL: $(foreach var, $(DLT), $(1)-$(var)-cum-ALLSPILL)
$(1)-DLT-cum-ALLSPILL_%: $(foreach var, $(DLT), $(1)-$(var)-cum-ALLSPILL_%);
$(1)-FJ-cum-ALLSPILL: $(foreach var, $(FJ), $(1)-$(var)-cum-ALLSPILL)
$(1)-FJ-cum-ALLSPILL_%: $(foreach var, $(FJ), $(1)-$(var)-cum-ALLSPILL_%);
$(1)-COREF-cum-ALLSPILL: $(foreach var, $(COREF), $(1)-$(var)-cum-ALLSPILL)
$(1)-COREF-cum-ALLSPILL_%: $(foreach var, $(COREF), $(1)-$(var)-cum-ALLSPILL_%);
$(1)-SEMDIST-cum-ALLSPILL: $(foreach var, $(SEMDIST), $(1)-$(var)-cum-ALLSPILL)
$(1)-SEMDIST-cum-ALLSPILL_%: $(foreach var, $(SEMDIST), $(1)-$(var)-cum-ALLSPILL_%);
$(1)-MEM-cum-ALLSPILL: $(1)-DLT-cum-ALLSPILL $(1)-FJ-cum-ALLSPILL $(1)-COREF-cum-ALLSPILL $(1)-SEMDIST-cum-ALLSPILL
$(1)-MEM-cum-ALLSPILL_%: $(1)-DLT-cum-ALLSPILL_% $(1)-FJ-cum-ALLSPILL_% $(1)-COREF-cum-ALLSPILL_% $(1)-SEMDIST-cum-ALLSPILL_%;

$(1)-KITCHENSINK: $(1)-MEM-ALLSPILL $(1)-MEM-cum-ALLSPILL
$(1)-KITCHENSINK_%: $(1)-MEM-ALLSPILL_% $(1)-MEM-cum-ALLSPILL_%;
endef


################################################################################
#
#  Combined toks file creation
#
################################################################################

define KernelBlock
RTCorpus:
  blocktitle: Reading Time Corpus
  instance_of: Corpus
  paramtype: Dropdown
ETCorpus:
  blocktitle: Eye-Tracking Corpus
  instance_of: RTCorpus
  paramtype: Dropdown
SPRCorpus:
  blocktitle: SPR Corpus
  instance_of: RTCorpus
  paramtype: Dropdown
endef

## copy partition ini into project genmodel directory
scripts/%.lmerform: $(RT-SCRIPTS)/%.lmerform |  scripts
	cp $< $@

genmodel/rename_cols.params: | genmodel
	echo '' > $@

# Pastes left-corner predictors to existing identically tokenized tokmeasures file
%.lc.tokmeasures: %.tokmeasures $$(word 1, $$(subst ., ,%)).lc.tokmeasures
	cat $< | paste -d' ' - <(cut -f 2- -d' ' $(word 2, $^)) > $@

# Computes left-corner predictors
%.lc.tokmeasures:  %.senttoks  $(RT-SCRIPTS)/calcEmbd.py  %.casp.nopunc.-u0_-c0_mlpdecpars
	cat $<  |  grep -v '!ARTICLE'  |  python $(word 2, $^) $(word 3, $^)  >  $@
    
%.lc.tokmeasures:  %.senttoks  $(RT-SCRIPTS)/calcEmbd.py  %.nopunc.-u0_-c0_mlpdecpars
	cat $<  |  python $(word 2, $^) $(word 3, $^)  >  $@

# Pastes DLT predictors to existing identically tokenized tokmeasures file
%.dlt.tokmeasures: %.tokmeasures $$(word 1, $$(subst ., ,%)).dlt.tokmeasures
	cat $< | paste -d' ' - <(cut -f 2- -d' ' $(word 2, $^)) > $@
 
# Computes DLT predictors
%.dlt.tokmeasures:  %.casp.senttrees  $(RT-SCRIPTS)/calcDLT.py
	cat $<  |  grep -v '!ARTICLE'  |  python2 $(word 2, $^)  >  $@
#%.dlt.tokmeasures: %.linetrees $(RT-SCRIPTS)/calcDLT.py
#	cat $< | python2 $(word 2, $^) > $@

# Pastes GloVe predictors to existing identically tokenized tokmeasures file
%.glove.tokmeasures: $$(basename $$(basename %)).tokmeasures $$(word 1, $$(subst ., ,%)).gigaword$$(suffix $$(basename $$*))$$(suffix $$*).glove.tokmeasures
	cat $< | paste -d' ' - <(cat $(word 2, $^) | sed 's/\t/ /g'| cut -f 3- -d' ') > $@

# Pastes Word2Vec predictors to existing identically tokenized tokmeasures file
%.word2vec.tokmeasures: $$(basename $$(basename %)).tokmeasures $$(word 1, $$(subst ., ,%)).gigaword$$(suffix $$(basename $$*))$$(suffix $$*).word2vec.tokmeasures
	cat $< | paste -d' ' - <(cat $(word 2, $^) | sed 's/\t/ /g' | cut -f 3- -d' ') > $@

# Pastes word embedding predictors to existing identically tokenized tokmeasures file
%.embeddings.tokmeasures: $$(basename $$(basename $$(basename %))).tokmeasures $$(word 1, $$(subst ., ,%))$$(suffix $$(basename $$(basename $$*)))$$(suffix $$(basename $$*))$$(suffix $$*).embeddings.tokmeasures
	cat $< | paste -d' ' - <(cat $(word 2, $^) | sed 's/\t/ /g' | cut -f 3- -d' ') > $@

%.embeddings.tokmeasures: $$(basename $$(basename %)).tokmeasures $$(word 1, $$(subst ., ,%)).gigaword$$(suffix $$(basename $$*))$$(suffix $$*).embeddings.tokmeasures
	cat $< | paste -d' ' - <(cat $(word 2, $^) | sed 's/\t/ /g' | cut -f 3- -d' ') > $@

# Pastes coreference predictors to existing identically tokenized tokmeasures file
# Pastes coreference predictors to existing identically tokenized tokmeasures file
%.coref.tokmeasures: %.tokmeasures $$(word 1, $$(subst ., ,%)).coref.tokmeasures
	cat $< | paste -d' ' - <(cat $(word 2, $^) | sed 's/\t/ /g') > $@

# Computes coreference predictors 
%.coref.tokmeasures: $(RT-SCRIPTS)/extract_coref_predictors.py %.coref.fromlinetrees.numbered.editabletrees
	python $^ > $@

# Pastes parser complexity predictors to existing identically tokenized tokmeasures file
%_parsed.tokmeasures: $$(basename %).tokmeasures \
$$(word 1, $$(subst ., ,%))$$(suffix $$*)_parsed.tokmeasures \
$(RT-SCRIPTS)/filter_cols.py
	cat $< | paste -d' ' - <(cat $(word 2, $^) | python2 $(word 3, $^) -x word sentid) > $@

# Computes parser complexity predictors 
%_parsed.tokmeasures: %_parsed.tokdecs $(RT-SCRIPTS)/filter_cols.awk
	cat $< | awk -f $(word 2, $^) -v cols=word:totsurp | grep -v '!ARTICLE' > $@

# Merges surprisal columns from a parser trained on multiple grammars
%.merged_parsed.tokmeasures: $(RT-SCRIPTS)/filter_cols.awk $$(foreach grammar,$$(subst _, ,$$(subst .,,$$(suffix $$*))),genmodel/$$(notdir $$(subst GRAMMAR,$$(grammar),$$(basename $$*)))_parsed.tokmeasures)
	cat $(word 2, $^) | sed '1s/totsurp/$(subst -,,$(word 1, $(subst _, ,$(subst .,,$(suffix $*)))))surp/' | $(foreach grammar,$(wordlist 2,100,$(subst _, ,$(subst .,,$(suffix $*)))), $(MERGESURP)) > $@

# Merges surprisal columns from multiple parsers trained on a grammar
%.parsermerged_parsed.tokmeasures: $(RT-SCRIPTS)/filter_cols.awk $$(foreach parser,$$(subst -_+,-+,$$(subst +,_+,$$(subst _, ,$$(subst .,,$$(suffix $$*))))),genmodel/$$(notdir $$(subst PARSER,$$(parser),$$(basename $$*)))_parsed.tokmeasures)
	cat $(word 2, $^) | sed '1s/totsurp/$(subst -,,$(word 1, $(subst _, ,$(subst .,,$(suffix $*)))))surp/' | $(foreach parser,$(wordlist 2,100,$(subst _, ,$(subst .,,$(suffix $*)))), $(MERGEPARSERSURP)) > $@

# %.tokmeasures with a time column, requires %.time.tokmeasures with timestamps for tokens in stimulus
%.t.tokmeasures: %.tokmeasures $$(word 1, $$(subst ., , %)).time.tokmeasures
	cat $(word 1, $^) | paste -d' ' - <(cut -f 3- -d' ' $(word 2, $^)) > $@

# Shortcut: With co-ref features
%.syn.tokmeasures: %_parsed.dlt.lc.100.1_2_3_inf.word2vec.coref.tokmeasures
	mv $^ $@

# Shortcut: Without co-ref features
%.syn.tokmeasures: %_parsed.dlt.lc.100.1_2_3_inf.word2vec.tokmeasures
	mv $^ $@

# Returns syntactic categories from gold "nol4train" linetrees
%.syncat.tokmeasures: %.casp.nol4train.senttrees $(RT-SCRIPTS)/getSyncat.py
	cat $<  |  grep -v '!ARTICLE'  |  python $(word 2, $^)  >  $@

# Pastes syntactic categories to existing identically tokenized tokmeasures file
%.syncat.tokmeasures: %.tokmeasures $$(word 1, $$(subst ., ,%)).syncat.tokmeasures
	cat $< | paste -d' ' - <(cut -f 2- -d' ' $(word 2, $^)) > $@

%.itemmeasures: $$(DEP-RHACKS) $$(basename %).tokmeasures $$(word 1, $$(subst ., , %))$$(subst -,.,$$(suffix $$*)).ngram.itemmeasures \
$(RT-SCRIPTS)/roll_toks.py $(RT-SCRIPTS)/filter_cols.py $(RT-SCRIPTS)/addCols.r $(abspath $(RHACKS-SCRIPTS)/mer-utils.R) $(abspath $(RHACKS-SCRIPTS)/regression-utils.R) \
$$(word 1, $$(subst ., , %))$$(suffix $$(subst -,.,$$(suffix $$*))).unigram.itemmeasures
	cat $< | python2 $(word 3, $^) $(word 2, $^) sentid embddepthMin timestamp -e latin-1 | \
	paste -d' ' - <(cut -f 2- -d' ' $(word 2, $^)) | \
        paste -d' ' - <(cut -f 2- -d' ' $(word 8, $^)) | python2 $(word 4, $^) -d | \
	$(word 5, $^) > $@

%.t.itemmeasures: $$(basename %).t.tokmeasures $$(word 1, $$(subst ., , %))$$(subst -,.,$$(suffix $$*)).ngram.itemmeasures \
$(RT-SCRIPTS)/roll_toks.py $(RT-SCRIPTS)/filter_cols.py $(RT-SCRIPTS)/addCols.r $(abspath $(RHACKS-SCRIPTS)/mer-utils.R) $(abspath $(RHACKS-SCRIPTS)/regression-utils.R) \
$$(word 1, $$(subst ., , %))$$(suffix $$(subst -,.,$$(suffix $$*))).unigram.itemmeasures
	cat $< | python2 $(word 3, $^) $(word 2, $^) sentid embddepthMin timestamp -e latin-1 | \
	paste -d' ' - <(cut -f 2- -d' ' $(word 2, $^)) | \
        paste -d' ' - <(cut -f 2- -d' ' $(word 8, $^)) | python2 $(word 4, $^) -d | \
	$(word 5, $^) > $@

%.t.itemmeasures: $$(basename %).tokmeasures $$(word 1, $$(subst ., , %))$$(subst -,.,$$(suffix $$*)).ngram.t.itemmeasures \
$(RT-SCRIPTS)/roll_toks.py $(RT-SCRIPTS)/filter_cols.py $(RT-SCRIPTS)/addCols.r $(abspath $(RHACKS-SCRIPTS)/mer-utils.R) $(abspath $(RHACKS-SCRIPTS)/regression-utils.R) \
$$(word 1, $$(subst ., , %))$$(suffix $$(subst -,.,$$(suffix $$*))).unigram.itemmeasures
	cat $< | python2 $(word 3, $^) $(word 2, $^) sentid embddepthMin timestamp -e latin-1 | \
	paste -d' ' - <(cut -f 2- -d' ' $(word 2, $^)) | \
        paste -d' ' - <(cut -f 2- -d' ' $(word 8, $^)) | python2 $(word 4, $^) -d | \
	$(word 5, $^) > $@

%.r.itemmeasures: $$(basename %).tokmeasures $$(word 1, $$(subst ., , %))$$(subst -,.,$$(suffix $$*)).ngram.itemmeasures $$(basename %).roark.itemmeasures \
$(RT-SCRIPTS)/roll_toks.py $(RT-SCRIPTS)/filter_cols.py $(RT-SCRIPTS)/addCols.r $(abspath $(RHACKS-SCRIPTS)/mer-utils.R) $(abspath $(RHACKS-SCRIPTS)/regression-utils.R) \
$$(word 1, $$(subst ., , %))$$(suffix $$(subst -,.,$$(suffix $$*))).unigram.itemmeasures
	cat $< | python2 $(word 4, $^) $(word 2, $^) sentid embddepthMin timestamp -e latin-1 | \
	paste -d' ' - <(cut -f 2- -d' ' $(word 2, $^)) | paste -d' ' - $(word 3, $^) | \
        paste -d' ' - <(cut -f 2- -d' ' $(word 8, $^)) | python2 $(word 5, $^) -d | \
	$(word 6, $^) > $@

%.docids.itemmeasures: %.itemmeasures $$(word 1,$$(subst ., ,%)).docids.itemmeasures $(RT-SCRIPTS)/merge_tables.py
	python $(word 3, $^) $(word 1, $^) $(word 2, $^) word sentpos sentid > $@


define CompositeBlock
RTDataParams:
  blockseq:
  - blocktype: RTCorpus
  - blocktype: ParseParams
    kwargs:
      innerdelim: '-'
      instancename: Surprisal Metrics
  - blocktype: NgramParams
    kwargs:
      innerdelim: '-'
  blocktitle: Reading Time Data Parameters
endef
define TargetBlock
RTCoreEvmeasures:
  blocktitle: Reading Time Data
  targetsuffix: .core.evmeasures
  family: Event Data
  blockseq:
  - blocktype: RTDataParams
    kwargs:
      innerdelim: .
endef

%.itemmeasures: %.sentitems $(RT-SCRIPTS)/lineitems2itemmeasures.py
	cat $<  |  grep -v '!ARTICLE'  |  python $(word 2, $^) >  $@

%parsed.itemmeasures: %parsed.tokmeasures  $$(basename %).itemmeasures  $(RT-SCRIPTS)/roll_toks.py
	cat $< | python2 $(word 3, $^) $(word 2, $^) sentid embddepthMin timestamp -e latin-1  >  $@

%.itemmeasures: %.tokmeasures $$(basename %).itemmeasures $(RT-SCRIPTS)/roll_toks.py
	cat $< | python2 $(word 3, $^) $(word 2, $^) sentid embddepthMin timestamp -e latin-1 > $@

%.all-itemmeasures:  $(RT-SCRIPTS)/evmeas2uniqcols.py  $(RT-SCRIPTS)/filter_cols.py  $(RT-SCRIPTS)/addCols.r  $(abspath $(RHACKS-SCRIPTS)/mer-utils.R) $(abspath $(RHACKS-SCRIPTS)/regression-utils.R) \
                         $$(patsubst $$(pc),$$(notdir $$(word 1,$$(subst ., ,$$*))).$$(pc).itemmeasures,$$(wordlist 2,$$(words $$(subst ., ,$$*)),$$(subst ., ,%))) \
			 $$(word 1, $$(subst ., ,%)).itemmeasures
	paste -d ' ' $(wordlist 6,$(words $^),$^)  |  python2 $(word 1,$^)  |  python $(word 2,$^)  |   $(word 3,$^)  >  $@
#%.core-itemmeasures: $$(patsubst $$(pc),$$(word 1,$$(subst ., ,%)).$$(pc).itemmeasures,$$(wordlist 2,$$(words $$(subst ., ,%)),$$(subst ., ,%))) 
#	paste $^  >  $@

%.delim.lineitems: %.itemmeasures $(RT-SCRIPTS)/itemmeasures2delimlineitems.py
	cat $< | python $(word 2, $^) > $@

## TO SPEC
.PRECIOUS: %.core-evmeasures
%.core-evmeasures:  %.all-itemmeasures  $$(word 1, $$(subst ., , %)).evmeasures  $(RT-SCRIPTS)/merge_tables.py
	python $(word 3, $^) $< $(word 2, $^) sentid sentpos  >  $@

## PRD/RESMEASURES SPEC
.PRECIOUS: %.prdmeasures
%.prdmeasures:  %.all-itemmeasures  $$(word 1, $$(subst ., , %)).evmeasures  $(RT-SCRIPTS)/merge_tables.py
	python $(word 3, $^) $< $(word 2, $^) sentid sentpos  >  $@
%.resmeasures:  %.all-itemmeasures  $$(word 1, $$(subst ., , %)).evmeasures  $(RT-SCRIPTS)/merge_tables.py
	cp $(word 2,$^) $@
#	python $(word 3, $^) $< $(word 2, $^) sentid sentpos  >  $@
#	ln -sf $(abspath $(word 2,$^)) $(abspath $@)


## TO SPEC
%.filt-evmeasures: $$(basename %).core-evmeasures scripts/$$(word 1, $$(subst _, ,$$(subst .,,$$(suffix $$*)))).lmeform \
$(RT-SCRIPTS)/inferPredictors.py \
$(RT-SCRIPTS)/filter_cols.awk \
$(RT-SCRIPTS)/accumulateMetrics.py \
$(RT-SCRIPTS)/rm_unfix_items.py \
$(RT-SCRIPTS)/futureMetrics.py \
$(RT-SCRIPTS)/spilloverMetrics.py \
$(RT-SCRIPTS)/parsePredictors.py \
$(RT-SCRIPTS)/rm_na_items.py
	$(eval CORECOLS := $(sort $(shell cat $(word 2, $^) | python $(word 3, $^) 1 ) $(shell echo $(subst +, ,$(word 2, $(subst _, ,$(subst .,,$(suffix $*))))) | python $(word 9, $^))))
	$(eval COLS := $(sort $(shell cat $(word 2, $^) | python $(word 3, $^)) $(subst +, ,$(word 2, $(subst _, ,$(subst .,,$(suffix $*)))))))
	@echo Extracting columns: $(CORECOLS)
	@echo Computing columns: $(COLS)
	awk -f $(word 4, $^) -v cols=$(subst $(SPACE),:,$(strip $(CORECOLS))) $< | \
	python $(word 5, $^) -c $(shell echo $(COLS) | python $(word 9, $^) -a) | \
	python $(word 6, $^) | \
	python $(word 7, $^) -I -c $(shell echo $(COLS) | python $(word 9, $^) -f) | \
	python $(word 8, $^) -I -C $(shell echo $(COLS) | python $(word 9, $^) -s) | \
	awk -f $(word 4, $^) -v cols=$(subst $(SPACE),:,$(strip $(COLS))) | \
	python $(word 10, $^) > $@



## change .itemmeasures to .core-itemmeasures
.PRECIOUS: %.core.evmeasures
%.core.evmeasures: %.itemmeasures $$(word 1, $$(subst ., , %)).evmeasures \
$(RT-SCRIPTS)/merge_tables.py
	python $(word 3, $^) $< $(word 2, $^) sentid sentpos > $@

## identical except filt.,core. <-> filt-,core-
%.filt.evmeasures: $$(basename %).core.evmeasures scripts/$$(word 1, $$(subst _, ,$$(subst .,,$$(suffix $$*)))).lmeform \
$(RT-SCRIPTS)/inferPredictors.py \
$(RT-SCRIPTS)/filter_cols.awk \
$(RT-SCRIPTS)/accumulateMetrics.py \
$(RT-SCRIPTS)/rm_unfix_items.py \
$(RT-SCRIPTS)/futureMetrics.py \
$(RT-SCRIPTS)/spilloverMetrics.py \
$(RT-SCRIPTS)/parsePredictors.py \
$(RT-SCRIPTS)/rm_na_items.py
	$(eval CORECOLS := $(sort $(shell cat $(word 2, $^) | python $(word 3, $^) 1 ) $(shell echo $(subst +, ,$(word 2, $(subst _, ,$(subst .,,$(suffix $*))))) | python $(word 9, $^))))
	$(eval COLS := $(sort $(shell cat $(word 2, $^) | python $(word 3, $^)) $(subst +, ,$(word 2, $(subst _, ,$(subst .,,$(suffix $*)))))))
	@echo Extracting columns: $(CORECOLS)
	@echo Computing columns: $(COLS)
	awk -f $(word 4, $^) -v cols=$(subst $(SPACE),:,$(strip $(CORECOLS))) $< | \
	python $(word 5, $^) -c $(shell echo $(COLS) | python $(word 9, $^) -a) | \
	python $(word 6, $^) | \
	python $(word 7, $^) -I -c $(shell echo $(COLS) | python $(word 9, $^) -f) | \
	python $(word 8, $^) -I -C $(shell echo $(COLS) | python $(word 9, $^) -s) | \
	awk -f $(word 4, $^) -v cols=$(subst $(SPACE),:,$(strip $(COLS))) | \
	python $(word 10, $^) > $@

%.train.evmeasures: %.evmeasures $(RT-SCRIPTS)/split.py
	cat $(word 1, $^) | python $(word 2, $^) train > $@

%.dev.evmeasures: %.evmeasures $(RT-SCRIPTS)/split.py
	cat $(word 1, $^) | python $(word 2, $^) dev > $@

%.test.evmeasures: %.evmeasures $(RT-SCRIPTS)/split.py
	cat $(word 1, $^) | python $(word 2, $^) test > $@

# Spillover evaluation for a baseline model, comparing all permutations of adjacent spillover values
# Because the *.lmeform names are unpredictable this is a weird target that will not be able to clean
# up the intermediate *.lmeform files it creates. To delete them manually, just run:
#     
#     rm -f scripts/*SP.lmeform
#
# NOTE: this assumes you have not created any *.lmeform files of your own that end in *SP.lmeform.
# If you have, it would be best to rename them before running this target.
BASENAME-NB := genmodel/dundee.wsj02to21-gcg15-decoupled-fg-3sm-bd-x+efabp-+c_+b5000.syn.5-kenlm
BASELINE-SPILL-EVAL-%: $(RT-SCRIPTS)/permuteSpillover.py $(RT-SCRIPTS)/$$(word 1, $$(subst _, ,%)).lmeform \
$$(foreach b, \
$$(shell python $$(RT-SCRIPTS)/permuteSpillover.py -m0 -M1 $$(RT-SCRIPTS)/$$(word 1, $$(subst _, ,$$*)).lmeform scripts) \
$$(shell python $$(RT-SCRIPTS)/permuteSpillover.py -m1 -M2 $$(RT-SCRIPTS)/$$(word 1, $$(subst _, ,$$*)).lmeform scripts) \
$$(shell python $$(RT-SCRIPTS)/permuteSpillover.py -m2 -M3 $$(RT-SCRIPTS)/$$(word 1, $$(subst _, ,$$*)).lmeform scripts), \
$$(BASENAME-NB).$$(basename $$(notdir $$(b))).-$$(word 2, $$(subst _, ,$$*)).lm);

BASELINE-SPILL-EVAL-%: $(RT-SCRIPTS)/permuteSpillover.py $(CURDIR)/scripts/$$(word 1, $$(subst _, ,%)).lmeform \
$$(foreach b, \
$$(shell python $$(RT-SCRIPTS)/permuteSpillover.py -m0 -M1 $$(RT-SCRIPTS)/$$(word 1, $$(subst _, ,$$*)).lmeform scripts) \
$$(shell python $$(RT-SCRIPTS)/permuteSpillover.py -m1 -M2 $$(RT-SCRIPTS)/$$(word 1, $$(subst _, ,$$*)).lmeform scripts) \
$$(shell python $$(RT-SCRIPTS)/permuteSpillover.py -m2 -M3 $$(RT-SCRIPTS)/$$(word 1, $$(subst _, ,$$*)).lmeform scripts), \
$$(BASENAME-NB).$$(basename $$(notdir $$(b))).-$$(word 2, $$(subst _, ,$$*)).lm);

################################################################################
#
#  Utility recipes
#
################################################################################

%.evmeasures.tsv: %.evmeasures
	cat $< | sed 's/ /\t/g' > $@

%.tokmeasures.tsv: %.tokmeasures
	cat $< | sed 's/ /\t/g' > $@

