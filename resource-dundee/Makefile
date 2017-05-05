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
#  MB-DOC 
#
################################################################################

def ParamVal :=
Dundee:
  kernel: ETCorpus
  text: Dundee (eye-tracking corpus)
  value: dundee
endef

################################################################################
#
#  Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:

DUNDEESECTS = 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20
DUNDEESUBJS = sa sb sc sd se sf sg sh si sj
DUNDEESUBJSFILT := sb sc se sf sg sh si sj

MERGESURP = python $(word 1, $^) --input $(subst GRAMMAR,$(grammar),$(basename $*)).tokmeasures --i totsurp | \
sed '1s/.*/$(subst -,,$(grammar))totsurp/' | paste -d' ' $@ - > $@.tmp; \
mv $@{.tmp,}; \
python $(word 1, $^) --input $(subst GRAMMAR,$(grammar),$(basename $*)).tokmeasures --i cumtotsurp | \
sed '1s/.*/$(subst -,,$(grammar))cumtotsurp/' | paste -d' ' $@ - > $@.tmp; \
mv $@{.tmp,};

SPILL-DUNDEE := S1 S2 S3
DLT-DUNDEE := dlt dltc dltcv dltv dltm dltcm dltcvm dltvm
FJ-DUNDEE := fjprob endembd embdlen embdlendr embdlendrv noF noFlen noFlendr noFlendrv reinst reinstlen reinstlendr reinstlendrv
ALL-MEM = $(DLT-DUNDEE) $(FJ-DUNDEE)
ALL-MEM-SPILL = $(ALL-MEM) $(foreach var,$(ALL-MEM),$(foreach spill,$(SPILL-DUNDEE),$(var)$(spill)))


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

define CONFIGWARN =

ATTENTION! I had to create $(CONFIG),
which contains an incorrect default value (".").
Targets with dependencies to this file will fail until it is fixed.

endef

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-dundee-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
DUNDEEDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(DUNDEEDIR))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(DUNDEEDIR)$(MSG3))
endif
endif


    
%/user-dundee-directory.txt: | %
	echo '/home/corpora/original/english/dundee' > $@
    
################################################################################
#
#  Eye-tracking data formatting items
#
################################################################################

#### obtain lexicon of terms that appear more than 5 times
%.wordcounts: %.linetoks $(DUNDEE-SCRIPTS)/buildlexicon.py
	python3 $(word 2,$^) $< -u5 > $@

genmodel/dundee.%.textdata: $(DUNDEEDIR)/tx%wrdp.dat $(CONFIGDIR)/user-dundee-directory.txt $(GENERAL-SCRIPTS)/convert_latin-1.py | genmodel
	echo 'WORD ' > $@
	cat $(word 1,$^) | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;" | python3 $(word 3, $^) >> $@

genmodel/dundee.textdata:  $(foreach sect,$(DUNDEESECTS),genmodel/dundee.$(sect).textdata) | genmodel
	cat $^  >  $@

genmodel/dundee.%.eventdata: $(DUNDEEDIR)/%ma1p.dat $(CONFIGDIR)/user-dundee-directory.txt | genmodel
	cat $(word 1,$^) | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;" > $@

genmodel/dundee.%.eventdata:  $(foreach sect,$(DUNDEESECTS),genmodel/dundee.%$(sect).eventdata) | genmodel
	cat $^  >  $@

genmodel/dundee.%.eyedata: $(DUNDEEDIR)/%ma2p.dat $(CONFIGDIR)/user-dundee-directory.txt | genmodel
	cat $(word 1,$^) | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;" > $@

genmodel/dundee.%.eyedata: $(foreach sect,$(DUNDEESECTS),genmodel/dundee.%$(sect).eyedata) | genmodel
	cat $^ > $@

genmodel/dundee.%.tokfixns: $(DUNDEE-SCRIPTS)/builddundeetokfixns.py genmodel/dundee.textdata \
genmodel/dundee.%.eyedata genmodel/dundee.%.eventdata \
$(DUNDEE-SCRIPTS)/filterLexicon.py genmodel/wsj02to21.wordcounts | genmodel
	python3 $< $(word 2,$^) $(word 3,$^) $(word 4,$^) | python3 $(word 5,$^) $(word 6,$^) > $@

genmodel/dundee.%_gp.src.evmeasures: $(DUNDEE-SCRIPTS)/calcGPtimetoks.py genmodel/dundee.%.tokfixns \
genmodel/dundee.textdata $(RT-SCRIPTS)/sents2sentids.py genmodel/dundee.lineitems | genmodel
	paste -d' ' <(python3 $< $(word 2, $^) $(word 3, $^)) <(cat $(word 5, $^) | python $(word 4, $^) | cut -d' ' -f 2-) > $@

genmodel/dundee.%_fp.src.evmeasures: $(DUNDEE-SCRIPTS)/calcFPtimetoks.py genmodel/dundee.%.tokfixns \
genmodel/dundee.textdata $(RT-SCRIPTS)/sents2sentids.py genmodel/dundee.lineitems | genmodel
	paste -d' ' <(python3 $< $(word 2, $^) $(word 3, $^)) <(cat $(word 5, $^) | python $(word 4, $^) | cut -d' ' -f 2-) > $@

genmodel/dundee.%.evmeasures: $(RT-SCRIPTS)/concat_tables.py $(foreach subj, $(DUNDEESUBJS), genmodel/dundee.$(subj)_%.src.evmeasures) | genmodel
	python $^ > $@

genmodel/dundee.src.evmeasures: genmodel/dundee.fp.evmeasures genmodel/dundee.gp.evmeasures $(RT-SCRIPTS)/rename_cols.py $(RT-SCRIPTS)/filter_cols.awk
	cat $< | python $(word 3, $^) fdur fdurFP | paste -d' ' - <(awk -f $(word 4, $^) -v cols=fdur:fileid $(word 2, $^) | \
	python $(word 3, $^) fdur fdurGP fileid docid) > $@


################################################################################
#
#  Dundee sents files
#
################################################################################    

dundee%.linetoks: $(DUNDEEDIR)/tx$$*wrdp.dat $(DUNDEE-SCRIPTS)/builddundeecorpus.py \
$(GENERAL-SCRIPTS)/convert_latin-1.py $(CONFIGDIR)/user-dundee-directory.txt
	cat $< | python $(word 2,$^) > $@.raw
	cat $@.raw | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;s/^ //" | python3 $(word 3,$^) | \
	sed 's/garbage\/sentimental/garbage \/ sentimental/g' \
	| sed 's/Oxford-educated/Oxford- educated/g' \
	| sed 's/4'\''s/4 '\''s/g' \
	| sed 's/'\''s-find-a-cure-fast/'\''s- find- a- cure- fast/g' \
	| sed 's/\([Aa]\)lcohol-related/\1lcohol- related/g' \
	| sed 's/and-mouth/and- mouth/g' \
	| sed 's/fork-lift/fork- lift/g' \
	| sed 's/in-the-box/in- the- box/g' \
	| sed 's/potato-and/potato- and/g' \
	| sed 's/Faction\/Dogs/Faction \/ Dogs/g' \
	| sed 's/three- year-old/three- year- old/g' \
	> $@
	rm -f $@.raw

%dundee.linetoks: $$(foreach sect,$$(DUNDEESECTS),%dundee$$(sect).linetoks)
	cat $^ > $@

%dundee.lineitems: %dundee.linetoks $(RT-SCRIPTS)/toks2sents.py %dundee.textdata
	cat <(echo word) <(cat $(word 3, $^) | awk '!/^WORD/') | python $(word 2, $^) $< > $@

################################################################################
#
#  Param files for constructing experiment data table
#
################################################################################    

genmodel/dundee.merge_tables.params: | genmodel
	echo 'word sentid sentpos' > $@

genmodel/dundee.accumulateMetrics.params: | genmodel
	echo 'fwprob5 totsurp' $(ALL-MEM-SPILL) > $@

genmodel/dundee.spilloverMetrics.params: | genmodel
	echo '-n1 -p fwprob5 totsurp' > $@

genmodel/dundee.futureMetrics.params: | genmodel
	echo 'cumfwprob5 cumtotsurp' > $@

genmodel/dundee.rm_unfix_items.params: | genmodel
	echo '' > $@

################################################################################
#
#  Recipes for gold-standard annotated dundee corpus
#
################################################################################

%dundee.gold.linetrees: $(GCG-SCRIPTS)/plugLeaves.py $(RESOURCE-DUNDEE)/srcmodel/dundee.stripped.linetrees %dundee.linetoks
	python $^ > $@

%dundee.gold.edit.editabletrees: $$(dir %)dundee.gold.linetrees bin/indent
	cat $(word 1, $^) | bin/indent > $@

%.goldrules: %.linetrees $(LTREES-SCRIPTS)/editabletrees2linetrees.pl $(GENERAL-SCRIPTS)/trees2rules.pl
	cat $(word 1, $^) | perl $(word 2, $^) | perl $(word 3, $^) | sort | uniq -c > $@
