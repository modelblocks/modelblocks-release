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
#  Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:
.DELETE_ON_ERROR:

## WS: VPATH MAKES IT IMPOSSIBLE TO CREATE LOCAL SCRIPTS DIRECTORY IN PROJECT DIRECTORY
#VPATH += $(RESOURCE-DUNDEE)

DUNDEESECTS = 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20
DUNDEESUBJS = sa sb sc sd se sf sg sh si sj
DUNDEESUBJSFILT := sb sc se sf sg sh si sj

DUNDEE-BASENAME := genmodel/dundee.wsj02to21-GRAMMAR-nol-prtrm-3sm-synproc-+c_+u_+b5000.gcg15_nodashtags.merged.syn.5-kenlm.eyeGP
DUNDEE-LMEDEFAULT := -NSCFl
$(eval $(call RT-EXPERIMENTS,dundee,$(DUNDEE-BASENAME),$(DUNDEE-LMEDEFAULT)))

################################################################################
#
#  User-specific parameter files (not shared; created by default with default values)
#
#  These parameter files differ from user to user, and should not be checked in.
#  This script just establishes 'official' default values for these parameters.
#
################################################################################

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-dundee-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(call CONFIGWARN, $(CONFIG)))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
DUNDEEDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(DUNDEEDIR))))
RESOURCE-NAME := The Dundee eye-tracking corpus
define RESOURCE-DESCR =

NAME: The Dundee eye-tracking corpus
MB POINTER FILE: config/user-dundee-directory.txt
AVAILABILITY: Unreleased
DESCRIPTION: A corpus of eye-tracking measures from 10 subjects who read
newspaper articles (Kennedy et al, 2003).
endef
DUNDEE-ERROR := $(call DEPWARN, $(RESOURCE-NAME), $(CONFIG), $(RESOURCE-DESCR))
DEP-DUNDEE = $(error $(DUNDEE-ERROR))
endif
endif


    
%/user-dundee-directory.txt: | %
	echo '/fs/project/lin-dept/corpora/original/english/dundee' > $@
    
################################################################################
#
#  Eye-tracking data formatting items
#
################################################################################

## copy partition ini into project genmodel directory
genmodel/%.ini: $(DUNDEE-SCRIPTS)/%.ini
	cp $< $@

#### obtain lexicon of terms that appear more than 5 times
%.wordcounts: %.linetoks $(DUNDEE-SCRIPTS)/buildlexicon.py
	python3 $(word 2,$^) $< -u5 > $@

genmodel/dundee.%.textdata: $$(DEP-DUNDEE) $(DUNDEEDIR)/tx%wrdp.dat $(CONFIGDIR)/user-dundee-directory.txt $(GENERAL-SCRIPTS)/convert_cp437.py | genmodel
	echo 'WORD ' > $@
	cat $(word 1, $^) | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;" | python3 $(word 3, $^) >> $@

genmodel/dundee.textdata:  $(foreach sect,$(DUNDEESECTS),genmodel/dundee.$(sect).textdata) | genmodel
	cat $^  >  $@

genmodel/dundee.%.eventdata: $$(DEP-DUNDEE) $(DUNDEEDIR)/%ma1p.dat $(CONFIGDIR)/user-dundee-directory.txt | genmodel
	cat $(word 1, $^) | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;" > $@

genmodel/dundee.%.eventdata:  $(foreach sect,$(DUNDEESECTS),genmodel/dundee.%$(sect).eventdata) | genmodel
	cat $^  >  $@

genmodel/dundee.%.eyedata: $$(DEP-DUNDEE) $(DUNDEEDIR)/%ma2p.dat $(CONFIGDIR)/user-dundee-directory.txt | genmodel
	cat $(word 1, $^) | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;" > $@

genmodel/dundee.%.eyedata: $(foreach sect,$(DUNDEESECTS),genmodel/dundee.%$(sect).eyedata) | genmodel
	cat $^ > $@

genmodel/dundee.%.tokfixns: $(DUNDEE-SCRIPTS)/builddundeetokfixns.py genmodel/dundee.textdata \
genmodel/dundee.%.eyedata genmodel/dundee.%.eventdata \
$(DUNDEE-SCRIPTS)/filterLexicon.py genmodel/wsj02to21.wordcounts | genmodel
	python3 $< $(word 2,$^) $(word 3,$^) $(word 4,$^) | python3 $(word 5,$^) $(word 6,$^) > $@

genmodel/dundee.%_gp.evmeasures: $(DUNDEE-SCRIPTS)/calcGPtimetoks.py genmodel/dundee.%.tokfixns \
genmodel/dundee.textdata $(RT-SCRIPTS)/sents2sentids.py genmodel/dundee.lineitems | genmodel
	paste -d' ' <(python3 $< $(word 2, $^) $(word 3, $^)) <(cat $(word 5, $^) | python $(word 4, $^) | cut -d' ' -f 2-) > $@

genmodel/dundee.%_fp.evmeasures: $(DUNDEE-SCRIPTS)/calcFPtimetoks.py genmodel/dundee.%.tokfixns \
genmodel/dundee.textdata $(RT-SCRIPTS)/sents2sentids.py genmodel/dundee.lineitems | genmodel
	paste -d' ' <(python3 $< $(word 2, $^) $(word 3, $^)) <(cat $(word 5, $^) | python $(word 4, $^) | cut -d' ' -f 2-) > $@

genmodel/dundee.%.evmeasures: $(RT-SCRIPTS)/concat_tables.py $(foreach subj, $(DUNDEESUBJS), genmodel/dundee.$(subj)_%.evmeasures) | genmodel
	python $^ > $@

#genmodel/dundee.evmeasures: genmodel/dundee.fp.evmeasures genmodel/dundee.gp.evmeasures $(RT-SCRIPTS)/rename_cols.py $(RT-SCRIPTS)/filter_cols.awk $(DTSR-SCRIPTS)/compute_timestamps_from_RT.py \
#$(RT-SCRIPTS)/add_sentence_starts_and_ends.py
#	cat $< | python $(word 3, $^) fdur fdurFP | paste -d' ' - <(awk -f $(word 4, $^) -v cols=fdur:fileid $(word 2, $^) | \
#	python $(word 3, $^) fdur fdurGP fileid docid) | python $(word 5, $^) | python $(word 6, $^) > $@

%dundee.evmeasures: $$(DEP-DUNDEE) $(DUNDEE-SCRIPTS)/process_dundee.py $(DUNDEEDIR) %dundee.sentitems
	#python3 $(word 1,$^) $(word 2,$^) <( cat $(word 3,$^) | grep -v '!ARTICLE' )  >  $@
	python3 $(word 1,$^) $(word 2,$^) <( cat $(word 3,$^) )  >  $@


################################################################################
#
#  Sents files
#
################################################################################    

# genmodel/dundee%.linetoks: $$(DEP-DUNDEE) $(DUNDEEDIR)/tx$$*wrdp.dat $(DUNDEE-SCRIPTS)/builddundeecorpus.py \
# $(GENERAL-SCRIPTS)/convert_latin-1.py $(CONFIGDIR)/user-dundee-directory.txt
# 	cat $(word 1, $^) | python2 $(word 2,$^) -e latin-1 > $@.raw
# 	cat $@.raw | perl -pe "s/\(/-LRB-/g;s/\)/-RRB-/g;s/  / /g;s/^ //" | python3 $(word 3,$^) | \
# 	sed 's/garbage\/sentimental/garbage \/ sentimental/g' \
# 	| sed 's/Oxford-educated/Oxford- educated/g' \
# 	| sed 's/4'\''s/4 '\''s/g' \
# 	| sed 's/'\''s-find-a-cure-fast/'\''s- find- a- cure- fast/g' \
# 	| sed 's/\([Aa]\)lcohol-related/\1lcohol- related/g' \
# 	| sed 's/and-mouth/and- mouth/g' \
# 	| sed 's/fork-lift/fork- lift/g' \
# 	| sed 's/in-the-box/in- the- box/g' \
# 	| sed 's/potato-and/potato- and/g' \
# 	| sed 's/Faction\/Dogs/Faction \/ Dogs/g' \
# 	| sed 's/three- year-old/three- year- old/g' \
# 	> $@
# 	rm -f $@.raw
# 
# genmodel/dundee.linetoks: $$(foreach sect,$$(DUNDEESECTS),genmodel/dundee$$(sect).linetoks) | genmodel
# 	cat $^ > $@
# 
# %dundee.lineitems: %dundee.linetoks $(RT-SCRIPTS)/toks2sents.py %dundee.textdata
# 	cat <(echo word) <(cat $(word 3, $^) | awk '!/^WORD/')  |  python2 $(word 2, $^) $< -e latin-1  > $@

genmodel/dundee%.sentitems:  $(RESOURCE-DUNDEE)/srcmodel/dundee.text.headers  $(DUNDEEDIR)/tx%wrdp.dat  $(GENERAL-SCRIPTS)/convert_cp437.py  $(DUNDEE-SCRIPTS)/findBreaks.py
	cat $(word 1,$^) $(word 2,$^)  |  perl -pe 's/ +/ /g'  |  python3 $(word 3,$^)  |  python3 $(word 4,$^) > $@
%dundee.sentitems:  $$(foreach sect,$$(DUNDEESECTS),genmodel/dundee$$(sect).sentitems) | genmodel
	cat $^  >  $@

## s/\(/-LRB- /g;s/\)/ -RRB-/g;        |  sed 's/``/'\''/g;s/'\'\''/'\''/g;s/(/-LRB-/g;s/)/-RRB-/g'
%dundee.senttoks:  %dundee.sentitems  $(TOKENIZERDIR)/ptb_tokenizer.sed
	cat $<  \
	|  perl -pe "s/  / /g;s/^ //" \
	|  sed 's/garbage\/sentimental/garbage \/ sentimental/g' \
	|  sed 's/Oxford-educated/Oxford- educated/g' \
	|  sed 's/4'\''s/4 '\''s/g' \
	|  sed 's/'\''s-find-a-cure-fast/'\''s- find- a- cure- fast/g' \
	|  sed 's/\([Aa]\)lcohol-related/\1lcohol- related/g' \
	|  sed 's/and-mouth/and- mouth/g' \
	|  sed 's/fork-lift/fork- lift/g' \
	|  sed 's/in-the-box/in- the- box/g' \
	|  sed 's/potato-and/potato- and/g' \
	|  sed 's/Faction\/Dogs/Faction \/ Dogs/g' \
	|  sed 's/three- year-old/three- year- old/g'\
  |  perl -pe "s/(\W)'(\w)/\1' \2/g"  |  $(word 2,$^)  |  perl -pe "s/can not/cannot/g;s/lock- ' em-up/lock-'em-up/g;s/ d ' / d'/g;s/([^ ])\. '/\1 \. '/g;s/([^ ])\. -/\1 \. -/g;s/''/' '/g;s/Qa ' ida/Qa'ida/g;s/-LSB- /-LSB-/g;s/ -RSB-/-RSB-/g;s/4 ' s/4 's/g;s/let's/let 's/g;s/ O ' / O'/g;s/o ' clock/o'clock/g;s/\bY ' know/Y' know/g;s/\.\. \./.../g"  |  sed "s/^' */' /g;s/$$ /$$/g"  |  sed 's/! ARTICLE/!ARTICLE/'  |  perl -pe 's/ +$$//g'  >  $@

################################################################################
#
#  Recipes for gold-standard annotated dundee corpus
#
################################################################################

# %dundee.gold.linetrees: $(GCG-SCRIPTS)/plugLeaves.py $(RESOURCE-DUNDEE)/srcmodel/dundee.stripped.linetrees %dundee.linetoks
# 	python $^ | awk 'NF' | awk '!/ARTICLE/' > $@
# 
# %dundee.delim.gold.linetrees: $(GCG-SCRIPTS)/plugLeaves.py $(RESOURCE-DUNDEE)/srcmodel/dundee.stripped.linetrees %dundee.linetoks
# 	python $^ | awk 'NF' > $@
# 
# genmodel/dundee.delim.linetoks: $(LTREES-SCRIPTS)/linetrees2linetoks.py genmodel/dundee.delim.gold.linetrees
# 	python $^ > $@
# 
# %dundee.edit.editabletrees: $$(dir %)dundee.gold.linetrees bin/indent
# 	cat $(word 1, $^) | bin/indent > $@
# 
# %.goldrules: %.linetrees $(LTREES-SCRIPTS)/editabletrees2linetrees.pl $(GENERAL-SCRIPTS)/trees2rules.pl
# 	cat $(word 1, $^) | perl $(word 2, $^) | perl $(word 3, $^) | sort | uniq -c > $@

#%dundee.gold.senttrees: $(GCG-SCRIPTS)/plugLeaves.py $(RESOURCE-DUNDEE)/srcmodel/dundee.stripped.linetrees %dundee.senttoks
#	python2 $^ | awk 'NF' | awk '!/ARTICLE/' > $@

%dundee.casp.notoks.senttrees:  $(RESOURCE-DUNDEE)/srcmodel/dundee.stripped.senttrees  $(RESOURCE-GCG)/scripts/lengthentrees.py
	cat $<  |  python3 $(word 2,$^)  >  $@

%dundee.itemmeasures: %dundee.sentitems $(RT-SCRIPTS)/lineitems2itemmeasures.py
	cat $<  |  python3 $(word 2, $^) >  $@
	#cat $<  |  grep -v '!ARTICLE'  |  python $(word 2, $^) >  $@

