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
#
################################################################################

################################################################################
#
#  Macros & variables
#
################################################################################

.SUFFIXES:
.SECONDEXPANSION:

GENMODEL := genmodel

################################################################################
#
#  Code compilation items
#
################################################################################

# CODE REVIEW: some of these are bogus!!!!
SHELL = /bin/bash
#INCLUDES = -I$(abspath $(MBDIR)/resource-rvtl) #-Iinclude -I../slush/include #-I/sw/include #-I/Users/dingcheng/Documents/boost/boost_1_44_0
#CFLAGS = -Wall `cat user-cflags.txt` -g #-DNDEBUG -O3 #-DNOWARNINGS #-g #
#CC = g++

#### bin directory (ignored by git b/c empty)
bin:
	if [ ! -d $@ ]; then mkdir $@; fi

bin/evalb: $(RESOURCE-LTREES)/src/evalb.c | bin
	gcc -Wall -g -o $@ $<

#### c++ executables
.PRECIOUS: bin/%
bin/%:  $(RESOURCE-LTREES)/src/$$(notdir $$@).cpp  $(CONFIGDIR)/user-cflags.txt  |  bin
	g++ -I$(abspath $(MBDIR)/resource-rvtl) -Wall `cat $(word 2,$^)` -g -lm  $<  -o $@

################################################################################
#
#  Tree manipulations
#
################################################################################

#### obtain (flat) linetrees from (indented) editabletrees
%.linetrees:  %.editabletrees  $(LTREES-SCRIPTS)/editabletrees2linetrees.pl
	cat $<  |  perl -pe 's/^[0-9]*://'  |  perl $(word 2,$^)  >  $@
%.fromsenttrees.toktrees:  %.senttrees  bin/indent
	cat $<  |  $(word 2,$^)  >  $@
%.numbered.toktrees:  %.toktrees  $(LTREES-SCRIPTS)/make-trees-numbered.pl
	cat $<  |  perl $(word 2,$^)  >  $@

### HAPPENS TOO MUCH; JUST CREATE MAKE ITEM IF YOU NEED IT
##### generalized operations on linetrees
#%.linetrees:  $(LTREES-SCRIPTS)/make-trees-$$(subst .,,$$(suffix $$*)).pl  $$(basename %).linetrees
#	cat $(word 2,$^)  |  perl $(word 1,$^)  >  $@
%.lower.linetrees:  $(LTREES-SCRIPTS)/make-trees-lower.py  %.linetrees
	cat $(word 2,$^)  |  python $(word 1,$^)  >  $@
#### generate right-branching linetrees from linetoks
%.rb.linetrees : %.linetoks $(LTREES-SCRIPTS)/sents2rb.py
	cat $< | python $(word 2, $^) > $@

#### use only the first/last N trees
%first.senttrees:  $$(basename %).senttrees
	head -$(subst .,,$(suffix $*)) $<  >  $@

%onward.senttrees: $$(basename %).senttrees
	tail -n +$(subst .,,$(suffix $*)) $<  >  $@

%last.senttrees:  $$(basename %).senttrees
	tail -$(subst .,,$(suffix $*)) $<  >  $@

%maxwords.senttrees: $$(basename %).senttrees
	cat $< | perl -na -e "if (split(/[\)]+/)<=1+$(subst .,,$(suffix $*))) {print $$_;}"  >  $@ 

#### remove unary branches
%.nounary.senttrees:  %.senttrees  $(LTREES-SCRIPTS)/killUnaries.pl
	cat $<  |  perl $(word 2,$^)  >  $@

#### use treebank reannotation
%.nodashtags.linetrees:  %.linetrees
	cat $<  |  perl -pe 's/([^\(])[-=][^ \)]+ /\1 /g'  >  $@

#### obtain tokenized input sentences from linetrees

#### This item is used to flatten trees from treebank corpora *in the genmodel directory* into tokenized parser input...
#%.linetoks:  $(GENMODEL)/%.linetrees   #$(GENMODEL)/$$(subst +,-,$$(subst -,.,$$(notdir $$*.linetrees)))
%.senttoks:  $(GENMODEL)/$$(notdir $$*.senttrees)   #$(GENMODEL)/$$(subst +,-,$$(subst -,.,$$(notdir $$*.senttrees)))
	cat $<  |  sed 's/(-NONE-[^)]*)//g'  \
		|  sed 's/([^ ]* //g;s/)//g'  |  sed 's/  */ /g;s/^ *//;s/ *$$//;'  \
		|  sed 's/!unf! *//g' >  $@
#### This item is used to flatten trees from treebank corpora into tokenized parser input...
%.senttoks:  %.senttrees
	cat $<  |  sed 's/(-NONE-[^)]*)//g'  \
		|  sed 's/([^ ]* //g;s/)//g'  |  sed 's/  */ /g;s/^ *//;s/ *$$//;'  \
		|  sed 's/!unf! *//g' >  $@
%.morph.linetoks: %.linetoks
	morfessor -t $^ -T $^ --output-format={analysis}' ' --output-newlines > $@

%.delim.linetoks: %.linetoks $(LTREES-SCRIPTS)/linetoks2delimlinetoks.py
	cat $< | python $(word 2,$^) > $@

#### add leaves to non-copyright-infringing defoliated trees...
%.senttrees:  %.notoks.senttrees  $(GENMODEL)/$$(word 1,$$(subst ., ,$$(notdir $$*))).senttoks  $(GCG-SCRIPTS)/plugLeaves.py
	python3 $(word 3,$^) $(word 1,$^) $(word 2,$^)  >  $@
#### remove leaves from senttrees...
%.fromsenttrees.notoks.senttrees:  %.senttrees
	cat $<  |  perl -pe 's/ [^)(]+/ LEAF!/g'  >  $@

#### reverse the order of tokens by sentence in an input linetoks file
%.rev.linetoks: %.linetoks
	tac $^ | awk '{for (i=NF; i > 1; i--) printf("%s ", $$i); print $$1;}' > $@

#### (DEPRECATED) obtain input sentences from linetrees
%.sents:  $(GENMODEL)/$$(subst +,-,$$(subst -,.,$$(notdir $$*.linetrees)))
	cat $<  |  sed 's/(-NONE-[^)]*)//g'  \
		|  sed 's/([^ ]* //g;s/)//g'  |  sed 's/  */ /g;s/^ *//;s/ *$$//;'  \
		|  sed 's/!unf! *//g' >  $@
%.sents:  %.linetrees
	cat $<  |  sed 's/(-NONE-[^)]*)//g'  \
		|  sed 's/([^ ]* //g;s/)//g'  |  sed 's/  */ /g;s/^ *//;s/ *$$//;'  \
		|  sed 's/!unf! *//g' >  $@

#### unlabeled trees (for evaluation)
%.nolabel.linetrees:  %.linetrees
	cat $^  |  perl -pe 's/\(\)//'| perl -pe 's/\([^ ]*/\(X/g'  >  $@

%.nopunc.senttrees: %.senttrees $(LTREES-SCRIPTS)/make-trees-nounary.pl
	cat $<  |  perl $(word 2, $^ ) |  perl -pe "s/ \([^ ()]+ (,|\.|\`\`|\`|--|-RRB-|-LRB-|-LCB-|-RCB-|''|'|\.\.\.|\?|\!|\:|\;)\)//g" | perl $(word 2, $^) > $@

%.nopunc.linetoks: %.linetoks
	cat $<  |  perl -pe "s/ (,|\.|\`\`|\`|--|-RRB-|-LRB-|-LCB-|-RCB-|''|'|\.\.\.|\?|\!|\:|\;|\(|\))//g" > $@

#%.blankfail.linetrees: %.linetrees
#	cat $< | sed 's/^.*FAIL.*$$//g' > $@

%.nocurr.linetrees: %.linetrees $(LTREES-SCRIPTS)/make-trees-nounary.pl
	cat $< | sed 's/\(\$$ \+[^ ()]\+\)//g;' | perl $(word 2, $^) > $@

%.replparens.linetoks: %.linetoks
	cat $<  |  sed 's/(/-LRB-/g; s/)/-RRB-/g;'  > $@

# Shortcut for %.nt.lower.nounary.nolabel.nopunc.linetrees.
%.induc.linetrees: %.nt.lower.nounary.nolabel.nopunc.linetrees
	cat $< > $@

%.fromdeps.linetrees: %.tokdeps $(LTREES-SCRIPTS)/deps2trees.py
	cat $< | sed 's/<num>/-NUM-/g' | python $(word 2, $^) -f stanford > $@

#### generates unlabeled stanford dependencies from linetrees file
%.tokdeps: %.linetrees $(LTREES-SCRIPTS)/trees2deps.py %.head.model
	cat $< | python3 $(word 2, $^) $(word 3, $^) > $@


#### generates a CoNLL-style table of PoS tags from linetrees
%.conll: %.linetrees $(LTREES-SCRIPTS)/get_pos.py
	cat $< | python $(word 2, $^) -i > $@

#### use only the first/last N sents
%first.senttoks:  $$(basename %).senttoks
	head -$(subst .,,$(suffix $*)) $<  >  $@
%onward.senttoks: $$(basename %).senttoks
	tail -n +$(subst .,,$(suffix $*)) $<  >  $@
%last.senttoks:  $$(basename %).senttoks
	tail -$(subst .,,$(suffix $*)) $<  >  $@
#### (DEPRECATED) use only the first/last N trees
%first.sents:  $$(basename %).sents
	head -$(subst .,,$(suffix $*)) $<  >  $@
%onward.sents: $$(basename %).linetoks
	tail -n +$(subst .,,$(suffix $*)) $<  >  $@
%last.sents:  $$(basename %).linetoks
	tail -$(subst .,,$(suffix $*)) $<  >  $@

#### rules
%.rules: %.senttrees  $(LTREES-SCRIPTS)/trees2rules.pl
	cat $<  |  perl $(word 2,$^)  >  $@
#	cat $< | sed 's/:/-COLON-/g;s/=/-EQUALS-/g' |  perl $(word 2,$^)  >  $@

%.model: %.rules
	cat $< | sort | uniq -c | sort -nr | awk '{"wc -l $< | cut -d\" \" -f1" | getline t; u = $$1; $$1 = u/t; print;}' | awk '{p = $$1; for (i=1;i<NF;i++) $$i=$$(i+1);$$NF="="; $$(NF + 1)=p; tmp=$$2;$$2=$$3;$$3=tmp;$$1="R";print;}' > $@

%.head.model: %.model $(LTREES-SCRIPTS)/rules2headmodel.py
	cat $< | python3 $(word 2, $^) > $@


################################################################################
#
#  Items for parse evaluation
#
################################################################################    

## TO SPEC
.PRECIOUS: %syneval
%syneval:  bin/evalb \
           $(RESOURCE-GENERAL)/srcmodel/new.prm \
           genmodel/$$(subst, +,-,$$(subst -,.,$$(basename $$(basename $$*))))$$(subst _,.,$$(suffix $$*))senttrees \
           $$(subst +,-,$$(subst -,.,$$(word 1,$$(subst ., ,%))))$$(suffix $$(basename $$*))$$(subst _,.,$$(suffix $$*))senttrees
	$< -p $(word 2,$^) $(word 3,$^) $(word 4,$^) > $@

##### syntactic evaluation           #genmodel/$$(subst -,.,$$(basename $$(basename $$(basename $$(notdir $$*))))).linetrees  %.linetrees
#.PRECIOUS: %.syneval
##### <dashed-alpha-testset><dotted-beta-edits-to-gold><dashed-gamma-model>.syneval -> calls <alpha><beta>.linetrees gold, <alpha><gamma>.linetrees hypoth
#%.syneval:  bin/evalb \
#            $(RESOURCE-GENERAL)/srcmodel/new.prm \
#            genmodel/$$(subst -,.,$$(basename $$*)).linetrees \
#            $$(subst -,.,$$(word 1,$$(subst ., ,%)))$$(suffix $$*).linetrees
#	$< -p $(word 2,$^) $(word 3,$^) $(word 4,$^) > $@
#.PRECIOUS: %.nounary_syneval
#%.nounary_syneval:  bin/evalb \
#                    $(RESOURCE-GENERAL)/srcmodel/new.prm \
#                    genmodel/$$(subst -,.,$$(basename $$*)).nounary.linetrees \
#                    $$(subst -,.,$$(word 1,$$(subst ., ,%)))$$(suffix $$*).nounary.linetrees
#	$< -p $(word 2,$^) $(word 3,$^) $(word 4,$^) > $@
#.PRECIOUS: %.noberkstuff_nounary_syneval
#%.noberkstuff_nounary_syneval:  bin/evalb \
#                                $(RESOURCE-GENERAL)/srcmodel/new.prm \
#                                genmodel/$$(subst -,.,$$(basename $$*)).nounary.linetrees \
#                                $$(subst -,.,$$(word 1,$$(subst ., ,%)))$$(suffix $$*).noberkstuff.nounary.linetrees
#	$< -p $(word 2,$^) $(word 3,$^) $(word 4,$^) > $@

%.syneval.errs:  $(LTREES-SCRIPTS)/tree-compare.py \
                 genmodel/$$(subst -,.,$$(basename $$*)).linetrees \
                 $$(subst -,.,$$(word 1,$$(subst ., ,%)))$$(suffix $$*).linetrees
	python $^  >  $@
%.nounary_syneval.errs:  $(LTREES-SCRIPTS)/tree-compare.py \
                         genmodel/$$(subst -,.,$$(basename $$*)).nounary.linetrees \
                         $$(subst -,.,$$(word 1,$$(subst ., ,%)))$$(suffix $$*).nounary.linetrees
	python $^  >  $@
%.noberkstuff_nounary_syneval.errs:  $(LTREES-SCRIPTS)/tree-compare.py \
                                     genmodel/$$(subst -,.,$$(basename $$*)).nounary.linetrees \
                                     $$(subst -,.,$$(word 1,$$(subst ., ,%)))$$(suffix $$*).noberkstuff.nounary.linetrees
	python $^  >  $@

#%.syneval:  bin/evalb \
#            $(RESOURCE-GENERAL)/srcmodel/new.prm \
#            genmodel/$$(subst -,.,$$(basename $$(basename $$(basename $$(notdir $$*))))).linetrees \
#            $$(subst -,.,$$(word 1,$$(subst ., ,%)))$$(suffix $$(basename $$(basename $$*)))$$(suffix $$(basename $$*))$$(suffix $$*).linetrees
#	$< -p $(word 2,$^) $(word 3,$^) $(word 4,$^) > $@
##### syntactic evaluation for nounary post-process
#%.nounary.syneval:  bin/evalb \
#            $(RESOURCE-GENERAL)/srcmodel/new.prm \
#            genmodel/$$(subst -,.,$$(basename $$(basename $$(basename $$(notdir $$*))))).linetrees \
#            $$(subst -,.,$$(word 1,$$(subst ., ,%)))$$(suffix $$(basename $$(basename $$*)))$$(suffix $$(basename $$*))$$(suffix $$*).nounary.linetrees
#	$< -p $(word 2,$^) $(word 3,$^) $(word 4,$^) > $@
##### syntactic evaluation for nounary nolabel post-process
#%.nounary.nolabel.syneval:  bin/evalb \
#            $(RESOURCE-GENERAL)/srcmodel/new.prm \
#            genmodel/$$(subst -,.,$$(basename $$(basename $$(basename $$(notdir $$*))))).linetrees \
#            $$(subst -,.,$$(word 1,$$(subst ., ,%)))$$(suffix $$(basename $$(basename $$*)))$$(suffix $$(basename $$*))$$(suffix $$*).nounary.nolabel.linetrees
#	$< -p $(word 2,$^) $(word 3,$^) $(word 4,$^) > $@
#
##### syntactic evaluation for nounary nolabel nopunctuation post-process
#%.nounary.nolabel.nopunc.syneval:  bin/evalb \
#            $(RESOURCE-GENERAL)/srcmodel/new.prm \
#            genmodel/$$(subst -,.,$$(basename $$(basename $$(basename $$(notdir $$*))))).linetrees \
#            $$(subst -,.,$$(word 1,$$(subst ., ,%)))$$(suffix $$(basename $$(basename $$*)))$$(suffix $$(basename $$*))$$(suffix $$*).nounary.nolabel.nopunc.linetrees
#	$< -p $(word 2,$^) $(word 3,$^) $(word 4,$^) > $@

# <common-prefix-can-contain-periods>.<unique-gold-params>.<unique-test-params>.<common-suffix>.syneval
.PRECIOUS: %.new.syneval
%.new.syneval: bin/evalb \
$(RESOURCE-GENERAL)/srcmodel/new.prm \
$$(subst ..,.,$$(basename $$(basename $$*))$$(suffix $$*).linetrees) \
$$(subst ..,.,$$(basename $$(basename $$(basename %)))$$(suffix $$(basename $$(basename $$*)))$$(suffix $$*).linetrees)
	$< -p $(word 2,$^) $(word 3,$^) $(word 4,$^) > $@

# Generates a bunch of constituency-based eval metrics
.PRECIOUS: %.constiteval.txt
#### the bizarre syntax of these prereqs is to allow the eval to go through even if
#### the two names share no common suffix (it converts '..' to '.')
%.constiteval.txt: $(LTREES-SCRIPTS)/constit_eval.py genmodel/$$(subst ..,.,$$(subst -,.,$$(basename $$(basename $$*)))$$(subst -,.,$$(suffix $$*)).linetrees) \
$$(subst ..,.,$$(subst -,.,$$(subst .,,$$(suffix $$(basename %))))$$(subst -,.,$$(suffix $$*)).linetrees) $$(RESOURCE-LTREES)/scripts/filter_reannot_fails.py | genmodel
	python $< $(word 2, $^) <(cat $(word 3, $^) | python $(word 4, $^) $(word 2, $^)) > $@

# Generates a bunch of constituency-based eval metrics with verbose output
.PRECIOUS: %.d.constiteval.txt
#### the bizarre syntax of these prereqs is to allow the eval to go through even if
#### the two names share no common suffix (it converts '..' to '.')
%.d.constiteval.txt: $(LTREES-SCRIPTS)/constit_eval.py genmodel/$$(subst ..,.,$$(subst -,.,$$(basename $$(basename $$*)))$$(subst -,.,$$(suffix $$*)).linetrees) \
$$(subst ..,.,$$(subst -,.,$$(subst .,,$$(suffix $$(basename %))))$$(subst -,.,$$(suffix $$*)).linetrees) $$(RESOURCE-LTREES)/scripts/filter_reannot_fails.py | genmodel
	python $< -d $(word 2, $^) <(cat $(word 3, $^) | python $(word 4, $^) $(word 2, $^)) > $@

%.constitevaltable.txt: $(LTREES-SCRIPTS)/constitevals2table.py %.constitevallist
	python $^ > $@

%.learning_curves: $(LTREES-SCRIPTS)/plot_learning_curves.r %.constitevaltable.txt
	$< $(abspath $(word 2, $^))

#### syntactic evaluation when gold and test have different naming conventions
.PRECIOUS: %.diffname.syneval
#### the bizarre syntax of these prereqs is to allow the eval to go through even if
#### the two names share no common suffix (it converts '..' to '.')
%.diffname.syneval: bin/evalb $(RESOURCE-GENERAL)/srcmodel/new.prm genmodel/$$(subst ..,.,$$(subst -,.,$$(basename $$(basename $$*)))$$(subst -,.,$$(suffix $$*)).linetrees) \
$$(subst ..,.,$$(subst -,.,$$(subst .,, $$(suffix $$(basename %))))$$(subst -,.,$$(suffix $$*)).linetrees) $$(RESOURCE-LTREES)/scripts/filter_reannot_fails.py | genmodel
	$< -p $(word 2, $^) $(word 3, $^) <(cat $(word 4, $^) | python $(word 5, $^) $(word 3, $^)) > $@
# $(wordlist 2,4, $^) > $@


################################################################################
#
#  Items for significance testing
#
################################################################################    

#### signif testing
%.bootstrapsignif:  $(LTREES-SCRIPTS)/compare.pl \
                    $$(basename $$(basename $$*))$$(suffix $$*) \
                    $$(basename $$(basename $$(basename $$*)))$$(suffix $$(basename $$*))$$(suffix $$*)
	perl $(word 1,$^) $(word 2,$^) $(word 3,$^)  >  $@

#         $$(word 1,$$(subst .., ,%)).$$(word 2,$$(subst .., ,$$*)).$$(word 4,$$(subst .., ,$$*)) \
#         $$(word 1,$$(subst .., ,%)).$$(word 3,$$(subst .., ,$$*)).$$(word 4,$$(subst .., ,$$*))

