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

MSG1 := The current config file, 
MSG2 := , points to a non-existent location (
MSG3 := ). Fix before re-running make, or (if not needed) set to ".".

define CONFIGWARN =

ATTENTION! I had to create $(CONFIG),
which contains an incorrect default value (".").
Targets with dependencies to this file will fail until it is fixed.

endef

ifndef MAKECONFIG
CONFIG := $(CONFIGDIR)/user-berkeleyparserjar-directory.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
BERKPARSEDIR := $(shell cat $(CONFIG))
ifeq (, $(firstword $(wildcard $(BERKPARSEDIR))))
$(error $(MSG1)$(CONFIG)$(MSG2)$(BERKPARSEDIR)$(MSG3))
endif

CONFIG := $(CONFIGDIR)/user-javaflags.txt
ifeq (,$(firstword $(wildcard $(CONFIG))))
$(info $(CONFIGWARN))
DUMMY := $(shell $(MAKE) $(CONFIG) MAKECONFIG=1)
endif
JAVAFLAGS := $(shell cat $(CONFIG))
endif

%/user-berkeleyparserjar-directory.txt: | %
	echo '/home/compling/berkeleyparser' > $@

%/user-javaflags.txt: | %
	echo '-Xmx8g' > $@

################################################################################
#
#  Code compilation items
#
################################################################################

#### berkeley parser shortcut "executable" -- external jarfile
.PRECIOUS: bin/parser-fullberk
bin/parser-fullberk: $(CONFIGDIR)/user-javaflags.txt $(CONFIGDIR)/user-berkeleyparserjar-directory.txt \
$(BERKPARSEDIR)/berkeleyParser.jar  |  bin  #$$(abspath $$(dir $$@))
	echo "#!/bin/sh" > $@
	echo "java $(shell cat $(word 1,$^)) -cp $(word 3,$^) edu.berkeley.nlp.PCFGLA.BerkeleyParser -substates -gr \$$1" >> $@
	chmod u+x $@

bin/parser-vitberk: $(CONFIGDIR)/user-javaflags.txt $(CONFIGDIR)/user-berkeleyparserjar-directory.txt \
$(BERKPARSEDIR)/berkeleyParser.jar  |  bin  #$$(abspath $$(dir $$@))
	echo "#!/bin/sh" > $@
	echo "java $(shell cat $(word 1,$^)) -cp $(word 3,$^) edu.berkeley.nlp.PCFGLA.BerkeleyParser -viterbi -substates -gr \$$1" >> $@
	chmod u+x $@

################################################################################
#
#  Berkelely LV PCFG trainer
#
#  to construct the following file types:
#    <x>.<n>sm.gr    : serialized java data structures
#
################################################################################

#### add extra paren and change '-' ops to '+' for berkeley trainer
%.extrparplusops.linetrees:  %.linetrees
	cat $<  |  sed 's/^\((.*\)$$/(\1)/g'  |  perl -pe 's/\-(?=[a-z][^\)]* )/+/g'  >  $@

#### sm grammar using berkeley trainer
## no validation -- external jarfile
define ParamVal
SM:
  cascade: SplitMerge
  kernel: ModelOpt
  text: Split-merge iterations
  value: ''
endef
define KernelBlock
ModelOpt:
  blocktitle: Model Options
  paramtype: Dropdown
  nargs: '*'
SplitMerge:
  blocktitle: SplitMerge
  nodelimiter: true
  paramtype: Integer
  paramval:
  - after: true
    descr: ''
    text: Split Merge Iterations
    value: sm
endef
%sm.gr:  $(CONFIGDIR)/user-javaflags.txt  $(CONFIGDIR)/user-berkeleyparserjar-directory.txt  \
$(BERKPARSEDIR)/berkeleyParser.jar  $$(basename %).extrparplusops.linetrees
	java  $(shell cat $(word 1,$^))  -cp $(word 3,$^)  edu.berkeley.nlp.PCFGLA.GrammarTrainer  -SMcycles $(subst .,,$(suffix $*))  -path $(word 4,$^)  -treebank SINGLEFILE  -out $@
## validation -- external version
%sm.gr:  $(CONFIGDIR)/user-javaflags.txt  $(CONFIGDIR)/user-berkeleyparserjar-directory.txt  \
$(BERKPARSEDIR)/berkeleyParser.jar  $$(basename %).extrparplusops.linetrees $$(basename %)last.extrparplusops.linetrees
	java  $(shell cat $(word 1,$^))  -cp $(word 3,$^)  edu.berkeley.nlp.PCFGLA.GrammarTrainer  -SMcycles $(subst .,,$(suffix $*))  -path $(word 4,$^)  -validation $(word 5,$^)  -treebank SINGLEFILE  -out $@

#### silly shortcut for berkeley parser
%.fullberk.model:  %.gr
	ln -sf $(notdir $<) $@
%.vitberk.model:  %.gr
	ln -sf $(notdir $<) $@

#### obtain text grammar from javadump
%.splits %.grammar %.lexicon:  $(CONFIGDIR)/user-javaflags.txt  $(CONFIGDIR)/user-berkeleyparserjar-directory.txt  $(BERKPARSEDIR)/berkeleyParser.jar  %.gr
	java  $(shell cat $(word 1,$^))  -cp $(word 3,$^)  edu.berkeley.nlp.PCFGLA.WriteGrammarToTextFile  $(word 4,$^)  $(basename $@)

################################################################################
#
#  Berkeley parser
#
#  to construct the following file types:
#    <x>.linetoks                        : sentences, one per line, consisting of only tokenized words, delimited by spaces
#    <w>.<x>.<y>.parsed.linetrees : .linetrees file resulting from applying parser-<y>, and model <x>.<y>.model (with hyphenated x), to <w>.linetoks file
#                                       (e.g. wsj23.wsj02to21-0sm.berk.parsed.linetrees
#    <x>.syneval : evaluation report for parsing
#
################################################################################

##### obtain model-specific parser output by running sentences through parser given flags and model, from including makefile's genmodel dir:
##### <testset>.<trainset>.<model>.(<params>_)parsed  ---->  genmodel/<testset>.linetoks  bin/parser-<model>  genmodel/<trainset>.<model>.model
# $$(basename $$(basename $$(basename %))).$$(findstring hy,$$*)sents
.PRECIOUS: %parsed.output
## obsolete
#%parsed.output: $$(basename $$(basename $$(basename %))).linetoks \
#		$(BIN)/parser-$$(subst .,,$$(suffix $$(basename $$*))) \
#		$(GENMODEL)/$$(subst -,.,$$(subst .,,$$(suffix $$(basename $$(basename $$*)))))$$(suffix $$(basename $$*)).model
#	@echo "WARNING: long build for '$@'!  Press CTRL-C to abort!"
#	@sleep 5
#	cat $<  |  $(word 2,$^)  $(subst _, ,$(subst .,,$(suffix $*)))  $(word 3,$^)  >  $@
## this is to spec; above is deprecated
%parsed.output: $(BIN)/parser-$$(subst .,,$$(suffix $$(basename $$(subst +,-,$$(subst -,.,$$*))))) \
		$$(basename %).linetoks \
		$(GENMODEL)/$$(basename $$(subst +,-,$$(subst -,.,$$(subst .,,$$(suffix $$*))))).model
	cat $(word 2,$^)  |  $(word 1,$^) $(subst _, ,$(subst +,-,$(subst .,,$(suffix $(subst -,.,$*))))) $(word 3,$^)  >  $@
## broken now
#%parsed.output: $$(basename $$(basename %)).linetoks \
#		$(BIN)/parser-$$(subst .,,$$(suffix $$(basename $$*))) \
#		$(GENMODEL)/$$(basename $$(basename $$(subst -,.,$$(subst .,,$$(suffix $$*)))))$$(suffix $$(basename $$*)).model
#	@echo "WARNING: long build for '$@'!  Press CTRL-C to abort!"
#	@sleep 5
#	cat $<  |  $(word 2,$^)  $(subst _, ,$(subst .,,$(suffix $(subst -,.,$*))))  $(word 3,$^)  >  $@

#### obtain linetrees by removing extra parens and converting '+'s in categories to '-'s
#%berk.parsed.linetrees:  $$(word 1,$$(subst ., ,%))$$(suffix $$(basename $$*))$$(suffix $$*)berk.parsed.output
%berk.parsed.linetrees:  %berk.parsed.output
	cat $<  |  perl -pe 's/\+(?=[^\)]* )/\-/g'  |  perl -pe 's/^ *\( *//;s/ *\) *$$//;s/-\d+ / /g'  >  $@
## this is to spec; above is deprecated
%berk-parsed.linetrees:  %berk-parsed.output
	cat $<  |  perl -pe 's/\+(?=[^\)]* )/\-/g'  |  perl -pe 's/^ *\( *//;s/ *\) *$$//;s/-\d+ / /g'  >  $@

#### Remove Berkeley split-merge annotation information
%.noberkstuff.linetrees: %.linetrees
	cat $^ | perl -pe 's/\^[^ ]*//g;s/_[0-9]+ / /g' | python $(RESOURCE-LVPCFG)/scripts/removeAt.py > $@ 


