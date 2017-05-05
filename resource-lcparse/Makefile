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
#  include $(RESOURCE-LTREES)/Makefile
#  include $(RESOURCE-GCG)/Makefile
#  include $(RESOURCE-LVPCFG)/Makefile
#
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
INCLUDE := $(RESOURCE-LCPARSE)/include
BIN     := bin

include $(wildcard $(RESOURCE-LCPARSE)/*/*.d)      ## don't comment out, it breaks make!

################################################################################
#
#  Code compilation items
#
################################################################################

SHELL = /bin/bash
#INCLUDES = -I$(INCLUDE) -I$(abspath $(MBDIR)/resource-rvtl)
#CFLAGS = $(INCLUDES) -Wall `cat user-cflags.txt` -fpermissive
#CC = g++

#### obtain c++ dependencies by compiling local source and include files
.PRECIOUS: $(RESOURCE-LCPARSE)/%.d
$(RESOURCE-LCPARSE)/%.d:  $(RESOURCE-LCPARSE)/%.cpp  |  $(INCLUDE)
	echo '$*.d: \' > $@   #' ##
	echo `g++ -I$| -I$(abspath $(MBDIR)/resource-rvtl) -std=c++11 -MM $<` | sed 's/^.*:.*\.cpp */ $$(wildcard /;s/\\ *//g;s/$$/)/' >> $@
	cat $@ | sed 's/\.d:/\.cpp:/' >> $@
#       $(CC) -MM $< | sed 's/^.*://' >> $@

#### obtain binaries by compiling local source and include files
.PRECIOUS: bin/%
bin/%:  $(RESOURCE-LCPARSE)/src/$$(notdir $$@).cpp  $(RESOURCE-LCPARSE)/src/$$(notdir $$@).d  $(CONFIGDIR)/user-cflags.txt  |  bin  $(INCLUDE)  #$$(abspath $$(dir $$@))  #$$(realpath $$(dir $$@))  #$(BIN)  #$(CURDIR)/$$(notdir $$(dir $$@))  #
	g++ -I$(word 2,$|) -I$(abspath $(MBDIR)/resource-rvtl) -Wall `cat $(word 3,$^)` -fpermissive -std=c++11 -lm $< -o $@

################################################################################
#
#  Model building items
#
################################################################################

#### obtain '.model' format pcfg (with unaries) from berkeley text grammar (converting '+' categories back to '-')
%.x-ccu.model:  %.grammar  %.lexicon  $(LCPARSE-SCRIPTS)/berkgrammar2ckygr.py  $(LCPARSE-SCRIPTS)/berklexicon2ckylex.py
	cat $(word 1,$^)  |  sed 's/[+]/-/g'  |  python3 $(word 3,$^)  >   $@
	cat $(word 2,$^)  |  sed 's/[+]/-/g'  |  python3 $(word 4,$^)  >>  $@

#### obtain strict binary-branching pcfg model (with no unaries) from pcfg model with unaries
%.x-cc.model:  %.x-ccu.model  $(LCPARSE-SCRIPTS)/ccu2cc.py
	cat $<  |  python3 $(word 2,$^)  >  $@

#### obtain branch- and depth-annotated grammar with preterm nodes generated seperately
def ParamVal :=
BD:
  kernel: ModelOpt
  text: Annotate branching direction and depth
  value: bd
endef
%.bd.x-ccp.model:  %.x-cc.model  $(LCPARSE-SCRIPTS)/pcfg2pxmodel.py
	cat $<  |  python3 $(word 2,$^)  >  $@

#### obtain incremental probability model from strict binary-branching pcfg
%.x-fabp.probmodel:  %.x-ccp.model  bin/calc-fabp-model
	cat $<  |  grep -v '^PX '  | sed 's/^Cr : \(.*\^[Ll],1\) =/CC REST^R,0 : \1 REST^R,0 =/;s/ - - / -\^.,. -\^.,. /' \
                |  sed 's/CC \(.*\)\^\(.\),\(.\) : \(.*\)\^.,. \(.*\)\^.,. = \(.*\)/CC \2 \3 \1 : \4 \5 = \6/' |  $(word 2,$^)  >  $@

#### obtain sorted model for A* algo, and rename for parser that incrementally tracks entities
%.x-efabp.model:  %.x-fabp.probmodel  $(LCPARSE-SCRIPTS)/sortbyprob.pl
	cat $<  |  perl $(word 2,$^)  >  $@

################################################################################
#
#  Parsing items
#
################################################################################

def ParamVal :=
EFABP:
  kernel: Parser
  text: EFABP parser (van Schijndel et al (2013)
  value: x+efabp
Fullberk:
  kernel: Parser
  text: Full Berkeley parser
  value: fullberk
Vitberk:
  kernel: Parser
  text: Viterbi Berkeley parser
  value: vitberk
endef
def KernelBlock :=
Grammar:
  blocktitle: Grammar
  paramtype: Dropdown
Parser:
  blocktitle: Parser
  paramtype: Dropdown
ParserParams:
  blocktitle: Parser Parameters
  paramtype: Text
  paramval:
  - text: Underscore-delimited command-line arguments to the parser
    value: ''
endef
def CompositeBlock :=
ParseParams:
  blockseq:
  - blocktype: Corpus
    kwargs:
      instancename: Training
  - blocktype: Grammar
  - blocktype: TreeOpt
    kwargs:
      innerdelim: '-'
  - blocktype: ModelOpt
    kwargs:
      innerdelim: '-'
  - blocktype: Parser
  - blocktype: ParserParams
  blocktitle: Parser Parameters
endef
def TargetBlock :=
ParseLinetrees:
  blockseq:
  - blocktype: Corpus
    kwargs:
      instancename: Test
  - blocktype: String
    value: .
  - blocktype: ParseParams
    kwargs:
      innerdelim: '-'
  blocktitle: Parse (Linetrees)
  targetsuffix: _parsed.linetrees
endef

##### obtain model-specific parser output by running sentences through parser given flags and model, from including makefile's genmodel dir:
##### <testset>.<trainset>.<model>.(<params>_)parsed  ---->  genmodel/<testset>.linetoks  bin/parser-<model>  genmodel/<trainset>.<model>.model
# $$(basename $$(basename $$(basename %))).$$(findstring hy,$$*)sents
.PRECIOUS: %parsed.output
%parsed.output:  $$(basename $$(basename $$(basename %))).linetoks \
                 $(BIN)/parser-$$(subst .,,$$(suffix $$(basename $$*))) \
                 $$(subst -,.,$$(subst .,,$$(suffix $$(basename $$(basename $$*)))))$$(suffix $$(basename $$*)).model
	@echo "WARNING: long build for '$@'!  Press CTRL-C to abort!"
	@sleep 5
	cat $<  |  $(word 2,$^)  $(subst _, ,$(subst .,,$(suffix $*)))  $(word 3,$^)  >  $@

#### obtain linetrees by converting output using script:
# %parsed.linetrees: $$(word 1,$$(subst ., ,%))$$(suffix $$(basename $$*))$$(suffix $$*)parsed.output \
#                    $(LCPARSE-SCRIPTS)/$$(lastword $$(subst -, ,$$(basename $$*)))out2linetrees.py \
#                    $(LCPARSE-SCRIPTS)/remove-at-cats.pl
# Example: wsj23.wsj02to21-gcg14b-1671-3sm-bd.x-efabp.-c_-b2000_parsed.linetrees
%parsed.linetrees: %parsed.output $(LCPARSE-SCRIPTS)/$$(lastword $$(subst -, ,$$(basename $$*)))out2linetrees.py  $(LCPARSE-SCRIPTS)/remove-at-cats.pl $(LCPARSE-SCRIPTS)/spargeComplexity.py
	#both sed calls are necessary since the patterns overlap, so two passes with sed are necessary to convert all plusops to dashops
	cat $<  |  python $(word 4, $^) -o  |  python3 $(word 2,$^)  | sed 's/\^.,.//g;s/\^g//g;s/\_[0-9]*//g;s/\([^+ ]\)+\([^+ ]\)/\1-\2/g;' | sed 's/\([^+ ]\)+\([^+ ]\)/\1-\2/g;'  |  perl $(word 3,$^)  >  $@

## squirrely old parser item...
.PRECIOUS: %parsed.output
%parsed.output:  bin/parser-$$(subst .,,$$(suffix $$(basename $$(subst +,-,$$(subst -,.,$$*))))) \
                  $$(basename %).linetoks \
                  genmodel/$$(basename $$(subst +,-,$$(subst -,.,$$(subst .,,$$(suffix $$*))))).model
	cat $(word 2,$^)  |  $(word 1,$^) $(subst _, ,$(subst +,-,$(subst .,,$(suffix $(subst -,.,$*))))) $(word 3,$^)  >  $@

## support squirrely old efabp output item from new syneval item...
%parsed.linetrees: %parsed.output $(LCPARSE-SCRIPTS)/$$(lastword $$(subst +, ,$$(basename $$(subst -,.,$$*))))out2linetrees.py  $(LCPARSE-SCRIPTS)/remove-at-cats.pl $(LCPARSE-SCRIPTS)/spargeComplexity.py
	#both sed calls are necessary since the patterns overlap, so two passes with sed are necessary to convert all plusops to dashops
	cat $<  |  python $(word 4, $^) -o  |  python3 $(word 2,$^)  | sed 's/\^.,.//g;s/\^g//g;s/\_[0-9]*//g;s/\([^+ ]\)+\([^+ ]\)/\1-\2/g;' | sed 's/\([^+ ]\)+\([^+ ]\)/\1-\2/g;'  |  perl $(word 3,$^)  >  $@



