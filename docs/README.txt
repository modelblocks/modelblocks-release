README for ModelBlocks Assistant

ModelBlocks Assistant is (sort of) a documentation generator for ModelBlocks.
It uses markups in the code throughout the repository to generate a browser-
based GUI that automatically fills out GNU Make targets for experiments
according to the user's needs.

Makefile markups are in YAML format. To scan the repo and regenerate
ModelBlocks Assistant, run

make mbassist

from this directory.

## For ModelBlocks Assistant Users

As is clear from the ModelBlocks Assistant interface, ModelBlocks Assistant
assumes a modular design in which 'blocks' of the model can be combined,
rearranged, and reparameterized in well-defined ways.

Once you've built your experiment, simply click "Copy to Clipboard" and
paste the result into a UNIX shell at the relevant experiment directory.
If all is correct, you should just be able to press "Enter" and let the
magic happen.

Please note the following two important points:
1. ModelBlocks Assistant does not generate all possible ModelBlocks recipes,
but instead focuses on those that the documenters felt people were likely
to want as a final product. If there is a particular target you want/need,
that ModelBlocks Assistant does not support, it may be necessary to browse
the docs and Makefiles for clues about how to build it.

2. ModelBlocks Assistant tries to structure model-building so that it's
not as easy to go wrong. However, valid output targets are not guaranteed,
and it may be necessary to consult the documentation if a ModelBlocks
Assistant--generated target is still failing.

## For ModelBlocks documentation writers
If you've created experiments in ModelBlocks and would like to mark
them up so that they can be populated into ModelBlocks Assistant, the
following should help you get started.

### ModelBlocks Assistant data structures
ModelBlocks Assistant assumes four main data structures that correspond
to components of target:

1. ParamVal:
A ParamVal object represents a value that a particular variable can
take on. For example 'PTB' is a ParamVal for the 'Grammar' variable,
since it is an available grammar to use (e.g. for parsing).

To define a ParamVal in a Makefile, you need to wrap it in the following
Make variable definition:

def ParamVal
 ...
endef

ParamVals have a type, along with the following required fields:

kernel: the variable type that the ParamVal instantiates (can be a list)
text: short descriptive text
value: the actual string value that will represent the ParamVal in the output.

ParamVals also have the following optional fields:

descr: longer description/documentation. Nothing is currently done with this yet.
cascade: (list of) other ParamVal that are only relevant if the current ParamVal is used.
nodelimiter: Eliminate separator from inside complex variables (boolean, omission considered false).

2. KernelBlock:
A KernelBlock is a primitive block type that represents a variable to be
instantiated in the output. Values are automatically propagated to KernelBlocks
according to their 'kernel' fields. For example, if the ParamVal NaturalStories
has 'Corpus' as its 'kernel' field, the ModelBlocks Assistant will attach
NaturalStories to the Corpus KernelBlock as a value that it can take on.

To define a KernelBlock in a Makefile, you need to wrap it in the following
Make variable definition:

def KernelBlock
 ...
endef

KernelBlocks have a type, along with the following required fields:

blocktitle: short descriptive text
paramtype: currently supports 'Dropdown', 'Multiselect', 'Integer', 'Text', and 'Boolean'
paramval: list of ParamVal, can be singleton

In practice, paramval can be omitted if at least one ParamVal object lists
the KernelBlock in its 'kernel' field, since it will be automatically pulled in.
But they can also be specified in place. This requires care on the part of the
documentation writer because ModelBlocks Assistant does not include many
checks for errors here (e.g. empty paramval with no calling ParamVal objects,
redundant ParamVal, or multiple ParamVal for 'Integer', 'Text', or 'Boolean'
types, which are only defined for singleton paramval). Ill-formed documentation
can cause ModelBlocks Assistant not to build.

KernelBlocks also have the following optional fields:

descr: longer description/documentation. Nothing is currently done with this yet.
nargs: number of such KernelBlocks to stack. Defaults to 1, can also be "*" or "?".
instance_of: allows for subclassing/inheritance so that values can trickle up.

3. CompositeBlock:
A CompositeBlock represents a reusable chunk of output that will occur in many
different targets (for example, a chunk of parser training parameters).
CompositeBlocks don't take on values directly, they merely define a sequence
of KernelBlocks and/or other CompositeBlocks for ModelBlocks Assistant to build
out. CompositeBlocks must be acyclic (a block cannot call itself as a child,
nor can any of its descendents), otherwise ModelBlocks Assistant won't load.

To define a CompositeBlock in a Makefile, you need to wrap it in the following
Make variable definition:

def CompositeBlock
 ...
endef

CompositeBlocks have a type, along with the following required fields:

blocktitle: short descriptive text
blockseq: sequence of blocktypes that make up the CompositeBlock

Text can be defined in the blockseq in the following manner:

- blocktype: String
  value: '.some.text.'

However, note that calling Blocks can pass a delimiter to the CompositeBlock,
so using Sring blocks for delimiters in the sequence should generally be
avoided.

CompositeBlocks also have the following optional fields:

descr: longer description/documentation. Nothing is currently done with this yet.
nargs: number of such CompositeBlocks to stack. Defaults to 1, can also be "*" or "?".

4. TargetBlocks
A TargetBlock represents a complete target. All TargetBlocks annotated in the
repository are automatically offered as build options by ModelBlocks Assistant.
TargetBlocks are identical to CompositeBlocks except that they must provide
the following additional required field:

targetsuffix: the part of the target following '%' in the Make recipe.

Also, since TargetBlocks are not called by other blocks, delimiters
must be inserted manually using String blocks, as described above.
