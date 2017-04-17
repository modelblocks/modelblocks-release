QUICK START
===========

Basic File Structure
--------------------
Modelblocks is composed of `HUB` and `TERMINAL` subdirectories.
`HUB` subdirectories build resources that are used by a variety of
projects (e.g., the GCG-reannotated WSJ corpus or the left-corner
parser). `TERMINALS` contain individual projects. By convention,
all `HUB` directory names begin with the string "resource-".

Sandbox Generation
------------------
To begin, you'll need a `TERMINAL` subdirectory devoted to your project
which is appropriately linked to the `HUB` directories. To easily
accomplish this, you should generate a sandbox directory.

    > cd sandbox
    > make PROJECT
    > cd PROJECT

Link to Hub Directories
-----------------------
You are now in a sandbox project subdirectory named `PROJECT`.
Feel free to name your sandbox directory whatever you like.
Now edit the Makefile in PROJECT to uncomment any resources you need.
You can find a list of resources in the `RESOURCES` file.

Link to External Resources
--------------------------
Modelblocks needs to be told where to access tools that don't come
bundled with it, such as the Penn Treebank. When make is first invoked
in a project directory, it will create a variety of `user-*.txt` files 
in the `config` directory at the Modelblocks root, which each point to 
a given external resource. The full list of external dependencies will 
be output at the console. For example, `user-treebank-directory.txt` 
points to the Wall Street Journal corpus. You need to edit these files 
to point to your local locations of each resource.

Demo Run
------------------------
Most likely, you'll want to extract incremental complexity metrics
for some corpus (referred to as `NEWCORPUS`). You'll need to get your
corpus into `PROJECT/genmodel/NEWCORPUS.linetoks`. The sents file needs
to have one sentence per line and be tokenized comparably to the
data used to train the complexity metrics. This guide will assume
that you will use the Penn Treebank for training. In that case,
`NEWCORPUS.sents` will need to use the PTB tokenization scheme.
You can get a PTB tokenizer from:
https://github.com/vansky/extended_penn_tokenizer

To use the tokenizer:

    > extended_penn_tokenizer/ptb_tokenizer.sed < NEWCORPUS.untoksents > NEWCORPUS.linetoks

To get complexity metrics for the corpus, uncomment the following
HUB directories in your Makefile:

    > RESOURCE-TOKENIZER
    > RESOURCE-LTREES
    > RESOURCE-LVPCFG
    > RESOURCE-LCPARSE
    > RESOURCE-TREEBANK

Then run this make command:

    > make genmodel/NEWCORPUS.wsj02to21-nodashtags-5sm-bd.x-efabp.-c_-b5000_parsed.output