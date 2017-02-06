QUICK START
===========

Basic File Structure
--------------------
Modelblocks is composed of `HUB` and `TERMINAL` subdirectories.
`HUB` subdirectories build resources that are used by a variety of
projects (e.g., the GCG-reannotated WSJ corpus or the left-corner
parser). `TERMINALS` contain individual projects. By convention,
all `HUB` directory names begin with the string "resource-".

To begin, you'll need a `TERMINAL` subdirectory devoted to your project
which is appropriately linked to the `HUB` directories. To easily
accomplish this, copy `vanschijndelschuler15` and rename the result
for your project. For this quickstart, we'll call your project
`newproj`.

    > cp -r vanschijndelschuler15 newproj
    > cd newproj

Modelblocks needs to be told where to access tools that don't come
bundled with it, such as the Penn Treebank. When make is first invoked
in a project directory, it will create a variety of `user-*.txt` files 
in the `config` directory at the Modelblocks root, which each point to 
a given external resource. The full list of external dependencies will 
be output at the console. For example, `user-treebank-directory.txt` 
points to the Wall Street Journal corpus. You need to edit these files 
to point to your local locations of each resource.

To view precisely which versions of each external resource are expected
by the Modelblocks code, simply type "make -n cocomo" from any project
directory, which will print a list of the resource paths used internally
in the CoCoMo lab.

Test Corpus Construction
------------------------
Most likely, you'll want to extract incremental complexity metrics
for some corpus (referred to as `NEWCORPUS`). You'll need to get your
corpus into `newproj/genmodel/NEWCORPUS.linetoks`. The sents file needs
to have one sentence per line and be tokenized comparably to the
data used to train the complexity metrics. This guide will assume
that you will use the Penn Treebank for training. In that case,
`NEWCORPUS.sents` will need to use the PTB tokenization scheme.
You can get a PTB tokenizer from:
https://github.com/vansky/extended_penn_tokenizer

to use the tokenizer:

    > extended_penn_tokenizer/ptb_tokenizer.sed < NEWCORPUS.untoksents > NEWCORPUS.linetoks

