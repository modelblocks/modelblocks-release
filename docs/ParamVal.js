/////////////////////////////////////
//
//  PARAMVAL DEFINITIONS
//
/////////////////////////////////////

ParamVal = {}

/////////////////////////////////////
//
//  Corpora 
//
/////////////////////////////////////

ParamVal.Alice = {
  kernel: 'fMRICorpus',
  value: 'alice',
  text: 'Alice in Wonderland, Chapter 1',
};
ParamVal.Dundee = {
  kernel: 'ETCorpus',
  value: 'dundee',
  text: 'Dundee (eye-tracking corpus)',
};
ParamVal.NaturalStories = {
  kernel: 'SPRCorpus',
  value: 'naturalstories',
  text: 'Natural Stories (self-paced reading corpus)',
};
ParamVal.UCL = {
  kernel: ['ETCorpus', 'SPRCorpus'],
  value: 'ucl',
  text: 'UCL (eye-tracking)',
};
ParamVal.WSJ02to21 = {
  kernel: 'TextCorpus',
  value: 'wsj02to21',
  text: 'Wall Street Journal training set (sections 2-21)',
};
ParamVal.WSJ23 = {
  kernel: 'TextCorpus',
  value: 'wsj23',
  text: 'Wall Street Journal test set (section 23)',
}

/////////////////////////////////////
//
//  Grammar
//
/////////////////////////////////////

ParamVal.GCG16 = {
  kernel: 'GCGSpec',
  value: '13',
  text: '2013 spec'
};

ParamVal.GCG15 = {
  kernel: 'GCGSpec',
  value: '13',
  text: '2013 spec'
};

ParamVal.GCG14 = {
  kernel: 'GCGSpec',
  value: '13',
  text: '2013 spec'
};

ParamVal.GCG13 = {
  kernel: 'GCGSpec',
  value: '13',
  text: '2013 spec'
};

ParamVal.GCG = {
  kernel: 'Grammar',
  value: 'gcg',
  text: 'Generalized Categorial Grammar (GCG)',
  descr: '',
  cascade: ['GCGSpec', 'Decoupled']
};

ParamVal.PTB = {
  kernel: 'Grammar',
  value: 'nodashtags',
  text: 'Penn Treebank',
  descr: '',
};

ParamVal.First = {
  kernel: 'TreeOpt',
  value: '',
  text: 'Keep only the first n trees',
  cascade: 'First'
};

ParamVal.Last = {
  kernel: 'TreeOpt',
  value: '',
  text: 'Keep only the last n trees',
  cascade: 'Last'
};

ParamVal.Onward = {
  kernel: 'TreeOpt',
  value: '',
  text: 'Discard the first n trees',
  cascade: 'Onward'
};

ParamVal.Maxword = {
  kernel: 'TreeOpt',
  value: '',
  text: 'Keep only trees with fewer than n words',
  cascade: 'Maxwords'
};

ParamVal.Fromdeps = {
  kernel: 'TreeOpt',
  value: 'fromdeps',
  text: 'Extract trees from a dependency graph',
};

ParamVal.Nolabel = {
  kernel: 'TreeOpt',
  value: 'nolabel',
  text: 'Remove non-terminal category labels'
};

ParamVal.Nopunc = {
  kernel: 'TreeOpt',
  value: 'nopunc',
  text: 'Remove punctuation'
};

ParamVal.Nounary = {
  kernel: 'TreeOpt',
  value: 'nounary',
  text: 'Remove unary branches',
};

/////////////////////////////////////
//
//  Parser
//
/////////////////////////////////////


ParamVal.EFABP = {
  kernel: 'Parser',
  value: 'x+efabp-',
  text: 'EFABP parser (van Schijndel et al (2013)',
  cascade: ['C', 'BeamSize']
};

ParamVal.Fullberk = {
  kernel: 'Parser',
  value: 'fullberk-',
  text: 'Full Berkeley parser',
};

ParamVal.Synproc = {
  kernel: 'Parser',
  value: 'synproc-',
  text: 'Incremental syntactic processing parser (synproc)',
  cascade: ['C', 'BeamSize']
};
    
ParamVal.Vitberk = {
  kernel: 'Parser',
  value: 'vitberk-',
  text: 'Viterbi Berkeley parser'
};

/////////////////////////////////////
//
//  MODEL OPTIONS
//
/////////////////////////////////////

ParamVal.FG = {
  kernel: 'ModelOpt',
  value: 'fg',
  text: 'Filler-gap transform (inserts stack elements for long-distance dependencies)'
};

ParamVal.SM = {
  kernel: 'ModelOpt',
  value: '',
  text: 'Split-merge iterations',
  cascade: 'SplitMerge'
};

ParamVal.BD = {
  kernel: 'ModelOpt',
  value: 'bd',
  text: 'Annotate branching direction and depth',
};

/////////////////////////////////////
//
//  N-gram options
//
/////////////////////////////////////

ParamVal.SRILM = {
  kernel: 'NgramModel',
  value: 'srilm',
  text: 'SRILM'
};

ParamVal.KENLM = {
  kernel: 'NgramModel',
  value: 'kenlm',
  text: 'KENLM'
};









/////////////////////////////////////
//
//  KERNEL BLOCK CLASS DEFINITIONS
//
/////////////////////////////////////

KernelBlockDefs = {}

/////////////////////////////////////
//
//  Corpus 
//
/////////////////////////////////////

KernelBlockDefs.ETCorpus = {
  blocktitle: 'ETCorpus',
  paramtype: 'Dropdown',
  instance_of: 'Corpus'
}

KernelBlockDefs.SPRCorpus = {
  blocktitle: 'SPRCorpus',
  paramtype: 'Dropdown',
  instance_of: 'Corpus'
}

KernelBlockDefs.fMRICorpus = {
  blocktitle: 'fMRICorpus',
  paramtype: 'Dropdown',
  instance_of: 'Corpus'
}

KernelBlockDefs.TextCorpus = {
  blocktitle: 'TextCorpus',
  paramtype: 'Dropdown',
  instance_of: 'Corpus'
}

KernelBlockDefs.Corpus = {
  blocktitle: 'Corpus',
  paramtype: 'Dropdown',
}

/////////////////////////////////////
//
//  Grammar 
//
/////////////////////////////////////

KernelBlockDefs.GCGSpec = {
  blocktitle: 'GCG',
  paramtype: 'Dropdown'
};

KernelBlockDefs.Decoupled = {
  blocktitle: 'Decoupled',
  paramtype: 'Boolean',
  paramval: [
    {
      value: '-decoupled',
      text: 'Decoupled'
    }
  ]
}

KernelBlockDefs.Grammar = {
  blocktitle: 'Grammar',
  paramtype: 'Dropdown'
};

/////////////////////////////////////
//
//  Model Options 
//
/////////////////////////////////////

KernelBlockDefs.ModelOpt = {
    blocktitle: 'Model Options',
    paramtype: 'Multiselect'
};

KernelBlockDefs.SplitMerge = {
  blocktitle: 'SplitMerge',
  paramtype: 'Integer',
  nodelimiter: true,
  paramval: [
    {
      value: 'sm',
      text: 'Split Merge Iterations',
      descr: '',
      after: true
    }
  ]
}

/////////////////////////////////////
//
//  Parser Options 
//
/////////////////////////////////////


KernelBlockDefs.Parser = {
    blocktitle: 'Parser',
    paramtype: 'Dropdown'
};

KernelBlockDefs.C = {
  blocktitle: 'C',
  paramtype: 'Boolean',
  paramval: [
    {
      value: '+c',
      text: 'Output complexity metrics',
      descr: '',
    }
  ]
}

KernelBlockDefs.BeamSize = {
  blocktitle: 'BeamSize',
  paramtype: 'Integer',
  paramval: [
    {
      value: '+b',
      text: 'Beam Size',
      descr: ''
    }
  ]
}

/////////////////////////////////////
//
//  Tree Options
//
/////////////////////////////////////


KernelBlockDefs.TreeOpt = {
  blocktitle: 'Tree Options',
  paramtype: 'Dropdown',
  nargs: '*',
};

KernelBlockDefs.First = {
  blocktitle: 'First',
  paramtype: 'Integer',
  paramval: [
    {
      value: 'first',
      text: 'First (n)',
      descr: '',
      after: true
    }
  ]
};

KernelBlockDefs.Last = {
  blocktitle: 'Last',
  paramtype: 'Integer',
  paramval: [
    {
      value: 'last',
      text: 'Last (n)',
      descr: '',
      after: true
    }
  ]
};

KernelBlockDefs.Onward = {
  blocktitle: 'Onward',
  paramtype: 'Integer',
  paramval: [
    {
      value: 'onward',
      text: 'Onward (n)',
      descr: '',
      after: true
    }
  ]
};

KernelBlockDefs.Maxwords = {
  blocktitle: 'Maxwords',
  paramtype: 'Integer',
  paramval: [
    {
      value: 'maxwords',
      text: 'Maxwords (n)',
      descr: '',
      after: true
    }
  ]
};

/////////////////////////////////////
//
//  LMEFIT Parameters
//
/////////////////////////////////////

KernelBlockDefs.BaselineFormula = {
  blocktitle: 'LME Baseline Formula',
  paramtype: 'Text',
  paramval: [
    {
      value: '',
      text: 'Basename of LME formula file'
    }
  ]
};

KernelBlockDefs.LMEArgs = {
  blocktitle: 'LME command-line arguments',
  paramtype: 'Text',
  paramval: [
    {
      value: '',
      text: 'Underscore-delimited command-line args to the LME regression',
      descr: 'Run bare MB/resource-lmefit/scripts/evmeasures2lmefit.r for documentation'
    }
  ]
}

/////////////////////////////////////
//
//  N-gram options
//
/////////////////////////////////////

KernelBlockDefs.NgramOrder = {
  blocktitle: 'N-gram order',
  paramtype: 'Integer',
  paramval: [
    {
      value: '',
      text: 'N-gram order'
    }
  ]
};

KernelBlockDefs.NgramModel = {
  blocktitle: 'N-gram model',
  paramtype: 'Dropdown',
};

KernelBlockDefs.NgramIZ = {
  blocktitle: 'N-gram IZ',
  paramtype: 'Boolean',
  nargs: '?',
  paramval: [
    {
      value: 'iz',
      text: 'Add item/zone fields (used only for Natural Stories)'
    }
  ]
}



