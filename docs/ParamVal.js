/////////////////////////////////////
//
//  PARAMVAL DEFINITIONS
//
/////////////////////////////////////

ParamVal = {}

/////////////////////////////////////
//
//  CORPUS
//
/////////////////////////////////////

ParamVal.Corpus = {
  paramtype: 'Dropdown',
  nargs: 1,
  V: {
    alice: { 
      value: 'alice',
      text: 'Alice in Wonderland, Chapter 1',
      descr: '',
    },
    dundee: {
      value: 'dundee',
      text: 'Dundee (eye-tracking corpus)',
      descr: '',
    },
    naturalstories: {
      value: 'naturalstories',
      text: 'Natural Stories (self-paced reading corpus)',
      descr: '',
    },
    ucl: {
      value: 'ucl',
      text: 'UCL (eye-tracking)',
      descr: '',
    },
    wsj02to21: {
      value: 'wsj02to21',
      text: 'Wall Street Journal training set (sections 2-21)',
      descr: '',
    },
    wsj23: { 
      value: 'wsj23',
      text: 'Wall Street Journal test set (section 23)',
      descr: '',
    }
  }
}

/////////////////////////////////////
//
//  TREE OPTIONS
//
/////////////////////////////////////

ParamVal.first = {
  paramtype: 'Integer',
  V: {
    0: {
      value: 'first',
      text: 'First (n)',
      descr: '',
      after: true
    }
  }
}
ParamVal.last = {
  paramtype: 'Integer',
  V: {
    0: {
      value: 'last',
      text: 'Last (n)',
      descr: '',
      after: true
    }
  }
}
ParamVal.onward = {
  paramtype: 'Integer',
  V: {
    0: {
      value: 'onward',
      text: 'Onward (n)',
      descr: '',
      after: true
    }
  }
}
ParamVal.maxwords = {
  paramtype: 'Integer',
  V: {
    0: {
      value: 'maxwords',
      text: 'Maxwords (n)',
      descr: '',
      after: true
    }
  }
}
ParamVal.TreeOpt = {
  paramtype: 'Dropdown',
  nargs: '*',
  V: {
    first: {
      value: '',
      text: 'Keep only the first n trees',
      cascade: [ParamVal.first],
    },
    last: {
      value: '',
      text: 'Keep only the last n trees',
      cascade: [ParamVal.last],
    },
    onward: {
      value: '',
      text: 'Discard the first n trees',
      cascade: [ParamVal.onward],
    },
    maxword: {
      value: '',
      text: 'Keep only trees with fewer than n words',
      cascade: [ParamVal.maxwords],
    },
    fromdeps: {
      value: 'fromdeps',
      text: 'Extract trees from a dependency graph',
      descr: ''
    },
    nolabel: {
      value: 'nolabel',
      text: 'Remove non-terminal category labels',
      descr: ''
    },
    nopunc: {
      value: 'nopunc',
      text: 'Remove punctuation',
      descr: '',
    },
    nounary: {
      value: 'nounary',
      text: 'Remove unary branches',
      descr: ''
    }
  }
}

/////////////////////////////////////
//
//  GRAMMAR
//
/////////////////////////////////////

ParamVal.GCG = {
  paramtype: 'Dropdown',
  nargs: 1,
  V: {
    16: {
      value: '16',
      text: '2016 spec',
      descr: '',
    },
    15: {
      value: '15',
      text: '2015 spec',
      descr: '',
    },
    14: {
      value: '14',
      text: '2014 spec',
      descr: '',
    },
    13: {
      value: '13',
      text: '2013 spec',
      descr: '',
    }
  }
}
ParamVal.Decoupled = {
  paramtype: 'Boolean',
  V: {
    0: {
      value: '-decoupled',
      text: 'Decoupled',
      descr: '',
    }
  }
}
ParamVal.Grammar = {
  paramtype: 'Dropdown',
  nargs: 1,
  V: {
    PTB: {
      value: 'nodashtags',
      text: 'Penn Treebank',
      descr: '',
    },
    GCG: {
      value: 'gcg',
      text: 'Generalized Categorial Grammar (GCG)',
      descr: '',
      cascade: [ParamVal.GCG, ParamVal.Decoupled]
    }
  }
}

/////////////////////////////////////
//
//  PARSER
//
/////////////////////////////////////

ParamVal.C = {
  paramtype: 'Boolean',
  V: {
    0: {
      value: '+c',
      text: 'Output complexity metrics',
      descr: '',
    }
  }
}
ParamVal.BeamSize = {
  paramtype: 'Integer',
  V: {
    0: {
      value: '+b',
      text: 'Beam Size',
      descr: '',
    }
  }
}
ParamVal.Parser = {
  paramtype: 'Dropdown',
  nargs: 1,
  V: {
    EFABP: {
      value: 'x+efabp-',
      text: 'EFABP parser (van Schijndel et al (2013)',
      descr: '',
      cascade: [ParamVal.C, ParamVal.BeamSize]
    },
    fullberk: {
      value: 'fullberk-',
      text: 'Full Berkeley parser',
      descr: '',
    },
    synproc: {
      value: 'synproc-',
      text: 'Incremental syntactic processing parser (synproc)',
      descr: '',
      cascade: [ParamVal.C, ParamVal.BeamSize]
    },
    vitberk: {
      value: 'vitberk-',
      text: 'Viterbi Berkeley parser',
      descr: '',
    }
  }
}

/////////////////////////////////////
//
//  MODEL OPTIONS
//
/////////////////////////////////////

ParamVal.SplitMerge = {
  paramtype: 'Integer',
  V: {
    0: {
      value: 'sm',
      text: 'Split Merge Iterations',
      descr: '',
      after: true
    }
  }
}
ParamVal.ModelOpt = {
  paramtype: 'Multiselect',
  V: {
    FG: {
      value: 'fg',
      text: 'Filler-gap transform (inserts stack elements for long-distance dependencies)',
      descr: '',
    },
    SM: {
      value: '',
      text: 'Split-merge iterations',
      descr: '',
      cascade: [ParamVal.SplitMerge]
    },
    BD: {
      value: 'bd',
      text: 'Annotate branching direction and depth',
      descr: '',
    }
  }
}

/////////////////////////////////////
//
//  NGRAM OPTIONS
//
/////////////////////////////////////

ParamVal.NgramOrder = {
  paramtype: 'Integer',
  V: {
    0: {
      value: '',
      text: 'N-gram order'
    }
  }
}
ParamVal.NgramModel = {
  paramtype: 'Dropdown',
  V: {
    srilm: {
      value: 'srilm',
      text: 'SRILM'
    },
    kenlm: {
      value: 'kenlm',
      text: 'KENLM'
    }
  }
}
ParamVal.NgramIZ = {
  paramtype: 'Boolean',
  V: {
    0: {
      value: 'iz',
      text: 'Add item/zone fields (used only for Natural Stories)'
    }
  }
}

/////////////////////////////////////
//
//  LMEFIT PARAMETERS
//
/////////////////////////////////////

ParamVal.BaselineFormula = {
  paramtype: 'Text',
  V: {
    0: {
      value: '',
      text: 'Basename of LME formula file'
    }
  }
}
ParamVal.LMEArgs = {
  paramtype: 'Text',
  V: {
    0: {
      value: '',
      text: 'Underscore-delimited command-line args to the LME regression',
      descr: 'Run bare MB/resource-lmefit/scripts/evmeasures2lmefit.r for documentation'
    }
  }
}

