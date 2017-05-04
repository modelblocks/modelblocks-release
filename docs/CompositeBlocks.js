/////////////////////////////////////
//
//  COMPOSITE BLOCK CLASSES
//
/////////////////////////////////////

CompositeBlockDefs = {
  ParseParams: {
    blocktitle: 'Parser Parameters',
    blockseq: [
      {
        blocktype: 'Corpus',
        kwargs: {
          instancename: 'Training'
        }
      },
      {
        blocktype: 'String',
        value: '-'
      },
      {
        blocktype: 'Grammar',
      },
      {
        blocktype: 'TreeOpt',
        kwargs: {
          delim: '-'
        }
      },
      {
        blocktype: 'ModelOpt',
        kwargs: {
          delim: '-'
        }
      },
      {
        blocktype: 'String',
        value: '-'
      },
      {
        blocktype: 'Parser',
        kwargs: {
          delim: '_'
        }
      }
    ]
  },
  NgramParams: {
    blocktitle: 'N-gram Parameters',
    blockseq: [
      {
        blocktype: 'NgramOrder'
      },
      {
        blocktype: 'NgramModel'
      },
      {
        blocktype: 'NgramIZ'
      }
    ]
  },
  RTDataParams: {
    blocktitle: 'Reading Time Data Parameters',
    blockseq: [
      {
        blocktype: 'Corpus',
        kwargs: {
          instancename: 'Reading Time',
        }
      },
      {
        blocktype: 'String',
        value: '.'
      },
      {
        blocktype: 'ParseParams',
        kwargs: {
          instancename: 'Surprisal Metrics'
        }
      },
      {
        blocktype: 'String',
        value: '.'
      },
      {
        blocktype: 'NgramParams',
        kwargs: {
          delim: '-'
        }
      }
    ]
  }
}

