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
        blocktype: 'Grammar'
      },
      {
        blocktype: 'TreeOpt',
        kwargs: {
          innerdelim: '-'
        }
      },
      {
        blocktype: 'ModelOpt',
        kwargs: {
          innerdelim: '-'
        }
      },
      {
        blocktype: 'Parser',
        kwargs: {
          innerdelim: '_'
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
        blocktype: 'ParseParams',
        kwargs: {
          instancename: 'Surprisal Metrics',
          innerdelim: '-'
        }
      },
      {
        blocktype: 'NgramParams',
        kwargs: {
          innerdelim: '-'
        }
      }
    ]
  }
}

