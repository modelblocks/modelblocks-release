/////////////////////////////////////
//
// FINAL TARGET CLASSES 
//
/////////////////////////////////////

TargetBlockDefs = {
  Linetoks: {
    blocktitle: 'Linetoks',
    targetsuffix: '.linetoks',
    blockseq: [
      {
        blocktype: 'Corpus',
      },
      {
        blocktype: 'TreeOpt',
        kwargs: {
          delim: '.'
        }
      }
    ]
  },
  Linetrees: {
    blocktitle: 'Linetrees',
    targetsuffix: '.linetrees',
    blockseq: [
      {
        blocktype: 'Corpus',
      },
      {
        blocktype: 'TreeOpt',
        kwargs: {
          delim: '.'
        }
      }
    ]
  },
  ParseLinetrees: {
    blocktitle: 'Parse (Linetrees)',
    targetsuffix: '_parsed.linetrees',
    blockseq: [
      {
        blocktype: 'Corpus',
        kwargs: {
          instancename: 'Test'
        }
      },
      {
        blocktype: 'String',
        value: '.'
      },
      {
        blocktype: 'ParseParams',
      }
    ]
  },
  RTCoreEvmeasures: {
    blocktitle: 'Reading Time Data',
    targetsuffix: '.core.evmeasures',
    blockseq: [
      {
        blocktype: 'RTDataParams',
      }
    ]
  },
  LMEFit: {
    blocktitle: 'LME Fit',
    targetsuffix: '.lmefit',
    blockseq: [
      {
        blocktype: 'RTDataParams',
      },
      {
        blocktype: 'String',
        value: '.'
      },
      {
        blocktype: 'BaselineFormula'
      },
      {
        blocktype: 'String',
        value: '.'
      },
      {
        blocktype: 'LMEArgs'
      }
    ]
  },
  Tokdeps: {
    blocktitle: 'Tokdeps',
    targetsuffix: '.tokdeps',
    blockseq: [
      {
        blocktype: 'Corpus',
      },
      {
        blocktype: 'TreeOpt',
        kwargs: {
          delim: '.'
        }
      }
    ]
  }
}

