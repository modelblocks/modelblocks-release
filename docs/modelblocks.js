function updateTarget(div, newtext) {
  div.text(newtext);
}

function getValString(select, delim='') {
  vals = []
  select.find(':selected').each(function() {text = $(this).text(); if (text != '') {vals.push(delim + $(this).attr('value'));}})
  return vals.join('')
}

function copyToClipboard(element) {
  var $temp = $("<input>");
  $("body").append($temp);
  $temp.val($(element).text()).select();
  document.execCommand("copy");
  $temp.remove();
}

function addBlock(blocks, block, id, name, parameters, target, mult=false, delim='') {
  if (mult) {
    new_block = $('<div id="' + id + '_' + block + '_select" class="paramselect"><h4>' +
                    name + '</h4><select multiple><option></option></select></div>');
  } else {
    new_block = $('<div id="' + id + '_' + block + '_select" class="paramselect"><h4>' +
                    name + '</h4><select><option>' + blocks[block]['default'] + '</option></select></div>');
  }
  for (i in blocks[block]['options']) {
    new_block.find('select').append('<option value="' + blocks[block]['options'][i] + '">' + i + '</option>');
  }
  new_span = $('<span id="' + id + '_' + block + '">' + blocks[block]['default'] + '</span>')
  target.append(new_span)
  parameters.append(new_block)
  new_block.change(function () {updateTarget($('span#' + id + '_' + block), getValString($('div#' + id + '_' + block + '_select'), delim))});
}

blocks = {
'corpus': {
           'default': '&lt;CORPUS&gt;',
           'options': {
                       'Alice in Wonderland, Chapter 1': 'alice',
                       'Dundee (eye-tracking corpus)': 'dundee',
                       'Natural Stories (self-paced reading corpus)': 'naturalstories',
                       'UCL (eye-tracking)': 'ucl',
                       'Wall Street Journal training set (sections 2-21)': 'wsj02to21',
                       'Wall Street Journal test set (section 23)': 'wsj23'
                      }
          },

'grammar': {
            'default': '&lt;GRAMMAR&gt;',
            'options': {
                        'Penn Treebank': 'nodashtags',
                        'GCG (2015 specification)': 'gcg15',
                        'GCG (2014 specification)': 'gcg14',
                        'GCG (2013 specification)': 'gcg13'
                       }
           },

'tree_opts': {
             'default': '&lt;TREE-OPTIONS&gt;',
             'options': {
                         'Remove unary branches': 'nounary',
                         'Remove non-terminal category labels': 'nolabel',
                         'Remove punctuation': 'nopunc'
                        }
            },

'model_opts': {
             'default': '&lt;MODEL-OPTIONS&gt;',
             'options': {
                         'Filler-gap transform (inserts stack elements for long-distance dependencies)': 'fg',
                         'Split-merge iterations = 3': '3sm',
                         'Annotate branching direction and depth': 'bd'
                        }
            },

'parser': {
            'default': '&lt;PARSER&gt;',
            'options': {
                        'EFABP parser (van Schijndel et al (2013)': 'x+efabp',
                        'Full Berkeley parser': 'fullberk',
                        'Incremental syntactic processing parser (synproc)': 'synproc',
                        'Viterbi Berkeley parser': 'vitberk'
                       }
           },

'parser_opts': {
             'default': '&lt;PARSER-OPTIONS&gt;',
             'options': {
                         'C': '+c',
                         'Beam width 2000': '+b2000'
                        }
            }
}

parameters = $('div#parameters'); 

target = $('div#target')
target.append('<span>genmodel/</span>')

addBlock(blocks, 'corpus', 'test', 'Test Corpus', parameters, target)
target.append('.')
addBlock(blocks, 'corpus', 'train', 'Training Corpus', parameters, target)
target.append('-')
addBlock(blocks, 'grammar', 'train', 'Grammar', parameters, target)
addBlock(blocks, 'tree_opts', 'train', 'Training Tree Options', parameters, target, mult=true, delim='-')
addBlock(blocks, 'model_opts', 'train', 'Training Model Options', parameters, target, mult=true, delim='-')
target.append('-')
addBlock(blocks, 'parser', 'train', 'Parser', parameters, target)
target.append('-')
addBlock(blocks, 'parser_opts', 'train', 'Parser Options', parameters, target, mult=true, delim='_')
target.append('_parsed.linetrees')

$('button#copier').click(function() {copyToClipboard(target);})
