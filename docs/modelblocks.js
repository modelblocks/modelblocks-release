function updateTarget(div, newtext) {
  div.text(newtext);
}

function getValString(select, delim='-') {
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

corpora = {
           'default': '&lt;CORPUS&gt;',
           'options': {
                       'Alice in Wonderland, Chapter 1': 'alice',
                       'Dundee (eye-tracking corpus)': 'dundee',
                       'Natural Stories (self-paced reading corpus)': 'naturalstories',
                       'Wall Street Journal training set (sections 2-21)': 'wsj02to21',
                       'Wall Street Journal test set (section 23)': 'wsj23'
                      }
          }

grammars = {
            'default': '&lt;GRAMMAR&gt;',
            'options': {
                        'Penn Treebank': 'nodashtags',
                        'GCG (2015 specification)': 'gcg15',
                        'GCG (2014 specification)': 'gcg14',
                        'GCG (2013 specification)': 'gcg13'
                       }
           }

tree_opts = {
             'default': '&lt;TREE-OPTIONS&gt;',
             'options': {
                         'Remove unary branches': 'nounary',
                         'Remove non-terminal category labels': 'nolabel',
                         'Remove punctuation': 'nopunc'
                        }
            }


model_opts = {
             'default': '&lt;MODEL-OPTIONS&gt;',
             'options': {
                         'Filler-gap transform (inserts stack elements for long-distance dependencies)': 'fg',
                         'Split-merge iterations = 3': '3sm',
                         'Annotate branching direction and depth': 'bd'
                        }
            }

parsers = {
            'default': '&lt;PARSER&gt;',
            'options': {
                        'EFABP parser (van Schijndel et al (2013)': 'x+efabp',
                        'Full Berkeley parser': 'fullberk',
                        'Incremental syntactic processing parser (synproc)': 'synproc',
                        'Viterbi Berkeley parser': 'vitburk'
                       }
           }

parser_opts = {
             'default': '&lt;PARSER-OPTIONS&gt;',
             'options': {
                         'C': '+c',
                         'Beam width 2000': '+b2000'
                        }
            }

parameters = $('div#parameters'); 

test_corpus = $('<div id="test_corpus_select" class="paramselect"><h4>Test Corpus</h4><select><option>' + corpora['default'] + '</option></select></div>');
for (i in corpora['options']) {
  test_corpus.find('select').append('<option value="' + corpora['options'][i] + '">' + i + '</option>');
}
test_corpus.change(function () {updateTarget($('span#test_corpus'), getValString(test_corpus, ''))});

test_tree_options = $('<div id="test_tree_options_select" class="paramselect"><h4>Test Tree Options</h4><select multiple><option></option></select></div>');
for (i in tree_opts['options']) {
  test_tree_options.find('select').append('<option value="' + tree_opts['options'][i] + '">' + i + '</option>');
}
test_tree_options.change(function () {updateTarget($('span#test_tree_options'), getValString(test_tree_options))});

train_corpus = $('<div id="train_corpus_select" class="paramselect"><h4>Training Corpus</h4><select><option>' + corpora['default'] + '</option></select></div>');
for (i in corpora['options']) {
  train_corpus.find('select').append('<option value="' + corpora['options'][i] + '">' + i + '</option>');
}
train_corpus.change(function () {updateTarget($('span#train_corpus'), getValString(train_corpus, ''))});

grammar = $('<div id="grammar_select" class="paramselect"><h4>Grammar</h4><select><option>' + grammars['default'] + '</option></select></div>');
for (i in grammars['options']) {
  grammar.find('select').append('<option value="' + grammars['options'][i] + '">' + i + '</option>');
}
grammar.change(function () {updateTarget($('span#grammar'), getValString(grammar, ''))});

train_tree_options = $('<div id="training_tree_options_select" class="paramselect"><h4>Training Tree Options</h4><select multiple><option></option></select></div>');
for (i in tree_opts['options']) {
  train_tree_options.find('select').append('<option value="' + tree_opts['options'][i] + '">' + i + '</option>');
}
train_tree_options.change(function () {updateTarget($('span#train_tree_options'), getValString(train_tree_options))});

train_model_options = $('<div id="training_model_options_select" class="paramselect"><h4>Training Model Options</h4><select multiple><option></option></select></div>');
for (i in model_opts['options']) {
  train_model_options.find('select').append('<option value="' + model_opts['options'][i] + '">' + i + '</option>');
}
train_model_options.change(function () {updateTarget($('span#train_model_options'), getValString(train_model_options))});

parser = $('<div id="parser_select" class="paramselect"><h4>Parser</h4><select><option value="' + parsers['default'] + '">' + parsers['default'] + '</option></select></div>');
for (i in parsers['options']) {
  parser.find('select').append('<option value="' + parsers['options'][i] + '">' + i + '</option>');
}
parser.change(function () {updateTarget($('span#parser'), getValString(parser, ''))});

parser_options = $('<div id="parser_options_select" class="paramselect"><h4>Parser Options</h4><select multiple><option></option></select></div>');
for (i in parser_opts['options']) {
  parser_options.find('select').append('<option value="' + parser_opts['options'][i] + '">' + i + '</option>');
}
parser_options.change(function () {updateTarget($('span#parser_options'), getValString(parser_options, '_'))});

parameters.append(test_corpus)
parameters.append(test_tree_options)
parameters.append(train_corpus)
parameters.append(grammar)
parameters.append(train_tree_options)
parameters.append(train_model_options)
parameters.append(parser)
parameters.append(parser_options)

test = ['a', 'b', 'c']

target = $('div#target')
target.append('genmodel/<span id="test_corpus">' + corpora['default'] + '</span>' +
              '<span id="test_tree_options">' + tree_opts['default'] + '</span>.' +
              '<span id="train_corpus">' + corpora['default'] + '</span>-' +
              '<span id="grammar">' + grammars['default'] + '</span>' +
              '<span id="train_tree_options">' + tree_opts['default'] + '</span>' +
              '<span id="train_model_options">' + model_opts['default'] + '</span>-' +
              '<span id="parser">' + parsers['default'] + '</span>-' +
              '<span id="parser_options">' + parser_opts['default'] + '</span>' +
              '_parsed.linetrees'
              )

$('button#copier').click(function() {copyToClipboard(target);})
