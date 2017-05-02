function updateTarget(div, newtext) {
  div.text(newtext);
}

function getValString(select, delim='-') {
  vals = []
  select.find(':selected').each(function() {text = $(this).text(); if (text != '') {vals.push(delim + $(this).text());}})
  return vals.join('')
}

corpora = {
           'default': '&lt;CORPUS&gt;',
           'options': [
                       'alice',
                       'dundee',
                       'naturalstories',
                       'wsj02to21',
                       'wsj23'
                      ]
          }

grammars = {
            'default': '&lt;GRAMMAR&gt;',
            'options': [
                        'nodashtags',
                        'gcg15',
                        'gcg14',
                        'gcg13'
                       ]
           }

tree_opts = {
             'default': '&lt;TREE-OPTIONS&gt;',
             'options': [
                         'nounary',
                         'nolabel',
                         'nopunc'
                        ]
            }


model_opts = {
             'default': '&lt;MODEL-OPTIONS&gt;',
             'options': [
                         'fg',
                         '3sm',
                         'bd'
                        ]
            }

parsers = {
            'default': '&lt;PARSER&gt;',
            'options': [
                        'x+efabp',
                        'fullberk',
                        'synprob',
                        'vitburk',
                       ]
           }

parser_opts = {
             'default': '&lt;PARSER-OPTIONS&gt;',
             'options': [
                         '+c',
                         '+b2000',
                        ]
            }

parameters = $('div#parameters'); 

test_corpus = $('<div id="test_corpus_select" class="paramselect"><h4>Test Corpus</h4><select><option>' + corpora['default'] + '</option></select></div>');
for (i in corpora['options']) {
  test_corpus.find('select').append('<option>' + corpora['options'][i] + '</option>');
}
test_corpus.change(function () {updateTarget($('span#test_corpus'), test_corpus.find(':selected').text())})

test_tree_options = $('<div id="test_tree_options_select" class="paramselect"><h4>Test Tree Options</h4><select multiple><option></option></select></div>');
for (i in tree_opts['options']) {
  test_tree_options.find('select').append('<option>' + tree_opts['options'][i] + '</option>');
}
test_tree_options.change(function () {updateTarget($('span#test_tree_options'), getValString(test_tree_options))})

train_corpus = $('<div id="train_corpus_select" class="paramselect"><h4>Training Corpus</h4><select><option>' + corpora['default'] + '</option></select></div>');
for (i in corpora['options']) {
  train_corpus.find('select').append('<option>' + corpora['options'][i] + '</option>');
}
train_corpus.change(function () {updateTarget($('span#train_corpus'), train_corpus.find(':selected').text())})

grammar = $('<div id="grammar_select" class="paramselect"><h4>Grammar</h4><select><option>' + grammars['default'] + '</option></select></div>');
for (i in grammars['options']) {
  grammar.find('select').append('<option>' + grammars['options'][i] + '</option>');
}
grammar.change(function () {updateTarget($('span#grammar'), grammar.find(':selected').text())})

train_tree_options = $('<div id="training_tree_options_select" class="paramselect"><h4>Training Tree Options</h4><select multiple><option></option></select></div>');
for (i in tree_opts['options']) {
  train_tree_options.find('select').append('<option>' + tree_opts['options'][i] + '</option>');
}
train_tree_options.change(function () {updateTarget($('span#train_tree_options'), getValString(train_tree_options))})

train_model_options = $('<div id="training_model_options_select" class="paramselect"><h4>Training Model Options</h4><select multiple><option></option></select></div>');
for (i in model_opts['options']) {
  train_model_options.find('select').append('<option>' + model_opts['options'][i] + '</option>');
}
train_model_options.change(function () {updateTarget($('span#train_model_options'), getValString(train_model_options))})

parser = $('<div id="parser_select" class="paramselect"><h4>Parser</h4><select><option>' + parsers['default'] + '</option></select></div>');
for (i in grammars['options']) {
  parser.find('select').append('<option>' + parsers['options'][i] + '</option>');
}
parser.change(function () {updateTarget($('span#parser'), parser.find(':selected').text())})

parser_options = $('<div id="parser_options_select" class="paramselect"><h4>Parser Options</h4><select multiple><option></option></select></div>');
for (i in parser_opts['options']) {
  parser_options.find('select').append('<option>' + parser_opts['options'][i] + '</option>');
}
parser_options.change(function () {updateTarget($('span#parser_options'), getValString(parser_options, '_'))})

parameters.append(test_corpus)
parameters.append(test_tree_options)
parameters.append(train_corpus)
parameters.append(grammar)
parameters.append(train_tree_options)
parameters.append(train_model_options)
parameters.append(parser)
parameters.append(parser_options)
parameters.append('<div>')

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
