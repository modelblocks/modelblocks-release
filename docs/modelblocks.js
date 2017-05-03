/////////////////////////////////////
//
//  KERNEL BLOCK CLASSES
//
/////////////////////////////////////

// Corpus class
function Corpus(args) {
  var self = this;
  if (args === undefined) {
    args = {};
  }
  if (args.value === undefined) {
    self.value = null;
  } else {
    self.value = args.value;
  }
  if (args.name === undefined) {
    self.name = '';
  } else {
    self.name = args.name;
  }
  if (args.descr === undefined) {
    self.descr = '';
  } else {
    self.descr = args.descr;
  }
  if (args.required === undefined) {
    self.required = true;
  } else {
    self.required = args.required;
  }
  if (args.delim === undefined) {
    self.delim = '';
  } else {
    self.delim = args.delim;
  }
  self.defaultdisplay = '<CORPUS>';
  self.blocktype = 'Corpus';
  if (self.value == null) {
    self.string = self.defaultdisplay;
  } else {
    self.string = self.value;
  }
  self.displayspan = $('<span>');
  self.displayspan.text(self.string);
  self.select = $('<select>');
  self.selectcontainer = $('<div class="paramselect"></div>');
  if (self.name != '') {
    self.selectcontainer.append('<h4>' + self.name + ' Corpus</h4>');
  } else {
    self.selectcontainer.append('<h4>Corpus</h4>');
  }
  self.selectcontainer.append(self.select);
  self.select.append('<option value="none"></option>');
  for (i in self.values) {
    self.select.append('<option value="' + i + '">' + self.values[i] + '</option>')
  }
  self.select.change(
    function () {
      updateTarget(self.displayspan, getValString(self.select, self.delim));
    }
  )
}

Corpus.prototype.values = {
                           'alice': 'Alice in Wonderland, Chapter 1',
                           'dundee': 'Dundee (eye-tracking corpus)',
                           'naturalstories': 'Natural Stories (self-paced reading corpus)',
                           'ucl': 'UCL (eye-tracking)',
                           'wsj02to21': 'Wall Street Journal training set (sections 2-21)',
                           'wsj23': 'Wall Street Journal test set (section 23)'
                          }

// Grammar class
function Grammar(args) {
  var self = this;
  if (args === undefined) {
    args = {};
  }
  if (args.value === undefined) {
    self.value = null;
  } else {
    self.value = args.value;
  }
  if (args.name === undefined) {
    self.name = '';
  } else {
    self.name = args.name;
  }
  if (args.descr === undefined) {
    self.descr = '';
  } else {
    self.descr = args.descr;
  }
  if (args.required === undefined) {
    self.required = true;
  } else {
    self.required = args.required;
  }
  if (args.delim === undefined) {
    self.delim = '';
  } else {
    self.delim = args.delim;
  }
  self.defaultdisplay = '<GRAMMAR>';
  self.blocktype = 'Grammar';
  if (self.value == null) {
    self.string = self.defaultdisplay;
  } else {
    self.string = self.value;
  }
  self.displayspan = $('<span>');
  self.displayspan.text(self.string);
  self.select = $('<select>');
  self.selectcontainer = $('<div class="paramselect"></div>');
  if (self.name != '') {
    self.selectcontainer.append('<h4>' + self.name + ' Grammar</h4>');
  } else {
    self.selectcontainer.append('<h4>Grammar</h4>');
  }
  self.selectcontainer.append(self.select);
  self.select.append('<option value="none"></option>');
  for (i in self.values) {
    self.select.append('<option value="' + i + '">' + self.values[i] + '</option>')
  }
  self.select.change(
    function () {
      updateTarget(self.displayspan, getValString(self.select, self.delim));
    }
  )
}

Grammar.prototype.values = {
                            'nodashtags': 'Penn Treebank',
                            'gcg15': 'GCG (2015 specification)',
                            'gcg14': 'GCG (2014 specification)',
                            'gcg13': 'GCG (2013 specification)'
                           }

// ModelOpts class
function ModelOpts(args) {
  var self = this;
  if (args === undefined) {
    args = {};
  }
  if (args.value === undefined) {
    self.value = null;
  } else {
    self.value = args.value;
  }
  if (args.name === undefined) {
    self.name = '';
  } else {
    self.name = args.name;
  }
  if (args.descr === undefined) {
    self.descr = '';
  } else {
    self.descr = args.descr;
  }
  if (args.required === undefined) {
    self.required = true;
  } else {
    self.required = args.required;
  }
  if (args.delim === undefined) {
    self.delim = '';
  } else {
    self.delim = args.delim;
  }
  self.defaultdisplay = '<MODEL-OPTIONS>';
  self.blocktype = 'ModelOpts';
  if (self.value == null) {
    self.string = self.defaultdisplay;
  } else {
    self.string = self.value;
  }
  self.displayspan = $('<span>');
  self.displayspan.text(self.string);
  self.select = $('<select multiple>');
  self.selectcontainer = $('<div class="paramselect"></div>');
  if (self.name != '') {
    self.selectcontainer.append('<h4>' + self.name + ' Tree Options</h4>');
  } else {
    self.selectcontainer.append('<h4>Model Options</h4>');
  }
  self.selectcontainer.append(self.select);
  self.select.append('<option value="none"></option>');
  for (i in self.values) {
    self.select.append('<option value="' + i + '">' + self.values[i] + '</option>')
  }
  self.select.change(
    function () {
      updateTarget(self.displayspan, getValString(self.select, self.delim));
    }
  )
}

ModelOpts.prototype.values = {
                              'fg': 'Filler-gap transform (inserts stack elements for long-distance dependencies)',
                              '3sm': 'Split-merge iterations = 3',
                              'bd': 'Annotate branching direction and depth'
                             }

// Parser class
function Parser(args) {
  var self = this;
  if (args === undefined) {
    args = {};
  }
  if (args.value === undefined) {
    self.value = null;
  } else {
    self.value = args.value;
  }
  if (args.name === undefined) {
    self.name = '';
  } else {
    self.name = args.name;
  }
  if (args.descr === undefined) {
    self.descr = '';
  } else {
    self.descr = args.descr;
  }
  if (args.required === undefined) {
    self.required = true;
  } else {
    self.required = args.required;
  }
  if (args.delim === undefined) {
    self.delim = '';
  } else {
    self.delim = args.delim;
  }
  self.defaultdisplay = '<PARSER>';
  self.blocktype = 'Parser';
  if (self.value == null) {
    self.string = self.defaultdisplay;
  } else {
    self.string = self.value;
  }
  self.displayspan = $('<span>');
  self.displayspan.text(self.string);
  self.select = $('<select>');
  self.selectcontainer = $('<div class="paramselect"></div>');
  if (self.name != '') {
    self.selectcontainer.append('<h4>' + self.name + ' Parser</h4>');
  } else {
    self.selectcontainer.append('<h4>Parser</h4>');
  }
  self.selectcontainer.append(self.select);
  self.select.append('<option value="none"></option>');
  for (i in self.values) {
    self.select.append('<option value="' + i + '">' + self.values[i] + '</option>')
  }
  self.select.change(
    function () {
      updateTarget(self.displayspan, getValString(self.select, self.delim));
    }
  )
}

Parser.prototype.values = {
                           'x+efabp': 'EFABP parser (van Schijndel et al (2013)',
                           'fullberk': 'Full Berkeley parser',
                           'synproc': 'Incremental syntactic processing parser (synproc)',
                           'vitberk': 'Viterbi Berkeley parser'
                          }

// ParserOpts class
function ParserOpts(args) {
  var self = this;
  if (args === undefined) {
    args = {};
  }
  if (args.value === undefined) {
    self.value = null;
  } else {
    self.value = args.value;
  }
  if (args.name === undefined) {
    self.name = '';
  } else {
    self.name = args.name;
  }
  if (args.descr === undefined) {
    self.descr = '';
  } else {
    self.descr = args.descr;
  }
  if (args.required === undefined) {
    self.required = true;
  } else {
    self.required = args.required;
  }
  if (args.delim === undefined) {
    self.delim = '';
  } else {
    self.delim = args.delim;
  }
  self.defaultdisplay = '<PARSER-OPTIONS>';
  self.blocktype = 'ParserOpts';
  if (self.value == null) {
    self.string = self.defaultdisplay;
  } else {
    self.string = self.value;
  }
  self.displayspan = $('<span>');
  self.displayspan.text(self.string);
  self.select = $('<select multiple>');
  self.selectcontainer = $('<div class="paramselect"></div>');
  if (self.name != '') {
    self.selectcontainer.append('<h4>' + self.name + ' Parser Options</h4>');
  } else {
    self.selectcontainer.append('<h4>Parser Options</h4>');
  }
  self.selectcontainer.append(self.select);
  self.select.append('<option value="none"></option>');
  for (i in self.values) {
    self.select.append('<option value="' + i + '">' + self.values[i] + '</option>')
  }
  self.select.change(
    function () {
      updateTarget(self.displayspan, getValString(self.select, self.delim));
    }
  )
}

ParserOpts.prototype.values = {
                           '+c': 'C',
                           '+b2000': 'Beam width = 2000',
                           '+b5000': 'Beam width = 5000'
                          }

// TreeOpts class
function TreeOpts(args) {
  var self = this;
  if (args === undefined) {
    args = {};
  }
  if (args.value === undefined) {
    self.value = null;
  } else {
    self.value = args.value;
  }
  if (args.name === undefined) {
    self.name = '';
  } else {
    self.name = args.name;
  }
  if (args.descr === undefined) {
    self.descr = '';
  } else {
    self.descr = args.descr;
  }
  if (args.required === undefined) {
    self.required = true;
  } else {
    self.required = args.required;
  }
  if (args.delim === undefined) {
    self.delim = '';
  } else {
    self.delim = args.delim;
  }
  self.defaultdisplay = '<TREE-OPTIONS>';
  self.blocktype = 'TreeOpts';
  if (self.value == null) {
    self.string = self.defaultdisplay;
  } else {
    self.string = self.value;
  }
  self.displayspan = $('<span>');
  self.displayspan.text(self.string);
  self.select = $('<select multiple>');
  self.selectcontainer = $('<div class="paramselect"></div>');
  if (self.name != '') {
    self.selectcontainer.append('<h4>' + self.name + ' Tree Options</h4>');
  } else {
    self.selectcontainer.append('<h4>Tree Options</h4>');
  }
  self.selectcontainer.append(self.select);
  self.select.append('<option value="none"></option>');
  for (i in self.values) {
    self.select.append('<option value="' + i + '">' + self.values[i] + '</option>')
  }
  self.select.change(
    function () {
      updateTarget(self.displayspan, getValString(self.select, self.delim));
    }
  )
}

TreeOpts.prototype.values = {
                             'fromdeps': 'Extract trees from a dependency graph',
                             'nolabel': 'Remove non-terminal category labels',
                             'nopunc': 'Remove punctuation',
                             'nounary': 'Remove unary branches'
                            }

/////////////////////////////////////
//
//  COMPOSITE BLOCK CLASSES
//
/////////////////////////////////////

// ParseParams

function ParseParams(args) {
  var self = this;
  if (args === undefined) {
    args = {};
  }
  if (args.name === undefined) {
    self.name = '';
  } else {
    self.name = args.name;
  }
  self.c = new Corpus({name:'Training'});
  self.g = new Grammar();
  self.to = new TreeOpts({delim: '-'});
  self.mo = new ModelOpts({delim: '-'});
  self.p = new Parser();
  self.po = new ParserOpts({delim: '_'});
  self.blocktype = 'ParseParams';
  self.displayspan = $('<span>');
  self.displayspan.append(self.c.displayspan);
  self.displayspan.append('-');
  self.displayspan.append(self.g.displayspan);
  self.displayspan.append(self.to.displayspan);
  self.displayspan.append(self.mo.displayspan);
  self.displayspan.append('-');
  self.displayspan.append(self.p.displayspan);
  self.displayspan.append('-');
  self.displayspan.append(self.po.displayspan);
  self.selectcontainer = $('<div class="paramselect"></div>');
  if (self.name != '') {
    self.selectcontainer.append('<h4>' + self.name + ' Parser Parameters</h4>');
  } else {
    self.selectcontainer.append('<h4>Parser Parameters</h4>');
  }
  self.selectcontainer.append(self.c.selectcontainer)
  self.selectcontainer.append(self.g.selectcontainer)
  self.selectcontainer.append(self.to.selectcontainer)
  self.selectcontainer.append(self.mo.selectcontainer)
  self.selectcontainer.append(self.p.selectcontainer)
  self.selectcontainer.append(self.po.selectcontainer)
}

/////////////////////////////////////
//
// FINAL TARGET CLASSES 
//
/////////////////////////////////////

// %.linetrees class
function Linetrees(args) {
  var self = this;
  if (args === undefined) {
    args = {};
  }
  if (args.name === undefined) {
    self.name = '';
  } else {
    self.name = args.name;
  }
  self.t = new Corpus({name: 'Test'});
  self.to = new TreeOpts({delim: '.'});
  self.blocktype = 'Linetrees';
  self.displayspan = $('<span>');
  self.displayspan.append('genmodel/');
  self.displayspan.append(self.t.displayspan);
  self.displayspan.append(self.to.displayspan);
  self.displayspan.append('.linetrees ');
  self.selectcontainer = $('<div class="targetparams"></div>');
  if (self.name != '') {
    self.selectcontainer.append('<h4>' + self.name + ' Linetrees Target (<span class="code">%.linetrees</span>)</h4>');
  } else {
    self.selectcontainer.append('<h4>Linetrees Target (<span class="code">%.linetrees</span>)</h4>');
  }
  self.selectcontainer.append(self.t.selectcontainer)
  self.selectcontainer.append(self.to.selectcontainer)
  self.trayicon = $('<button class="traybutton selected">');
  self.trayicon.text('LINETREES');
  self.trayicon.click(function () {
    var worksurface = $('div#worksurface');
    var tray = $('div#tray');
    tray.find('button').removeClass('selected');
    $(this).addClass('selected');
    worksurface.find('div.targetparams').detach();
    worksurface.append(self.selectcontainer);
  });
  self.deleter = $('<button class="trashbutton"><i class="fa fa-trash" aria-hidden="true"></i>DELETE</button>');
  self.selectcontainer.append(self.deleter)
  self.deleter.click(function() {
    self.displayspan.remove();
    self.trayicon.remove();
    self.selectcontainer.remove();
    self.deleter.remove();
  });
}
// %.parse.linetrees class
function ParseLinetrees(args) {
  var self = this;
  if (args === undefined) {
    args = {};
  }
  if (args.name === undefined) {
    self.name = '';
  } else {
    self.name = args.name;
  }
  self.t = new Corpus({name: 'Test'});
  self.pp = new ParseParams();
  self.blocktype = 'ParseLinetrees';
  self.displayspan = $('<span>');
  self.displayspan.append('genmodel/');
  self.displayspan.append(self.t.displayspan);
  self.displayspan.append('.');
  self.displayspan.append(self.pp.displayspan);
  self.displayspan.append('_parsed.linetrees ');
  self.selectcontainer = $('<div class="targetparams"></div>');
  if (self.name != '') {
    self.selectcontainer.append('<h4>' + self.name + ' Parse Target (<span class="code">%.parsed.linetrees</span>)</h4>');
  } else {
    self.selectcontainer.append('<h4>Parse Target (<span class="code">%parsed.linetrees</span>)</h4>');
  }
  self.selectcontainer.append(self.t.selectcontainer)
  self.selectcontainer.append(self.pp.selectcontainer)
  self.trayicon = $('<button class="traybutton selected">');
  self.trayicon.text('PARSE');
  self.trayicon.click(function () {
    var worksurface = $('div#worksurface');
    var tray = $('div#tray');
    tray.find('button').removeClass('selected');
    $(this).addClass('selected');
    worksurface.find('div.targetparams').detach();
    worksurface.append(self.selectcontainer);
  });
  self.deleter = $('<button class="trashbutton"><i class="fa fa-trash" aria-hidden="true"></i>DELETE</button>');
  self.selectcontainer.append(self.deleter)
  self.deleter.click(function() {
    self.displayspan.remove();
    self.trayicon.remove();
    self.selectcontainer.remove();
    self.deleter.remove();
  });
}

/////////////////////////////////////
//
// UTILITY FUNCTIONS 
//
/////////////////////////////////////

function updateTarget(div, newtext) {
  div.text(newtext);
}

function getValString(select, delim='') {
  var vals = []
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

/////////////////////////////////////
//
// PREPARE DOM 
//
/////////////////////////////////////

var worksurface = $('div#worksurface'); 
var tray = $('div#tray');
var target = $('div#target');
target.append('make ');

$('button#addParse').click(function() {
  worksurface.find('div.targetparams').detach();
  var parse = new ParseLinetrees();
  target.append(parse.displayspan);
  tray.find('button').removeClass('selected');
  tray.append(parse.trayicon);
  worksurface.append(parse.selectcontainer);
});

$('button#addLinetrees').click(function() {
  worksurface.find('div.targetparams').detach();
  var linetrees = new Linetrees();
  target.append(linetrees.displayspan);
  tray.find('button').removeClass('selected');
  tray.append(linetrees.trayicon);
  worksurface.append(linetrees.selectcontainer);
});

$('button#copier').click(function() {copyToClipboard(target);})
