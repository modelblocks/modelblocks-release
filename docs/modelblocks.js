/////////////////////////////////////
//
//  LIBRARY FUNCTIONS 
//
/////////////////////////////////////

(function( func ) {
    $.fn.addClass = function() { // replace the existing function on $.fn
        func.apply( this, arguments ); // invoke the original function
        this.trigger('classChanged'); // trigger the custom event
        return this; // retain jQuery chainability
    }
})($.fn.addClass); // pass the original function as an argument

(function( func ) {
    $.fn.removeClass = function() {
        func.apply( this, arguments );
        this.trigger('classChanged');
        return this;
    }
})($.fn.removeClass);

$.fn.hasAttr = function(name) {  
   attr = this.attr(name)
   return typeof attr !== typeof undefined && attr !== false;
};

/////////////////////////////////////
//
//  BLOCK CLASS CONSTRUCTORS
//
/////////////////////////////////////

function createBlockKernelClass(blocktitle, paramval) {
  function BlockKernel(kwargs) {
    var self = this;
    if (kwargs === undefined) {
      kwargs = {};
    }
    if (kwargs.value === undefined) {
      self.value = null;
    } else {
      self.value = kwargs.value;
    }
    if (kwargs.instancename === undefined) {
      self.instancename = '';
    } else {
      self.instancename = kwargs.instancename;
    }
    if (kwargs.descr === undefined) {
      self.descr = '';
    } else {
      self.descr = kwargs.descr;
    }
    if (kwargs.required === undefined) {
      self.required = true;
    } else {
      self.required = kwargs.required;
    }
    if (kwargs.delim === undefined) {
      self.delim = '';
    } else {
      self.delim = kwargs.delim;
    }
    self.placeholder = '<' + blocktitle.replace(' ', '-').toUpperCase() + '>';
    self.blocktype = blocktitle.replace(' ','');
    self.displayspan = $('<span>');
    self.defaultdisplay = $('<span class="defaultdisplay"></span>');
    self.defaultdisplay.text(self.placeholder);
    self.displayspan.append(self.defaultdisplay);
    self.showingplaceholder = true;
    self.paramcontainer = $('<div class="paramselect"></div>');
    if (self.instancename != '') {
      self.paramcontainer.append('<h4>' + self.instancename + ' ' + blocktitle + '</h4>');
    } else {
      self.paramcontainer.append('<h4>' + blocktitle + '</h4>');
    }
    if (self.paramval.nargs != '*') {
      buildParamUI(self.paramval,
                   self.paramcontainer,
                   self.displayspan,
                   null,
                   self.paramval.nargs,
                   self.delim);
    } else {
      console.log(self.placeholder);
      self.paramcontainer.append('<button class="adder">');
      self.paramcontainer.find('button').text(' Add ' + self.placeholder);
      self.paramcontainer.find('button').prepend('<i class="fa fa-plus-square" aria-hidden="true"></i>');
      self.paramcontainer.find('button').click(function() {
        buildParamUI(self.paramval,
                     self.paramcontainer,
                     self.displayspan,
                     null,
                     self.paramval.nargs,
                     self.delim,
                     true);
        
      });
    }
  }

  BlockKernel.prototype.paramval = paramval;
  BlockKernel.prototype.hideDefault = function () {
    var self = this;
    self.defaultdisplay.text('');
    self.showingplaceholder = false;
  }
  BlockKernel.prototype.showDefault = function () {
    self.defaultdisplay.detach();
    self.displayspan.text('');
    self.defaultdisplay.text(self.placeholder);
    self.displayspan.append(self.defaultdisplay);
    self.showingplaceholder = true;
  }


  return BlockKernel;
}

function buildParamUI(P, paramcontainer, displayspan, par, nargs, delim, delimfirst) {
  var newdisplayspan = $('<span>');
  displayspan.append(newdisplayspan);
  if (P.paramtype == 'Dropdown') {
    var param = $('<select><option value=""></option></select>');
    param.data('displayspan', newdisplayspan);
    param.data('cascade', []);
    paramcontainer.append(param);
    if (par == null) {
      param.addClass('selected');
    }
    for (var v in P.V) {
      var new_val = $('<option value="' + P.V[v].value + '">' + P.V[v].text + '</option>');
      if (P.V[v].cascade != null) {
        new_val.data('cascade', []);
        for (var c in P.V[v].cascade) {
          var new_param = buildParamUI(P.V[v].cascade[c],
                                       paramcontainer,
                                       displayspan,
                                       new_val,
                                       nargs,
                                       delim,
                                       true);
          new_val.data('cascade').push(new_param);
          param.data('cascade').push(new_param);
        }
      }
      param.append(new_val);
    }
    param.change(function() {
      processCascade(param)
      displayspan.find('span.defaultdisplay').text('');
      new_text = param.val();
      if (!(nargs == '*') || param.val() != '') {
        if (delim != null && (par != null || delimfirst)) {
          new_text = delim + new_text;
        }
        updateTarget(newdisplayspan, new_text);
      } else {
        param.remove();
        newdisplayspan.remove();
      }
    });
  } else if (P.paramtype == 'Multiselect') {
    var param = $('<select multiple><option value=""></option></select>');
    param.data('displayspan', newdisplayspan);
    param.data('cascade', []);
    paramcontainer.append(param);
    if (par == null) {
      param.addClass('selected');
    }
    for (var v in P.V) {
      var new_val = $('<option value="' + P.V[v].value + '">' + P.V[v].text + '</option>');
      var newnewdisplayspan = $('<span>');
      newdisplayspan.append(newnewdisplayspan);
      new_val.data('displayspan', newnewdisplayspan);
      if (P.V[v].cascade != null) {
        new_val.data('cascade', []);
        for (var c in P.V[v].cascade) {
          var new_param = buildParamUI(P.V[v].cascade[c], paramcontainer, newdisplayspan, new_val, delim);
          new_val.data('cascade').push(new_param);
          param.data('cascade').push(new_param);
        }
      }
      param.append(new_val);
    }
    param.change(function() {
      processCascade(param)
      displayspan.find('span.defaultdisplay').text('');
      vals = []
      param.find('option').each(function() {
        if ($(this).data('displayspan') != null) {
          $(this).data('displayspan').text('');
        }
      });
      param.find(':selected').each(function() {
        var new_text = $(this).attr('value');
        if (delim != null) {
          new_text = delim + new_text;
        }
        if ($(this).data('displayspan') != null) {
          $(this).data('displayspan').text(new_text);
        }
      })
    });
   } else if (P.paramtype == 'Boolean') {
    var param = $('<input type="checkbox" id="' + P.V[0].value + '" value="' + P.V[0].value + '">');
    var label = $('<label for="' + P.V[0].value + '">' + P.V[0].text + '</label>');
    param.on('classChanged', function() {
      if ($(this).hasClass('selected')) {
        label.addClass('selected');
      } else {
        label.removeClass('selected');
      }
    });
    paramcontainer.append(param);
    paramcontainer.append(label);
    param.change(function() {
      processCascade(param);
      displayspan.find('span.defaultdisplay').text('');
      if (param.is(':checked')) {
        new_text = P.V[0].value;
        if (delim != null && (par != null || delimfirst)) {
          new_text = delim + new_text;
        }
      } else {
        new_text = '';
      }
      updateTarget(newdisplayspan, new_text);
    });
  } else if (P.paramtype == 'Integer') {
    var param = $('<br><form><b>' + P.V[0].text + ':</b><br><input type="text" id="' + P.V[0].value + '" value=""><input type="submit"></form>');
    paramcontainer.append(param);
    param.submit(function(event) {
      event.preventDefault();
      processCascade(param);
      displayspan.find('span.defaultdisplay').text('');
      var val = param.find('input[type="text"]').val();
      if (val != '') {
        if (P.V[0].after) {
          new_text = val + P.V[0].value;
        } else {
          new_text = P.V[0].value + val;
        }
        if (delim != null && (par != null || delimfirst)) {
          new_text = delim+new_text;
        }
      }
      else {
        new_text = '';
      }
      updateTarget(newdisplayspan, new_text);
    });
  }
  return param;
}

function processCascade(param, multi) {
  var notcascade = [];
  param.find(':not(:selected)').each(function() {
    if ($(this).data('cascade') != null) {
      for (var c in $(this).data('cascade')) {
        notcascade.push($(this).data('cascade')[c]);
      }
    }
  });
  if (notcascade != null) {
    for (var i in notcascade) {
      var c = notcascade[i];
      if (c.attr('type') == 'checkbox') {
        c.removeAttr('checked');
      } else if (c.is('form')) {
        c.find('input[type="text"]').val('');
        c.find('input[type="submit"]').submit();
      } else {
        c.val('');
      }
      c.change();
      c.removeClass('selected');
    }
  }
  var cascade = [];
  param.find(':selected').each(function() {
    if ($(this).data('cascade') != null) {
      for (var c in $(this).data('cascade')) {
        cascade.push($(this).data('cascade')[c]);
      }
    }
  });
  if (cascade != null) {
    for (var i in cascade) {
      var c = cascade[i];
      c.addClass('selected');
    }
  }
}

function createBlockCompositeClass() {
}

/////////////////////////////////////
//
//  PARAMVAL DEFINITIONS
//
/////////////////////////////////////

ParamVal = {}
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
ParamVal.TreeOpt = {
  paramtype: 'Dropdown',
  nargs: '*',
  V: {
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
//  KERNEL BLOCK CLASS DEFINITIONS
//
/////////////////////////////////////

KernelBlockDefs = {
  Corpus: {
    blocktitle: 'Corpus',
    paramval: ParamVal.Corpus
  },
  Grammar: {
    blocktitle: 'Grammar',
    paramval: ParamVal.Grammar
  },
  ModelOpt: {
    blocktitle: 'Model Options',
    paramval: ParamVal.ModelOpt
  },
  Parser: {
    blocktitle: 'Parser',
    paramval: ParamVal.Parser
  },
  TreeOpt: {
    blocktitle: 'Tree Options',
    paramval: ParamVal.TreeOpt
  }
}

/////////////////////////////////////
//
//  POPULATE BLOCKS
//
/////////////////////////////////////

Blocks = {}
for (k in KernelBlockDefs) {
  Blocks[k] = createBlockKernelClass(KernelBlockDefs[k].blocktitle, KernelBlockDefs[k].paramval);
}

// TreeOpts class
function TreeOpts(kwargs) {
  var self = this;
  if (kwargs === undefined) {
    kwargs = {};
  }
  if (kwargs.value === undefined) {
    self.value = null;
  } else {
    self.value = kwargs.value;
  }
  if (kwargs.instancename === undefined) {
    self.instancename = '';
  } else {
    self.instancename = kwargs.instancename;
  }
  if (kwargs.descr === undefined) {
    self.descr = '';
  } else {
    self.descr = kwargs.descr;
  }
  if (kwargs.required === undefined) {
    self.required = true;
  } else {
    self.required = kwargs.required;
  }
  if (kwargs.delim === undefined) {
    self.delim = '';
  } else {
    self.delim = kwargs.delim;
  }
  self.placeholder = '<TREE-OPTIONS>';
  self.blocktype = 'TreeOpts';
  if (self.value == null) {
    self.string = self.placeholder;
  } else {
    self.string = self.value;
  }
  self.displayspan = $('<span>');
  self.displayspan.text(self.string);
  self.param = $('<select multiple>');
  self.paramcontainer = $('<div class="paramselect"></div>');
  if (self.instancename != '') {
    self.paramcontainer.append('<h4>' + self.instancename + ' Tree Options</h4>');
  } else {
    self.paramcontainer.append('<h4>Tree Options</h4>');
  }
  self.paramcontainer.append(self.param);
  self.param.append('<option value="none"></option>');
  for (i in self.values) {
    self.param.append('<option value="' + i + '">' + self.values[i] + '</option>')
  }
  self.param.change(
    function () {
      updateTarget(self.displayspan, getValString(self.param, self.delim));
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

function ParseParams(kwargs) {
  var self = this;
  if (kwargs === undefined) {
    kwargs = {};
  }
  if (kwargs.instancename === undefined) {
    self.instancename = '';
  } else {
    self.instancename = kwargs.instancename;
  }
  self.c = new Blocks.Corpus({instancename:'Training'});
  self.g = new Blocks.Grammar();
  self.to = new Blocks.TreeOpt({delim: '-'});
  self.mo = new Blocks.ModelOpt({delim: '-'});
  self.p = new Blocks.Parser({delim: '_'});
  self.blocktype = 'ParseParams';
  self.displayspan = $('<span>');
  self.displayspan.append(self.c.displayspan);
  self.displayspan.append('-');
  self.displayspan.append(self.g.displayspan);
  self.displayspan.append(self.to.displayspan);
  self.displayspan.append(self.mo.displayspan);
  self.displayspan.append('-');
  self.displayspan.append(self.p.displayspan);
  self.paramcontainer = $('<div class="paramselect"></div>');
  if (self.instancename != '') {
    self.paramcontainer.append('<h4>' + self.instancename + ' Parser Parameters</h4>');
  } else {
    self.paramcontainer.append('<h4>Parser Parameters</h4>');
  }
  self.paramcontainer.append(self.c.paramcontainer)
  self.paramcontainer.append(self.g.paramcontainer)
  self.paramcontainer.append(self.to.paramcontainer)
  self.paramcontainer.append(self.mo.paramcontainer)
  self.paramcontainer.append(self.p.paramcontainer)
}

/////////////////////////////////////
//
// FINAL TARGET CLASSES 
//
/////////////////////////////////////

// %.linetrees class
function Linetrees(kwargs) {
  var self = this;
  if (kwargs === undefined) {
    kwargs = {};
  }
  if (kwargs.instancename === undefined) {
    self.instancename = '';
  } else {
    self.instancename = kwargs.instancename;
  }
  self.t = new Blocks.Corpus({instancename: 'Test'});
  self.to = new Blocks.TreeOpt({delim: '.'});
  self.blocktype = 'Linetrees';
  self.displayspan = $('<span>');
  self.displayspan.append('genmodel/');
  self.displayspan.append(self.t.displayspan);
  self.displayspan.append(self.to.displayspan);
  self.displayspan.append('.linetrees ');
  self.paramcontainer = $('<div class="targetparams"></div>');
  if (self.instancename != '') {
    self.paramcontainer.append('<h4>' + self.instancename + ' Linetrees Target (<span class="code">%.linetrees</span>)</h4>');
  } else {
    self.paramcontainer.append('<h4>Linetrees Target (<span class="code">%.linetrees</span>)</h4>');
  }
  self.paramcontainer.append(self.t.paramcontainer)
  self.paramcontainer.append(self.to.paramcontainer)
  self.trayicon = $('<button class="traybutton selected">');
  self.trayicon.text('LINETREES');
  self.trayicon.click(function () {
    var worksurface = $('div#worksurface');
    var tray = $('div#tray');
    tray.find('button').removeClass('selected');
    $(this).addClass('selected');
    worksurface.find('div.targetparams').detach();
    worksurface.append(self.paramcontainer);
  });
  self.deleter = $('<button class="trashbutton"><i class="fa fa-trash" aria-hidden="true"></i>DELETE</button>');
  self.paramcontainer.append(self.deleter)
  self.deleter.click(function() {
    self.displayspan.remove();
    self.trayicon.remove();
    self.paramcontainer.remove();
    self.deleter.remove();
  });
}
// %.parse.linetrees class
function ParseLinetrees(kwargs) {
  var self = this;
  if (kwargs === undefined) {
    kwargs = {};
  }
  if (kwargs.instancename === undefined) {
    self.instancename = '';
  } else {
    self.instancename = kwargs.instancename;
  }
  self.t = new Blocks.Corpus({instancename: 'Test'});
  self.pp = new ParseParams();
  self.blocktype = 'ParseLinetrees';
  self.displayspan = $('<span>');
  self.displayspan.append('genmodel/');
  self.displayspan.append(self.t.displayspan);
  self.displayspan.append('.');
  self.displayspan.append(self.pp.displayspan);
  self.displayspan.append('_parsed.linetrees ');
  self.paramcontainer = $('<div class="targetparams"></div>');
  if (self.instancename != '') {
    self.paramcontainer.append('<h4>' + self.instancename + ' Parse Target (<span class="code">%.parsed.linetrees</span>)</h4>');
  } else {
    self.paramcontainer.append('<h4>Parse Target (<span class="code">%parsed.linetrees</span>)</h4>');
  }
  self.paramcontainer.append(self.t.paramcontainer)
  self.paramcontainer.append(self.pp.paramcontainer)
  self.trayicon = $('<button class="traybutton selected">');
  self.trayicon.text('PARSE');
  self.trayicon.click(function () {
    var worksurface = $('div#worksurface');
    var tray = $('div#tray');
    tray.find('button').removeClass('selected');
    $(this).addClass('selected');
    worksurface.find('div.targetparams').detach();
    worksurface.append(self.paramcontainer);
  });
  self.deleter = $('<button class="trashbutton"><i class="fa fa-trash" aria-hidden="true"></i>DELETE</button>');
  self.paramcontainer.append(self.deleter)
  self.deleter.click(function() {
    self.displayspan.remove();
    self.trayicon.remove();
    self.paramcontainer.remove();
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
  worksurface.append(parse.paramcontainer);
});

$('button#addLinetrees').click(function() {
  worksurface.find('div.targetparams').detach();
  var linetrees = new Linetrees();
  target.append(linetrees.displayspan);
  tray.find('button').removeClass('selected');
  tray.append(linetrees.trayicon);
  worksurface.append(linetrees.paramcontainer);
});

$('button#copier').click(function() {copyToClipboard(target);})
