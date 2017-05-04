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

function createBlockCompositeClass(blocktitle, blockseq) {
  function BlockComposite(kwargs) {
    var self = this;
    if (kwargs === undefined) {
      kwargs = {};
    }
    if (kwargs.instancename === undefined) {
      self.instancename = '';
    } else {
      self.instancename = kwargs.instancename;
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
    self.displayspan.append(self.defaultdisplay);
    self.showingplaceholder = true;
    self.paramcontainer = $('<div class="paramselect"></div>');
    if (self.instancename != '') {
      self.paramcontainer.append('<h4>' + self.instancename + ' ' + blocktitle + '</h4>');
    } else {
      self.paramcontainer.append('<h4>' + blocktitle + '</h4>');
    }
    
    for (var i in blockseq) {
      var b = blockseq[i];
      if (b.blocktype == 'String') {
        self.displayspan.append(b.value);
      } else {
        new_block = new Blocks[b.blocktype](b.kwargs);
        if (i > 0) {
          self.displayspan.append(self.delim);
        }
        self.displayspan.append(new_block.displayspan);
        self.paramcontainer.append(new_block.paramcontainer);
      }
    }
  }

  return BlockComposite;
}

function createBlockTargetClass(blocktitle, blocksuffix, blockseq) {
  function BlockTarget(kwargs) {
    var self = this;
    if (kwargs === undefined) {
      kwargs = {};
    }
    if (kwargs.instancename === undefined) {
      self.instancename = '';
    } else {
      self.instancename = kwargs.instancename;
    }
    self.placeholder = '<' + blocktitle.replace(' ', '-').toUpperCase() + '>';
    self.blocktype = blocktitle.replace(' ','');
    self.displayspan = $('<span>');
    self.defaultdisplay = $('<span class="defaultdisplay"></span>');
    self.displayspan.append('genmodel/');
    self.displayspan.append(self.defaultdisplay);
    self.showingplaceholder = true;
    self.paramcontainer = $('<div class="targetparams"></div>');
    if (self.instancename != '') {
      self.paramcontainer.append('<h4>' + self.instancename + ' ' + blocktitle + ' (<span class="code">&#37;' + blocksuffix + '</span>)</h4>');
    } else {
      self.paramcontainer.append('<h4>' + blocktitle + ' (<span class="code">&#37;' + blocksuffix + '</span>)</h4>');
    }
    
    self.trayicon = $('<button class="traybutton selected">');
    self.trayicon.text(blocktitle.replace(' ', '-').toUpperCase());
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
      $('div#tray').children().last().click();
    });

    for (var i in blockseq) {
      var b = blockseq[i];
      if (b.blocktype == 'String') {
        self.displayspan.append(b.value);
      } else {
        new_block = new Blocks[b.blocktype](b.kwargs);
        self.displayspan.append(new_block.displayspan);
        self.paramcontainer.append(new_block.paramcontainer);
      }
    }
    self.displayspan.append(blocksuffix + ' ');
  }

  return BlockTarget;
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

for (k in CompositeBlockDefs) {
  Blocks[k] = createBlockCompositeClass(CompositeBlockDefs[k].blocktitle, CompositeBlockDefs[k].blockseq);
}

for (k in TargetBlockDefs) {
  Blocks[k] = createBlockTargetClass(TargetBlockDefs[k].blocktitle, TargetBlockDefs[k].targetsuffix, TargetBlockDefs[k].blockseq);
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

function buildParamUI(P, paramcontainer, displayspan, par, nargs, delim, delimfirst) {
  if (delim == null) {
    delim = '';
  }
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
      processCascade(param);
      displayspan.find('span.defaultdisplay').text('');
      new_text = param.val();
      if (!(nargs == '*') || param.val() != '') {
        if (delim != '' && (par != null || delimfirst)) {
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
        if (delim != '') {
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
    if (par == null) {
      param.addClass('selected');
      label.addClass('selected');
    }
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
        if (delim != '' && (par != null || delimfirst)) {
          new_text = delim + new_text;
        }
      } else {
        new_text = '';
      }
      updateTarget(newdisplayspan, new_text);
    });
  } else if (P.paramtype == 'Integer' || P.paramtype == 'Text') {
    var param = $('<form><b>' + P.V[0].text + ': </b><input type="text" id="' + P.V[0].value + '" value=""><input type="submit"></form>');
    if (par == null) {
      param.addClass('selected');
    }
    paramcontainer.append(param);
    param.submit(function(event) {
      event.preventDefault();
      processCascade(param);
      displayspan.find('span.defaultdisplay').text('');
      var val = param.find('input[type="text"]').val();
      if (!(nargs == '*') || val != '') {
        if (val != '') {
          if (P.V[0].after) {
            new_text = val + delim + P.V[0].value;
          } else {
            new_text = P.V[0].value + delim + val;
          }
          if (delim != null && (par != null || delimfirst)) {
            new_text = delim+new_text;
          }
        }
        else {
          new_text = '';
        }
        updateTarget(newdisplayspan, new_text);
      } else {
        param.remove();
        newdisplayspan.remove();
      }
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

function buildAddButtons(TargetBlockDefs) {
  for (var k in TargetBlockDefs) {
    var buttonContainer = $('div#addButtonContainer');
    var worksurface = $('div#worksurface'); 
    var tray = $('div#tray');
    var target = $('div#target');
    var new_button = $('<button id="add' + k + '" class="traybutton"><i class="fa fa-plus-square" aria-hidden="true"></i> ' + TargetBlockDefs[k].blocktitle + '</button>');
    new_button.data('blocktype', k);
    buttonContainer.append(new_button);
    new_button.click(function() {
      var k = $(this).data('blocktype');
      var new_targ = new Blocks[k];
      worksurface.find('div.targetparams').detach();
      target.append(new_targ.displayspan);
      tray.find('button').removeClass('selected');
      tray.append(new_targ.trayicon);
      worksurface.append(new_targ.paramcontainer);
    });
  }
}

function clearWorkspace() {
  var tray = $('div#tray');
  var worksurface = $('div#worksurface');
  tray.find('button').each(function() {
    $(this).click();
    worksurface.find('div.targetparams').find('button.trashbutton').click();
  });
}

/////////////////////////////////////
//
// PREPARE DOM 
//
/////////////////////////////////////

$('div#target').append('make ');

buildAddButtons(TargetBlockDefs);

$('button#copier').click(function() {copyToClipboard(target);})
$('button#clear').click(clearWorkspace)
