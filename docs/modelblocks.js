/////////////////////////////////////
//
//  LIBRARY FUNCTIONS 
//
/////////////////////////////////////

(function( func ) {
    $.fn.addClass = function() {
        func.apply( this, arguments );
        this.trigger('classChanged');
        return this;
    }
})($.fn.addClass);

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

function createBlockKernelClass(blocktype,
                                blocktitle,
                                paramtype,
                                paramval,
                                nargs,
                                descr,
                                instance_of) {

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
    if (kwargs.innerdelim === undefined) {
      self.innerdelim = '';
    } else {
      self.innerdelim = kwargs.innerdelim;
    }
    if (kwargs.outerdelim === undefined) {
      self.outerdelim = '';
    } else {
      self.outerdelim = kwargs.outerdelim;
    }

    self.blocktype = blocktype;
    self.blocktitle = blocktitle;
    self.paramtype = paramtype;
    self.paramval = paramval;
    self.nargs = nargs;
    self.descr = descr;
    
    self.placeholder = '<' + self.blocktitle.replace(/ /g, '-').toUpperCase() + '>';
    self.displayspan = $('<span>');
    self.defaultdisplay = $('<span class="defaultdisplay"></span>');
    self.defaultdisplay.text(self.outerdelim + self.placeholder);
    self.displayspan.append(self.defaultdisplay);
    if (self.nargs == '*' || self.nargs == '?') {
      self.hideDefault();
    }
    self.showingplaceholder = true;
    
    self.paramcontainer = $('<div class="paramselect"></div>');
    if (self.instancename != '') {
      self.paramcontainer.append('<h4>' + self.instancename + ' ' + self.blocktitle + '</h4>');
    } else {
      self.paramcontainer.append('<h4>' + self.blocktitle + '</h4>');
    }
    if (self.nargs != '*') {
      buildParamUI(self.paramtype,
                   self.paramval,
                   self.paramcontainer,
                   self.displayspan,
                   null,
                   self.nargs,
                   self.outerdelim,
                   self.innerdelim);
    } else {
      self.paramcontainer.append('<button class="adder">');
      self.paramcontainer.find('button').text(' Add ' + self.placeholder);
      self.paramcontainer.find('button').prepend('<i class="fa fa-plus-square" aria-hidden="true"></i>');
      self.paramcontainer.find('button').click(function() {
        buildParamUI(self.paramtype,
                     self.paramval,
                     self.paramcontainer,
                     self.displayspan,
                     null,
                     self.nargs,
                     self.outerdelim,
                     self.innerdelim);
        
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

function createBlockCompositeClass(blocktype,
                                   blocktitle,
                                   blockseq,
                                   descr) {

  function BlockComposite(kwargs) {
    var self = this;
    if (kwargs === undefined) {
      var kwargs = {};
    }
    if (kwargs.instancename === undefined) {
      self.instancename = '';
    } else {
      self.instancename = kwargs.instancename;
    }
    if (kwargs.outerdelim == null) {
       self.outerdelim = '';
    } else {
       self.outerdelim = kwargs.outerdelim;
    }
    if (kwargs.innerdelim == null) {
      kwargs.innerdelim = '';
    } else {
      self.innerdelim = kwargs.innerdelim;
    }
    self.placeholder = '<' + blocktitle.replace(/ /g, '-').toUpperCase() + '>';
    self.blocktype = blocktype;
    self.displayspan = $('<span>');
    self.displayspan.text(self.outerdelim);
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
      } else if (b.blocktype == 'Any') {
        // TODO
      } else if (b.blocktype == 'Either') {
        // TODO
      } else {
        if (b.kwargs == null) {
          var bkwargs = {}
        } else {
          var bkwargs = $.extend({}, b.kwargs);
        }
        if (i > 0) {
          bkwargs.outerdelim = self.innerdelim;
        }
        new_block = new Blocks[b.blocktype](bkwargs);
        self.displayspan.append(new_block.displayspan);
        self.paramcontainer.append(new_block.paramcontainer);
      }
    }
  }

  return BlockComposite;
}

function createBlockTargetClass(blocktype,
                                blocktitle,
                                blocksuffix,
                                blockseq,
                                descr) {

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
    self.placeholder = '<' + blocktitle.replace(/ /g, '-').toUpperCase() + '>';
    self.blocktype = blocktype;
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
    self.trayicon.text(blocktitle.replace(/ /g, '-').toUpperCase());
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
      } else if (b.blocktype == 'Any') {
        // TODO
      } else if (b.blocktype == 'Either') {
        // TODO
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

// Add options to Kernel blocks
function addVal(p, k) {
  if (k != null) {
    console.log(k);
    if (KernelBlockDefs[k].paramval != null) {
      if (KernelBlockDefs[k].paramval[p] == null) {
        KernelBlockDefs[k].paramval[p] = ParamVal[p];
      }
    } else {
      KernelBlockDefs[k].paramval = {};
      KernelBlockDefs[k].paramval[p] = ParamVal[p];
    }
    console.log(KernelBlockDefs[k]);
  }
}

function trickleUp(p, kernels) {
  for (var k in kernels) {
    console.log(kernels);
    console.log(kernels[k]);
    addVal(p, kernels[k]);
    if (KernelBlockDefs[kernels[k]].instance_of != null) {
      var superblocks = [].concat(KernelBlockDefs[kernels[k]].instance_of);
      trickleUp(p, superblocks);
    }
  }
}

for (var p in ParamVal) {
  var kernels = [].concat(ParamVal[p].kernel);
  trickleUp(p, kernels);
}

// Create classes for each block type
Blocks = {}
for (var k in KernelBlockDefs) {
  Blocks[k] = createBlockKernelClass(k,
                                     KernelBlockDefs[k].blocktitle,
                                     KernelBlockDefs[k].paramtype,
                                     KernelBlockDefs[k].paramval,
                                     KernelBlockDefs[k].nargs,
                                     KernelBlockDefs[k].descr,
                                     KernelBlockDefs[k].instance_of);
}

for (var k in CompositeBlockDefs) {
  Blocks[k] = createBlockCompositeClass(k,
                                        CompositeBlockDefs[k].blocktitle,
                                        CompositeBlockDefs[k].blockseq,
                                        CompositeBlockDefs[k].descr);
}

for (var k in TargetBlockDefs) {
  Blocks[k] = createBlockTargetClass(k,
                                     TargetBlockDefs[k].blocktitle,
                                     TargetBlockDefs[k].targetsuffix,
                                     TargetBlockDefs[k].blockseq,
                                     TargetBlockDefs[k].descr);
}

UtilityBlockDefs = {
  Either: {
    blocktitle: 'Either'
  }
}

/////////////////////////////////////
//
// UTILITY FUNCTIONS 
//
/////////////////////////////////////

function blocksorter(a, b) {
  a.blocktitle.localecompare(b.blocktitle);
}

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

function buildParamUI(paramtype,
                      paramval,
                      paramcontainer,
                      displayspan,
                      ancestor,
                      nargs,
                      outerdelim,
                      innerdelim) {
  var newdisplayspan = $('<span>');
  displayspan.append(newdisplayspan);
  if (paramtype == 'Dropdown') {
    var param = $('<select><option value=""></option></select>');
    param.data('displayspan', newdisplayspan);
    param.data('cascade', []);
    paramcontainer.append(param);
    if (ancestor == null) {
      param.addClass('selected');
    }
    for (var v in paramval) {
      var new_val = $('<option value="' + paramval[v].value + '">' + paramval[v].text + '</option>');
      if (paramval[v].cascade != null) {
        var cascade = [].concat(paramval[v].cascade);
        new_val.data('cascade', []);
        for (var i in cascade) {
          console.log(cascade);
          var c = KernelBlockDefs[cascade[i]];
          console.log(c)
          if (c.nodelimiter) {
            var od = innerdelim;
            var id = '';
          } else {
            var od = outerdelim;
            var id = innerdelim;
          }
          var new_param = buildParamUI(c.paramtype,
                                       c.paramval,
                                       paramcontainer,
                                       displayspan,
                                       new_val,
                                       nargs,
                                       od,
                                       id);
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
        new_text = outerdelim + new_text;
        updateTarget(newdisplayspan, new_text);
      } else {
        param.remove();
        newdisplayspan.remove();
      }
    });
  } else if (paramtype == 'Multiselect') {
    var param = $('<select multiple><option value=""></option></select>');
    param.data('displayspan', newdisplayspan);
    param.data('cascade', []);
    paramcontainer.append(param);
    if (ancestor == null) {
      param.addClass('selected');
    }
    for (var v in paramval) {
      var new_val = $('<option value="' + paramval[v].value + '">' + paramval[v].text + '</option>');
      var newnewdisplayspan = $('<span>');
      newdisplayspan.append(newnewdisplayspan);
      new_val.data('displayspan', newnewdisplayspan);
      if (paramval[v].cascade != null) {
        var cascade = [].concat(paramval[v].cascade);
        new_val.data('cascade', []);
        for (var i in cascade) {
          if (c.nodelimiter) {
            var od = innerdelim;
            var id = '';
          } else {
            var od = outerdelim;
            var id = innerdelim;
          }
          var new_param = buildParamUI(c.paramtype,
                                       c.paramval,
                                       paramcontainer,
                                       newdisplayspan,
                                       new_val,
                                       nargs,
                                       od,
                                       id);
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
        new_text = outerdelim + new_text;
        if ($(this).data('displayspan') != null) {
          $(this).data('displayspan').text(new_text);
        }
      })
    });
  } else if (paramtype == 'Boolean') {
    var p = paramval[function() {for (var i in paramval) return i}()];
    id = '';
    if (ancestor != null) {
      id += ancestor.attr('value');
    }
    id += p.value;
    var param = $('<input type="checkbox" id="' + id + '" value="' + p.value + '">');
    var label = $('<label for="' + id + '">' + p.text + '</label>');
    if (ancestor == null) {
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
        var new_text = outerdelim + p.value;
      } else {
        var new_text = '';
      }
      updateTarget(newdisplayspan, new_text);
    });
  } else if (paramtype == 'Integer' || paramtype == 'Text') {
    var p = paramval[function() {for (var i in paramval) return i}()];
    var param = $('<form><b>' + p.text + ': </b><input type="text" id="' + p.value + '" value=""><input type="submit"></form>');
    if (ancestor == null) {
      param.addClass('selected');
    }
    paramcontainer.append(param);
    param.submit(function(event) {
      event.preventDefault();
      processCascade(param);
      displayspan.find('span.defaultdisplay').text('');
      var val = param.find('input[type="text"]').val();
      if (!(nargs == '*' || nargs == '?') || val != '') {
        if (val != '') {
          if (p.after) {
            new_text = outerdelim + val + innerdelim + p.value;
          } else {
            new_text = outerdelim + p.value + innerdelim + val;
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
  var optionContainer = $('div#optionContainer');
  for (var k in TargetBlockDefs) {
    var buttonContainer = $('div#addButtonContainer');
    var worksurface = $('div#worksurface'); 
    var tray = $('div#tray');
    var target = $('div#target');
    //var new_button = $('<button id="add' + k + '" class="traybutton"><i class="fa fa-plus-square" aria-hidden="true"></i> ' + TargetBlockDefs[k].blocktitle + '</button>');
    //new_button.data('blocktype', k);
    //buttonContainer.append(new_button);
    //new_button.click(function() {
    //  var t = $(this).data('blocktype');
    //  var new_targ = new Blocks[t];
    //  worksurface.find('div.targetparams').detach();
    //  target.append(new_targ.displayspan);
    //  tray.find('button').removeClass('selected');
    //  tray.append(new_targ.trayicon);
    //  worksurface.append(new_targ.paramcontainer);
    //});
    
    var new_opt = $('<div class="targetoption"</div>')
    new_opt.text(TargetBlockDefs[k].blocktitle);
    new_opt.data('blocktype', k);
    optionContainer.append(new_opt);
    new_opt.click(function() {
      var t = $(this).data('blocktype');
      var new_targ = new Blocks[t];
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
