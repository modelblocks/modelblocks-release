
###############################################################################
##                                                                           ##
## This file is part of ModelBlocks. Copyright 2009, ModelBlocks developers. ##
##                                                                           ##
##    ModelBlocks is free software: you can redistribute it and/or modify    ##
##    it under the terms of the GNU General Public License as published by   ##
##    the Free Software Foundation, either version 3 of the License, or      ##
##    (at your option) any later version.                                    ##
##                                                                           ##
##    ModelBlocks is distributed in the hope that it will be useful,         ##
##    but WITHOUT ANY WARRANTY; without even the implied warranty of         ##
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          ##
##    GNU General Public License for more details.                           ##
##                                                                           ##
##    You should have received a copy of the GNU General Public License      ##
##    along with ModelBlocks.  If not, see <http://www.gnu.org/licenses/>.   ##
##                                                                           ##
###############################################################################

use Getopt::Std;

getopts("d");

$DEBUG = 0;
if ($opt_d) {
    $DEBUG = 1;
}

sub debug {
    if ($DEBUG) {
        $msg = $_[1];
        print stderr $_[0] , " " , $msg, "\n";
    }
}

## for each tree...
while ( <> ) {
    s/[\[\]]/!/g;   # change bracks to exclams
    s/\^/!carat!/g; # change carats to exclams
    s/</!langle!/g; # change langle to exclams
    s/>/!rangle!/g; # change rangle to exclams
    s/=/!equals!/g; # change equals to exclams

    ## FIRST PASS: FLATTEN LEFT-BRANCHING TREE...
    ## mark top-level constituent as current...
    s/^ *\((.*)\) *$/\^\1\^/;
    ## mark all other constituents as internal...
    s/\(/\[/g;
    s/\)/\]/g;
    ## for each constituent, top-down...
    while ( $_ =~ /\^/ ) {
      ## mark all children of current...
      for ( $i=index($_,'^'),$d=0; $i<rindex($_,'^'); $i++ ) {
        if ( substr($_,$i,1)eq'[' ) { if($d==0){substr($_,$i,1)='<';} $d++; }
        if ( substr($_,$i,1)eq']' ) { $d--; if($d==0){substr($_,$i,1)='>';} }
      }
      #################### ADD SED RULES HERE: apply rules to angles (children) within braces (consituent)...
      debug(++$step, " flatt  $_");
      ## (recursive step) remove label-less complex left child of right parent and add slash to right child ...
      s/(?<=[\)>\]] )\(([^ \\]*) \^(?![^ ]*-l)([^ ]*) <([^ ]*) ([^>]*)> <([^>]*)>(.*)\^/\(\1 <\3 \4> <\2\\\3 \[\5\]>\6/;
      ## (base step) add slash to right child of simple left child of right parent...
      s/(?<=[\)>\]] )\^([^ \\]*) <([^ ]*) ([^\[>]*)> <([^ ]*\\[^>]*)>(.*)\^/\^\1 <\2 \3> <\1\\\2 \[\4\]>\5\^/;
#      s/(?<=[\)>\]] )\(([^ ]*) \^(?![^ ]*-l)([^ ]*) (<[^>]*>) (<[^>]*>)(.*)\^/\(\1 \3 <\1\\\2 \4>\5/;
      ####################
      ## delete all nested angle-brackets...
      while ( s/<([^<>]*)>([^<>]*)>/\[\1\]\2>/ ) { }
      ## mark current as external...
      s/\^(.*?)\^/\(\1\)/;
      ## mark first unexpanded child as current...
      s/<(.*?)>/\^\1\^/;
    }


#    ## FIRST PASS: FLATTEN LEFT-BRANCHING TREE...
#    ## for each constituent, bottom-up...
#    while ( $_ =~ /\([^\(\)]*\)/ ) {
#        ## convert outer parens to carets...
#        $_ =~ s/\(([^\(\)]*)\)/\^\1\^/;
#        #################### ADD SED RULES HERE: apply rules to angles (children) within braces (consituent)...
#        debug($step++, " flat  $_");
#        s/(?<= )\^([^ ]*) <(?![^ ]*-l)([^ ]*) ([^>]*\] [^>]*)> <([^>]*)>\^/<\1 \3 \[\1\\\2 \[\4\]\]>/;
#        s/\^([^ ]*) <([^ ]*) ([^>]*)> <([^>]*)>\^/\^\1 <\2 \3> <\1\\\2 \[\4\]>\^/;
#        #s/\^([^ ]*) <(?![^ ]*-l)([^ ]*) ([^>]*)> <([^>]*)>\^/\^\1 <\2 \3> <\1\\\2 \[\4\]>\^/;
#        #s/(?<=[^ \)>] ) *\^([^ ]*) <([^ l]*) ([^>]*\] [^>]*)> <([^>]*)>\^/<\1 \3 \[\1\\\2 \[\4\]\]>/;
#        #s/(?<=[^ \)>] ) *\^([^ ]*) <([^ l]*) ([^>]*)> <([^>]*)>\^/\^\1 <\2 \3> <\1\\\2 \[\4\]>\^/;
#        ####################
#        ## convert inner angles (if any) to bracks...
#        while ( s/(\^[^\^]*)<([^<>]*)>/\1\[\2\]/ ){}
#        ## convert outer carets to angles...
#        $_ =~ s/\^(.*)\^/<\1>/;
#    }
#    ## finish up...
#    $_ =~ s/</[/;
#    $_ =~ s/>/]/;
#    ## translate to parens again...
#    $_ =~ s/\[/\(/g;
#    $_ =~ s/\]/\)/g;

    ## SECOND PASS: BUILD RIGHT-BRANCHING TREE...
    ## for each constituent, bottom-up...
    while ( $_ =~ /\([^\(\)]*\)/ ) {
        ## convert outer parens to carets...
        $_ =~ s/\(([^\(\)]*)\)/\^\1\^/;
        #################### ADD SED RULES HERE: apply rules to angles (children) within braces (consituent)...
        debug($step++, " comb  $_");
#        s/\^(.*<.*) <([^ >]*)\\([^ >]*) ([^>]*)> (<([^ >]*)\\\2 [^>]*>)\^/\(\1 \^\6\\\3 \4 \5\^\)/;
        s/\^([^ ]*)(.*<.*) <([^ >]*)\\([^ >]*) ([^>]*)> (<([^ >]*) [^>]*>)\^/\(\1\2 \^\1\\\4 \5 \6\^\)/;
        s/\^([^ ]*) (<[^>]*> <[^ \\]* [^>]*>) (<\1\\([^ >]*) [^>]*>)\^/\(\1 \^\4 \2\^ \3\)/;
        #s/\^(.*<.*) <([^ >]*) ([^>]*)> (<([^ >]*)\\\2 [^>]*>)\^/\(\1 \^\5 <\2 \3> \4\^\)/;
        ####################
        ## convert inner angles (if any) to bracks...
        while ( s/(\^[^\^]*)<([^<>]*)>/\1\[\2\]/ ){}
        ## convert outer carets to angles...
        $_ =~ s/\^(.*)\^/<\1>/;
    }
    ## finish up...
    $_ =~ s/</[/;
    $_ =~ s/>/]/;
    ## translate to parens again...
    $_ =~ s/\[/\(/g;
    $_ =~ s/\]/\)/g;
    
    ## THIRD PASS: KILL UNARY BACKSLASHES...
    ## for each constituent, bottom-up...
    while ( $_ =~ /\([^\(\)]*\)/ ) {
        ## convert outer parens to carets...
        $_ =~ s/\(([^\(\)]*)\)/\^\1\^/;
        #################### ADD SED RULES HERE: apply rules to angles (children) within braces (consituent)...
        debug($step++, " noun  $_");
        s/\^[^ >]*\\[^ >]* <([^>]*)>\^/\^\1\^/;
        ####################
        ## convert inner angles (if any) to bracks...
        while ( s/(\^[^\^]*)<([^<>]*)>/\1\[\2\]/ ){}
        ## convert outer carets to angles...
        $_ =~ s/\^(.*)\^/<\1>/;
    }
    ## finish up...
    $_ =~ s/</[/;
    $_ =~ s/>/]/;
    ## translate to parens again...
    $_ =~ s/\[/\(/g;
    $_ =~ s/\]/\)/g;

    ## FOURTH PASS: CLEAN UP LEFT-SIDE OPERATION LABELS
    s/-l.\\/\\/g;
    s/(\\[^ ]*)-l./\1/g;

    ## output...
    print $_;
}
