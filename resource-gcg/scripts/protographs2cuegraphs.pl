
# remove commas within tokens (e.g. "10,000")
while ( s/:(\S*),/:\1-comma-/ ) { }

########## INHERITANCE DEPENDENCIES

# node identification/equivalence...
#s/(\S+),=,(\S+)/\1,e,\2/g;

# cin deps...
s/(\S+)r,\d+,(\S+)c(?=\s)/\2,c,\1/g;
#s/(\S+)r,\d+,(\S+)c(?=\s)/\2r,c,\1r \2s,c,\1s/g;

# re-sort...
#s/,c,/r,00,/g;
$_ = join(' ',sort(split())) . "\n";
#s/r,00,/,c,/g;

# localize root pointer...
s/00,1,([^ ]*) /00,1,\1s /;

# use eventuality vertices (##r or ##e) instead of lexical items (##)...
s/(\d+)(,0,(?:[^DNU]|D-aN|N-b\{N-aD\})\S+)/\1r\2/g;
s/(\d+)(,0,\S+.*) \1r,5,(\S+)/\1\2 \1e,6,\3/g;
s/(\d+)(,0,\S+.*) \1r,4,(\S+)/\1\2 \1e,5,\3/g;
s/(\d+)(,0,\S+.*) \1r,3,(\S+)/\1\2 \1e,4,\3/g;
s/(\d+)(,0,\S+.*) \1r,2,(\S+)/\1\2 \1e,3,\3/g;
s/(\d+)(,0,\S+.*) \1r,1,(\S+)/\1\2 \1e,2,\3/g;
s/(\d+)(,0,\S+)/\1e\2 \1e,1,\1r/g;
#while ( s/((\S+)e,0,N.* \2)r(,[1-9])/\1e\3/ ) { }

########## REMOVE PUNCT

# remove punctuation...
s/ (\S+),0,-[ A-Z]\S+( .*\1,\S+)?//g;

########## CAPS NORMALIZATION

# lowercase lexical item...
s/(\S+:)(\S+)/\1\L\2/g;

########## MORPH NORMALIZATION (ASSUMES "MORPHED" LINETREES)

# use stem form from morphed.linetrees...
while( s/,0,.(\S*?)-x.%(\S*)\|([^% ]*)%([^-: ]*)([^: ]*):(\S*)\L\2 /,0,\3\1\5:\6\4 /g ) { }
#### s/,0,[VBLG](\S+)-xX\S*\*(\S*)\*(\S*)(:\S*)\L\3 /,0,B\1\4\2 /g;
#### s/,0,N(\S*)-xX\S*\*(\S*)\*(\S*)(:\S*)\L\3 /,0,N\1\4\2 /g;

# remove -x...
s/-x:/:/g;

########## ARGUMENT PROPAGATION

$NULLRSG = '(?<=\{A-aN\}:)be|(?<=\{B-aN\}:)do|(?<=\{L-aN\}:)have|(?<=\{B-aN\}:)to';  # semantically empty -- remove and redirect impinging deps to second arg
$MARKERS = 'O-bN:of|D-aN:\'s|C-bV:that';                                             # semantically empty -- remove and redirect impinging deps to first arg
$ORDDETS = 'N-b\{N-aD\}:\S+';                                                        # redirect impinging deps to first arg
$RAISING = 'able|about|free|likely|necessary|unable' # adjs  # move subject to be subject of second arg
         . '|end|go|turn' # ditrans raisg verbs (end up, go on to, turn out to)
         . '|can|could|may|might|must|shall|should|will|would' # auxs / base-comp verbs
         . '|appear|become|begin|come|consider|continue|fall|feel|finish|get|go|grow|keep|look|prove|remain|sound|start|stay|stop|try|turn' # adj-comp verbs
         . 	'|appear|begin|come|continue|fail|get|go|grow|have|manage|need|prove|seem|start|tend|threaten|use'; # to-comp verbs
$SBJCTRL = 'enough|ready|willing'  # adjs  # coindex subject as subject of second arg
         . '|spend' # ditrans raisg verbs
         . '|avoid|enjoy' # adj-comp verbs
         . '|help' # base-comp verbs
         . '|aim|attempt|choose|claim|decide|help|hope|intend|learn|like|move|plan|prefer|pretend|promise|refuse|seek|try|volunteer|vote|wait|want|wish|work'; # to-comp verbs

# propagate arguments of empty raising, ordinary raising, and subject control verbs...
while ( # ordinary, to subject; or simple passive, to direct object; or predicative, to rin dependency...
        s/((\S+),0,\S+-aN-b\{.-aN\}\S*:(?:$NULLRSG|$RAISING|$SBJCTRL).* \2,1,(\d\S+).* \2,2,(\d+)s.* \4r,[=1-9],)-/\1\3/ ||
        s/((\S+),0,\S+-aN-b\{.-aN\}\S*:(?:$NULLRSG|$RAISING|$SBJCTRL).* \2,1,(\d\S+).* \2,2,(\d+)s.* (\d+),c,\4.* \5r,[=1-9],)-/\1\3/ ||
        # inverted, to subject; or simple passive, to direct object; or predicative, to rin dependency...
        s/((\d+)r,[=1-9],)- (.*(\S+),0,\S+-aN-b\{.-aN\}\S*:(?:$NULLRSG|$RAISING|$SBJCTRL).* \4,1,(\d\S+).* \4,2,\2s)/\1\5 \3/ ||
        s/((\d+),c,(\d+).* \2r,[=1-9],)- (.*(\S+),0,\S+-aN-b\{.-aN\}\S*:(?:$NULLRSG|$RAISING|$SBJCTRL).* \5,1,(\d\S+).* \5,2,\3s)/\1\6 \4/ ||
        # complex passive, to small clause subject...
        s/((\S+),0,\S+-aN-b\{.-aN\}\S*:(?:$NULLRSG|$RAISING|$SBJCTRL).* \2,1,(\d\S+).* \2,2,(\d+)s.* \4r,0,\S+-b[ABIG]:.* \4r,1,\? \4r,2,(\d+)s.* \5r,[=1],)-/\1\3/ ||
        s/((\S+),0,\S+-aN-b\{.-aN\}\S*:(?:$NULLRSG|$RAISING|$SBJCTRL).* \2,1,(\d\S+).* \2,2,(\d+)s.* (\d+),c,\4.* \5r,0,\S+-b[ABIG]:.* \5r,1,\? \5r,2,(\d+)s.* \6r,[=1],)-/\1\3/ ||
        # complex passive, to modifier object...
        s/((\S+),0,\S+-aN-b\{.-aN\}\S*:(?:$NULLRSG|$RAISING|$SBJCTRL).* \2,1,(\d\S+).* \2,2,(\d+)s.* \4r,0,\S+-aN:.* \4r,1,\? (\S+),0,R-aN-.* \5,1,\4r \5,2,)-/\1\3/ ||
        s/((\S+),0,\S+-aN-b\{.-aN\}\S*:(?:$NULLRSG|$RAISING|$SBJCTRL).* \2,1,(\d\S+).* \2,2,(\d+)s.* (\d+),c,\4.* \5r,0,\S+-aN:.* \5r,1,\? (\S+),0,R-aN-.* \6,1,\5r \6,2,)-/\1\3/
      ) { }

# passive (before morph norm)

# remove first arguments of ordinary raising verbs...
s/((\S+),0,\S+-aN-b\{\S+:(?:$RAISING)) \2,1,\d\S+/\1/g;

# redirect impinging dependencies to first argument of non-possessive determiners or empty markers...
while ( s/,(\d+)([rs])(.* \1[er],0,(?:$ORDDETS|$MARKERS) .*\1[er],1,(\d+)s)/,\4\2\3/ ||
        s/,(\d+)([rs])(.* (\d+)[er],0,(?:$ORDDETS|$MARKERS) .*(\d+),c,\4 .*\5[er],1,(\d+)s)/,\5\2\3/ ||
        s/ (\d+)([rs]),=(.* \1[er],0,(?:$ORDDETS|$MARKERS) .*\1[er],1,(\d+)s)/ \4\2,=\3/ ||
        s/ (\d+)([rs]),=(.* (\d+)[er],0,(?:$ORDDETS|$MARKERS) .*(\d+),c,\4 .*\5[er],1,(\d+)s)/ \5\2,=\3/ ||
        s/( (\d+)[er],0,(?:$ORDDETS|$MARKERS) .*\2[er],1,(\d+)s .*),\2([rs])/\1,\3\4/ ||
        s/( (\d+)[er],0,(?:$ORDDETS|$MARKERS) .*(\d+),c,\2 .*\3[er],1,(\d+)s .*),\2([rs])/\1,\4\5/ ||
        s/( (\d+)[er],0,(?:$ORDDETS|$MARKERS) .*\2[er],1,(\d+)s.*) \2([rs]),=/\1 \3\4,=/ ||
        s/( (\d+)[er],0,(?:$ORDDETS|$MARKERS) .*(\d+),c,\2 .*\3[er],1,(\d+)s.*) \2([rs]),=/\1 \4\5,=/
      ) { }
# redirect conj of empty raising
s/\b((\d+)[er],0,(?:$ORDDETS|$MARKERS) .*\2,c,(\d\S+) .*\2[er],1,(\d+)s.* \4[er],0[^ ]*)/\1 \4,c,\3/g;
# remove empty marker...
s/ (\S+),0,(?:$MARKERS) .*\1,1,\d\S+//g;

# redirect impinging dependencies to second argument of empty raising verbs...
while ( # if second arg is nominal redirect to eventuality...
        s/,(\d+)([rs])(.* \1r,0,\S+-aN-b\{.-aN\}\S*:(?:$NULLRSG) .*\1r,2,(\d+)s .*\4e,0)/,\4e\3/ ||
        s/,(\d+)([rs])(.* (\d+)r,0,\S+-aN-b\{.-aN\}\S*:(?:$NULLRSG) .*(\d+),c,\4 .*\5r,2,(\d+)s .*\6e,0)/,\5e\3/ ||
        s/( (\d+)r,0,\S+-aN-b\{.-aN\}\S*:(?:$NULLRSG) .*\2r,2,(\d+)s .*\3e,0.*),\2([rs])/\1,\3e/ ||
        s/( (\d+)r,0,\S+-aN-b\{.-aN\}\S*:(?:$NULLRSG) .*(\d+),c,\2 .*\3r,2,(\d+)s .*\5e,0.*),\2([rs])/\1,\4e/ ||
        # otherwise, redirect to it...
        s/,(\d+)([rs])(.* \1r,0,\S+-aN-b\{.-aN\}\S*:(?:$NULLRSG) .*\1r,2,(\d+)s)/,\4\2\3/ ||
        s/,(\d+)([rs])(.* (\d+)r,0,\S+-aN-b\{.-aN\}\S*:(?:$NULLRSG) .*(\d+),c,\4 .*\5r,2,(\d+)s)/,\5\2\3/ ||
        s/( (\d+)r,0,\S+-aN-b\{.-aN\}\S*:(?:$NULLRSG) .*\2r,2,(\d+)s .*),\2([rs])/\1,\3\4/ ||
        s/( (\d+)r,0,\S+-aN-b\{.-aN\}\S*:(?:$NULLRSG) .*(\d+),c,\2 .*\3r,2,(\d+)s .*),\2([rs])/\1,\4\5/
      ) { }
# redirect conj of empty raising
s/\b((\d+)[rs],0,\S+-aN-b\{.-aN\}\S*:(?:$NULLRSG).* \2,c,(\d\S+) .*\2[rs],2,(\d+)s.* \4e,0[^ ]*)/\1 \4e,c,\3s/g;
s/\b((\d+)[rs],0,\S+-aN-b\{.-aN\}\S*:(?:$NULLRSG).* \2,c,(\d\S+) .*\2[rs],2,(\d+)s.* \4r,0[^ ]*)/\1 \4,c,\3/g;
# remove empty raising verb...
s/ (\S+),0,\S+-aN-b\{\S+:(?:$NULLRSG) .*\1,[0-9],\d\S+//g;

# remove conj predicates...
s/ (\S+),X-cX-dX(\S)+//g;
# remove '?' arguments from passives...
s/ (\S+),\?//g;

# make rins explicit...
s/ (\d+)e,(1),(\d+)r/ \1e,\2,\3r \3s,r,\3r/g;
s/ (\d+)r,(0),(\S+)/ \1r,\2,\3 \1s,r,\1r/g;

# make cins explicit...
s/ (\d+),c,(\d+)/ \1r,c,\2r \1s,c,\2s/g;

# remove coref in cases that are already handled by syntax in predication...
s/(([^ ]*)r,=,.*?\2e,0,(?:U|N-aD))-n[0-9]+:/\1:/g;
# make coref links explicit...
s/([^ ]*)(e,0,[UND][^ ]*)-n([^ ]*)(:[^ ]*)/\1\2\4 \1r,:,\3s/g;
s/([^ ]*)(r,0,[^UND][^ ]*)-n([^ ]*)(:[^ ]*)/\1\2\4 \1r,:,\3s/g;   # this would be anaphora like 'similarly' not observed in annotations

# substitute ='s...
s/,=,(\S+)[rs]/,r,\1s/g;
# while ( s/(\b(\S+),=,([^-]\S+\b).*) \2,/\1 \3,/ ) { }
# while ( s/(\b(\S+),=,([^-]\S+\b).*),\2 /\1,\3 / ) { }
# while ( s/ (\S+),(.*\b\1,=,([^-]\S+\b))/ \3,\2/ ) { }
# while ( s/,(\S+) (.*\b\1,=,([^-]\S+\b))/,\3 \2/ ) { }
s/ \S+,=,\S+//g;

