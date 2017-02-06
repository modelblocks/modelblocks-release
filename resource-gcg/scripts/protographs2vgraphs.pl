
$COPS = '(?:be|is|\'s|was|are|\'re|were|been|being)';

# delete all punctuation
s/ (\w*)\/0\/[-,\.\!\?\;\:\`\'].*? \1r\/1\/\w*//g;

# propagate subject to object control complement (+ via conjunction left + via conjunction right + via passive + via tough construction)
while ( s/((\w*)\/0\/[^ ]*-b{.-a(?![^}]*-g).* \2r\/1\/([^- ]+) \2r\/2\/(\d+)(?!.*\4r\/.\/\3).*? \4r\/.\/)-/\1\3/ ||
        s/((\w*)\/0\/[^ ]*-b{.-a(?![^}]*-g).* \2r\/1\/(\w+) \2r\/2\/(\d+).* (\d+)r\/.\/)-(?= .*\4r\/.\/\5c)/\1\3/ ||
        s/((\w*)\/0\/[^ ]*-b{.-a(?![^}]*-g).* \2r\/1\/(\w+) \2r\/2\/(\d+).* \4r\/.\/(\d+)c.* \5r\/.\/)-/\1\3/ ||
        s/((\w*)\/0\/[^ ]*-b{.-a(?![^}]*-g).* \2r\/1\/(\w+) \2r\/2\/(\d+).* \4r\/1\/\? \4r\/2\/(\d+).* \5r\/.\/)-/\1\3/ ||
        s/((\w*)\/0\/[^ ]*-b{.-a[^{}]*-g.* \2r\/1\/(\w+) \2r\/2\/(\d+).* \4r\/1\/\* \4r\/2\/(\d+).* \5r\/.\/)-/\1\3/ ) {} # print "====> $_"; }

# # propagate subject to object control complement (+ via conjunction)
# while ( s/((\w*)\/0\/[^ ]*-b{.-a.* \2r\/1\/(\w+) \2r\/2\/(\d+).* \4r\/.\/)-/\1\3/ ||
#         s/((\w*)\/0\/[^ ]*-b{.-a.* \2r\/1\/(\w+) \2r\/2\/(\d+).* (\d+)r\/.\/)-(?= .*\4r\/.\/\5c)/\1\3/ ||
#         s/((\w*)\/0\/[^ ]*-b{.-a.* \2r\/1\/(\w+) \2r\/2\/(\d+).* \4r\/.\/(\d+)c.* \5r\/.\/)-/\1\3/ ) {}

# # treat copulas as identity with complement
# s/(([^ ]*)\/0\/.-aN-b{A-aN}[^ ]*:$COPS.*) \2\/1\/[^ ]* \2\/2\/([^ ]*)/\1 \2\/=\/\3/g;
# 
# # preceding/succeeding deps to identity antecedent replaced with consequent
# while ( s/\/([0-9]*) (.* \1\/=\/([0-9]*)(?![0-9]))/\/\3 \2/ ) {}
# while ( s/( ([0-9]*)\/=\/([0-9]*) .*[0-9]*\/[0-9]*)\/\2/\1\/\3/ ) {}
# 
# # conjunctions
# s/X-cX-dX:[^ ]*/X-cX-dX/g;
# while ( s/(([0-9]*)\/0\/X-cX-dX.*) \2\/[0-9]\/([0-9]*)/\1 \3\/&\/\2/ ) {}
# 

