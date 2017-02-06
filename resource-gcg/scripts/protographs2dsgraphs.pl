
# propagate subject to object control complement (+ in subj-aux inverted clause + via conjunction left + via conjunction right + via passive + via tough construction)
while ( s/((\w*)\/0\/[^ ]*-b{.-a(?![^}]*-g).* \2r\/1\/([^- ]+) \2r\/2\/(\d+)(?!.*\4r\/.\/\3).*? \4r\/.\/)-/\1\3/ ||
        s/(\w*)r\/1\/-(.* (\w*)\/0\/[^ ]*-b{.-a(?![^}]*-g).* \3r\/1\/([^- ]+) \3r\/2\/\1s)/\1r\/1\/\4\2/ ||
        s/((\w*)\/0\/[^ ]*-b{.-a(?![^}]*-g).* \2r\/1\/(\w+) \2r\/2\/(\d+).* (\d+)r\/.\/)-(?= .*\4r\/.\/\5c)/\1\3/ ||
        s/((\w*)\/0\/[^ ]*-b{.-a(?![^}]*-g).* \2r\/1\/(\w+) \2r\/2\/(\d+).* \4r\/.\/(\d+)c.* \5r\/.\/)-/\1\3/ ||
        s/((\w*)\/0\/[^ ]*-b{.-a(?![^}]*-g).* \2r\/1\/(\w+) \2r\/2\/(\d+).* \4r\/1\/\? \4r\/2\/(\d+).* \5r\/.\/)-/\1\3/ ||
        s/((\w*)\/0\/[^ ]*-b{.-a[^{}]*-g.* \2r\/1\/(\w+) \2r\/2\/(\d+).* \4r\/1\/\* \4r\/2\/(\d+).* \5r\/.\/)-/\1\3/ ) {} # print "====> $_"; }

# remove restrictor/scope/conjunct tags
s/ (\w*)[crs](\/[^0][0-9]*\/\w+)[crs](?=[^\w])/ \1\2/g;

# remove dependencies to unknown referents
s/ \w*\/[^0][0-9]*\/[-\?\*](?=[^\w])//g;

