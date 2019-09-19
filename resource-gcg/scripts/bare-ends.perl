# follows some space-based indenter and bumps the close parens down to the next line

local $/; $_ = <>; # slurp!

s/                  # whenever you find
    ( \)+ )         # one or more close parens
    \n              # at the end of a line,
    ( [ ]+ )        # and the next line is indented with spaces
    \(              # out to an open paren,
/\n$2$1 \(/gx;      # move the close parens to the next line after the indent,
                    # and space off the open paren to preserve some little sanity
print;

# next in pipeline: yoink-word-rightward.perl, nathans-numberer.perl