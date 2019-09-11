# follows bare-ends, which got the word alone at the end of the line. Make them more readable now by putting some space between them and the syntax glarg.

my $TGT_COLUMN = 70;

while (<>) {
	my $follows_last_space = index scalar(reverse $_), ' ';
	my $last_space_position = length($_) - $follows_last_space;
	my $pad = '';
	if ($last_space_position < $TGT_COLUMN) {
		$pad = ' ' x ($TGT_COLUMN - $last_space_position);
	}
	s/ ([^ ]*)\n/$pad $1\n/;
	print;
}

# next up: nathans-numberer
