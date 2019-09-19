# follows incrsem/bin/indent

our $indent = 0;

while (<>) {
	# subst current indent with proper level
	my $pad = ' ' x $indent;
	s/^ */$pad/;
	print;
	# adjust indent level
	for my $char (split '') {
		$indent++ if $char eq '(';
		$indent-- if $char eq ')';
	}
}

# precedes bare-ends.perl