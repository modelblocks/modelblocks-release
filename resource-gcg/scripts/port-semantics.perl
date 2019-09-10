use strict; use warnings;

use open ':encoding(UTF-8)', ':std'; # assume all files are in UTF-8 (of which ASCII is a subset)

open my $syn_file, '<', $ARGV[0] or die "Could not open syntax file $ARGV[0]\n";
open my $sem_file, '<', $ARGV[1] or die "Could not open semantics file $ARGV[1]\n";

my $annot = qr/((?:-[mnsw]\d+r?)+) +\S+$/; # captures the whole sequence of annots

while (<$syn_file>) {
	unless (/\S/) {print; next} # if line is only whitespace, pass it right through
	my $sem = <$sem_file> until $sem =~ /\S/; # skip blank lines in sem file too
	$sem =~ s/-[ns]\?{4}//g; # strip leftover -n???? etc (from .annot-ready)
	# check for annots in $sem, capture if so
	if ($sem =~ /$annot/) { 
		my $tags = $1;
		s/(?= +\S+$)/$tags/; # and splice onto the syntax line
	}
	print;
}
