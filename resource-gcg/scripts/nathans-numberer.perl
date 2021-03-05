# not a hugely robust piece of code. Relies on lines starting with open paren 
# (produced by prettyprinter) to detect sentence boundary.
# Will silently do The Wrong Thing on a text longer than 99 sentences. 
my ($sentence, $word) = (0,1);

while (<>) {
	if (/^!ARTICLE/) {print "\n0000:!ARTICLE\n"; $sentence = -1; $word = 1; next}
	if (/^\(/) {print "\n"; $sentence++; $word = 1}
	printf '%02u%02u:%s', $sentence, $word++, $_;
}
