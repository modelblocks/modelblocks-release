my ($sentence, $word) = (0,1);

while (<>) {
	if (/^!ARTICLE/) {print "\n0000:!ARTICLE\n"; $sentence = -1; $word = 1; next}
	if (/^\(/) {print "\n"; $sentence++; $word = 1}
	printf '%02u%02u:%s', $sentence, $word++, $_;
}
