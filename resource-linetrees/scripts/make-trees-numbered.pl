
while ( <> ) {
  if ($_=~/!ARTICLE/) { $sentnum = -1; }
  if ($_=~/^\(/ || $_=~/^!ARTICLE/ || $_=~/^ *$/) { $sentnum++; $toknum = 1; }
  #print "$sentnum$toknum:$_";
  printf ( "%02d%02d:%s", $sentnum, $toknum, $_ );
  $toknum++;
}

