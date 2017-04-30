
$SHORT = '(?!<[aeiou])(?:[aeiou])';

while ( <> ) {

  ## remove old -lI tag
  s/-lI//g;

  ######## N -> N:

  #### irregular nouns:
  ## this
  s/\((N(?!-b{N-aD}))([^ %]*) ([Tt]his|[Ss]pecies)\)/\(\1\2-o\1%\L\3\E|N%\L\3\E \3\)/g;
  ## %y
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?[^aeou])(y|ies)\)/\(\1\2-o\1%\4|N%y \3y\)/g;
  ## %* (/es/ed/ing)
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?(?:s|focu))(s)(|es)\)/\(\1\2-o\1%\4\5|N%\4 \3\4\)/g;
  ## %z (/es/ed/ing)
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?z)(z)(|es)\)/\(\1\2-o\1%\4\5|N%\4 \3\4\)/g;
  ## %h (/es/ed/ing)
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?[cs](?<! ac))(h)(|es)\)/\(\1\2-o\1%\4\5|N%\4 \3\4\)/g;
  ## %x (/es/ed/ing)
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?)(x)(|es)\)/\(\1\2-o\1%\4\5|N%\4 \3\4\)/g;

  #### regular nouns:
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?)(|s)\)/\(\1\2-o\1%\4|N% \3\)/g;

  ######## N -> A:

  ## %iness|%y
  s/\((N)([^ ]*) ([^ ]*?)(iness)\)/\(\1\2-oN%\4|A%y \3y\)/g;
  ## %ness|%
  s/\((N)([^ ]*) ([^ ]*?)(ness)\)/\(\1\2-oN%\4|A% \3\)/g;
  ## %lty|%l
  s/\((N)([^ ]*) ([^ ]*?)(l)(ty)\)/\(\1\2-oN%\4\5|A%\4 \3\4\)/g;
  ## %bility|%ble
  s/\((N)([^ ]*) ([^ ]*?)(b)(ility)\)/\(\1\2-oN%\4\5|A%\4le \3\4le\)/g;
#  ## %ity|%e
#  s/\((N)([^ ]*) ([^ ]*?)(ity)\)/\(\1\2-oN%\4|A%e \3e\)/g;

  ######## R -> A:

  ## false cognates: early, only (not to become ear, on)
  s/\((R)([^ %]*) (early|only)()\)/\(A\2-o\1%\4|A% \3\)/g;
  ## well|good
  s/\((R)([^ %]*) ()(well)\)/\(A\2-o\1%\4|A%good good\)/g;
  ## easy|ily gay|ily
  s/\((R)([^ %]*) ([^ ]*?)(ily)\)/\(A\2-o\1%\4|A%y \3y\)/g;
  ## probable|y, simple|y, singly|e, not cheap|ly
  s/\((R)([^ %]*) ([^ ]*?[^aeiou][aeiou]|si[mn])([bgp]l)(y)\)/\(A\2-o\1%\4\5|A%\4e \3\4e\)/g;
  ## R%ly|A% -- note A% is not unique
  s/\((R)([^ %]*) ([^ ]*?)(ly)\)/\(A\2-o\1%\4|A% \3\)/g;
  ## simple adverbs -- note A% is not unique
  s/\((R)([^ %]*) ([^ ]*?)()\)/\(A\2-o\1%\4|A% \3\)/g;

  ######## A COMPARATIVE -> A:

  ## better, worse
  s/\((A)([^ ]*) ()([Bb]etter)\)/\(\1\2-oA%\L\4\E|A%good \3good\)/g;
  s/\((A)([^ ]*) ()([Ww]orse)\)/\(\1\2-oA%\L\4\E|A%bad \3bad\)/g;
  s/\((A)([^ ]*) ([Ff])(arther|urther)\)/\(\1\2-oA%\4|A%ar \3ar\)/g;
  ## %der comparatives
  s/\((A)([^ ]*) ([^ ]*?)(d)(der)\)/\(\1\2-o\1%\4\5|A%\4 \3\4\)/g;
  ## %ger comparatives
  s/\((A)([^ ]*) ([^ ]*?)(g)(ger)\)/\(\1\2-o\1%\4\5|A%\4 \3\4\)/g;
  ## %ter comparatives
  s/\((A)([^ ]*) ([^ ]*?)(t)(ter)\)/\(\1\2-o\1%\4\5|A%\4 \3\4\)/g;
  ## %ier comparatives
  s/\((A)([^ ]*) (?![CcFf]ourier)([^ ]*?)(ier)\)/\(\1\2-o\1%\4|A%y \3y\)/g;
  ## %r comparatives
  s/\((A)([^ ]*) (?![Uu]nderwater|[Ww]einer)([^ ]*?(?:os|in|arg|at|pl))(e)(r)\)/\(\1\2-o\1%\4\5|A%\4 \3\4\)/g;
  ## %er comparatives
  s/\((A)([^ ]*) (?![Aa]fter|[Aa]nother|[Ee]ager|[Ee]ither|[^ ]*[Ee]ver|[Ff]iller|[Ff]ormer|[Ff]ourier|[Gg]ender|[Ii]nner|[^ ]*[Oo]ther|[Oo]uter|[^ ]*[Oo]ver|[Oo]rder|[Pp]er|[Pp]roper|[Rr]ather|[Tt]ogether|[Uu]nder|[Uu]nderwater|[Uu]pper|[Ww]hether|[Cc]omputer|[Mm]eter|[Ww]einer)([^ ]*?)(er)\)/\(\1\2-o\1%\4|A% \3\)/g;

  ######## A SUPERLATIVE -> A:

  ## best, worst
  s/\((A)([^ ]*) ()(best)\)/\(\1\2-oA%\4|A%good \3good\)/g;
  s/\((A)([^ ]*) ()(worst)\)/\(\1\2-oA%\4|A%bad \3bad\)/g;
  ## %dest superlatives
  s/\((A)([^ ]*) ([^ ]*?)(d)(dest)\)/\(\1\2-o\1%\4\5|A%\4 \3\4\)/g;
  ## %gest superlatives
  s/\((A)([^ ]*) ([^ ]*?)(g)(gest)\)/\(\1\2-o\1%\4\5|A%\4 \3\4\)/g;
  ## %test superlatives
  s/\((A)([^ ]*) ([^ ]*?)(t)(test)\)/\(\1\2-o\1%\4\5|A%\4 \3\4\)/g;
  ## %iest superlatives
  s/\((A)([^ ]*) ([^ ]*?)(iest)\)/\(\1\2-o\1%\4|A%y \3y\)/g;
  ## %st superlatives
  s/\((A)([^ ]*) ([^ ]*?(?:os|in|arg|at|pl))(e)(st)\)/\(\1\2-o\1%\4\5|A%\4 \3\4\)/g;
  ## %est superlatives
  s/\((A)([^ ]*) (?![Ww]est|[Mm]odest)([^ ]*?)(est)\)/\(\1\2-o\1%\4|A% \3\)/g;

  ######## A NEGATIVE -> A

  $NOTUN = '(?!canny|der\)|dercut|derlie|derline|derly|derpin|derscore|derstand|dertake|ited?\)|til\)|less\)|iqu)';

  ## un%
  s/\((A)([^ ]*) ([Uu]n)$NOTUN([^ ]*?)()\)/\(\1\2-o\1NEG%:\3\4\5|\1%:un% \4\5\)/g;

  ######## V|B|L|G -> B:

  #### irregular verbs:
  ## arise/rise
  s/\(([BVLG])([^ %]*) ([Aa]?[Rr])(ise|ises|ose|isen|ising)\)/\(B\2-o\1%r\4|B%rise \3ise\)/g;
  ## awake
  s/\(([BVLG])([^ %]*) ([Aa]?[Ww])(aken|akes|ake|oke|akened|akening)\)/\(B\2-o\1%aw\4|B%awaken \3aken\)/g;
  ## be
  s/\(([BVLG])([^ %]*) ()(\'m|\'re|(?<=\{A-a.\} )\'s|[Bb]e|[Aa]m|[Ii]s|[Aa]re|[Ww]as|[Ww]ere|[Bb]een|[Bb]eing)\)/\(B\2-o\1%\L\4\E|B%be \3be\)/g;
  ## bear
  s/\(([BVLG])([^ %]*) ([Bb])(ear|ears|ore|orne|earing)\)/\(B\2-o\1%b\4|B%bear \3ear\)/g;
  ## beat
  s/\(([BVLG])([^ %]*) ([Bb]eat)(|s|en|ing)\)/\(B\2-o\1%eat\4|B%eat \3\)/g;
  ## begin/spin
  s/\(([BVLG])([^ %]*) ([Bb]eg|[Ss]p)(in|ins|an|un|inning)\)/\(B\2-o\1%\4|B%in \3in\)/g;
  ## bleed/breed/feed/speed
  s/\(([BVLG])([^ %]*) ([Bb]le|[Bb]re|[Ff]e|[Ss]pe)(ed|eds|d|eding)\)/\(B\2-o\1%e\4|B%eed \3ed\)/g;
  ## blow/grow/know/throw
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:[Bb]l|[Gg]r|[Tt]hr|[Kk]n))(ow|ows|ew|own|owing)\)/\(B\2-o\1%\4|B%ow \3ow\)/g;
  ## bid/rid
  s/\(([BVLG])([^ %]*) ([^ f]*?[BbRr]id)(|s|ding)\)/\(B\2-o\1%id\4|B%id \3\)/g;
  ## break/speak
  s/\(([BVLG])([^ %]*) ([Bb]r|[Ss]p)(eak|eaks|oke|oken|eaking)\)/\(B\2-o\1%\4|B%eak \3eak\)/g;
  ## bring
  s/\(([BVLG])([^ %]*) ([Bb]r)(ing|ings|ung|ought|inging)\)/\(B\2-o\1%br\4|B%bring \3ing\)/g;
  ## build/rebuild
  s/\(([BVLG])([^ %]*) ([^ ]*?[Bb]uil)(d|ds|t|ding)\)/\(B\2-o\1%buil\4|B%build \3d\)/g;
  ## buy
  s/\(([BVLG])([^ %]*) ([Bb])(uy|uys|ought|uying)\)/\(B\2-o\1%b\4|B%buy \3uy\)/g;
  ## catch
  s/\(([BVLG])([^ %]*) ([Cc])(atch|atches|aught|atching)\)/\(B\2-o\1%c\4|B%catch \3atch\)/g;
  ## choose
  s/\(([BVLG])([^ %]*) ([Cc]ho)(ose|oses|se|sen|osing)\)/\(B\2-o\1%cho\4|B%choose \3ose\)/g;
  ## cling/fling/ring/sing/spring/sting/swing/wring
  s/\(([BVLG])([^ %]*) ([Cc]l|[Ff]l|[Rr]|[Ss]|[Ss]pr|[Ss]t|[Ss]w|[Ww]r)(ing|ings|ang|ung|inging)\)/\(B\2-o\1%\4|B%ing \3ing\)/g;
  ## creep/keep/sleep/sweep/weep
  s/\(([BVLG])([^ %]*) ([Cc]re|[Kk]e|[Ss]le|[Ss]we|[Ww]e)(ep|eps|pt|eping)\)/\(B\2-o\1%\4|B%ep \3ep\)/g;
  ## come/become/overcome
  s/\(([BVLG])([^ %]*) ((?:|[Bb]e|[Oo]ver)[Cc])(ome|omes|ame|omed|oming)\)/\(B\2-o\1%c\4|B%come \3ome\)/g;
  ## deal
  s/\(([BVLG])([^ %]*) ([Dd]eal)(|s|t|ing)\)/\(B\2-o\1%eal\4|B%eal \3\)/g;
  ## die/lie/tie (lie as in fib)
  s/\(([BVLG])([^ %]*) ([DdLlTt])(ie|ies|ied|ying)\)/\(B\2-o\1%\4|B%ie \3ie\)/g;
  ## dig
  s/\(([BVLG])([^ %]*) ([Dd])(ig|igs|ug|igging)\)/\(B\2-o\1%d\4|B%dig \3ig\)/g;
  ## do/undo/outdo
  s/\(([BVLG])([^ %]*) ((?:[Uu]n|[Oo]ut)?[Dd])(o|oes|id|one|oing)\)/\(B\2-o\1%d\4|B%do \3o\)/g;
  ## draw/withdraw
  s/\(([BVLG])([^ %]*) ([^ ]*?[Dd]r)(aw|aws|ew|awn|awing)\)/\(B\2-o\1%dr\4|B%draw \3aw\)/g;
  ## drink/sink/shrink
  s/\(([BVLG])([^ %]*) ([Dd]r|[Ss]|[Ss]hr)(ink|inks|ank|unk|inking)\)/\(B\2-o\1%\4|B%ink \3ink\)/g;
  ## drive/strive
  s/\(([BVLG])([^ %]*) ([Dd]r|[Ss]tr)(ive|ives|ove|iven|iving)\)/\(B\2-o\1%r\4|B%rive \3ive\)/g;
  ## eat
  s/\(([BVLG])([^ %]*) ()([Ee]at|[Ee]ats|[Aa]te|[Ee]aten|[Ee]ating)\)/\(B\2-o\1%\L\4\E|B%eat \3eat\)/g;
  ## fall
  s/\(([BVLG])([^ %]*) ([^ ]*?[Ff])(all|alls|ell|allen|alling)\)/\(B\2-o\1%f\4|B%fall \3all\)/g;
  ## feel/kneel
  s/\(([BVLG])([^ %]*) ([Ff]e|[Kk]ne)(el|els|lt|eling)\)/\(B\2-o\1%\4|B%el \3el\)/g;
  ## fight
  s/\(([BVLG])([^ %]*) ([Ff])(ight|ights|ought|ighting)\)/\(B\2-o\1%\4|B%ight \3ight\)/g;
  ## find/grind
  s/\(([BVLG])([^ %]*) ([Ff]|[Gg]r)(ind|inds|ound|inding)\)/\(B\2-o\1%\4|B%ind \3ind\)/g;
  ## flee
  s/\(([BVLG])([^ %]*) ([Ff]le)(e|es|d|eing)\)/\(B\2-o\1%fle\4|B%flee \3e\)/g;
  ## forbid
  s/\(([BVLG])([^ %]*) ([Ff]orb)(id|ids|ade|idden|idding)\)/\(B\2-o\1%forb\4|B%forbid \3id\)/g;
  ## frolic/panic/mimic
  s/\(([BVLG])([^ %]*) ([Ff]rolic|[Pp]anic|[Mm]imic)(|s|ked|king)\)/\(B\2-o\1%ic\4|B%ic \3\)/g;
  ## freeze
  s/\(([BVLG])([^ %]*) ([Ff]r)(eeze|eezes|oze|ozen|eezing)\)/\(B\2-o\1%fr\4|B%freeze \3eeze\)/g;
  ## get/forget
  s/\(([BVLG])([^ %]*) ((?:[^ ]*[Ff]or)?[Gg])(et|ets|ot|otten|etting)\)/\(B\2-o\1%g\4|B%get \3et\)/g;
  ## give
  s/\(([BVLG])([^ %]*) ([^ ]*?[Gg])(ive|iveth|ives|ave|iven|iving)\)/\(B\2-o\1%g\4|B%give \3ive\)/g;
  ## go/undergo
  s/\(([BVLG])([^ %]*) ([^ ]*?)([Gg]o|[Gg]oes|[Ww]ent|[Gg]one|[Gg]oing)\)/\(B\2-o\1%\L\4\E|B%go \3go\)/g;
  ## hang/overhang
  s/\(([BVLG])([^ %]*) ([Hh]|[Oo]verh)(ang|angs|ung|anged|anging)\)/\(B\2-o\1%h\4|B%hang \3ang\)/g;
  ## have
  s/\(([BVLG])([^ %]*) ()((?<=\{L-a.\} )\'d|\'s|\'ve|[Hh]ave|[Hh]as|[Hh]ad|[Hh]aving)\)/\(B\2-o\1%\L\4\E|B%have \3have\)/g;
  ## hear
  s/\(([BVLG])([^ %]*) ([Hh]ear)(|s|d|ing)\)/\(B\2-o\1%hear\4|B%hear \3\)/g;
  ## hew/sew/strew
  s/\(([BVLG])([^ %]*) ([HhSs]ew|[Ss]trew)(|s|ed|n|ing)\)/\(B\2-o\1%ew\4|B%ew \3\)/g;
  ## hide
  s/\(([BVLG])([^ %]*) ([Hh]id)(e|es||den|ing)\)/\(B\2-o\1%hid\4|B%hide \3e\)/g;
  ## hit
  s/\(([BVLG])([^ %]*) ([Hh]it)(|s||ting)\)/\(B\2-o\1%hit\4|B%hit \3\)/g;
  ## hold
  s/\(([BVLG])([^ %]*) ([^ ]*?[Hh])(old|olds|eld|olding)\)/\(B\2-o\1%h\4|B%hold \3old\)/g;
  ## lay
  s/\(([BVLG])([^ %]*) ([Ll])(ay|ays|aid|ain|aying)\)/\(B\2-o\1%l\4|B%lay \3ay\)/g;
  ## lead/plead/mislead
  s/\(([BVLG])([^ %]*) ((?:mis)?[Pp]?[Ll]e)(ad|ads|d|ading)\)/\(B\2-o\1%le\4|B%lead \3ad\)/g;
  ## leap/outleap
  s/\(([BVLG])([^ %]*) ([^ s]*?[Ll]e)(ap|aps|pt|aping)\)/\(B\2-o\1%le\4|B%leap \3ap\)/g;
  ## leave
  s/\(([BVLG])([^ %]*) ([Ll])(eave|eaves|eft|eaving)\)/\(B\2-o\1%l\4|B%leave \3eave\)/g;
  ## lend/send/spend
  s/\(([BVLG])([^ %]*) ((?:[Ll]|[Ss]|[Ss]p)en)(d|ds|t|ding)\)/\(B\2-o\1%en\4|B%end \3d\)/g;
  ## lie (as in recline)
  s/\(([BVLG])([^ %]*) ([Ll])(ie|ies|ay|ying)\)/\(B\2-o\1%l\4|B%lie \3ie\)/g;
  ## light/highlight/spotlight
  s/\(([BVLG])([^ %]*) ((?:high|moon|spot)?[Ll]i)(ght|ghts|t|ghting)\)/\(B\2-o\1%li\4|B%light \3ght\)/g;
  ## lose
  s/\(([BVLG])([^ %]*) ([Ll]os)(e|es|t|ing)\)/\(B\2-o\1%los\4|B%lose \3e\)/g;
  ## make
  s/\(([BVLG])([^ %]*) ([Mm]a)(ke|kes|de|king)\)/\(B\2-o\1%ma\4|B%make \3ke\)/g;
  ## mean
  s/\(([BVLG])([^ %]*) ([Mm]ean)(|s|t|ing)\)/\(B\2-o\1%mean\4|B%mean \3\)/g;
  ## meet
  s/\(([BVLG])([^ %]*) ([Mm]e)(et|ets|t|eting)\)/\(B\2-o\1%me\4|B%meet \3et\)/g;
  ## pay/say/overpay
  s/\(([BVLG])([^ %]*) ([^ ]*?[PpSs]a)(y|ys|id|ying)\)/\(B\2-o\1%a\4|B%ay \3y\)/g;
  ## prove
  s/\(([BVLG])([^ %]*) ((?:[Dd]is)?[Pp]rov)(e|es|ed|en|ing)\)/\(B\2-o\1%prov\4|B%prove \3e\)/g;
  ## quit
  s/\(([BVLG])([^ %]*) ([Qq]uit)(|s|ting)\)/\(B\2-o\1%quit\4|B%quit \3\)/g;
  ## ride
  s/\(([BVLG])([^ %]*) ([Rr]|[Oo]verr)(ide|ides|ode|idden|iding)\)/\(B\2-o\1%r\4|B%ride \3ide\)/g;
  ## run
  s/\(([BVLG])([^ %]*) ([Rr])(un|uns|an|unning)\)/\(B\2-o\1%r\4|B%run \3un\)/g;
  ## see/oversee
  s/\(([BVLG])([^ %]*) ([^ ]*?[Ss])(ee|ees|aw|een|eeing)\)/\(B\2-o\1%s\4|B%see \3ee\)/g;
  ## seek
  s/\(([BVLG])([^ %]*) ([Ss])(eek|eeks|ought|eeking)\)/\(B\2-o\1%s\4|B%seek \3eek\)/g;
  ## sell/tell
  s/\(([BVLG])([^ %]*) ([^ ]*?[SsTt])(ell|ells|old|elling)\)/\(B\2-o\1%\4|B%ell \3ell\)/g;
  ## shoot
  s/\(([BVLG])([^ %]*) ([Ss]ho)(ot|ots|t|otting)\)/\(B\2-o\1%sho\4|B%shoot \3ot\)/g;
  ## show
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:[Ss]how))(|s|ed|n|ing)\)/\(B\2-o\1%show\4|B%show \3\)/g;
  ## sit
  s/\(([BVLG])([^ %]*) ([Ss])(it|its|at|itting)\)/\(B\2-o\1%s\4|B%sit \3it\)/g;
  ## slay
  s/\(([BVLG])([^ %]*) ([Ss]l)(ay|ays|ayed|ain|aying)\)/\(B\2-o\1%sl\4|B%slay \3ay\)/g;
  ## sneak
  s/\(([BVLG])([^ %]*) ([Ss]n)(eak|eaks|uck|eaking)\)/\(B\2-o\1%sn\4|B%sneak \3eak\)/g;
  ## smite/write
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:[Ss]m|[Ww]r))(ite|ites|ote|itten|iting)\)/\(B\2-o\1%\4|B%ite \3ite\)/g;
  ## stand/understand
  s/\(([BVLG])([^ %]*) ([^ ]*?[Ss]t)(and|ands|ood|anding)\)/\(B\2-o\1%st\4|B%and \3and\)/g;
  ## steal
  s/\(([BVLG])([^ %]*) ([Ss]t)(eal|eals|ole|olen|ealing)\)/\(B\2-o\1%st\4|B%steal \3eal\)/g;
  ## stick
  s/\(([BVLG])([^ %]*) ([Ss]t)(ick|icks|uck|icking)\)/\(B\2-o\1%st\4|B%stick \3ick\)/g;
  ## strike
  s/\(([BVLG])([^ %]*) ([Ss]tr)(ike|ikes|uck|icken|iking)\)/\(B\2-o\1%str\4|B%strike \3ike\)/g;
  ## swear/tear
  s/\(([BVLG])([^ %]*) ([Ss]w|[Tt])(ear|ear|ore|orn|earing)\)/\(B\2-o\1%\4|B%ear \3ear\)/g;
  ## forsake/take/shake
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:[Tt]|[Ss]h|[Ff]ors))(ake|akes|aketh|ook|aken|aking)\)/\(B\2-o\1%\4|B%ake \3ake\)/g;
  ## teach
  s/\(([BVLG])([^ %]*) ([Tt])(each|eaches|aught|eaching)\)/\(B\2-o\1%t\4|B%teach \3each\)/g;
  ## think
  s/\(([BVLG])([^ %]*) ([Tt]h)(ink|inks|ought|inking)\)/\(B\2-o\1%th\4|B%think \3ink\)/g;
  ## tread
  s/\(([BVLG])([^ %]*) ([Tt]r)(ead|eads|od|eading)\)/\(B\2-o\1%tr\4|B%tread \3ead\)/g;
  ## weave
  s/\(([BVLG])([^ %]*) ([Ww])(eave|eaves|ove|oven|eaving)\)/\(B\2-o\1%w\4|B%weave \3eave\)/g;
  ## wreak
  s/\(([BVLG])([^ %]*) ([Ww]r)(eak|eaks|eaked|ought|eaking)\)/\(B\2-o\1%wr\4|B%wreak \3eak\)/g;
  ## will
  s/\(([BVLG])([^ %]*) ()(\'ll|[Ww]ill|[Ww]o)\)/\(B\2-o\1%\L\4\E|B%will \3will\)/g;
  ## win
  s/\(([BVLG])([^ %]*) ([Ww])(in|ins|on|un|inning)\)/\(B\2-o\1%w\4|B%win \3in\)/g;
  ## would
  s/\(([BVLG])([^ %]*) ()(\'d|[Ww]ould)\)/\(B\2-o\1%\L\4\E|B%would \3would\)/g;


  #### irregular in orthography only:
  ## Xd* -- shred/wed/wad
  s/\(([BVLG])([^ %]*) ([Ss]hred|[Ww]ed|[Ww]ad)(|s|ded|ding)\)/\(B\2-o\1%d\4|B%d \3\)/g;
  ## Xl* -- compel/propel/impel/repel
  s/\(([BVLG])([^ %]*) ([^ ]*pel)(|s|led|ling)\)/\(B\2-o\1%l\4|B%l \3\)/g;
  ## Xl* -- control/patrol, not stroll
  s/\(([BVLG])([^ %]*) ([^ ]*..trol)(|s|led|ling)\)/\(B\2-o\1%l\4|B%l \3\)/g;
  ## Xl* -- initial/total
  s/\(([BVLG])([^ %]*) ([^ ]*(?:ial|[Tt]otal))(|s|led|ling)\)/\(B\2-o\1%l\4|B%l \3\)/g;
  ## Xp* -- quip
  s/\(([BVLG])([^ %]*) ([^ ]*?uip)(|s|ped|ping)\)/\(B\2-o\1%p\4|B%p \3\)/g;
  ## Xr* -- infer|deter
  s/\(([BVLG])([^ %]*) ([Dd]eter|(?<= )aver|[^ ]*[^f]fer|proffer|[^ ]*cur)(|s|red|ring)\)/\(B\2-o\1%r\4|B%r \3\)/g;
  ## X* -- alter/bicker/audit/benefit
  s/\(([BVLG])([^ %]*) ([Aa]lter|[Bb]icker|[Aa]udit|[Bb]enefit)(|s|ed|ing)\)/\(B\2-o\1%\4|B% \3\)/g;
  ## Xshed* (/s/ding) shed
  s/\(([BVLG])([^ %]*) ([Ss]hed)(|s|ding)\)/\(B\2-o\1%shed\4|B%shed \3\)/g;
  ## Xtiptoe* (/s/d/ing) tiptoe
  s/\(([BVLG])([^ %]*) ([Tt]iptoe)(|s|d|ing)\)/\(B\2-o\1%tiptoe\4|B%tiptoe \3\)/g;
  ## x*e -- breathe/seethe/soothe/loathe/swathe/writhe/ache
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:breath|eeth|sooth|loath|swath|writh|(?<= )ach))(e|es|ed|ing)\)/\(B\2-o\1%\4|B%e \3e\)/g;
  ## X*e -- waste
  s/\(([BVLG])([^ %]*) ([Ww]ast)(e|es|ed|ing)\)/\(B\2-o\1%\4|B%e \3e\)/g;


  ## X*y
  s/\(([BVLG])([^ %]*) ([^ ]*?[^aeou])(y|ies|ied|ying)\)/\(B\2-o\1%\4|B%y \3y\)/g;

  ### double consonant
  ## Xb* (/s/bed/bing)
  s/\(([BVLG])([^ %]*) ([^ ]*?[^bmr]b)(|s|bed|bing)\)/\(B\2-o\1%b\4|B%b \3\)/g;
  ## Xd* (/s/ded/ding)
  s/\(([BVLG])([^ %]*) ([^ ]*?[^adelnrs'](?<![aeiou][aeiouw])d)(|s|ded|ding)\)/\(B\2-o\1%d\4|B%d \3\)/g;
  ## Xg* (/s/ged/ging)
  s/\(([BVLG])([^ %]*) ([^ ]*?[^gnrs]g)(|s|ged|ging)\)/\(B\2-o\1%g\4|B%g \3\)/g;
  ## Xk* (/s/ked/king)
  s/\(([BVLG])([^ %]*) ([^ ]*?ek)(|s|ked|king)\)/\(B\2-o\1%k\4|B%k \3\)/g;
  ## Xl* (/s/led/ling)
  s/\(([BVLG])([^ %]*) ([^ ]*?(?![aeiou][aeiouw]|e|l|r)l)(|s|led|ling)\)/\(B\2-o\1%l\4|B%l \3\)/g;
  ## Xm* (/s/med/ming)
  s/\(([BVLG])([^ %]*) ([^ ]*?[^lmrs'](?<![aeiou][aeiouw])m)(|s|med|ming)\)/\(B\2-o\1%m\4|B%m \3\)/g;
  ## Xn* (/s/ned/ning)
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:[^aeiou][aiu]n|(?<= )pen| con))(|s|ned|ning)\)/\(B\2-o\1%n\4|B%n \3\)/g;
  ## Xp* (/s/ped/ping)
  s/\(([BVLG])([^ %]*) ([^ ]*?[^lmprs](?<!elo)(?<![aeiou][aeiouw])p)(|s|ped|ping)\)/\(B\2-o\1%p\4|B%p \3\)/g;
  ## Xr* (/s/red/ring)
  s/\(([BVLG])([^ %]*) ([^ ]*?[^eor](?<![aeiou][aeiouw])(?<! pu)r)(|s|red|ring)\)/\(B\2-o\1%r\4|B%r \3\)/g;
  ## Xt* (/s/ted/ting)
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:[^aeiou][aiou]t|cquit|ffset|[fg]ret|abet|(?<= )[blnsBLNS]et)(?<!budget|target|.umpet|.rpret|..i[bcmrs]it|profit))(|s|ted|ting)\)/\(B\2-o\1%t\4|B%t \3\)/g;
  ## Xv* (/s/ved/ving)
  s/\(([BVLG])([^ %]*) ([^ ]*?[^v]v)(|s|ved|ving)\)/\(B\2-o\1%v\4|B%v \3\)/g;

  ## Xs* (/es/ed/ing)
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:ss|focus))(|es|ed|ing)\)/\(B\2-o\1%s\4|B%s \3\)/g;
  ## Xz* (/es/ed/ing)
  s/\(([BVLG])([^ %]*) ([^ ]*?zz)(|es|ed|ing)\)/\(B\2-o\1%z\4|B%z \3\)/g;
  ## Xh* (/es/ed/ing)
  s/\(([BVLG])([^ %]*) ([^ ]*?[cs](?<! ac)h)(|es|ed|ing)\)/\(B\2-o\1%h\4|B%h \3\)/g;
  ## Xx* (/es/ed/ing)
  s/\(([BVLG])([^ %]*) ([^ ]*?x)(|es|ed|ing)\)/\(B\2-o\1%x\4|B%x \3\)/g;

  ## Xee*
  s/\(([BVLG])([^ %]*) ([^ ]*?[^chn]ee)(|s|d|ing)\)/\(B\2-o\1%ee\4|B%ee \3\)/g;


  ## X*e (e|es|ed|ing) -- regular e, that would otherwise qualify as complex vowel or complex consonant
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:iec|oic|[nrs]c|[st]uad|(?<= )guid|af|ieg|[dlr]g|[bcdfgkptyz]l|ym|sum|[ct]ap|hop|[^s]ip|yp|uar|uir|sur|[aeiou]{2}s(?<!vous|bias)|ens|abus|ccus|mus|[Cc]reat|rmeat|[isu]at|uot|eav|iev|[aeo]lv|[ae]iv|rv|u|(?<= )ow|[es]iz|eez|ooz|yz))(e|es|ed|ing)\)/\(B\2-o\1%\4|B%e \3e\)/g;

  ## X* -- regular no e, with complex vowel or complex consonant
  s/\(([BVLG])([^ %]*) ([^ ]*?[aeiou]{2,}[bcdfghjklmnpqrstvwxyz](?<!oed))(|s|ed|ing)\)/\(B\2-o\1%\4|B% \3\)/g;
  s/\(([BVLG])([^ %]*) ([^ ]*?[bcdfghjklmnpqrstvwxyz][bcdfghjklmnpqrtvwxyz](?<![aeiu]ng|.bl))(|s|ed|ing)\)/\(B\2-o\1%\4|B% \3\)/g;

  ## X* -- regular no e, which would otherwise end in single vowel single consonant
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:log|[dnt]al|el|devil|eril|.[^r]lop|..sip|....[sd]on|.[dhkpst]en|[^r]ven|...[st]om|[ct]hor|..[bdhkmnptvw]er|ffer|nger|swer|censor|o[ln]or|a[bjv]or|rror|[fgkrv]et|osit|[ai]bit|erit|imit|mbast))(|s|ed|ing)\)/\(B\2-o\1%\4|B% \3\)/g;
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:hbor|(?<= )ping|mme.|gge.|cke.|icit|i[rv]al|ns..|amid|ofit|-bus|iphon|ilor|umpet|isit))(|s|ed|ing)\)/\(B\2-o\1%\4|B% \3\)/g;

  #### long + consonant + e regular
  ## X*e (e|es|ed|ing)
  s/\(([BVLG])([^ %]*) ([^ ]*?[bcdgklmnprstuvz](?<!l[aio]ng))(e|es|ed|ing)\)/\(B\2-o\1%\4|B%e \3e\)/g;

  #### X* -- default no e regular
  ## X*
  s/\(([BVLG])([^ %]*) ([^ ]*?)(|s|ed|ing)\)/\(B\2-o\1%\4|B% \3\)/g;

  ######## NEGATIVE FOR VERBS

  ## un%
  s/\((B)([^ ]*) ([Uu]n)$NOTUN([^ ]*?)()\)/\(\1\2-o\1NEG%:\3\4\5|\1%:un% \4\5\)/g;

  ######## REPETITIVE

  $NORE = '(?!ap|ach|ad|alize|ason|buff|buke|call|cede|ceive|cite|cognize|commend|cord|cruit|ctify|deem|duce|fer|flect|fresh|fund|fuse|gard|gister|ly|gulate|ject|late|lease|main|mark|member|mind|move|new|novate|plicate|ply|port|present|prove|quest|quire|sign|sist|solve|sonate|spond|store|strict|sult|sume|tain|taliate|tard|tire|tort|turn|view|vise|vive|ward)';

  ## re%
  s/\((B)([^ ]*) ([Rr]e)$NORE([^ ]*?)()\)/\(B\2-o\1REP%:\3\4\5|B%:re% \4\5\)/g;

  ######## CAUSAL, INCHOATIVE, STATIVE

  ## unergatives
  @CI = ('abate','accelerate','adapt','adjust','advance','arm','audition',
         'balance','balloon','band','begin','benefit','bleed','blow','blur','boom','bounce','bow','branch','break','brew','buckle','budge','burn','buzz',
         'capitalize','careen','change','cheer','circulate','clash','cling','close','cohere','coincide','collapse','collect','concentrate','confer','conform','continue','contract','convert','coordinate','crack','crash','crawl','creak','crest','crumble',
         'dance','debut','decrease','default','defect','degenerate','deteriorate','develop','differentiate','diminish','dip','disappear','disarm','dissipate','dissociate','dissolve','distribute','diverge','diversify','double','dress','drift','drill','drive',
         'ease','ebb','economize','edge','embark','emerge','end','engage','enroll','erode','erupt','evaporate','evolve','expand','explode','extend',
         'fade','fail','fester','finish','fire','firm','fit','flash','flatten','flinch','flip','float','flow','fly','focus','fold','form','formulate','freeze',
         'gather','glaze','glide','group','grow',
         'hang','head','hold','hum','hurt',
         'improve','inch','increase','inflate','integrate','intensify','invest',
         'kowtow',
         'land','leap','leapfrog','level','light','liquify','liquidate','lodge',
         'maneuver','materialize','mature','melt','merge','mesh','migrate','militate','mobilize','move',
         'nosedive',
         'operate','order','originate',
         'panic','parachute','part','peak','pile','plummet','plunge','point','pop','pose','premiere','prepare','press','progress','pull',
         'qualify','quit',
         'rally','range','rank','rebound','register','relax','renew','resolve','rest','restructure','resume','retail','retire','retreat','return','reverberate','reverse','revive','revolve','roll','run','rush',
         'segregate','separate','set','settle','shift','shine','shiver','shrink','side','sink','sit','skid','slide','slip','slog','slow','slump','smahs','smoke','soar','sort','spin','split','spread','stampede','stand','start','steam','steer','stem','step','stick','stop','strain','stray','stretch','subscribe','surface','surge','sway','swell','swing','switch',
         'team','terminate','tilt','tiptoe','touch','transfer','translate','triple','tumble','turn','twitch',
         'undulate','unite','unravel',
         'vacillate','vary','venture','volunteer',
         'wade','walk','wane','wedge','widen','wind','withdraw','work','worry','worsen','zoom');
  foreach $c (@CI) {
    s/\((B-aN)-bN([^ ]*) ($c)\)/\(\1\2-oBCAU%\3|B%$c $c\)/g;
  }

  ##         sign type    causative     inchoative    bstative  astative
  @CISA = ([ '-aN',      'awaken',     'awaken',     '',       'awake'   ],
           [ '-aN-bPup', 'blow',       'blow',       '',       ''        ],
           [ '-aN',      'cheer',      'cheer',      '',       ''        ],
           [ '-aN',      'cool',       'cool',       '',       'cool'    ],
           [ '-aN',      'complete',   'complete',   '',       'complete'],
           [ '-aN',      'drop',       'fall',       '',       ''        ],
           [ '-aN',      'dry',        'dry',        '',       'dry'     ],
           [ '-aN-bN',   'feed',       'eat',        '',       ''        ],
           [ '-aN-bN',   'gain',       'gain',       '',       ''        ],
           [ '-aN-bN',   'give',       'get',        'have',   'with'    ],
           [ '-aN',      'heat',       'heat',       '',       'hot'     ],
           [ '-aN',      'kill',       'die',        '',       'dead'    ],
           [ '-aN',      'narrow',     'narrow',     '',       'narrow'  ],
           [ '-aN',      'open',       'open',       '',       'open'    ],
           [ '-aN-bN',   'show',       'see',        '',       ''        ],
           [ '-aN',      'shut',       'shut',       '',       'shut'    ],
           #[ '-aN',      'strengthen', 'strengthen', '',       'strong'  ],
           [ '-aN-bC',   'teach',      'learn',      'know',   ''        ],
           [ '-aN-bPup', 'wake',       'wake',       '',       ''        ],
           [ '-aN',      'warm',       'warm',       '',       'warm'    ],
           [ '-aN',      'wet',        '',           '',       'wet'     ]);
  for( $i=0; $i<scalar(@CISA); $i++ ) {
    if( length($CISA[$i][1])>0 && length($CISA[$i][2])>0 ) { s/\(B($CISA[$i][0])-bN([^ ]*) ($CISA[$i][1])\)/\(B\1\2-oBCAU%\3|B%$CISA[$i][2] $CISA[$i][2]\)/g; }
    if( length($CISA[$i][2])>0 && length($CISA[$i][3])>0 ) { s/\(B($CISA[$i][0])([^ ]*) ($CISA[$i][2])\)/\(B\1\2-oBINC%\3|B%$CISA[$i][3] $CISA[$i][3]\)/g; }
    if( length($CISA[$i][3])>0 && length($CISA[$i][4])>0 ) { s/\(B($CISA[$i][0])([^ ]*) ($CISA[$i][3])\)/\(A\1\2-oB%\3|A%$CISA[$i][4] $CISA[$i][4]\)/g; }
    if( length($CISA[$i][2])>0 && length($CISA[$i][4])>0 ) { s/\(B($CISA[$i][0])([^ ]*) ($CISA[$i][2])\)/\(A\1\2-oBINC%\3|A%$CISA[$i][4] $CISA[$i][4]\)/g; }
  }

  ## %ize
#  s/\((B)([^ ]*) ([^ ]*?)(ize)\)/\(A\2-o\1%\4|A% \3\)/g;
  ## %engthen
  s/\((B)([^ ]*) ([^ ]*?)()(engthen)\)/\(A\2-o\1%\4\5|A%\4ong \3\4ong\)/g;
  ## %ipen
  s/\((B)([^ ]*) ([^ ]*?ip)(e)(n)\)/\(A\2-o\1%\4\5|A%\4 \3\4\)/g;
  ## %dden
  s/\((B)([^ ]*) ([^ ]*?)(d)(den)\)/\(A\2-o\1%\4\5|A%\4 \3\4\)/g;
  ## %tten
  s/\((B)([^ ]*) ([^ ]*?)(t)(ten)\)/\(A\2-o\1%\4\5|A%\4 \3\4\)/g;
  ## %en
  s/\((B)([^ ]*) (?![^ ]*open|threaten|happen)([^ ]*?)(en)\)/\(A\2-o\1%\4|A% \3\)/g;

  ######## CLEANUP

  ## convert lemma w tag -o back to observed form w tag -x, working bottom-up
  while( s/\((.)([^ ]*)-o(.)([^% ]*)%:([^% ]*)%([^\| ]*)\|(.)%:([^% ]*)%([^- ]*)(-[^ ]*?)? \8([^\)]*)\9\)/\(\3\2-x\3\4%:\5%\6\|\7%:\8%\9\10 \5\11\6\)/g ||
         s/\((.)([^ ]*)-o(.)([^% ]*)%([^\| ]*)\|(.)%([^- ]*)(-[^ ]*?)? ([^\)]*)\7\)/\(\3\2-x\3\4%\5\|\6%\7\8 \9\5\)/g ) { }

  #### remove -x tag
  s/\(([BVLGAR][^ ]*)-x-([^\(\)]*)\)/\(\1-\2\)/g;

  # #### remove empty morphemes
  # s/ \([^ ]* \*\)//g;

  print $_;
}
