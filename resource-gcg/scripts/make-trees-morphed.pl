
$SHORT = '(?!<[aeiou])(?:[aeiou])';

while ( <> ) {

  ## remove old -lI tag
  s/-lI//g;

  ######## NOUNS:

  #### irregular nouns:
  ## this
  s/\((N)([^ %]*) ([Tt]his)\)/\(\1\2-x\1%\L\3\E|N%\L\3\E \3\)/g;
  ## X*y
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?[^aeou])(y|ies)\)/\(\1\2-x\1%\4|N%y \3\4\)/g;
  ## Xs* (/es/ed/ing)
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?(?:ss|focus))(|es)\)/\(\1\2-x\1%s\4|N%s \3\4\)/g;
  ## Xz* (/es/ed/ing)
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?zz)(|es)\)/\(\1\2-x\1%z\4|N%z \3\4\)/g;
  ## Xh* (/es/ed/ing)
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?[cs](?<! ac)h)(|es)\)/\(\1\2-x\1%h\4|N%h \3\4\)/g;
  ## Xx* (/es/ed/ing)
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?x)(|es)\)/\(\1\2-x\1%x\4|N%x \3\4\)/g;

  #### regular nouns:
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?)(|s)\)/\(\1\2-x\1%\4|N% \3\4\)/g;


  ######## VERBS:

  #### irregular verbs:
  ## arise/rise
  s/\(([BVLG])([^ %]*) ([Aa]?[Rr])(ise|ises|ose|isen|ising)\)/\(\1\2-x\1%r\4|B%rise \3\4\)/g;
  ## awake
  s/\(([BVLG])([^ %]*) ([Aa]?[Ww])(aken|akes|ake|oke|akened|akening)\)/\(\1\2-x\1%aw\4|B%awaken \3\4\)/g;
  ## be
  s/\(([BVLG])([^ %]*) ()(\'m|\'re|(?<=\{A-a.\} )\'s|[Bb]e|[Aa]m|[Ii]s|[Aa]re|[Ww]as|[Ww]ere|[Bb]een|[Bb]eing)\)/\(\1\2-x\1%\L\4\E|B%be \3\4\)/g;
  ## bear
  s/\(([BVLG])([^ %]*) ([Bb])(ear|ears|ore|orne|earing)\)/\(\1\2-x\1%b\4|B%bear \3\4\)/g;
  ## beat
  s/\(([BVLG])([^ %]*) ([Bb]eat)(|s|en|ing)\)/\(\1\2-x\1%eat\4|B%eat \3\4\)/g;
  ## begin/spin
  s/\(([BVLG])([^ %]*) ([Bb]eg|[Ss]p)(in|ins|an|un|inning)\)/\(\1\2-x\1%\4|B%in \3\4\)/g;
  ## bleed/breed/feed/speed
  s/\(([BVLG])([^ %]*) ([Bb]le|[Bb]re|[Ff]e|[Ss]pe)(ed|eds|d|eding)\)/\(\1\2-x\1%e\4|B%eed \3\4\)/g;
  ## blow/grow/know/throw
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:[Bb]l|[Gg]r|[Tt]hr|[Kk]n))(ow|ows|ew|own|owing)\)/\(\1\2-x\1%\4|B%ow \3\4\)/g;
  ## bid/rid
  s/\(([BVLG])([^ %]*) ([^ f]*?[BbRr]id)(|s|ding)\)/\(\1\2-x\1%id\4|B%id \3\4\)/g;
  ## break/speak
  s/\(([BVLG])([^ %]*) ([Bb]r|[Ss]p)(eak|eaks|oke|oken|eaking)\)/\(\1\2-x\1%\4|B%eak \3\4\)/g;
  ## bring
  s/\(([BVLG])([^ %]*) ([Bb]r)(ing|ings|ung|ought|inging)\)/\(\1\2-x\1%br\4|B%bring \3\4\)/g;
  ## build/rebuild
  s/\(([BVLG])([^ %]*) ([^ ]*?[Bb]uil)(d|ds|t|ding)\)/\(\1\2-x\1%buil\4|B%build \3\4\)/g;
  ## buy
  s/\(([BVLG])([^ %]*) ([Bb])(uy|uys|ought|uying)\)/\(\1\2-x\1%b\4|B%buy \3\4\)/g;
  ## catch
  s/\(([BVLG])([^ %]*) ([Cc])(atch|atches|aught|atching)\)/\(\1\2-x\1%c\4|B%catch \3\4\)/g;
  ## choose
  s/\(([BVLG])([^ %]*) ([Cc]ho)(ose|oses|se|sen|osing)\)/\(\1\2-x\1%cho\4|B%choose \3\4\)/g;
  ## cling/fling/ring/sing/spring/sting/swing/wring
  s/\(([BVLG])([^ %]*) ([Cc]l|[Ff]l|[Rr]|[Ss]|[Ss]pr|[Ss]t|[Ss]w|[Ww]r)(ing|ings|ang|ung|inging)\)/\(\1\2-x\1%\4|B%ing \3\4\)/g;
  ## creep/keep/sleep/sweep/weep
  s/\(([BVLG])([^ %]*) ([Cc]re|[Kk]e|[Ss]le|[Ss]we|[Ww]e)(ep|eps|pt|eping)\)/\(\1\2-x\1%\4|B%ep \3\4\)/g;
  ## come/become/overcome
  s/\(([BVLG])([^ %]*) ((?:|[Bb]e|[Oo]ver)[Cc])(ome|omes|ame|omed|oming)\)/\(\1\2-x\1%c\4|B%come \3\4\)/g;
  ## deal
  s/\(([BVLG])([^ %]*) ([Dd]eal)(|s|t|ing)\)/\(\1\2-x\1%eal\4|B%eal \3\4\)/g;
  ## die/lie/tie (lie as in fib)
  s/\(([BVLG])([^ %]*) ([DdLlTt])(ie|ies|ied|ying)\)/\(\1\2-x\1%\4|B%ie \3\4\)/g;
  ## dig
  s/\(([BVLG])([^ %]*) ([Dd])(ig|igs|ug|igging)\)/\(\1\2-x\1%d\4|B%dig \3\4\)/g;
  ## do/undo/outdo
  s/\(([BVLG])([^ %]*) ((?:[Uu]n|[Oo]ut)?[Dd])(o|oes|id|one|oing)\)/\(\1\2-x\1%d\4|B%do \3\4\)/g;
  ## draw/withdraw
  s/\(([BVLG])([^ %]*) ([^ ]*?[Dd]r)(aw|aws|ew|awn|awing)\)/\(\1\2-x\1%dr\4|B%draw \3\4\)/g;
  ## drink/sink/shrink
  s/\(([BVLG])([^ %]*) ([Dd]r|[Ss]|[Ss]hr)(ink|inks|ank|unk|inking)\)/\(\1\2-x\1%\4|B%ink \3\4\)/g;
  ## drive/strive
  s/\(([BVLG])([^ %]*) ([Dd]r|[Ss]tr)(ive|ives|ove|iven|iving)\)/\(\1\2-x\1%r\4|B%rive \3\4\)/g;
  ## eat
  s/\(([BVLG])([^ %]*) ()([Ee]at|[Ee]ats|[Aa]te|[Ee]aten|[Ee]ating)\)/\(\1\2-x\1%\L\4\E|B%eat \3\4\)/g;
  ## fall
  s/\(([BVLG])([^ %]*) ([^ ]*?[Ff])(all|alls|ell|allen|alling)\)/\(\1\2-x\1%f\4|B%fall \3\4\)/g;
  ## feel/kneel
  s/\(([BVLG])([^ %]*) ([Ff]e|[Kk]ne)(el|els|lt|eling)\)/\(\1\2-x\1%\4|B%el \3\4\)/g;
  ## fight
  s/\(([BVLG])([^ %]*) ([Ff])(ight|ights|ought|ighting)\)/\(\1\2-x\1%\4|B%ight \3\4\)/g;
  ## find/grind
  s/\(([BVLG])([^ %]*) ([Ff]|[Gg]r)(ind|inds|ound|inding)\)/\(\1\2-x\1%\4|B%ind \3\4\)/g;
  ## flee
  s/\(([BVLG])([^ %]*) ([Ff]le)(e|es|d|eing)\)/\(\1\2-x\1%fle\4|B%flee \3\4\)/g;
  ## forbid
  s/\(([BVLG])([^ %]*) ([Ff]orb)(id|ids|ade|idden|idding)\)/\(\1\2-x\1%forb\4|B%forbid \3\4\)/g;
  ## frolic/panic/mimic
  s/\(([BVLG])([^ %]*) ([Ff]rolic|[Pp]anic|[Mm]imic)(|s|ked|king)\)/\(\1\2-x\1%ic\4|B%ic \3\4\)/g;
  ## freeze
  s/\(([BVLG])([^ %]*) ([Ff]r)(eeze|eezes|oze|ozen|eezing)\)/\(\1\2-x\1%fr\4|B%freeze \3\4\)/g;
  ## get/forget
  s/\(([BVLG])([^ %]*) ((?:[^ ]*[Ff]or)?[Gg])(et|ets|ot|otten|etting)\)/\(\1\2-x\1%g\4|B%get \3\4\)/g;
  ## give
  s/\(([BVLG])([^ %]*) ([^ ]*?[Gg])(ive|iveth|ives|ave|iven|iving)\)/\(\1\2-x\1%g\4|B%give \3\4\)/g;
  ## go/undergo
  s/\(([BVLG])([^ %]*) ([^ ]*?)([Gg]o|[Gg]oes|[Ww]ent|[Gg]one|[Gg]oing)\)/\(\1\2-x\1%\L\4\E|B%go \3\4\)/g;
  ## hang/overhang
  s/\(([BVLG])([^ %]*) ([Hh]|[Oo]verh)(ang|angs|ung|anged|anging)\)/\(\1\2-x\1%h\4|B%hang \3\4\)/g;
  ## have
  s/\(([BVLG])([^ %]*) ()((?<=\{L-a.\} )\'d|\'s|\'ve|[Hh]ave|[Hh]as|[Hh]ad|[Hh]aving)\)/\(\1\2-x\1%\L\4\E|B%have \3\4\)/g;
  ## hear
  s/\(([BVLG])([^ %]*) ([Hh]ear)(|s|d|ing)\)/\(\1\2-x\1%hear\4|B%hear \3\4\)/g;
  ## hew/sew/strew
  s/\(([BVLG])([^ %]*) ([HhSs]ew|[Ss]trew)(|s|ed|n|ing)\)/\(\1\2-x\1%hew\4|B%hew \3\4\)/g;
  ## hide
  s/\(([BVLG])([^ %]*) ([Hh]id)(e|es||den|ing)\)/\(\1\2-x\1%hid\4|B%hide \3\4\)/g;
  ## hit
  s/\(([BVLG])([^ %]*) ([Hh]it)(|s||ting)\)/\(\1\2-x\1%hit\4|B%hit \3\4\)/g;
  ## hold
  s/\(([BVLG])([^ %]*) ([^ ]*?[Hh])(old|olds|eld|olding)\)/\(\1\2-x\1%h\4|B%hold \3\4\)/g;
  ## lay
  s/\(([BVLG])([^ %]*) ([Ll])(ay|ays|aid|ain|aying)\)/\(\1\2-x\1%l\4|B%lay \3\4\)/g;
  ## lead/plead/mislead
  s/\(([BVLG])([^ %]*) ((?:mis)?[Pp]?[Ll]e)(ad|ads|d|ading)\)/\(\1\2-x\1%le\4|B%lead \3\4\)/g;
  ## leap/outleap
  s/\(([BVLG])([^ %]*) ([^ s]*?[Ll]e)(ap|aps|pt|aping)\)/\(\1\2-x\1%le\4|B%leap \3\4\)/g;
  ## leave
  s/\(([BVLG])([^ %]*) ([Ll])(eave|eaves|eft|eaving)\)/\(\1\2-x\1%l\4|B%leave \3\4\)/g;
  ## lend/send/spend
  s/\(([BVLG])([^ %]*) ((?:[Ll]|[Ss]|[Ss]p)en)(d|ds|t|ding)\)/\(\1\2-x\1%en\4|B%end \3\4\)/g;
  ## lie (as in recline)
  s/\(([BVLG])([^ %]*) ([Ll])(ie|ies|ay|ying)\)/\(\1\2-x\1%l\4|B%lie \3\4\)/g;
  ## light/highlight/spotlight
  s/\(([BVLG])([^ %]*) ((?:high|moon|spot)?[Ll]i)(ght|ghts|t|ghting)\)/\(\1\2-x\1%li\4|B%light \3\4\)/g;
  ## lose
  s/\(([BVLG])([^ %]*) ([Ll]os)(e|es|t|ing)\)/\(\1\2-x\1%los\4|B%lose \3\4\)/g;
  ## make
  s/\(([BVLG])([^ %]*) ([Mm]a)(ke|kes|de|king)\)/\(\1\2-x\1%ma\4|B%make \3\4\)/g;
  ## mean
  s/\(([BVLG])([^ %]*) ([Mm]ean)(|s|t|ing)\)/\(\1\2-x\1%mean\4|B%mean \3\4\)/g;
  ## meet
  s/\(([BVLG])([^ %]*) ([Mm]e)(et|ets|t|eting)\)/\(\1\2-x\1%me\4|B%meet \3\4\)/g;
  ## pay/say/overpay
  s/\(([BVLG])([^ %]*) ([^ ]*?[PpSs]a)(y|ys|id|ying)\)/\(\1\2-x\1%a\4|B%ay \3\4\)/g;
  ## prove
  s/\(([BVLG])([^ %]*) ((?:[Dd]is)?[Pp]rov)(e|es|ed|en|ing)\)/\(\1\2-x\1%prov\4|B%prove \3\4\)/g;
  ## quit
  s/\(([BVLG])([^ %]*) ([Qq]uit)(|s|ting)\)/\(\1\2-x\1%quit\4|B%quit \3\4\)/g;
  ## ride
  s/\(([BVLG])([^ %]*) ([Rr]|[Oo]verr)(ide|ides|ode|idden|iding)\)/\(\1\2-x\1%r\4|B%ride \3\4\)/g;
  ## run
  s/\(([BVLG])([^ %]*) ([Rr])(un|uns|an|unning)\)/\(\1\2-x\1%r\4|B%run \3\4\)/g;
  ## see/oversee
  s/\(([BVLG])([^ %]*) ([^ ]*?[Ss])(ee|ees|aw|een|eeing)\)/\(\1\2-x\1%s\4|B%see \3\4\)/g;
  ## seek
  s/\(([BVLG])([^ %]*) ([Ss])(eek|eeks|ought|eeking)\)/\(\1\2-x\1%s\4|B%seek \3\4\)/g;
  ## sell/tell
  s/\(([BVLG])([^ %]*) ([^ ]*?[SsTt])(ell|ells|old|elling)\)/\(\1\2-x\1%\4|B%ell \3\4\)/g;
  ## shoot
  s/\(([BVLG])([^ %]*) ([Ss]ho)(ot|ots|t|otting)\)/\(\1\2-x\1%sho\4|B%shoot \3\4\)/g;
  ## show
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:[Ss]how))(|s|ed|n|ing)\)/\(\1\2-x\1%show\4|B%show \3\4\)/g;
  ## sit
  s/\(([BVLG])([^ %]*) ([Ss])(it|its|at|itting)\)/\(\1\2-x\1%s\4|B%sit \3\4\)/g;
  ## slay
  s/\(([BVLG])([^ %]*) ([Ss]l)(ay|ays|ayed|ain|aying)\)/\(\1\2-x\1%sl\4|B%slay \3\4\)/g;
  ## sneak
  s/\(([BVLG])([^ %]*) ([Ss]n)(eak|eaks|uck|eaking)\)/\(\1\2-x\1%sn\4|B%sneak \3\4\)/g;
  ## smite/write
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:[Ss]m|[Ww]r))(ite|ites|ote|itten|iting)\)/\(\1\2-x\1%\4|B%ite \3\4\)/g;
  ## stand/understand
  s/\(([BVLG])([^ %]*) ([^ ]*?[Ss]t)(and|ands|ood|anding)\)/\(\1\2-x\1%st\4|B%stand \3\4\)/g;
  ## steal
  s/\(([BVLG])([^ %]*) ([Ss]t)(eal|eals|ole|olen|ealing)\)/\(\1\2-x\1%st\4|B%steal \3\4\)/g;
  ## stick
  s/\(([BVLG])([^ %]*) ([Ss]t)(ick|icks|uck|icking)\)/\(\1\2-x\1%st\4|B%stick \3\4\)/g;
  ## strike
  s/\(([BVLG])([^ %]*) ([Ss]tr)(ike|ikes|uck|icken|iking)\)/\(\1\2-x\1%str\4|B%strike \3\4\)/g;
  ## swear/tear
  s/\(([BVLG])([^ %]*) ([Ss]w|[Tt])(ear|ear|ore|orn|earing)\)/\(\1\2-x\1%\4|B%ear \3\4\)/g;
  ## forsake/take/shake
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:[Tt]|[Ss]h|[Ff]ors))(ake|akes|aketh|ook|aken|aking)\)/\(\1\2-x\1%\4|B%ake \3\4\)/g;
  ## teach
  s/\(([BVLG])([^ %]*) ([Tt])(each|eaches|aught|eaching)\)/\(\1\2-x\1%t\4|B%teach \3\4\)/g;
  ## think
  s/\(([BVLG])([^ %]*) ([Tt]h)(ink|inks|ought|inking)\)/\(\1\2-x\1%th\4|B%think \3\4\)/g;
  ## tread
  s/\(([BVLG])([^ %]*) ([Tt]r)(ead|eads|od|eading)\)/\(\1\2-x\1%tr\4|B%tread \3\4\)/g;
  ## weave
  s/\(([BVLG])([^ %]*) ([Ww])(eave|eaves|ove|oven|eaving)\)/\(\1\2-x\1%w\4|B%weave \3\4\)/g;
  ## wreak
  s/\(([BVLG])([^ %]*) ([Ww]r)(eak|eaks|eaked|ought|eaking)\)/\(\1\2-x\1%wr\4|B%wreak \3\4\)/g;
  ## will
  s/\(([BVLG])([^ %]*) ()(\'ll|[Ww]ill|[Ww]o)\)/\(\1\2-x\1%\L\4\E|B%will \3\4\)/g;
  ## win
  s/\(([BVLG])([^ %]*) ([Ww])(in|ins|on|un|inning)\)/\(\1\2-x\1%w\4|B%win \3\4\)/g;
  ## would
  s/\(([BVLG])([^ %]*) ()(\'d|[Ww]ould)\)/\(\1\2-x\1%\L\4\E|B%would \3\4\)/g;


  #### irregular in orthography only:
  ## Xd* -- shred/wed/wad
  s/\(([BVLG])([^ %]*) ([Ss]hred|[Ww]ed|[Ww]ad)(|s|ded|ding)\)/\(\1\2-x\1%d\4|B%d \3\4\)/g;
  ## Xl* -- compel/propel/impel/repel
  s/\(([BVLG])([^ %]*) ([^ ]*pel)(|s|led|ling)\)/\(\1\2-x\1%l\4|B%l \3\4\)/g;
  ## Xl* -- control/patrol, not stroll
  s/\(([BVLG])([^ %]*) ([^ ]*..trol)(|s|led|ling)\)/\(\1\2-x\1%l\4|B%l \3\4\)/g;
  ## Xl* -- initial/total
  s/\(([BVLG])([^ %]*) ([^ ]*(?:ial|[Tt]otal))(|s|led|ling)\)/\(\1\2-x\1%l\4|B%l \3\4\)/g;
  ## Xp* -- quip
  s/\(([BVLG])([^ %]*) ([^ ]*?uip)(|s|ped|ping)\)/\(\1\2-x\1%p\4|B%p \3\4\)/g;
  ## Xr* -- infer|deter
  s/\(([BVLG])([^ %]*) ([Dd]eter|(?<= )aver|[^ ]*[^f]fer|proffer|[^ ]*cur)(|s|red|ring)\)/\(\1\2-x\1%r\4|B%r \3\4\)/g;
  ## X* -- alter/bicker/audit/benefit
  s/\(([BVLG])([^ %]*) ([Aa]lter|[Bb]icker|[Aa]udit|[Bb]enefit)(|s|ed|ing)\)/\(\1\2-x\1%\4|B% \3\4\)/g;
  ## Xshed* (/s/ding) shed
  s/\(([BVLG])([^ %]*) ([Ss]hed)(|s|ding)\)/\(\1\2-x\1%shed\4|B%shed \3\4\)/g;
  ## Xtiptoe* (/s/d/ing) tiptoe
  s/\(([BVLG])([^ %]*) ([Tt]iptoe)(|s|d|ing)\)/\(\1\2-x\1%tiptoe\4|B%tiptoe \3\4\)/g;
  ## x*e -- breathe/seethe/soothe/loathe/swathe/writhe/ache
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:breath|eeth|sooth|loath|swath|writh|(?<= )ach))(e|es|ed|ing)\)/\(\1\2-x\1%\4|B%e \3\4\)/g;
  ## X*e -- waste
  s/\(([BVLG])([^ %]*) ([Ww]ast)(e|es|ed|ing)\)/\(\1\2-x\1%\4|B%e \3\4\)/g;


  ## X*y
  s/\(([BVLG])([^ %]*) ([^ ]*?[^aeou])(y|ies|ied|ying)\)/\(\1\2-x\1%\4|B%y \3\4\)/g;

  ### double consonant
  ## Xb* (/s/bed/bing)
  s/\(([BVLG])([^ %]*) ([^ ]*?[^bmr]b)(|s|bed|bing)\)/\(\1\2-x\1%b\4|B%b \3\4\)/g;
  ## Xd* (/s/ded/ding)
  s/\(([BVLG])([^ %]*) ([^ ]*?[^adelnrs'](?<![aeiou][aeiouw])d)(|s|ded|ding)\)/\(\1\2-x\1%d\4|B%d \3\4\)/g;
  ## Xg* (/s/ged/ging)
  s/\(([BVLG])([^ %]*) ([^ ]*?[^gnrs]g)(|s|ged|ging)\)/\(\1\2-x\1%g\4|B%g \3\4\)/g;
  ## Xk* (/s/ked/king)
  s/\(([BVLG])([^ %]*) ([^ ]*?ek)(|s|ked|king)\)/\(\1\2-x\1%k\4|B%k \3\4\)/g;
  ## Xl* (/s/led/ling)
  s/\(([BVLG])([^ %]*) ([^ ]*?(?![aeiou][aeiouw]|e|l|r)l)(|s|led|ling)\)/\(\1\2-x\1%l\4|B%l \3\4\)/g;
  ## Xm* (/s/med/ming)
  s/\(([BVLG])([^ %]*) ([^ ]*?[^lmrs'](?<![aeiou][aeiouw])m)(|s|med|ming)\)/\(\1\2-x\1%m\4|B%m \3\4\)/g;
  ## Xn* (/s/ned/ning)
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:[^aeiou][aiu]n|(?<= )pen| con))(|s|ned|ning)\)/\(\1\2-x\1%n\4|B%n \3\4\)/g;
  ## Xp* (/s/ped/ping)
  s/\(([BVLG])([^ %]*) ([^ ]*?[^lmprs](?<!elo)(?<![aeiou][aeiouw])p)(|s|ped|ping)\)/\(\1\2-x\1%p\4|B%p \3\4\)/g;
  ## Xr* (/s/red/ring)
  s/\(([BVLG])([^ %]*) ([^ ]*?[^eor](?<![aeiou][aeiouw])(?<! pu)r)(|s|red|ring)\)/\(\1\2-x\1%r\4|B%r \3\4\)/g;
  ## Xt* (/s/ted/ting)
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:[^aeiou][aiou]t|cquit|ffset|[fg]ret|abet|(?<= )[blnsBLNS]et)(?<!budget|target|.umpet|.rpret|..i[bcmrs]it|profit))(|s|ted|ting)\)/\(\1\2-x\1%t\4|B%t \3\4\)/g;
  ## Xv* (/s/ved/ving)
  s/\(([BVLG])([^ %]*) ([^ ]*?[^v]v)(|s|ved|ving)\)/\(\1\2-x\1%v\4|B%v \3\4\)/g;

  ## Xs* (/es/ed/ing)
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:ss|focus))(|es|ed|ing)\)/\(\1\2-x\1%s\4|B%s \3\4\)/g;
  ## Xz* (/es/ed/ing)
  s/\(([BVLG])([^ %]*) ([^ ]*?zz)(|es|ed|ing)\)/\(\1\2-x\1%z\4|B%z \3\4\)/g;
  ## Xh* (/es/ed/ing)
  s/\(([BVLG])([^ %]*) ([^ ]*?[cs](?<! ac)h)(|es|ed|ing)\)/\(\1\2-x\1%h\4|B%h \3\4\)/g;
  ## Xx* (/es/ed/ing)
  s/\(([BVLG])([^ %]*) ([^ ]*?x)(|es|ed|ing)\)/\(\1\2-x\1%x\4|B%x \3\4\)/g;

  ## Xee*
  s/\(([BVLG])([^ %]*) ([^ ]*?[^chn]ee)(|s|d|ing)\)/\(\1\2-x\1%ee\4|B%ee \3\4\)/g;


  ## X*e (e|es|ed|ing) -- regular e, that would otherwise qualify as complex vowel or complex consonant
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:iec|oic|[nrs]c|[st]uad|(?<= )guid|af|ieg|[dlr]g|[bcdfgkptyz]l|ym|sum|[ct]ap|hop|[^s]ip|yp|uar|uir|sur|[aeiou]{2}s(?<!vous|bias)|ens|abus|ccus|mus|[Cc]reat|rmeat|[isu]at|uot|eav|iev|[aeo]lv|[ae]iv|rv|u|(?<= )ow|[es]iz|eez|ooz|yz))(e|es|ed|ing)\)/\(\1\2-x\1%\4|B%e \3\4\)/g;

  ## X* -- regular no e, with complex vowel or complex consonant
  s/\(([BVLG])([^ %]*) ([^ ]*?[aeiou]{2,}[bcdfghjklmnpqrstvwxyz](?<!oed))(|s|ed|ing)\)/\(\1\2-x\1%\4|B% \3\4\)/g;
  s/\(([BVLG])([^ %]*) ([^ ]*?[bcdfghjklmnpqrstvwxyz][bcdfghjklmnpqrtvwxyz](?<![aeiu]ng|.bl))(|s|ed|ing)\)/\(\1\2-x\1%\4|B% \3\4\)/g;

  ## X* -- regular no e, which would otherwise end in single vowel single consonant
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:log|[dnt]al|el|devil|eril|.[^r]lop|..sip|....[sd]on|.[dhkpst]en|[^r]ven|...[st]om|[ct]hor|..[bdhkmnptvw]er|ffer|nger|swer|censor|o[ln]or|a[bjv]or|rror|[fgkrv]et|osit|[ai]bit|erit|imit|mbast))(|s|ed|ing)\)/\(\1\2-x\1%\4|B% \3\4\)/g;
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:hbor|(?<= )ping|mme.|gge.|cke.|icit|i[rv]al|ns..|amid|ofit|-bus|iphon|ilor|umpet|isit))(|s|ed|ing)\)/\(\1\2-x\1%\4|B% \3\4\)/g;

  #### long + consonant + e regular
  ## X*e (e|es|ed|ing)
  s/\(([BVLG])([^ %]*) ([^ ]*?[bcdgklmnprstuvz](?<!l[aio]ng))(e|es|ed|ing)\)/\(\1\2-x\1%\4|B%e \3\4\)/g;

  #### X* -- default no e regular
  ## X*
  s/\(([BVLG])([^ %]*) ([^ ]*?)(|s|ed|ing)\)/\(\1\2-x\1%\4|B% \3\4\)/g;


  #### remove -x tag
  s/\(([BVLGA][^ ]*)-x-([^\(\)]*)\)/\(\1-\2\)/g;

  # #### remove empty morphemes
  # s/ \([^ ]* \*\)//g;

  print $_;
}
