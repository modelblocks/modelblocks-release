
$SHORT = '(?!<[aeiou])(?:[aeiou])';

while ( <> ) {

  ## remove old -lI tag
  s/-lI//g;

  ######## NOUNS:

  #### irregular nouns:
  ## this
  s/\((N)([^ \*]*) ([Tt]his)\)/\(\1\2-xX\L\3\E** \3\)/g;
  ## X*y
  s/\((N(?!-b{N-aD}))([^ \*]*) ([^ ]*?[^aeou])(y|ies)\)/\(\1\2-xX*y*\4 \3\4\)/g;
  ## Xs* (/es/ed/ing)
  s/\((N(?!-b{N-aD}))([^ \*]*) ([^ ]*?(?:ss|focus))(|es)\)/\(\1\2-xXs**\4 \3\4\)/g;
  ## Xz* (/es/ed/ing)
  s/\((N(?!-b{N-aD}))([^ \*]*) ([^ ]*?zz)(|es)\)/\(\1\2-xXz**\4 \3\4\)/g;
  ## Xh* (/es/ed/ing)
  s/\((N(?!-b{N-aD}))([^ \*]*) ([^ ]*?[cs](?<! ac)h)(|es)\)/\(\1\2-xXh**\4 \3\4\)/g;
  ## Xx* (/es/ed/ing)
  s/\((N(?!-b{N-aD}))([^ \*]*) ([^ ]*?x)(|es)\)/\(\1\2-xXx**\4 \3\4\)/g;

  #### regular nouns:
  s/\((N(?!-b{N-aD}))([^ \*]*) ([^ ]*?)(|s)\)/\(\1\2-xX**\4 \3\4\)/g;


  ######## VERBS:

  #### irregular verbs:
  ## arise/rise
  s/\(([BVLG])([^ \*]*) ([Aa]?[Rr])(ise|ises|ose|isen|ising)\)/\(\1\2-xXr*ise*\4 \3\4\)/g;
  ## awake
  s/\(([BVLG])([^ \*]*) ([Aa]?[Ww])(aken|akes|ake|oke|akened|akening)\)/\(\1\2-xXaw*aken*\4 \3\4\)/g;
  ## be
  s/\(([BVLG])([^ \*]*) ()(\'m|\'re|(?<=\{A-a.\} )\'s|[Bb]e|[Aa]m|[Ii]s|[Aa]re|[Ww]as|[Ww]ere|[Bb]een|[Bb]eing)\)/\(\1\2-xX*be*\L\4\E \3\4\)/g;
  ## bear
  s/\(([BVLG])([^ \*]*) ([Bb])(ear|ears|ore|orne|earing)\)/\(\1\2-xXb*ear*\4 \3\4\)/g;
  ## beat
  s/\(([BVLG])([^ \*]*) ([Bb]eat)(|s|en|ing)\)/\(\1\2-xXeat**\4 \3\4\)/g;
  ## begin/spin
  s/\(([BVLG])([^ \*]*) ([Bb]eg|[Ss]p)(in|ins|an|un|inning)\)/\(\1\2-xX*in*\4 \3\4\)/g;
  ## bleed/breed/feed/speed
  s/\(([BVLG])([^ \*]*) ([Bb]le|[Bb]re|[Ff]e|[Ss]pe)(ed|eds|d|eding)\)/\(\1\2-xXe*ed*\4 \3\4\)/g;
  ## blow/grow/know/throw
  s/\(([BVLG])([^ \*]*) ([^ ]*?(?:[Bb]l|[Gg]r|[Tt]hr|[Kk]n))(ow|ows|ew|own|owing)\)/\(\1\2-xX*ow*\4 \3\4\)/g;
  ## bid/rid
  s/\(([BVLG])([^ \*]*) ([^ f]*?[BbRr]id)(|s|ding)\)/\(\1\2-xXid**\4 \3\4\)/g;
  ## break/speak
  s/\(([BVLG])([^ \*]*) ([Bb]r|[Ss]p)(eak|eaks|oke|oken|eaking)\)/\(\1\2-xX*eak*\4 \3\4\)/g;
  ## bring
  s/\(([BVLG])([^ \*]*) ([Bb]r)(ing|ings|ung|ought|inging)\)/\(\1\2-xXbr*ing*\4 \3\4\)/g;
  ## build/rebuild
  s/\(([BVLG])([^ \*]*) ([^ ]*?[Bb]uil)(d|ds|t|ding)\)/\(\1\2-xXbuil*d*\4 \3\4\)/g;
  ## buy
  s/\(([BVLG])([^ \*]*) ([Bb])(uy|uys|ought|uying)\)/\(\1\2-xXb*uy*\4 \3\4\)/g;
  ## catch
  s/\(([BVLG])([^ \*]*) ([Cc])(atch|atches|aught|atching)\)/\(\1\2-xXc*atch*\4 \3\4\)/g;
  ## choose
  s/\(([BVLG])([^ \*]*) ([Cc]ho)(ose|oses|se|sen|osing)\)/\(\1\2-xXcho*ose*\4 \3\4\)/g;
  ## cling/fling/ring/sing/spring/sting/swing/wring
  s/\(([BVLG])([^ \*]*) ([Cc]l|[Ff]l|[Rr]|[Ss]|[Ss]pr|[Ss]t|[Ss]w|[Ww]r)(ing|ings|ang|ung|inging)\)/\(\1\2-xX*ing*\4 \3\4\)/g;
  ## creep/keep/sleep/sweep/weep
  s/\(([BVLG])([^ \*]*) ([Cc]re|[Kk]e|[Ss]le|[Ss]we|[Ww]e)(ep|eps|pt|eping)\)/\(\1\2-xX*ep*\4 \3\4\)/g;
  ## come/become/overcome
  s/\(([BVLG])([^ \*]*) ((?:|[Bb]e|[Oo]ver)[Cc])(ome|omes|ame|omed|oming)\)/\(\1\2-xXc*ome*\4 \3\4\)/g;
  ## deal
  s/\(([BVLG])([^ \*]*) ([Dd]eal)(|s|t|ing)\)/\(\1\2-xXeal**\4 \3\4\)/g;
  ## die/lie/tie (lie as in fib)
  s/\(([BVLG])([^ \*]*) ([DdLlTt])(ie|ies|ied|ying)\)/\(\1\2-xX*ie*\4 \3\4\)/g;
  ## dig
  s/\(([BVLG])([^ \*]*) ([Dd])(ig|igs|ug|igging)\)/\(\1\2-xXd*ig*\4 \3\4\)/g;
  ## do/undo/outdo
  s/\(([BVLG])([^ \*]*) ((?:[Uu]n|[Oo]ut)?[Dd])(o|oes|id|one|oing)\)/\(\1\2-xXd*o*\4 \3\4\)/g;
  ## draw/withdraw
  s/\(([BVLG])([^ \*]*) ([^ ]*?[Dd]r)(aw|aws|ew|awn|awing)\)/\(\1\2-xXdr*aw*\4 \3\4\)/g;
  ## drink/sink/shrink
  s/\(([BVLG])([^ \*]*) ([Dd]r|[Ss]|[Ss]hr)(ink|inks|ank|unk|inking)\)/\(\1\2-xX*ink*\4 \3\4\)/g;
  ## drive/strive
  s/\(([BVLG])([^ \*]*) ([Dd]r|[Ss]tr)(ive|ives|ove|iven|iving)\)/\(\1\2-xXr*ive*\4 \3\4\)/g;
  ## eat
  s/\(([BVLG])([^ \*]*) ()([Ee]at|[Ee]ats|[Aa]te|[Ee]aten|[Ee]ating)\)/\(\1\2-xX*eat*\L\4\E \3\4\)/g;
  ## fall
  s/\(([BVLG])([^ \*]*) ([^ ]*?[Ff])(all|alls|ell|allen|alling)\)/\(\1\2-xXf*all*\4 \3\4\)/g;
  ## feel/kneel
  s/\(([BVLG])([^ \*]*) ([Ff]e|[Kk]ne)(el|els|lt|eling)\)/\(\1\2-xX*el*\4 \3\4\)/g;
  ## fight
  s/\(([BVLG])([^ \*]*) ([Ff])(ight|ights|ought|ighting)\)/\(\1\2-xX*ight*\4 \3\4\)/g;
  ## find/grind
  s/\(([BVLG])([^ \*]*) ([Ff]|[Gg]r)(ind|inds|ound|inding)\)/\(\1\2-xX*ind*\4 \3\4\)/g;
  ## flee
  s/\(([BVLG])([^ \*]*) ([Ff]le)(e|es|d|eing)\)/\(\1\2-xXfle*e*\4 \3\4\)/g;
  ## forbid
  s/\(([BVLG])([^ \*]*) ([Ff]orb)(id|ids|ade|idden|idding)\)/\(\1\2-xXforb*id*\4 \3\4\)/g;
  ## frolic/panic/mimic
  s/\(([BVLG])([^ \*]*) ([Ff]rolic|[Pp]anic|[Mm]imic)(|s|ked|king)\)/\(\1\2-xXic**\4 \3\4\)/g;
  ## freeze
  s/\(([BVLG])([^ \*]*) ([Ff]r)(eeze|eezes|oze|ozen|eezing)\)/\(\1\2-xXfr*eeze*\4 \3\4\)/g;
  ## get/forget
  s/\(([BVLG])([^ \*]*) ((?:[^ ]*[Ff]or)?[Gg])(et|ets|ot|otten|etting)\)/\(\1\2-xXg*et*\4 \3\4\)/g;
  ## give
  s/\(([BVLG])([^ \*]*) ([^ ]*?[Gg])(ive|iveth|ives|ave|iven|iving)\)/\(\1\2-xXg*ive*\4 \3\4\)/g;
  ## go/undergo
  s/\(([BVLG])([^ \*]*) ([^ ]*?)([Gg]o|[Gg]oes|[Ww]ent|[Gg]one|[Gg]oing)\)/\(\1\2-xX*go*\L\4\E \3\4\)/g;
  ## hang/overhang
  s/\(([BVLG])([^ \*]*) ([Hh]|[Oo]verh)(ang|angs|ung|anged|anging)\)/\(\1\2-xXh*ang*\4 \3\4\)/g;
  ## have
  s/\(([BVLG])([^ \*]*) ()((?<=\{L-a.\} )\'d|\'s|\'ve|[Hh]ave|[Hh]as|[Hh]ad|[Hh]aving)\)/\(\1\2-xX*have*\L\4\E \3\4\)/g;
  ## hear
  s/\(([BVLG])([^ \*]*) ([Hh]ear)(|s|d|ing)\)/\(\1\2-xXhear**\4 \3\4\)/g;
  ## hew/sew/strew
  s/\(([BVLG])([^ \*]*) ([HhSs]ew|[Ss]trew)(|s|ed|n|ing)\)/\(\1\2-xXhew**\4 \3\4\)/g;
  ## hide
  s/\(([BVLG])([^ \*]*) ([Hh]id)(e|es||den|ing)\)/\(\1\2-xXhid*e*\4 \3\4\)/g;
  ## hit
  s/\(([BVLG])([^ \*]*) ([Hh]it)(|s||ting)\)/\(\1\2-xXhit**\4 \3\4\)/g;
  ## hold
  s/\(([BVLG])([^ \*]*) ([^ ]*?[Hh])(old|olds|eld|olding)\)/\(\1\2-xXh*old*\4 \3\4\)/g;
  ## lay
  s/\(([BVLG])([^ \*]*) ([Ll])(ay|ays|aid|ain|aying)\)/\(\1\2-xXl*ay*\4 \3\4\)/g;
  ## lead/plead/mislead
  s/\(([BVLG])([^ \*]*) ((?:mis)?[Pp]?[Ll]e)(ad|ads|d|ading)\)/\(\1\2-xXle*ad*\4 \3\4\)/g;
  ## leap/outleap
  s/\(([BVLG])([^ \*]*) ([^ s]*?[Ll]e)(ap|aps|pt|aping)\)/\(\1\2-xXle*ap*\4 \3\4\)/g;
  ## leave
  s/\(([BVLG])([^ \*]*) ([Ll])(eave|eaves|eft|eaving)\)/\(\1\2-xXl*eave*\4 \3\4\)/g;
  ## lend/send/spend
  s/\(([BVLG])([^ \*]*) ((?:[Ll]|[Ss]|[Ss]p)en)(d|ds|t|ding)\)/\(\1\2-xXen*d*\4 \3\4\)/g;
  ## lie (as in recline)
  s/\(([BVLG])([^ \*]*) ([Ll])(ie|ies|ay|ying)\)/\(\1\2-xXl*ie*\4 \3\4\)/g;
  ## light/highlight/spotlight
  s/\(([BVLG])([^ \*]*) ((?:high|moon|spot)?[Ll]i)(ght|ghts|t|ghting)\)/\(\1\2-xXli*ght*\4 \3\4\)/g;
  ## lose
  s/\(([BVLG])([^ \*]*) ([Ll]os)(e|es|t|ing)\)/\(\1\2-xXlos*e*\4 \3\4\)/g;
  ## make
  s/\(([BVLG])([^ \*]*) ([Mm]a)(ke|kes|de|king)\)/\(\1\2-xXma*ke*\4 \3\4\)/g;
  ## mean
  s/\(([BVLG])([^ \*]*) ([Mm]ean)(|s|t|ing)\)/\(\1\2-xXmean**\4 \3\4\)/g;
  ## meet
  s/\(([BVLG])([^ \*]*) ([Mm]e)(et|ets|t|eting)\)/\(\1\2-xXme*et*\4 \3\4\)/g;
  ## pay/say/overpay
  s/\(([BVLG])([^ \*]*) ([^ ]*?[PpSs]a)(y|ys|id|ying)\)/\(\1\2-xXa*y*\4 \3\4\)/g;
  ## prove
  s/\(([BVLG])([^ \*]*) ((?:[Dd]is)?[Pp]rov)(e|es|ed|en|ing)\)/\(\1\2-xXprov*e*\4 \3\4\)/g;
  ## quit
  s/\(([BVLG])([^ \*]*) ([Qq]uit)(|s|ting)\)/\(\1\2-xXquit**\4 \3\4\)/g;
  ## ride
  s/\(([BVLG])([^ \*]*) ([Rr]|[Oo]verr)(ide|ides|ode|idden|iding)\)/\(\1\2-xXr*ide*\4 \3\4\)/g;
  ## run
  s/\(([BVLG])([^ \*]*) ([Rr])(un|uns|an|unning)\)/\(\1\2-xXr*un*\4 \3\4\)/g;
  ## see/oversee
  s/\(([BVLG])([^ \*]*) ([^ ]*?[Ss])(ee|ees|aw|een|eeing)\)/\(\1\2-xXs*ee*\4 \3\4\)/g;
  ## seek
  s/\(([BVLG])([^ \*]*) ([Ss])(eek|eeks|ought|eeking)\)/\(\1\2-xXs*eek*\4 \3\4\)/g;
  ## sell/tell
  s/\(([BVLG])([^ \*]*) ([^ ]*?[SsTt])(ell|ells|old|elling)\)/\(\1\2-xX*ell*\4 \3\4\)/g;
  ## shoot
  s/\(([BVLG])([^ \*]*) ([Ss]ho)(ot|ots|t|otting)\)/\(\1\2-xXsho*ot*\4 \3\4\)/g;
  ## show
  s/\(([BVLG])([^ \*]*) ([^ ]*?(?:[Ss]how))(|s|ed|n|ing)\)/\(\1\2-xXshow**\4 \3\4\)/g;
  ## sit
  s/\(([BVLG])([^ \*]*) ([Ss])(it|its|at|itting)\)/\(\1\2-xXs*it*\4 \3\4\)/g;
  ## slay
  s/\(([BVLG])([^ \*]*) ([Ss]l)(ay|ays|ayed|ain|aying)\)/\(\1\2-xXsl*ay*\4 \3\4\)/g;
  ## sneak
  s/\(([BVLG])([^ \*]*) ([Ss]n)(eak|eaks|uck|eaking)\)/\(\1\2-xXsn*eak*\4 \3\4\)/g;
  ## smite/write
  s/\(([BVLG])([^ \*]*) ([^ ]*?(?:[Ss]m|[Ww]r))(ite|ites|ote|itten|iting)\)/\(\1\2-xX*ite*\4 \3\4\)/g;
  ## stand/understand
  s/\(([BVLG])([^ \*]*) ([^ ]*?[Ss]t)(and|ands|ood|anding)\)/\(\1\2-xXst*and*\4 \3\4\)/g;
  ## steal
  s/\(([BVLG])([^ \*]*) ([Ss]t)(eal|eals|ole|olen|ealing)\)/\(\1\2-xXst*eal*\4 \3\4\)/g;
  ## stick
  s/\(([BVLG])([^ \*]*) ([Ss]t)(ick|icks|uck|icking)\)/\(\1\2-xXst*ick*\4 \3\4\)/g;
  ## strike
  s/\(([BVLG])([^ \*]*) ([Ss]tr)(ike|ikes|uck|icken|iking)\)/\(\1\2-xXstr*ike*\4 \3\4\)/g;
  ## swear/tear
  s/\(([BVLG])([^ \*]*) ([Ss]w|[Tt])(ear|ear|ore|orn|earing)\)/\(\1\2-xX*ear*\4 \3\4\)/g;
  ## forsake/take/shake
  s/\(([BVLG])([^ \*]*) ([^ ]*?(?:[Tt]|[Ss]h|[Ff]ors))(ake|akes|aketh|ook|aken|aking)\)/\(\1\2-xX*ake*\4 \3\4\)/g;
  ## teach
  s/\(([BVLG])([^ \*]*) ([Tt])(each|eaches|aught|eaching)\)/\(\1\2-xXt*each*\4 \3\4\)/g;
  ## think
  s/\(([BVLG])([^ \*]*) ([Tt]h)(ink|inks|ought|inking)\)/\(\1\2-xXth*ink*\4 \3\4\)/g;
  ## tread
  s/\(([BVLG])([^ \*]*) ([Tt]r)(ead|eads|od|eading)\)/\(\1\2-xXtr*ead*\4 \3\4\)/g;
  ## weave
  s/\(([BVLG])([^ \*]*) ([Ww])(eave|eaves|ove|oven|eaving)\)/\(\1\2-xXw*eave*\4 \3\4\)/g;
  ## wreak
  s/\(([BVLG])([^ \*]*) ([Ww]r)(eak|eaks|eaked|ought|eaking)\)/\(\1\2-xXwr*eak*\4 \3\4\)/g;
  ## will
  s/\(([BVLG])([^ \*]*) ()(\'ll|[Ww]ill|[Ww]o)\)/\(\1\2-xX*will*\L\4\E \3\4\)/g;
  ## win
  s/\(([BVLG])([^ \*]*) ([Ww])(in|ins|on|un|inning)\)/\(\1\2-xXw*in*\4 \3\4\)/g;
  ## would
  s/\(([BVLG])([^ \*]*) ()(\'d|[Ww]ould)\)/\(\1\2-xX*would*\L\4\E \3\4\)/g;


  #### irregular in orthography only:
  ## Xd* -- shred/wed/wad
  s/\(([BVLG])([^ \*]*) ([Ss]hred|[Ww]ed|[Ww]ad)(|s|ded|ding)\)/\(\1\2-xXd**\4 \3\4\)/g;
  ## Xl* -- compel/propel/impel/repel
  s/\(([BVLG])([^ \*]*) ([^ ]*pel)(|s|led|ling)\)/\(\1\2-xXl**\4 \3\4\)/g;
  ## Xl* -- control/patrol, not stroll
  s/\(([BVLG])([^ \*]*) ([^ ]*..trol)(|s|led|ling)\)/\(\1\2-xXl**\4 \3\4\)/g;
  ## Xl* -- initial/total
  s/\(([BVLG])([^ \*]*) ([^ ]*(?:ial|[Tt]otal))(|s|led|ling)\)/\(\1\2-xXl**\4 \3\4\)/g;
  ## Xp* -- quip
  s/\(([BVLG])([^ \*]*) ([^ ]*?uip)(|s|ped|ping)\)/\(\1\2-xXp**\4 \3\4\)/g;
  ## Xr* -- infer|deter
  s/\(([BVLG])([^ \*]*) ([Dd]eter|(?<= )aver|[^ ]*[^f]fer|proffer|[^ ]*cur)(|s|red|ring)\)/\(\1\2-xXr**\4 \3\4\)/g;
  ## X* -- alter/bicker/audit/benefit
  s/\(([BVLG])([^ \*]*) ([Aa]lter|[Bb]icker|[Aa]udit|[Bb]enefit)(|s|ed|ing)\)/\(\1\2-xX**\4 \3\4\)/g;
  ## Xshed* (/s/ding) shed
  s/\(([BVLG])([^ \*]*) ([Ss]hed)(|s|ding)\)/\(\1\2-xXshed**\4 \3\4\)/g;
  ## Xtiptoe* (/s/d/ing) tiptoe
  s/\(([BVLG])([^ \*]*) ([Tt]iptoe)(|s|d|ing)\)/\(\1\2-xXtiptoe**\4 \3\4\)/g;
  ## x*e -- breathe/seethe/soothe/loathe/swathe/writhe/ache
  s/\(([BVLG])([^ \*]*) ([^ ]*?(?:breath|eeth|sooth|loath|swath|writh|(?<= )ach))(e|es|ed|ing)\)/\(\1\2-xX*e*\4 \3\4\)/g;
  ## X*e -- waste
  s/\(([BVLG])([^ \*]*) ([Ww]ast)(e|es|ed|ing)\)/\(\1\2-xX*e*\4 \3\4\)/g;


  ## X*y
  s/\(([BVLG])([^ \*]*) ([^ ]*?[^aeou])(y|ies|ied|ying)\)/\(\1\2-xX*y*\4 \3\4\)/g;

  ### double consonant
  ## Xb* (/s/bed/bing)
  s/\(([BVLG])([^ \*]*) ([^ ]*?[^bmr]b)(|s|bed|bing)\)/\(\1\2-xXb**\4 \3\4\)/g;
  ## Xd* (/s/ded/ding)
  s/\(([BVLG])([^ \*]*) ([^ ]*?[^adelnrs'](?<![aeiou][aeiouw])d)(|s|ded|ding)\)/\(\1\2-xXd**\4 \3\4\)/g;
  ## Xg* (/s/ged/ging)
  s/\(([BVLG])([^ \*]*) ([^ ]*?[^gnrs]g)(|s|ged|ging)\)/\(\1\2-xXg**\4 \3\4\)/g;
  ## Xk* (/s/ked/king)
  s/\(([BVLG])([^ \*]*) ([^ ]*?ek)(|s|ked|king)\)/\(\1\2-xXk**\4 \3\4\)/g;
  ## Xl* (/s/led/ling)
  s/\(([BVLG])([^ \*]*) ([^ ]*?(?![aeiou][aeiouw]|e|l|r)l)(|s|led|ling)\)/\(\1\2-xXl**\4 \3\4\)/g;
  ## Xm* (/s/med/ming)
  s/\(([BVLG])([^ \*]*) ([^ ]*?[^lmrs'](?<![aeiou][aeiouw])m)(|s|med|ming)\)/\(\1\2-xXm**\4 \3\4\)/g;
  ## Xn* (/s/ned/ning)
  s/\(([BVLG])([^ \*]*) ([^ ]*?(?:[^aeiou][aiu]n|(?<= )pen| con))(|s|ned|ning)\)/\(\1\2-xXn**\4 \3\4\)/g;
  ## Xp* (/s/ped/ping)
  s/\(([BVLG])([^ \*]*) ([^ ]*?[^lmprs](?<!elo)(?<![aeiou][aeiouw])p)(|s|ped|ping)\)/\(\1\2-xXp**\4 \3\4\)/g;
  ## Xr* (/s/red/ring)
  s/\(([BVLG])([^ \*]*) ([^ ]*?[^eor](?<![aeiou][aeiouw])(?<! pu)r)(|s|red|ring)\)/\(\1\2-xXr**\4 \3\4\)/g;
  ## Xt* (/s/ted/ting)
  s/\(([BVLG])([^ \*]*) ([^ ]*?(?:[^aeiou][aiou]t|cquit|ffset|[fg]ret|abet|(?<= )[blnsBLNS]et)(?<!budget|target|.umpet|.rpret|..i[bcmrs]it|profit))(|s|ted|ting)\)/\(\1\2-xXt**\4 \3\4\)/g;
  ## Xv* (/s/ved/ving)
  s/\(([BVLG])([^ \*]*) ([^ ]*?[^v]v)(|s|ved|ving)\)/\(\1\2-xXv**\4 \3\4\)/g;

  ## Xs* (/es/ed/ing)
  s/\(([BVLG])([^ \*]*) ([^ ]*?(?:ss|focus))(|es|ed|ing)\)/\(\1\2-xXs**\4 \3\4\)/g;
  ## Xz* (/es/ed/ing)
  s/\(([BVLG])([^ \*]*) ([^ ]*?zz)(|es|ed|ing)\)/\(\1\2-xXz**\4 \3\4\)/g;
  ## Xh* (/es/ed/ing)
  s/\(([BVLG])([^ \*]*) ([^ ]*?[cs](?<! ac)h)(|es|ed|ing)\)/\(\1\2-xXh**\4 \3\4\)/g;
  ## Xx* (/es/ed/ing)
  s/\(([BVLG])([^ \*]*) ([^ ]*?x)(|es|ed|ing)\)/\(\1\2-xXx**\4 \3\4\)/g;

  ## Xee*
  s/\(([BVLG])([^ \*]*) ([^ ]*?[^chn]ee)(|s|d|ing)\)/\(\1\2-xXee**\4 \3\4\)/g;


  ## X*e (e|es|ed|ing) -- regular e, that would otherwise qualify as complex vowel or complex consonant
  s/\(([BVLG])([^ \*]*) ([^ ]*?(?:iec|oic|[nrs]c|[st]uad|(?<= )guid|af|ieg|[dlr]g|[bcdfgkptyz]l|ym|sum|[ct]ap|hop|[^s]ip|yp|uar|uir|sur|[aeiou]{2}s(?<!vous|bias)|ens|abus|ccus|mus|[Cc]reat|rmeat|[isu]at|uot|eav|iev|[aeo]lv|[ae]iv|rv|u|(?<= )ow|[es]iz|eez|ooz|yz))(e|es|ed|ing)\)/\(\1\2-xX*e*\4 \3\4\)/g;

  ## X* -- regular no e, with complex vowel or complex consonant
  s/\(([BVLG])([^ \*]*) ([^ ]*?[aeiou]{2,}[bcdfghjklmnpqrstvwxyz](?<!oed))(|s|ed|ing)\)/\(\1\2-xX**\4 \3\4\)/g;
  s/\(([BVLG])([^ \*]*) ([^ ]*?[bcdfghjklmnpqrstvwxyz][bcdfghjklmnpqrtvwxyz](?<![aeiu]ng|.bl))(|s|ed|ing)\)/\(\1\2-xX**\4 \3\4\)/g;

  ## X* -- regular no e, which would otherwise end in single vowel single consonant
  s/\(([BVLG])([^ \*]*) ([^ ]*?(?:log|[dnt]al|el|devil|eril|.[^r]lop|..sip|....[sd]on|.[dhkpst]en|[^r]ven|...[st]om|[ct]hor|..[bdhkmnptvw]er|ffer|nger|swer|censor|o[ln]or|a[bjv]or|rror|[fgkrv]et|osit|[ai]bit|erit|imit|mbast))(|s|ed|ing)\)/\(\1\2-xX**\4 \3\4\)/g;
  s/\(([BVLG])([^ \*]*) ([^ ]*?(?:hbor|(?<= )ping|mme.|gge.|cke.|icit|i[rv]al|ns..|amid|ofit|-bus|iphon|ilor|umpet|isit))(|s|ed|ing)\)/\(\1\2-xX**\4 \3\4\)/g;

  #### long + consonant + e regular
  ## X*e (e|es|ed|ing)
  s/\(([BVLG])([^ \*]*) ([^ ]*?[bcdgklmnprstuvz](?<!l[aio]ng))(e|es|ed|ing)\)/\(\1\2-xX*e*\4 \3\4\)/g;

  #### X* -- default no e regular
  ## X*
  s/\(([BVLG])([^ \*]*) ([^ ]*?)(|s|ed|ing)\)/\(\1\2-xX**\4 \3\4\)/g;


  #### remove -x tag
  s/\(([BVLGA][^ ]*)-x-([^\(\)]*)\)/\(\1-\2\)/g;

  # #### remove empty morphemes
  # s/ \([^ ]* \*\)//g;

  print $_;
}
