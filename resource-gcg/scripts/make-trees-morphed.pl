
use Getopt::Std;

getopts("s");
$SEM = $opt_s;
print( $SEM );

$SHORT = '(?!<[aeiou])(?:[aeiou])';

while ( <> ) {

#  ## remove old -lI tag
#  s/-lI[^\)]* //g;

#  ## lowercase all words -- U (uppercase) category in gcg16 lets us reconstruct capitalization for proper names if we want it
#  s/ ((?!-)[^\(\)]*)\)/ \L\1\E\)/gi;

  ######## N -> N:

  #### irregular nouns:
  ## this
  s/\((N(?!-b{N-aD}))([^ %]*) ([Tt]his|[^ ]*[Ss]pecies|[^ ]*[Ss]eries)\)/\(\1\2-o\1%\3|N%\3 \3\)/gi;
  ## analysis (%is|%es)
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?(?:aeg|bas|chass|bet|cris|gnos|cler|lys|mes|phas|ps|thes))()(is|es)\)/\(\1\2-o\1%\4\5|N%\4is \3\4is\)/gi;
  ## beastie (%|%s)
  s/\((N(?!-b{N-aD}))([^ %]*) (lie|pie|tie|[^ ]*?(?:beastie|calorie|cookie|goodie|movie|prarie|talkie|yippie|yuppie|zombie))()(|s)\)/\(\1\2-o\1%\4\5|N%\4 \3\4\)/gi;
  ## bus (%s|%ses)
  s/\((N(?!-b{N-aD}))([^ %]*) (bu|[^ ]*?(?:s|canva|[^x]cu|iri|nu|pu|plu|tu))(s)(|es)\)/\(\1\2-o\1%\4\5|N%\4 \3\4\)/gi;
  ## echo (%o|%oes)
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?(?:ech|embarg|grott|her|potat|tomat|vet))(o)(|es)\)/\(\1\2-o\1%\4\5|N%\4 \3\4\)/gi;
  ## leaf (%f|%ves)
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?(?:ar|el|ol|[Ll]ea|thei))()(f|ves)\)/\(\1\2-o\1%\4\5|N%\4f \3\4f\)/gi;
  ## life,wife (%fe|%ves)
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?(?:[Ll]i|[Ww]i))()(fe|ves)\)/\(\1\2-o\1%\4\5|N%\4fe \3\4fe\)/gi;
  ## man (%an|%en)
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?(?:[Ww]o|))([Mm])(an|en)\)/\(\1\2-o\1%\4\5|N%\4an \3\4an\)/gi;
  ## %z|%zes
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?z)(z)(|es)\)/\(\1\2-o\1%\4\5|N%\4 \3\4\)/gi;
  ## %h|%hes
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?[cs](?<! ac))(h)(|es)\)/\(\1\2-o\1%\4\5|N%\4 \3\4\)/gi;
  ## %x|%xes
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?)(x)(|es)\)/\(\1\2-o\1%\4\5|N%\4 \3\4\)/gi;
  ## %y|%ies
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?[^aeou])(y|ies)\)/\(\1\2-o\1%\4|N%y \3y\)/gi;
  ## atlas,... (%|%es)
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?(?:autics|bias|bris|botics|enis|ennis|etics|estos|ethos|itis|itics|matics|ntics|o[mn]ics|ssis|stics|thics|tlas|ysics))(|es)\)/\(\1\2-o\1%\4|N% \3\)/gi;

  #### regular nouns:
  s/\((N(?!-b{N-aD}))([^ %]*) ([^ ]*?[^u])(|s)\)/\(\1\2-o\1%\4|N% \3\)/gi;

  ######## ADJECTIVAL NOMINALIZATION  A -> N:

  ## %ate|%acy (adequate)
  s/\((N)([^ ]*) ([^ ]*?adequ)(acy)\)/\(\1\2-oN%\4|A%ate \3ate\)/gi;
  ## %iness|%y (busy,holy)
  s/\((N)([^ ]*) ([^ ]*?)(iness)\)/\(\1\2-oN%\4|A%y \3y\)/gi;
  ## %ness|% (open)
  s/\((N)([^ ]*) ([^ ]*?)(ness)\)/\(\1\2-oN%\4|A% \3\)/gi;
  ## %lty|%l (royal)
  s/\((N)([^ ]*) (?!salty)([^ ]*?)(l)(ty)\)/\(\1\2-oN%\4\5|A%\4 \3\4\)/gi;
  ## %bility|%ble (lovable)
  s/\((N)([^ ]*) ([^ ]*?)(b)(ility)\)/\(\1\2-oN%\4\5|A%\4le \3\4le\)/gi;
#  ## %ity|%e
#  s/\((N)([^ ]*) ([^ ]*?)(ity)\)/\(\1\2-oN%\4|A%e \3e\)/gi;

  ######## ADVERBALIZATION  A -> R:

  ## false cognates: early, only (not to become ear, on)
  s/\((R)([^ %]*) (early|only)()\)/\(A\2-o\1%\4|A% \3\)/gi;
  ## well|good
  s/\((R)([^ %]*) ()(well)\)/\(A\2-o\1%\4|A%good good\)/gi;
  ## %ily|%y (easy)
  s/\((R)([^ %]*) ([^ ]*?)(ily)\)/\(A\2-o\1%\4|A%y \3y\)/gi;
  ## %[bgpt]ly|%[bgpt]le (probable,simple,single, not cheap)
  s/\((R)([^ %]*) ([^ ]*?[^aeiou][aeiou]|[^ ]*am|[^ ]*ia|[^ ]*ou|si[mn]|[^ ]*ua)(btl|[bgp]l)(y)\)/\(A\2-o\1%\4\5|A%\4e \3\4e\)/gi;
  ## %uly|%e (duly|truly|wholly)
  s/\((R)([^ %]*) ([^ ]*d|tr|wh)(u|ol)(ly)\)/\(A\2-o\1%\4\5|A%\4e \3\4e\)/gi;
  ## %lly|%ll (fully)
  s/\((R)([^ %]*) (fu)(ll)(y)\)/\(A\2-o\1%\4\5|A%\4 \3\4\)/gi;
  ## R%ly|A% -- note A% is not unique
  s/\((R)([^ %]*) ([^ ]*?)(ly)\)/\(A\2-o\1%\4|A% \3\)/gi;
  ## simpliciter adverbs -- note A% is not unique
  s/\((R)([^ %]*) ([^ ]*?)()\)/\(A\2-o\1%\4|A% \3\)/gi;

  ######## A -> A COMPARATIVE:

  ## better, worse
  s/\((A)([^ ]*) ()([Bb]etter)\)/\(\1\2-oA%\4|A%good \3good\)/gi;
  s/\((A)([^ ]*) ()([Ww]orse)\)/\(\1\2-oA%\4|A%bad \3bad\)/gi;
  s/\((A)([^ ]*) ([Ff])(arther|urther)\)/\(\1\2-oA%\4|A%ar \3ar\)/gi;
  ## %der comparatives
  s/\((A)([^ ]*) ([^ ]*?)(d)(der)\)/\(\1\2-o\1%\4\5|A%\4 \3\4\)/gi;
  ## %ger comparatives
  s/\((A)([^ ]*) ([^ ]*?)(g)(ger)\)/\(\1\2-o\1%\4\5|A%\4 \3\4\)/gi;
  ## %ner comparatives
  s/\((A)([^ ]*) (?!inner)([^ ]*?)(n)(ner)\)/\(\1\2-o\1%\4\5|A%\4 \3\4\)/gi;
  ## %ter comparatives
  s/\((A)([^ ]*) (?!latter|utter)([^ ]*?)(t)(ter)\)/\(\1\2-o\1%\4\5|A%\4 \3\4\)/gi;
  ## %ier comparatives
  s/\((A)([^ ]*) (?![cf]ourier|cavalier)([^ ]*?)(ier)\)/\(\1\2-o\1%\4|A%y \3y\)/gi;
  ## %r comparatives
  s/\((A)([^ ]*) (?![^ ]*[Gg]reat|[Uu]nderwat|[Ww]ein)([^ ]*?(?:af|am|arg|at|av|fre|in|os|pl|tl|ur))(e)(r)\)/\(\1\2-o\1%\4\5|A%\4 \3\4\)/gi;
  ## %er comparatives
  s/\((A)([^ ]*) (?!after|another|cavalier|courier|eager|n?either|elder|[^ ]*ever|fourier|filler|former|fourier|gender|hyper|inner|latter|lavender|[^ ]*luster|[^ ]*other|outer|[^ ]*over|order|polyester|per|[^ ]*proper|rubber|rather|sheer|sinister|silver|sober|somber|summer|super|tender|[^ ]*together|under|underwater|upper|utter|whether|computer|meter|weiner|winter)([^ ]*?)(er)\)/\(\1\2-o\1%\4|A% \3\)/gi;

  ######## A -> A SUPERLATIVE:

  ## best, worst
  s/\((A)([^ ]*) ()(best)\)/\(\1\2-oA%\4|A%good \3good\)/gi;
  s/\((A)([^ ]*) ()(worst)\)/\(\1\2-oA%\4|A%bad \3bad\)/gi;
  s/\((A)([^ ]*) ([Ff])(arthest|urthest)\)/\(\1\2-oA%\4|A%ar \3ar\)/gi;
  ## %dest superlatives
  s/\((A)([^ ]*) ([^ ]*?)(d)(dest)\)/\(\1\2-o\1%\4\5|A%\4 \3\4\)/gi;
  ## %gest superlatives
  s/\((A)([^ ]*) ([^ ]*?)(g)(gest)\)/\(\1\2-o\1%\4\5|A%\4 \3\4\)/gi;
  ## %nest superlatives
  s/\((A)([^ ]*) ([^ ]*?)(n)(nest)\)/\(\1\2-o\1%\4\5|A%\4 \3\4\)/gi;
  ## %test superlatives
  s/\((A)([^ ]*) ([^ ]*?)(t)(test)\)/\(\1\2-o\1%\4\5|A%\4 \3\4\)/gi;
  ## %iest superlatives
  s/\((A)([^ ]*) ([^ ]*?)(iest)\)/\(\1\2-o\1%\4|A%y \3y\)/gi;
  ## %st superlatives
  s/\((A)([^ ]*) (?![^ ]*[Gg]reat)([^ ]*?(?:af|am|arg|at|av|fre|in|os|pl|tl|ur))(e)(st)\)/\(\1\2-o\1%\4\5|A%\4 \3\4\)/gi;
  ## %est superlatives
  s/\((A)([^ ]*) (?!earnest|[^ ]*honest|lest|manifest|[^ ]*modest|[^ ]*west)([^ ]*?)(est)\)/\(\1\2-o\1%\4|A% \3\)/gi;

  ######## A -> A NEGATIVE

  $NOTUN = '(?!canny|der\)|dercut|derlie|derline|derly|derpin|derscore|derstand|dertake|ited?\)|iversal|til\)|less\)|iqu)';

  ## un%
  s/\((A)([^ ]*) ([Uu]n)$NOTUN([^ ]*?)()\)/\(\1\2-o\1%:\3%|\1NEG%:% \4\)/gi;

  ######## DEVERBAL NOMINALIZATIONS B -> N:

  ## %lysis|%lyze (analyze,electrolyze)
  s/\((N)([^ ]*) ([^ ]*ana|[^ ]*ectro)(ly)(sis)\)/\(B\2-o\1%\4\5|BNOM%\4ze \3\4ze\)/gi;
  ## %asis|%ase (base)
  s/\((N)([^ ]*) ([Bb])(as)(is)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %irth|%ear (bear)
  s/\((N)([^ ]*) ([Bb])()(irth)\)/\(B\2-o\1%\4\5|BNOM%\4ear \3\4ear\)/gi;
  ## %lief|%lieve (believe,relieve)
  s/\((N)([^ ]*) ([% ]*)(lie)(ve)\)/\(B\2-o\1%\4\5|BNOM%\4f \3\4f\)/gi;
  ## %eath|%eathe (breathe)
  s/\((N)([^ ]*) ([Bb]r)(eath)(e)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %oice|%oose (choose)
  s/\((N)([^ ]*) ([Cc]h)(o)(ice)\)/\(B\2-o\1%\4\5|BNOM%\4ose \3\4ose\)/gi;
  ## %osure|%ose (close,compose)
  s/\((N)([^ ]*) ([^ ]*)(os)(ure)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %parison|%pare (compare)
  s/\((N)([^ ]*) ([Cc]om)(par)(ison)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %aint|%ain (complain,constrain,restrain)
  s/\((N)([^ ]*) ([^ ]*(?:pl|str))(ain)(t)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %ormity|%orm (conform)
  s/\((N)([^ ]*) ([^ ]*onf)(orm)(ity)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %cism|%cise (criticize)
  s/\((N)([^ ]*) ([^ ]*iti)(cis)(m)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %livery|%liver (deliver)
  s/\((N)([^ ]*) ([^ ]*)(liver)(y)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %eath|%ie (die)
  s/\((N)([^ ]*) ([Dd])()(eath)\)/\(B\2-o\1%\4\5|BNOM%\4ie \3\4ie\)/gi;
  ## %overy|%over (discover,recover)
  s/\((N)([^ ]*) ([^ ]*c)(over)(y)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %sis|%size (emphasize,metamorphasize)
  s/\((N)([^ ]*) ([^ ]*pha)(si)(s)\)/\(B\2-o\1%\4\5|BNOM%\4ze \3\4ze\)/gi;
  ## %ailure|%ail (fail)
  s/\((N)([^ ]*) ([^ ]*)(ail)(ure)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %light|%ly (fly)
  s/\((N)([^ ]*) ([Ff])(l)(ight)\)/\(B\2-o\1%\4\5|BNOM%\4y \3\4y\)/gi;
  ## %light|%lee (flee)
  s/\((N)([^ ]*) ([Ff])(l)(ight)\)/\(B\2-o\1%\4\5|BNOM%\4ee \3\4ee\)/gi;
  ## %ift|%ive (give)
  s/\((N)([^ ]*) ([Gg])(i)(ft)\)/\(B\2-o\1%\4\5|BNOM%\4ve \3\4ve\)/gi;
  ## %rowth|%row (grow)
  s/\((N)([^ ]*) ([Gg])(row)(th)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %owledge|%ow (know)
  s/\((N)([^ ]*) ([Kk]n)(ow)(ledge)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %eadership|%ead (lead)
  s/\((N)([^ ]*) ([Ll])(ead)(ership)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %ife|%ive (live)
  s/\((N)([^ ]*) ([Ll])(i)(fe)\)/\(B\2-o\1%\4\5|BNOM%\4ve \3\4ve\)/gi;
  ## %oss|%ose (lose)
  s/\((N)([^ ]*) ([Ll])(os)(s)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %rriage|%rry (marry)
  s/\((N)([^ ]*) ([Mm]a)(rr)(iage)\)/\(B\2-o\1%\4\5|BNOM%\4y \3\4y\)/gi;
  ## %orship|%or (mentor)
  s/\((N)([^ ]*) ([Mm]entor|sponsor)()(ship)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %erger|%erge (merge)
  s/\((N)([^ ]*) ([Mm])(erge)(r)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %lea|%lead (plead)
  s/\((N)([^ ]*) ([Pp])(lea)(d)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %leasure|%lease (please)
  s/\((N)([^ ]*) ([Pp])(leas)(ure)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %perity|%per (prosper)
  s/\((N)([^ ]*) ([Pp]ros)(per)(ity)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %roof|%rove (prove)
  s/\((N)([^ ]*) ([Pp])(ro)(of)\)/\(B\2-o\1%\4\5|BNOM%\4ve \3\4ve\)/gi;
  ## %uit|%ue (pursue,sue)
  s/\((N)([^ ]*) (?![Ll]aw)([^ ]*[Ss])(u)(it)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %sponce|%spond (respond)
  s/\((N)([^ ]*) ([Rr]e)(spon)(ce)\)/\(B\2-o\1%\4\5|BNOM%\4d \3\4d\)/gi;
  ## %iezure|%ieze (sieze)
  s/\((N)([^ ]*) ([Ss])(iez)(ure)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %ale|%ell (sell,tell)
  s/\((N)([^ ]*) ([SsTt])()(ale)\)/\(B\2-o\1%\4\5|BNOM%\4ell \3\4ell\)/gi;
  ## %ervice|%erve (serve)
  s/\((N)([^ ]*) ([Ss])(erv)(ice)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %ot|%oot (shoot)
  s/\((N)([^ ]*) ([Ss]h)(o)(t)\)/\(B\2-o\1%\4\5|BNOM%\4ot \3\4ot\)/gi;
  ## %ong|%ing (sing)
  s/\((N)([^ ]*) ([Ss])()(ong)\)/\(B\2-o\1%\4\5|BNOM%\4ing \3\4ing\)/gi;
  ## %peech|%eak (speak)
  s/\((N)([^ ]*) ([Ss])(pe)(ech)\)/\(B\2-o\1%\4\5|BNOM%\4ak \3\4ak\)/gi;
  ## %timony|%tify (testify)
  s/\((N)([^ ]*) ([Tt]es)(ti)(mony)\)/\(B\2-o\1%\4\5|BNOM%\4fy \3\4fy\)/gi;
  ## %ought|%ink (think)
  s/\((N)([^ ]*) ([Tt]h)()(ought)\)/\(B\2-o\1%\4\5|BNOM%\4ink \3\4ink\)/gi;
  ## %nion|%nite (union)
  s/\((N)([^ ]*) ([Uu])(ni)(on)\)/\(B\2-o\1%\4\5|BNOM%\4te \3\4te\)/gi;

  ## %dgment|%dge (judge)
  s/\((N)([^ ]*) ([^ ]*?[Jj]u)(dg)(ment)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %mament|%m (arm)
  s/\((N)([^ ]*) ([^ ]*?)(m)(ament)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %ument|%ue (argue)
  s/\((N)([^ ]*) ([^ ]*?[Aa]rg)(u)(ment)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %ment|% (improve)
  s/\((N)([^ ]*) ([^ ]*?(?:[Aa]bate|edge|ess|ieve|ise|unce|ange|djust|dorse|gree|rm|etter|ppoint|urtail|evelop|iscern|mploy|ngage|nroll|arass|mprove|nfringe|ndict|nstall|nvest|agage|ove|pay|rage|ocure|ish|nforce|equire|lace|etire|ettle|tate|ship|reat))()(ment)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;

  ## %ayal|%ay (portray)
  s/\((N)([^ ]*) ([^ ]*?portr)(ay)(al)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %ttal|%t (qcquit)
  s/\((N)([^ ]*) ([^ ]*?cqui)(t)(tal)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %ial|%y (try,retry)
  s/\((N)([^ ]*) ((?:[Rr]e)?[Tt]r)()(ial)\)/\(B\2-o\1%\4\5|BNOM%\4y \3\4y\)/gi;
  ## %sal|%se (arouse,espouse,reverse)
  s/\((N)([^ ]*) ([^ ]*?(?:rou|spou|ever))(s)(al)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %ssal|%ss (dismiss)
  s/\((N)([^ ]*) ([^ ]*?(?:dismi))(ss)(al)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %tal|%te (recite)
  s/\((N)([^ ]*) ([^ ]*?(?:reci))(t)(al)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %val|%ve (approve,arrive,revive,survive)
  s/\((N)([^ ]*) ([^ ]*?(?:ppro|rri|evi|urvi))(v)(al)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %wal|%w (renew,withdraw)
  s/\((N)([^ ]*) ([^ ]*?(?:ene|ithdra))(w)(al)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;

  ## %rance|%re (assure)
  s/\((N)([^ ]*) ([^ ]*?(?:ssur))()(ance)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %er|%rance (enter)
  s/\((N)([^ ]*) ([^ ]*?(?:nt))()(rance)\)/\(B\2-o\1%\4\5|BNOM%\4er \3\4er\)/gi;
  ## %nce|%nd (defend,respond)
  s/\((N)([^ ]*) ([^ ]*?(?:efe|respo))(n)(ce)\)/\(B\2-o\1%\4\5|BNOM%\4d \3\4d\)/gi;
  ## %iance|%y (comply,rely,vary)
  s/\((N)([^ ]*) ([^ ]*?(?:ompl|el|ar))()(iance)\)/\(B\2-o\1%\4\5|BNOM%\4y \3\4y\)/gi;
  ## %ence|%e (cohere,diverge,interfere,reside)
  s/\((N)([^ ]*) ([^ ]*?(?:oher|nterfer|iverg|resid))()(ence)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %enance|%ain (maintain)
  s/\((N)([^ ]*) ([^ ]*?(?:aint))()(enance)\)/\(B\2-o\1%\4\5|BNOM%\4ain \3\4ain\)/gi;
  ## %ance|% (allow)
  s/\((N)([^ ]*) ([^ ]*?(?:llow|ppear|ssist|isturb|esist|erform))()(ance)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %ence|% (exist)
  s/\((N)([^ ]*) ([^ ]*?(?:oincid|orrespond|onfer|(?<!nd)epend|iffer|xist|nsist|ccur|ersist))()(ence)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;

  ## %llation|%l (cancel)
  s/\((N)([^ ]*) ([^ ]*?(?:cance))(l)(lation)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %ssation|%ase (cease)
  s/\((N)([^ ]*) ([^ ]*?(?:ce))()(ssation)\)/\(B\2-o\1%\4\5|BNOM%\4ase \3\4ase\)/gi;
  ## %anation|%ain (explain)
  s/\((N)([^ ]*) ([^ ]*?(?:expl))(a)(nation)\)/\(B\2-o\1%\4\5|BNOM%\4in \3\4in\)/gi;
  ## %otation|%oat (float)
  s/\((N)([^ ]*) ([^ ]*?(?:fl))(o)(tation)\)/\(B\2-o\1%\4\5|BNOM%\4at \3\4at\)/gi;
  ## %pation|%py (occupy)
  s/\((N)([^ ]*) ([^ ]*?(?:occu))(p)(ation)\)/\(B\2-o\1%\4\5|BNOM%\4y \3\4y\)/gi;
  ## %amaition|%aim (proclaim)
  s/\((N)([^ ]*) ([^ ]*?(?:procl))()(amation)\)/\(B\2-o\1%\4\5|BNOM%\4aim \3\4aim\)/gi;
  ## %lation|%al (reveal)
  s/\((N)([^ ]*) ([^ ]*?(?:reve))()(lation)\)/\(B\2-o\1%\4\5|BNOM%\4al \3\4al\)/gi;
  ## %ration|%er (sequester)
  s/\((N)([^ ]*) ([^ ]*?(?:sequest))()(ration)\)/\(B\2-o\1%\4\5|BNOM%\4er \3\4er\)/gi;
  ## %ssion|%ed (succeed)
  s/\((N)([^ ]*) ([^ ]*?(?:succe))()(ssion)\)/\(B\2-o\1%\4\5|BNOM%\4ed \3\4ed\)/gi;
  ## %lution|%lve (evolve,solve)
  s/\((N)([^ ]*) ([^ ]*?(?:[Ssv]o))(l)(ution)\)/\(B\2-o\1%\4\5|BNOM%\4ve \3\4ve\)/gi;
  ## %[st]ion|%d (attend,expand,extend,intend,suspend)
  s/\((N)([^ ]*) ([^ ]*?(?:atten|expan|exten|inten|suspen))()([st]ion)\)/\(B\2-o\1%\4\5|BNOM%\4d \3\4d\)/gi;
  ## %etition|%ete (competition)
  s/\((N)([^ ]*) ([^ ]*?(?:omp))(et)(ition)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %ddition|%dd (addition)
  s/\((N)([^ ]*) ([^ ]*?)(dd)(ition)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %lication|%lish (publish)
  s/\((N)([^ ]*) ([^ ]*?b)(li)(cation)\)/\(B\2-o\1%\4\5|BNOM%\4sh \3\4sh\)/gi;
  ## %eption|%eive (conceive,perceive)
  s/\((N)([^ ]*) ([^ ]*?[^x]c)(e)(ption)\)/\(B\2-o\1%\4\5|BNOM%\4ive \3\4ive\)/gi;
  ## %umption|%ume (assume,consume)
  s/\((N)([^ ]*) ([^ ]*?s)(um)(ption)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %ention|%ene (convene,intervene)
  s/\((N)([^ ]*) ([^ ]*?erv)(en)(tion)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %faction|%fy (satisfaction)
  s/\((N)([^ ]*) ([^ ]*?)((?<! )f)(action)\)/\(B\2-o\1%\4\5|BNOM%\4y \3\4y\)/gi;
  ## %tion|%t (contract)
  s/\((N)([^ ]*) (?!faction|fiction|fraction|friction|jurisdiction|section)([^ ]*?(?:ac|ibi|bor|eac|ec|ep|ic|ser|ven|ruc|up))(t)(ion)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %cession|%cede (concede,recede)
  s/\((N)([^ ]*) ([^ ]*?(?:))(ce)(ssion)\)/\(B\2-o\1%\4\5|BNOM%\4de \3\4de\)/gi;
  ## %ssion|%ss (discuss,obsess,profess,possess)
  s/\((N)([^ ]*) ([^ ]*?(?:bse|fe|re|scu|sse))(ss)(ion)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %ration|%er (administer,register)
  s/\((N)([^ ]*) ([^ ]*?(?:ist))()(ration)\)/\(B\2-o\1%\4\5|BNOM%\4er \3\4er\)/gi;
  ## %ission|%it (admit,emit,omit,permit)
  s/\((N)([^ ]*) ([^ ]*?(?:[deor]m))(i)(ssion)\)/\(B\2-o\1%\4\5|BNOM%\4t \3\4t\)/gi;
  ## %ption|%be (subscribe)
  s/\((N)([^ ]*) ([^ ]*?(?:scri))()(ption)\)/\(B\2-o\1%\4\5|BNOM%\4be \3\4be\)/gi;
  ## %ction|%ce (introduce,produce,reduce)
  s/\((N)([^ ]*) ([^ ]*?(?:du))(c)(tion)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %sion|%de (decide,explode,evade,provide,persuade)
  s/\((N)([^ ]*) ([^ ]*?(?:deci|ivi|clu|lo|ovi|ro|va|ua))()(sion)\)/\(B\2-o\1%\4\5|BNOM%\4de \3\4de\)/gi;
  ## %mption|%em (redeem)
  s/\((N)([^ ]*) ([^ ]*?(?:rede))()(mption)\)/\(B\2-o\1%\4\5|BNOM%\4em \3\4em\)/gi;
  ## %llion|%l (rebel)
  s/\((N)([^ ]*) ([^ ]*?(?:rebe))(l)(lion)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %sition|%re (acquire)
  s/\((N)([^ ]*) ([^ ]*?(?:acqui))()(sition)\)/\(B\2-o\1%\4\5|BNOM%\4re \3\4re\)/gi;
  ## %sition|%se (oppose)
  s/\((N)([^ ]*) ([^ ]*?(?:oppo))(s)(ition)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %sion|%se (fuse,revise)
  s/\((N)([^ ]*) ([^ ]*?(?:fu|evi))(s)(ion)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %nition|%nize (recognize)
  s/\((N)([^ ]*) ([^ ]*?(?:ecog))(ni)(tion)\)/\(B\2-o\1%\4\5|BNOM%\4ze \3\4ze\)/gi;
  ## %ication|%y (apply,classify,imply,multiply)
  s/\((N)([^ ]*) ([^ ]*?(?:if|mpl|tipl|ppl|[^u]pl))()(ication)\)/\(B\2-o\1%\4\5|BNOM%\4y \3\4y\)/gi;
  ## %[ou]tion|%[ou]te (contribute,distribute,emote,pollute,promote,prosecute)
  s/\((N)([^ ]*) ([^ ]*?(?:c|[Ee]m|ib|om|oll))([ou]t)(ion)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %lsion|%lse (convulse)
  s/\((N)([^ ]*) ([^ ]*?(?:u))(ls)(ion)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %inition|%ine (define,refine)
  s/\((N)([^ ]*) ([^ ]*?)(in)(ition)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## %iation|%y (vary)
  s/\((N)([^ ]*) ([^ ]*?(?:var))()(iation)\)/\(B\2-o\1%\4\5|BNOM%\4y \3\4y\)/gi;
  ## %ation|% (limit)
  s/\((N)([^ ]*) ([^ ]*?(?:demn|est|firm|[Ff]orm|front|port|[Ll]imit|lant|empt|ider|pect|pret|ound|ment|mend|[Rr]esign|sent|stall|sult|icit|tard|surp|x))()(ation)\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;
  ## %ation|%e (combine)
  s/\((N)([^ ]*) ([^ ]*?(?:[^t]amin|bin|clar|clin|cit|lleg|[dr]evalu|determin|dispens|fam|figur|grad|inton|is|iz|not|nsol|pil|pir|plor|quot|prepar|riv|rs|rv|sens|stor|tinu|[^e]valu|vit|ut))()(ation)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;
  ## ation|%ate (satiate)
  s/\((N)([^ ]*) (?!constellation|corporation|destination|indignation|nation|ostentation|reparation|salvation|[^ ]*station|trepidation)([^ ]*?)(at)(ion)\)/\(B\2-o\1%\4\5|BNOM%\4e \3\4e\)/gi;

  $NOMINALS = '(account|advance|aid|aim|alarm|answer|appeal|arrest|audition|bail|balance|bargain|bend|benefit|bet|bid|bite|blame|blunder|blur|bounce|bow|branch|break|brew|bribe|bumble|burn|buzz|call|care|challenge|change|chat|cheat|check|cheer|chew|clash|claim|climb|cling|close|collapse|combat|comment|compromise|consent|control|cost|count|cover|crack|crash|crawl|creak|crest|crumble|crust|cry|cut|dance|deal|debut|decrease|default|defect|demand|deposit|design|dip|dislike|display|drift|drink|drive|drop|dump|ease|ebb|edge|end|escape|estimate|exit|fade|fall|favor|fear|fight|find|finish|fit|flash|flinch|flip|float|flow|focus|fold|freeze|fret|frolic|gain|gamble|glaze|glide|gnaw|grimace|guess|hang|help|hint|hit|hold|holler|homer|hum|hurt|increase|influence|issue|joke|jump|knock|kowtow|lack|laugh|leap|leapfrog|limit|loan|look|mail|maneuver|manufacture|mesh|miss|moan|murder|offer|order|output|overbid|override|panic|pass|pay|peak|pick|pinch|plummet|plunge|pop|pose|practice|premiere|press|profit|promise|pull|pump|punch|purchase|push|quarrel|quote|rage|rain|rally|range|rank|reach|rebound|record|refocus|reform|rehash|release|remark|request|renege|reply|report|resort|rest|result|retreat|return|review|ride|rise|roll|roost|row|rule|run|rush|sanction|save|scream|scurry|search|shift|shine|shiver|skid|skim|slide|slip|slog|slump|smoke|sound|spin|split|spread|stampede|start|stay|step|stop|strain|stray|strike|study|stumble|supply|support|surge|survey|sway|swell|switch|take|talk|taste|tick|touch|trade|transfer|travel|trend|tumble|turn|twist|twitch|use|veto|view|vote|wade|wail|wait|walk|want|waste|watch|win|wonder|work|worry|yearn|yield|zoom)';
  s/\((N)([^ ]*) $NOMINALS()()\)/\(B\2-o\1%\4\5|BNOM%\4 \3\4\)/gi;

  s/BNOM/B/g;

  ######## B -> V|B|L|G:

  #### irregular verbs:
  ## arise/rise
  s/\(([BVLG])([^ %]*) ([Aa]?)([Rr])(ise|ises|ose|isen|ising)\)/\(B\2-o\1%\4\5|B%\4ise \3\4ise\)/gi;
  ## awake
  s/\(([BVLG])([^ %]*) ([Aa]?)([Ww])(aken|akes|ake|oke|akened|akening)\)/\(B\2-o\1%\4\5|B%\4aken \3\4aken\)/gi;
  ## be
  s/\(([BVLG])([^ %]*) ()(\'m|\'re|(?<=\{A-a.\} )\'s|[Bb]e|[Aa]m|[Ii]s|[Aa]re|[Ww]as|[Ww]ere|[Bb]een|[Bb]eing)\)/\(B\2-o\1%\4|B%be \3be\)/gi;
  ## bear
  s/\(([BVLG])([^ %]*) ([Bb])(ear|ears|ore|orne|earing)\)/\(B\2-o\1%\3\4|B%\3ear \3ear\)/gi;
  ## beat
  s/\(([BVLG])([^ %]*) ([Bb]eat)(|s|en|ing)\)/\(B\2-o\1%\3\4|B%\3 \3\)/gi;
  ## begin/spin
  s/\(([BVLG])([^ %]*) ([Bb]eg|[Ss]p)(in|ins|an|un|inning)\)/\(B\2-o\1%\4|B%in \3in\)/gi;
  ## bleed/breed/feed/speed
  s/\(([BVLG])([^ %]*) ([Bb]l|[Bb]r|[Ff]|[Ss]p)(e)(ed|eds|d|eding)\)/\(B\2-o\1%\4\5|B%\4ed \3\4ed\)/gi;
  ## blow/grow/know/throw
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:[Bb]l|[Gg]r|[Tt]hr|[Kk]n))(ow|ows|ew|own|owing)\)/\(B\2-o\1%\4|B%ow \3ow\)/gi;
  ## bid/rid
  s/\(([BVLG])([^ %]*) ([^ f]*?[BbRr])(id)(|s|ding)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## break/speak
  s/\(([BVLG])([^ %]*) ([Bb]r|[Ss]p)(eak|eaks|oke|oken|eaking)\)/\(B\2-o\1%\4|B%eak \3eak\)/gi;
  ## bring
  s/\(([BVLG])([^ %]*) ([Bb]r)(ing|ings|ung|ought|inging)\)/\(B\2-o\1%\3\4|B%\3ing \3ing\)/gi;
  ## build/rebuild
  s/\(([BVLG])([^ %]*) ([^ ]*?)([Bb]uil)(d|ds|t|ding)\)/\(B\2-o\1%\4\5|B%\4d \3\4d\)/gi;
  ## buy
  s/\(([BVLG])([^ %]*) ([Bb])(uy|uys|ought|uying)\)/\(B\2-o\1%\3\4|B%\3uy \3uy\)/gi;
  ## catch
  s/\(([BVLG])([^ %]*) ([Cc])(atch|atches|aught|atching)\)/\(B\2-o\1%\3\4|B%\3atch \3atch\)/gi;
  ## choose
  s/\(([BVLG])([^ %]*) ([Cc]ho)(ose|oses|se|sen|osing)\)/\(B\2-o\1%\3\4|B%\3ose \3ose\)/gi;
  ## cling/fling/ring/sing/spring/sting/swing/wring
  s/\(([BVLG])([^ %]*) ([Cc]l|[Ff]l|[Rr]|[Ss]|[Ss]pr|[Ss]t|[Ss]w|[Ww]r)(ing|ings|ang|ung|inging)\)/\(B\2-o\1%\4|B%ing \3ing\)/gi;
  ## creep/keep/sleep/sweep/weep
  s/\(([BVLG])([^ %]*) ([Cc]re|[Kk]e|[Ss]le|[Ss]we|[Ww]e)(ep|eps|pt|eping)\)/\(B\2-o\1%\4|B%ep \3ep\)/gi;
  ## come/become/overcome
  s/\(([BVLG])([^ %]*) (|[Bb]e|[Oo]ver)([Cc])(ome|omes|ame|omed|oming)\)/\(B\2-o\1%\4\5|B%\4ome \3\4ome\)/gi;
  ## deal
  s/\(([BVLG])([^ %]*) ([Dd])(eal)(|s|t|ing)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## belie/die/lie/tie/vie (lie as in fib)
  s/\(([BVLG])([^ %]*) (bel|[dltv])(ie|ies|ied|ying)\)/\(B\2-o\1%\4|B%ie \3ie\)/gi;
  ## dig
  s/\(([BVLG])([^ %]*) (d)(ig|igs|ug|igging)\)/\(B\2-o\1%\3\4|B%\3ig \3ig\)/gi;
  ## do/undo/outdo
  s/\(([BVLG])([^ %]*) ([Uu]n|[Oo]ut)?([Dd])(o|oes|id|one|oing)\)/\(B\2-o\1%\4\5|B%\4o \3\4o\)/gi;
  ## draw/withdraw
  s/\(([BVLG])([^ %]*) ([^ ]*?)([Dd]r)(aw|aws|ew|awn|awing)\)/\(B\2-o\1%\4\5|B%\4aw \3\4aw\)/gi;
  ## drink/sink/shrink
  s/\(([BVLG])([^ %]*) ([Dd]r|[Ss]|[Ss]hr)(ink|inks|ank|unk|inking)\)/\(B\2-o\1%\4|B%ink \3ink\)/gi;
  ## drive/strive
  s/\(([BVLG])([^ %]*) ([Dd]|[Ss]t)(r)(ive|ives|ove|iven|iving)\)/\(B\2-o\1%\4\5|B%\4ive \3\4ive\)/gi;
  ## eat
  s/\(([BVLG])([^ %]*) ()([Ee]at|[Ee]ats|[Aa]te|[Ee]aten|[Ee]ating)\)/\(B\2-o\1%\4|B%eat \3eat\)/gi;
  ## fall
  s/\(([BVLG])([^ %]*) ([^ ]*)?([Ff])(all|alls|ell|allen|alling)\)/\(B\2-o\1%\4\5|B%\4all \3\4all\)/gi;
  ## feel/kneel
  s/\(([BVLG])([^ %]*) ([Ff]e|[Kk]ne)(el|els|lt|eling)\)/\(B\2-o\1%\4|B%el \3el\)/gi;
  ## fight
  s/\(([BVLG])([^ %]*) ([Ff])(ight|ights|ought|ighting)\)/\(B\2-o\1%\4|B%ight \3ight\)/gi;
  ## find/grind
  s/\(([BVLG])([^ %]*) ([Ff]|[Gg]r)(ind|inds|ound|inding)\)/\(B\2-o\1%\4|B%ind \3ind\)/gi;
  ## flee
  s/\(([BVLG])([^ %]*) ([Ff]le)(e|es|d|eing)\)/\(B\2-o\1%\3\4|B%\3e \3e\)/gi;
  ## forbid
  s/\(([BVLG])([^ %]*) ([Ff]orb)(id|ids|ade|idden|idding)\)/\(B\2-o\1%\3\4|B%\3id \3id\)/gi;
  ## frolic/panic/mimic
  s/\(([BVLG])([^ %]*) ([Ff]rol|[Pp]an|[Mm]im)(ic)(|s|ked|king)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## freeze
  s/\(([BVLG])([^ %]*) ([Ff]r)(eeze|eezes|oze|ozen|eezing)\)/\(B\2-o\1%\3\4|B%\3eeze \3eeze\)/gi;
  ## get/forget
  s/\(([BVLG])([^ %]*) ([^ ]*[Ff]or)?([Gg])(et|ets|ot|otten|etting)\)/\(B\2-o\1%\4\5|B%\4et \3\4et\)/gi;
  ## give
  s/\(([BVLG])([^ %]*) ([^ ]*?)([Gg])(ive|iveth|ives|ave|iven|iving)\)/\(B\2-o\1%\4\5|B%\4ive \3\4ive\)/gi;
  ## go/undergo
  s/\(([BVLG])([^ %]*) ([^ ]*?)([Gg]o|[Gg]oes|[Ww]ent|[Gg]one|[Gg]oing)\)/\(B\2-o\1%\4|B%go \3go\)/gi;
  ## hang/overhang
  s/\(([BVLG])([^ %]*) (|[Oo]ver)(h)(ang|angs|ung|anged|anging)\)/\(B\2-o\1%\4\5|B%\4ang \3\4ang\)/gi;
  ## have
  s/\(([BVLG])([^ %]*) ()((?<=\{L-a.\} )\'d|\'s|\'ve|[Hh]ave|[Hh]as|[Hh]ad|[Hh]aving)\)/\(B\2-o\1%\4|B%have \3have\)/gi;
  ## hear
  s/\(([BVLG])([^ %]*) ([Hh]ear)(|s|d|ing)\)/\(B\2-o\1%\3\4|B%\3 \3\)/gi;
  ## hew/sew/strew
  s/\(([BVLG])([^ %]*) ([HhSs]|[Ss]tr)(ew)(|s|ed|n|ing)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## hide
  s/\(([BVLG])([^ %]*) ([Hh]id)(e|es||den|ing)\)/\(B\2-o\1%\3\4|B%\3e \3e\)/gi;
  ## hit
  s/\(([BVLG])([^ %]*) ([Hh]it)(|s||ting)\)/\(B\2-o\1%\3\4|B%\3 \3\)/gi;
  ## hold
  s/\(([BVLG])([^ %]*) ([^ ]*?)([Hh])(old|olds|eld|olding)\)/\(B\2-o\1%\4\5|B%\4old \3\4old\)/gi;
  ## lay
  s/\(([BVLG])([^ %]*) ([Ll])(ay|ays|aid|ain|aying)\)/\(B\2-o\1%\3\4|B%\3ay \3ay\)/gi;
  ## lead/plead/mislead
  s/\(([BVLG])([^ %]*) ((?:mis)?[Pp]?)([Ll]e)(ad|ads|d|ading)\)/\(B\2-o\1%\4\5|B%\4ad \3\4ad\)/gi;
  ## leap/outleap
  s/\(([BVLG])([^ %]*) ([^ s]*?)([Ll]ea)(p|ps|pt|ping)\)/\(B\2-o\1%\4\5|B%\4p \3\4p\)/gi;
  ## leave
  s/\(([BVLG])([^ %]*) ([Ll])(eave|eaves|eft|eaving)\)/\(B\2-o\1%\3\4|B%\3eave \3eave\)/gi;
  ## lend/send/spend
  s/\(([BVLG])([^ %]*) ([Ll]|[Ss]|[Ss]p)(en)(d|ds|t|ding)\)/\(B\2-o\1%\4\5|B%\4d \3\4d\)/gi;
  ## lie (as in recline)
  s/\(([BVLG])([^ %]*) ([Ll])(ie|ies|ay|ying)\)/\(B\2-o\1%\3\4|B%\3ie \3ie\)/gi;
  ## light/highlight/spotlight
  s/\(([BVLG])([^ %]*) (high|moon|spot)?([Ll]i)(ght|ghts|t|ghting)\)/\(B\2-o\1%\4\5|B%\4ght \3\4ght\)/gi;
  ## lose
  s/\(([BVLG])([^ %]*) ([Ll]os)(e|es|t|ing)\)/\(B\2-o\1%\3\4|B%\3e \3e\)/gi;
  ## make
  s/\(([BVLG])([^ %]*) ([Mm]a)(ke|kes|de|king)\)/\(B\2-o\1%\3\4|B%\3ke \3ke\)/gi;
  ## mean
  s/\(([BVLG])([^ %]*) ([Mm]ean)(|s|t|ing)\)/\(B\2-o\1%\3\4|B%\3 \3\)/gi;
  ## meet
  s/\(([BVLG])([^ %]*) ([Mm]e)(et|ets|t|eting)\)/\(B\2-o\1%\3\4|B%\3et \3et\)/gi;
  ## pay/say/overpay
  s/\(([BVLG])([^ %]*) ([^ ]*?[PpSs])(a)(y|ys|id|ying)\)/\(B\2-o\1%\4\5|B%\4y \3\4y\)/gi;
  ## prove
  s/\(([BVLG])([^ %]*) ([Dd]is)?([Pp]rov)(e|es|ed|en|ing)\)/\(B\2-o\1%\4\5|B%\4e \3\4e\)/gi;
  ## quit
  s/\(([BVLG])([^ %]*) ([Qq]uit)(|s|ting)\)/\(B\2-o\1%\3\4|B%\3 \3\)/gi;
  ## ride
  s/\(([BVLG])([^ %]*) (|[Oo]ver)(r)(ide|ides|ode|idden|iding)\)/\(B\2-o\1%\4\5|B%\4ide \3\4ide\)/gi;
  ## run
  s/\(([BVLG])([^ %]*) ([Rr])(un|uns|an|unning)\)/\(B\2-o\1%\3\4|B%\3un \3un\)/gi;
  ## see/oversee
  s/\(([BVLG])([^ %]*) ([^ ]*?)([Ss])(ee|ees|aw|een|eeing)\)/\(B\2-o\1%\4\5|B%\4ee \3\4ee\)/gi;
  ## seek
  s/\(([BVLG])([^ %]*) ([Ss])(eek|eeks|ought|eeking)\)/\(B\2-o\1%\3\4|B%\3eek \3eek\)/gi;
  ## sell/tell
  s/\(([BVLG])([^ %]*) ([^ ]*?[SsTt])(ell|ells|old|elling)\)/\(B\2-o\1%\4|B%ell \3ell\)/gi;
  ## shoot
  s/\(([BVLG])([^ %]*) ([Ss]ho)(ot|ots|t|otting)\)/\(B\2-o\1%\3\4|B%\3ot \3ot\)/gi;
  ## show
  s/\(([BVLG])([^ %]*) ([^ ]*?)([Ss]how)(|s|ed|n|ing)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## sit
  s/\(([BVLG])([^ %]*) ([Ss])(it|its|at|itting)\)/\(B\2-o\1%\3\4|B%\3it \3it\)/gi;
  ## slay
  s/\(([BVLG])([^ %]*) ([Ss]l)(ay|ays|ayed|ain|aying)\)/\(B\2-o\1%\3\4|B%\3ay \3ay\)/gi;
  ## sneak
  s/\(([BVLG])([^ %]*) ([Ss]n)(eak|eaks|uck|eaking)\)/\(B\2-o\1%\3\4|B%\3eak \3eak\)/gi;
  ## smite/write
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:[Ss]m|[Ww]r))(ite|ites|ote|itten|iting)\)/\(B\2-o\1%\4|B%ite \3ite\)/gi;
  ## stand/understand
  s/\(([BVLG])([^ %]*) ([^ ]*?)([Ss]t)(and|ands|ood|anding)\)/\(B\2-o\1%\4\5|B%\4and \3\4and\)/gi;
  ## steal
  s/\(([BVLG])([^ %]*) ([Ss]t)(eal|eals|ole|olen|ealing)\)/\(B\2-o\1%\3\4|B%\3eal \3eal\)/gi;
  ## stick
  s/\(([BVLG])([^ %]*) ([Ss]t)(ick|icks|uck|icking)\)/\(B\2-o\1%\3\4|B%\3ick \3ick\)/gi;
  ## strike
  s/\(([BVLG])([^ %]*) ([Ss]tr)(ike|ikes|uck|icken|iking)\)/\(B\2-o\1%\3\4|B%\3ike \3ike\)/gi;
  ## swear/shear/tear
  s/\(([BVLG])([^ %]*) ([Ss]w|[Ss]h|[Tt])(ear|ear|ore|orn|earing)\)/\(B\2-o\1%\4|B%ear \3ear\)/gi;
  ## forsake/take/shake
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:[Tt]|[Ss]h|[Ff]ors))(ake|akes|aketh|ook|aken|aking)\)/\(B\2-o\1%\4|B%ake \3ake\)/gi;
  ## teach
  s/\(([BVLG])([^ %]*) ([Tt])(each|eaches|aught|eaching)\)/\(B\2-o\1%\3\4|B%\3each \3each\)/gi;
  ## think
  s/\(([BVLG])([^ %]*) ([Tt]h)(ink|inks|ought|inking)\)/\(B\2-o\1%\3\4|B%\3ink \3ink\)/gi;
  ## tread
  s/\(([BVLG])([^ %]*) ([Tt]r)(ead|eads|od|eading)\)/\(B\2-o\1%\3\4|B%\3ead \3ead\)/gi;
  ## weave
  s/\(([BVLG])([^ %]*) ([Ww])(eave|eaves|ove|oven|eaving)\)/\(B\2-o\1%\3\4|B%\3eave \3eave\)/gi;
  ## wreak
  s/\(([BVLG])([^ %]*) ([Ww]r)(eak|eaks|eaked|ought|eaking)\)/\(B\2-o\1%\3\4|B%\3eak \3eak\)/gi;
  ## will
  s/\(([BVLG])([^ %]*) ()(\'ll|[Ww]ill|[Ww]o)\)/\(B\2-o\1%\4|B%will \3will\)/gi;
  ## win
  s/\(([BVLG])([^ %]*) ([Ww])(in|ins|on|un|inning)\)/\(B\2-o\1%w\4|B%win \3in\)/gi;
  ## would
  s/\(([BVLG])([^ %]*) ()(\'d|[Ww]ould)\)/\(B\2-o\1%\4|B%would \3would\)/gi;


  #### irregular in orthography only:
  ## Xd* -- shred/wed/wad
  s/\(([BVLG])([^ %]*) ([Ss]hre|[Ww]e|[Ww]a)(d)(|s|ded|ding)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## Xl* -- compel/propel/impel/repel, not spell
  s/\(([BVLG])([^ %]*) ([^ ]*..pe)(l)(|s|led|ling)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## Xl* -- control/patrol, not stroll
  s/\(([BVLG])([^ %]*) ([^ ]*..tro)(l)(|s|led|ling)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## Xl* -- initial/total
  s/\(([BVLG])([^ %]*) ([^ ]*(?:ia|[Tt]ota))(l)(|s|led|ling)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## Xp* -- quip
  s/\(([BVLG])([^ %]*) ([^ ]*?ui)(p)(|s|ped|ping)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## Xr* -- infer|deter
  s/\(([BVLG])([^ %]*) ([Dd]ete|(?<= )ave|[^ ]*[^f]fe|proffe|[^ ]*cu)(r)(|s|red|ring)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## X* -- alter/bicker/audit/benefit
  s/\(([BVLG])([^ %]*) ([Aa]lter|[Bb]icker|[Aa]udit|[Bb]enefit)(|s|ed|ing)\)/\(B\2-o\1%\4|B% \3\)/gi;
  ## Xshed* (/s/ding) shed
  s/\(([BVLG])([^ %]*) ([Ss]hed)(|s|ding)\)/\(B\2-o\1%\3\4|B%\3 \3\)/gi;
  ## Xtiptoe* (/s/d/ing) tiptoe
  s/\(([BVLG])([^ %]*) ([Tt]iptoe)(|s|d|ing)\)/\(B\2-o\1%\3\4|B%\3 \3\)/gi;
  ## x*e -- breathe/seethe/soothe/loathe/swathe/writhe/ache
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:breath|eeth|sooth|loath|swath|writh|(?<= )ach))(e|es|ed|ing)\)/\(B\2-o\1%\4|B%e \3e\)/gi;
  ## X*e -- waste
  s/\(([BVLG])([^ %]*) ([Ww]ast)(e|es|ed|ing)\)/\(B\2-o\1%\4|B%e \3e\)/gi;


  ## X*y
  s/\(([BVLG])([^ %]*) ([^ ]*?[^aeou])(y|ies|ied|ying)\)/\(B\2-o\1%\4|B%y \3y\)/gi;

  ### double consonant
  ## Xb* (/s/bed/bing)
  s/\(([BVLG])([^ %]*) ([^ ]*?[^bmr])(b)(|s|bed|bing)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## Xd* (/s/ded/ding)
  s/\(([BVLG])([^ %]*) ([^ ]*?[^adelnrs'](?<![aeiou][aeiouw])|embe)(d)(|s|ded|ding)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## Xg* (/s/ged/ging)
  s/\(([BVLG])([^ %]*) ([^ ]*?[^gnrs])(g)(|s|ged|ging)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## Xk* (/s/ked/king)
  s/\(([BVLG])([^ %]*) ([^ ]*?e)(k)(|s|ked|king)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## Xl* (/s/led/ling)
  s/\(([BVLG])([^ %]*) ([^ ]*?(?![aeiou][aeiouw]|e|l|r))(l)(|s|led|ling)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## Xm* (/s/med/ming)
  s/\(([BVLG])([^ %]*) ([^ ]*?[^lmrs'](?<![aeiou][aeiouw]))(m)(|s|med|ming)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## Xn* (/s/ned/ning)
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:[^aeiou][aiu]|(?<= )pe| co| do))(n)(|s|ned|ning)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## Xp* (/s/ped/ping)
  s/\(([BVLG])([^ %]*) ([^ ]*?[^lmprs](?<!elo)(?<![aeiou][aeiouw]))(p)(|s|ped|ping)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## Xr* (/s/red/ring)
  s/\(([BVLG])([^ %]*) ([^ ]*?[^eor](?<![aeiou][aeiouw])(?<! pu))(r)(|s|red|ring)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## Xt* (/s/ted/ting)
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:[^aeiou][aiou]|cqui|ffse|[fg]re|abe|(?<= )[blnsBLNS]e)(?<!budge|targe|.umpe|.rpre|..i[bcmrs]i|pro.i))(t)(|s|ted|ting)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## Xv* (/s/ved/ving)
  s/\(([BVLG])([^ %]*) ([^ ]*?[^v])(v)(|s|ved|ving)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;

  ## Xs* (/es/ed/ing)
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:s|focu))(s)(|es|ed|ing)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## Xz* (/es/ed/ing)
  s/\(([BVLG])([^ %]*) ([^ ]*?z)(z)(|es|ed|ing)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## Xh* (/es/ed/ing)
  s/\(([BVLG])([^ %]*) ([^ ]*?[cs](?<! ac))(h)(|es|ed|ing)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;
  ## Xx* (/es/ed/ing)
  s/\(([BVLG])([^ %]*) ([^ ]*?)(x)(|es|ed|ing)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;

  ## Xee*
  s/\(([BVLG])([^ %]*) ([^ ]*?[^chn])(ee)(|s|d|ing)\)/\(B\2-o\1%\4\5|B%\4 \3\4\)/gi;


  ## X*e (e|es|ed|ing) -- regular e, that would otherwise qualify as complex vowel or complex consonant
  s/\(([BVLG])([^ %]*) ([^ ]*?(?:(?<= )dy|eec|gaug|iec|oic|[nrs]c|[st]uad|(?<= )guid|af|ieg|[dlr]g|[bcdfgkptyz]l|ym|sum|[ct]ap|hop|[^s]ip|yp|uar|uir|sur|[aeiou]{2}s(?<!vous|bias)|ens|abus|ccus|mus|[Cc]reat|rmeat|[isu]at|uot|eav|iev|[aeo]lv|[ae]iv|rv|u|(?<= )ow|[es]iz|eez|ooz|yz))(e|es|ed|ing)\)/\(B\2-o\1%\4|B%e \3e\)/gi;

  ## X* -- regular no e, with complex vowel or complex consonant
  s/\(([BVLG])([^ %]*) ([^() ]*?[aeiou]{2,}[bcdfghjklmnpqrstvwxyz](?<!oed))(|s|ed|ing)\)/\(B\2-o\1%\4|B% \3\)/gi;
  s/\(([BVLG])([^ %]*) ([^() ]*?[bcdfghjklmnpqrstvwxyz][bcdfghjklmnpqrtvwxyz](?<![aeiu]ng|.bl))(|s|ed|ing)\)/\(B\2-o\1%\4|B% \3\)/gi;

  ## X* -- regular no e, which would otherwise end in single vowel single consonant
  s/\(([BVLG])([^ %]*) ([^() ]*?(?:log|[dnt]al|el|devil|(?<!p)edit|eril|.[^r]lop|..sip|....[sd]on|.[dhkpst]en|[^r]ven|...[st]om|[ct]hor|..[bdhkmnptvw]er|ffer|nger|swer|censor|ckon|mmon|o[ln]or|itor|umor|phan|rdon|eason|oison|[ar][bjv]or|rror|[fgkrv]et|osit|[ai]bit|erit|imit|mbast))(|s|ed|ing)\)/\(B\2-o\1%\4|B% \3\)/gi;
  s/\(([BVLG])([^ %]*) ([^() ]*?(?:hbor|(?<= )ping|mme.|gge.|cke.|icit|i[rv]al|ns..|amid|ofit|-bus|iphon|ilor|umpet|isit))(|s|ed|ing)\)/\(B\2-o\1%\4|B% \3\)/gi;

  #### long + consonant + e regular
  ## X*e (e|es|ed|ing)
  s/\(([BVLG])([^ %]*) ([^ ]*?[bcdgklmnprstuvz](?<!l[aio]ng))(e|es|ed|ing)\)/\(B\2-o\1%\4|B%e \3e\)/gi;

  #### X* -- default no e regular
  ## X*
  s/\(([BVLG])([^ %]*) ([^ ]*?)(|s|ed|ing)\)/\(B\2-o\1%\4|B% \3\)/gi;

  ######## NEGATIVE FOR VERBS

  ## un%
  s/\((B)([^ ]*) ([Uu]n)$NOTUN([^ ]*?)()\)/\(\1\2-o\1%:\L\3\E%|\1NEG%:% \4\)/gi;

  ######## REPETITIVE

  $NORE = '(?!ach|act|ad|alize|ap|ason|bel|buff|buke|call|cede|cess|ceive|cite|cogni[sz]e|commend|cord|cruit|ctify|deem|duce|fer|flect|fresh|fund|fuse|gard|gist|ly|gulate|inforce|ject|late|lease|main|mark|member|mind|move|new|novate|pel|plicate|ply|port|present|prove|pulse|pute|quest|quire|sign|sist|solve|sonate|spond|st|store|strict|sult|sume|tain|taliate|tard|tire|tort|turn|veal|view|vise|vive|volve|ward)';

  ## re%
  s/\((B)([^ ]*) ([Rr]e-?)$NORE([^ ]*?)()\)/\(B\2-o\1%:\L\3\E%|\1REP%:% \4\)/gi;

  ######## CAUSAL, INCHOATIVE, STATIVE
  if( $SEM ) {
    ## unergatives
    @CI = ('abate','accelerate','adapt','adjust','advance','arm','audition',
           'balance','balloon','band','begin','benefit','bleed','blow','blur','boom','bounce','bow','branch','break','brew','buckle','budge','burn','buzz',
           'capitalize','careen','change','cheer','circulate','clash','cling','close','cohere','coincide','collapse','collect','concentrate','confer','conform','continue','contract','convert','coordinate','crack','crash','crawl','creak','crest','crumble',
           'dance','debut','decrease','default','defect','degenerate','deteriorate','develop','differentiate','diminish','dip','disappear','disarm','dissipate','dissociate','dissolve','distribute','diverge','diversify','double','dress','drift','drill','drive',
           'ease','ebb','economize','embark','emerge','end','engage','enroll','erode','erupt','evaporate','evolve','expand','explode','extend',
           'fade','fail','fester','finish','fire','firm','fit','flash','flatten','flinch','flip','float','flow','fly','focus','fold','form','formulate','freeze',
           'gather','glaze','glide','group','grow',
           'hang','hurt',
           'improve','inch','increase','inflate','integrate','intensify','invest',
           'kowtow',
           'land','leapfrog','level','light','liquify','liquidate','lodge',
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
      s/\((B-aN)-bN([^ ]*) ($c)\)/\(\1\2-oB%|BCAU% $c\)/gi;
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
      if( length($CISA[$i][1])>0 && length($CISA[$i][2])>0 ) { s/\(B($CISA[$i][0])-bN([^ ]*) ($CISA[$i][1])\)/\(B\1\2-oB%\3|BCAU%$CISA[$i][2] $CISA[$i][2]\)/gi; }
      if( length($CISA[$i][2])>0 && length($CISA[$i][3])>0 ) { s/\(B($CISA[$i][0])([^ ]*) ($CISA[$i][2])\)/\(B\1\2-oB%\3|BINC%$CISA[$i][3] $CISA[$i][3]\)/gi; }
      if( length($CISA[$i][3])>0 && length($CISA[$i][4])>0 ) { s/\(B($CISA[$i][0])([^ ]*) ($CISA[$i][3])\)/\(A\1\2-oB%\3|A%$CISA[$i][4] $CISA[$i][4]\)/gi; }
      if( length($CISA[$i][2])>0 && length($CISA[$i][4])>0 ) { s/\(B($CISA[$i][0])([^ ]*) ($CISA[$i][2])\)/\(A\1\2-oB%\3|AINC%$CISA[$i][4] $CISA[$i][4]\)/gi; }
    }

    ## %en cau -> inc
    s/\((B)-aN-bN([^ ]*) (?!threaten)([^ ]*?en)\)/\(B-aN\2-o\1%|BCAU% \3\)/gi;

    ## %ize
#    s/\((B)([^ ]*) ([^ ]*?)(ize)\)/\(A\2-o\1%\4|AINC% \3\)/gi;
    ## %engthen
    s/\((B)([^ ]*) ([^ ]*?)()(engthen)\)/\(A\2-o\1%\4\5|AINC%\4ong \3\4ong\)/gi;
    ## %ipen
    s/\((B)([^ ]*) ([^ ]*?ip)(e)(n)\)/\(A\2-o\1%\4\5|AINC%\4 \3\4\)/gi;
    ## %dden
    s/\((B)([^ ]*) ([^ ]*?)(d)(den)\)/\(A\2-o\1%\4\5|AINC%\4 \3\4\)/gi;
    ## %tten
    s/\((B)([^ ]*) ([^ ]*?)(t)(ten)\)/\(A\2-o\1%\4\5|AINC%\4 \3\4\)/gi;
    ## %en
    s/\((B)([^ ]*) (?![^ ]*open|threaten|happen)([^ ]*?)(en)\)/\(A\2-o\1%\4|AINC% \3\)/gi;
  }

  ######## CLEANUP

  ## convert lemma w tag -o back to observed form w tag -x, working bottom-up
  while( s/\((.)([^ ]*)-o(.)([^% ]*)%:([^% ]*)%([^\| ]*)\|([^% ]*)%:([^% ]*)%([^- ]*)(-[^ ]*?)? \8([^\)]*)\9\)/\($3$2-x$3$4%:$5%$6\|$7%:$8%$9$10 $5$11$6\)/gi ||
         s/\((.)([^ ]*)-o(.)([^% ]*)%([^\| ]*)\|([^% ]*)%([^- ]*)(-[^ ]*?)? ([^\)]*)\7\)/\(\3\2-x\3\4%\5\|\6%\7\8 \9\5\)/gi ) { }
  ## convert morph tags to lowercase
  s/(%[A-Za-z]+)/\L\1\E/g;

  #### remove -x tag
  s/\(([BVLGAR][^ ]*)-x-([^\(\)]*)\)/\(\1-\2\)/gi;

  # #### remove empty morphemes
  # s/ \([^ ]* \*\)//gi;

  print $_;
}
