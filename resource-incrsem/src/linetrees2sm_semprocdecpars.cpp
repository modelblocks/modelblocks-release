////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  This file is part of ModelBlocks. Copyright 2009, ModelBlocks developers. //
//                                                                            //
//  ModelBlocks is free software: you can redistribute it and/or modify       //
//  it under the terms of the GNU General Public License as published by      //
//  the Free Software Foundation, either version 3 of the License, or         //
//  (at your option) any later version.                                       //
//                                                                            //
//  ModelBlocks is distributed in the hope that it will be useful,            //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of            //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             //
//  GNU General Public License for more details.                              //
//                                                                            //
//  You should have received a copy of the GNU General Public License         //
//  along with ModelBlocks.  If not, see <http://www.gnu.org/licenses/>.      //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#define ARMA_64BIT_WORD
#include <iostream>
#include <fstream>
#include <list>
#include <regex>
using namespace std;
#include <armadillo>
using namespace arma;
#define DONT_USE_UNMAPPABLE_TUPLES
#include <nl-randvar.h>
#include <nl-string.h>
#include <Delimited.hpp>
bool STORESTATE_TYPE = true;
bool STORESTATE_CHATTY = true;
int FEATCONFIG = 0;
bool INTERSENTENTIAL = true;
#include <StoreState.hpp>
#ifdef DENSE_VECTORS
#include <SemProcModels_dense.hpp>
#else
#include <SemProcModels_sparse.hpp>
#endif
#include <Beam.hpp>
#include <BerkUnkWord.hpp>
#include <Tree.hpp>
#include <ZeroPad.hpp>
int COREF_WINDOW = INT_MAX;

////////////////////////////////////////////////////////////////////////////////

L getLink( L l ) {
  if( string::npos != l.find("-n") ) {
    std::smatch sm;
    std::regex re( "(.*)-n([0-9]+).*" ); //get consecutive numbers after a "-n"
    if( std::regex_search( l, sm, re ) && sm.size() > 2 ) { 
      return( sm.str(2) ); 
    }
  }
  return( "" );
} 

L removeLink( L l ) {
  if( string::npos != l.find("-n") ) {
    std::smatch sm;
    std::regex re ( "(.*?)-n([0-9]+).*" ); //get consecutive numbers after a "-n"
    if( std::regex_search( l, sm, re ) && sm.size() > 2 ) { return( sm.str(1) ); }
  } 
  return( l );
}

////////////////////////////////////////////////////////////////////////////////

map<L,double> mldLemmaCounts;
int MINCOUNTS = 100;
bool ABLATE_UNARY = false;

////////////////////////////////////////////////////////////////////////////////

inline string regex_escape(const string& string_to_escape) {
    return regex_replace( string_to_escape, regex("([.^$|()\\[\\]{}*+?\\\\])"), "\\$1" );
}

////////////////////////////////////////////////////////////////////////////////

CVar getCat ( const L& l ) {
  return regex_replace( regex_replace( l, regex("-x[^} ][^ |]*[|][^- ]*"), string("") ), regex("-l."), string("") ).c_str();
}

////////////////////////////////////////////////////////////////////////////////

O getOp ( const L& l, const L& lSibling, const L& lParent ) {
//  if( string::npos != l.find("-lN") or string::npos != l.find("-lG") or string::npos != l.find("-lH") or string::npos != l.find("-lR") ) return 'N';
  if( string::npos != l.find("-lG") ) return 'G';
  if( string::npos != l.find("-lH") ) return 'H';
  if( string::npos != l.find("-lR") ) return 'R';
  if( string::npos != l.find("-lV") ) return 'V';  // NOTE: may not be used.
  if( string::npos != l.find("-lD") ) return 'N';
  if( string::npos != l.find("-lN") ) return 'N';
  if( string::npos != lSibling.find("-lU") ) return ( getCat(l).getSynArgs()==1 ) ? 'U' : 'u';
  if( string::npos != l.find("-lI") ) return 'I';
  if( string::npos != l.find("-lC") ) return 'C';
  if( string::npos == l.find("-l")  or string::npos != l.find("-lS") or string::npos != l.find("-lU") ) return O_I;
  if( string::npos != l.find("-lM") or string::npos != l.find("-lQ") ) return 'M';
  if( (string::npos != l.find("-lA") or string::npos != l.find("-lI")) and string::npos != lParent.find("\\") ) return '0'+getCat( string(lParent,lParent.find("\\")+1).c_str() ).getSynArgs();
  if( (string::npos != l.find("-lA") or string::npos != l.find("-lI")) and string::npos == lParent.find('\\') ) return '0'+getCat( lSibling ).getSynArgs();
  cout << "(WARNING: unhandled -l tag in label \"" << l << "\"" << " in binary branch -- assuming identity.)"<<endl;
  cerr << "WARNING: unhandled -l tag in label \"" << l << "\"" << " in binary branch -- assuming identity."<<endl;
  return O_I;
}

////////////////////////////////////////////////////////////////////////////////

string getUnaryOp ( const Tree<L>& tr ) {
  if( string::npos != L(tr.front()).find("-lV") ) return "V";
  if( string::npos != L(tr.front()).find("-lZ") ) return "Z";
  if( string::npos != L(tr.front()).find("-lQ") ) return "O";
  N n =  CVar( removeLink(tr).c_str() ).getLastNonlocal();
  if( n == N_NONE ) return "";
  if( (/*tr.front().size()==0 ||*/ tr.size()==1 and tr.front().size()==1 and tr.front().front().size()==0) and n == N("-rN") ) return "0";
  if( string::npos != L(tr.front()).find("-lE") )
    return ( CVar(removeLink(tr.front()).c_str()).getSynArgs() > CVar(removeLink(tr).c_str()).getSynArgs() ) ? (string(1,'0'+CVar(removeLink(tr.front()).c_str()).getSynArgs())) : "M";
  else return "";
}

////////////////////////////////////////////////////////////////////////////////

pair<K,CVar> getPred ( const L& lP, const L& lW ) {
  CVar c = getCat ( lP );

  // If punct, but not special !-delimited label...
  if ( ispunct(lW[0]) && ('!'!=lW[0] || lW.size()==1) ) return pair<K,CVar>(K::kBot,c);

  cout<<"reducing "<<lP<<" now "<<c;
  string sLemma = lW;  transform(sLemma.begin(), sLemma.end(), sLemma.begin(), [](unsigned char c) { return std::tolower(c); });
  string sCat = c.getString();
  string sPred = sCat + ':' + sLemma;
  cout<<" to "<<sCat<<endl;

  smatch m; for( string s=lP; regex_match(s,m,regex("^(.*?)-x([^} ][^| ]*[|](?:(?!-[a-zA-Z])[^ }])*)(.*?)$")); s=m[3] ) {
    string sX = m[2];
    smatch mX;
    cout<<"applying "<<sX<<" to "<<sPred;
    if( regex_match( sX, mX, regex("^(.*)%(.*)%(.*)[|](.*)%(.*)%(.*)$") ) )        // transfix (prefix+infix+suffix) rule application
      sPred = regex_replace( sPred, regex("^"+regex_escape(mX[1])+"(.*)"+regex_escape(mX[2])+"(.*)"+regex_escape(mX[3])+"$"), string(mX[4])+"$1"+string(mX[5])+"$2"+string(mX[6]) );
    if( regex_match( sX, mX, regex("^(.*)[%](.*)[|](.*)[%](.*)$") ) )              // circumfix (prefix+suffix) rule application
      sPred = regex_replace( sPred, regex("^"+regex_escape(mX[1])+"(.*)"+regex_escape(mX[2])+"$"), string(mX[3])+"$1"+string(mX[4]) );
    cout<<" obtains "<<sPred<<endl;
  }

  int iSplit = sPred.find( ":", 1 );
  sCat  = sPred.substr( 0, iSplit );
  sLemma = sPred.substr( iSplit+1 );
  if ( mldLemmaCounts.find(sLemma)==mldLemmaCounts.end() || mldLemmaCounts[sLemma]<MINCOUNTS ) sLemma = "!unk!";
  if ( isdigit(lW[0]) )                                                                        sLemma = "!num!";

  return pair<K,CVar>( ( sCat + ':' + sLemma + '_' + ((lP[0]=='N') ? '1' : '0') ).c_str(), c );
}

////////////////////////////////////////////////////////////////////////////////

EMat matE;
OFunc funcO;

NModel modN;
FModel modF;
JModel modJ;

////////////////////////////////////////////////////////////////////////////////

void calcContext ( Tree<L>& tr, 
                   map<string,int>& annot2tdisc, vector<trip<Sign,W,K>>& antecedentCandidates, int& tDisc, const int sentnum, map<string,HVec>& annot2kset,
		   int& wordnum, bool isFailTree, std::set<int>& excludedIndices, // coref related: 
       bool ABLATE_UNARY,                                             // whether or not to remove unary features for n,f,j submodels
		   int s=1, int d=0, string e="", L l=L() ) {                     // side, depth, unary (e.g. extraction) operators, ancestor label.
       
  static F          f;
  static string     eF;
  static Sign       aPretrm;
  static StoreState q;

  if( l==L() ) l = removeLink(tr);

  // At unary preterminal...
  if ( tr.size()==1 && tr.front().size()==0 ) {
    wordnum++;  // increment word index at terminal (sentence-level) one-indexing
    tDisc++;    // increment discourse-level word index. one-indexing
    string annot    = getLink( tr );  //if( annot == currentloc ) annot = "";
    f               = 1 - s;
    eF              = e + getUnaryOp( tr );
    pair<K,CVar> kc = getPred ( removeLink(tr), removeLink(tr.front()) );
    K k             = (FEATCONFIG & 8 && kc.first.getString()[2]!='y') ? K::kBot : kc.first;
    bool validIntra = false;

    std::string annotSentIdx = annot.substr(0,annot.size()-2); //get all but last two...
    if (annotSentIdx == std::to_string(sentnum)) validIntra = true;
    if (INTERSENTENTIAL == true) validIntra = true;
    const HVec& hvAnt = validIntra == true ? annot2kset[annot] : hvTop;
    bool nullAnt = (hvAnt.empty()) ? true : false;
    const string currentloc = std::to_string(sentnum) + ZeroPadNumber(2, wordnum); // be careful about where wordnum get initialized and incremented - starts at 1 in main, so get it before incrementing below with "wordnum++"
    if (annot != "")  {
      annot2kset[currentloc] = hvAnt;
    }
    annot2kset[currentloc] = HVec(k, matE, funcO); //add current k
    annot2tdisc[currentloc] = tDisc; //map current sent,word index to discourse word counter
    W histword(""); //histword will track most recent observed word whose k is unk. will be stored for correct antecedent only.
    if (not isFailTree) {
      // Print preterminal / fork-phase predictors...
      FPredictorVec lfp( modF, hvAnt, nullAnt, q ); 
      cout<<"----"<<q<<endl;

      // Print antecedent list...
      for( int i = tDisc; (i > 0 and tDisc-i <= COREF_WINDOW); i-- ) {  //only look back COREF_WINDOW antecedents at max
        if( excludedIndices.find(i) != excludedIndices.end() ) {  //skip indices which have already been found as coreference indices.  this prevents negative examples for non most recent corefs. //TODO Could also try making multiple positive examples, creating positive for each pair in chain
          continue; 
        }
        else {
          trip<Sign,W,K> candidate; //represents the candidate's Sign (used for features generation), histword, aka most recent observed word where k was unk (antunk or unk), and the k of the candidate
          int nLabel = 0; //correct/incorrect label for N model
          if (i < tDisc) {
            candidate = antecedentCandidates[i-1]; //there are one fewer candidates than tDisc value.  e.g., second word only has one previous candidate.
          }
          else {
            candidate = trip<Sign,W,K>(Sign(/*hvBot*/HVec(),cTop,S_A), W(""), k); //Sign(hvTop, "NONE", "/"); //null antecedent generated at first iteration, where i=tDisc. Sign consists of: kset, type (syncat), side (A/B)
            if (annot == "") {
              nLabel = 1; //null antecedent is correct choice, "1" when no annotation 
            }
          }
          
          //check for non-null coreference 
          if ((i == annot2tdisc[annot]) and (annot != "")) {
            nLabel = 1;
            excludedIndices.insert(annot2tdisc[annot]); //add blocking index here once find true, annotated coref. e.g. word 10 is coref with word 5. add annot2tdisc[annot] (5) to list of excluded.
            //set k to kAntUnk if is coreferent, and most recent antecedent unk k in chain has observed word (histword) that matches w_t
            //actually can store unk k logic by storing obsword as histword (candidate.second()) only following valid unk k conditions.
            //that is, store obsword as histword if current candidate isUnk, else store most recent non-empty histword in chain as histword. else store empty string as histword

            if (not k.isUnk()) {
              histword = candidate.second(); //inherit histword as most recent unk obsword
            }
            if (candidate.second() == W(removeLink(tr.front()).c_str())) { // and candidate.third().isUnk()) {  
              k = kAntUnk;
            }
          }


          bool corefON = ((i==tDisc) ? 0 : 1); //whether current antecedent is non-null or not
          NPredictorVec npv( modN, candidate.first(), corefON, tDisc - i, q, ABLATE_UNARY ); 
          cout << "N " << pair<const NModel&,const NPredictorVec&>(modN,npv) << " : " << nLabel << endl; //i-1 because that's candidate index 
        } //single candidate output
      } //all previous antecedent candidates output

#ifdef DENSE_VECTORS
      cout << "F " << lfp << " " << f << "&" << e << "&" << k << endl; // modF.getResponseIndex(f,e.c_str(),k);
      cout << "P " << PPredictorVec(f,e.c_str(),k,q) << " : " << getCat(removeLink(l)) /*getCat(l)*/     << endl;
      if (k != kAntUnk) { 
        cout << "W " << e << " " << k << " " << getCat(removeLink(l)) /*getCat(l)*/           << " : " << removeLink(tr.front())  << endl;
      }
#else
      cout << "F " << pair<const FModel&,const FPredictorVec&>(modF,lfp) << " : f" << f << "&" << e << "&" << k << endl;  modF.getResponseIndex(f,e.c_str(),k);
      cout << "P " << PPredictorVec(f,e.c_str(),k,q) << " : " << getCat(removeLink(l)) << endl;
      if (k != kAntUnk) { 
        cout << "W " << e << " " << k << " " << getCat(removeLink(l)) << " : " << removeLink(tr.front())  << endl;
      }
#endif
      q = StoreState( q, hvAnt, eF.c_str(), k, getCat(removeLink(l)), matE, funcO );
      aPretrm = q.back().apex().back();
    } else {
      aPretrm = Sign();
    }
    //current candidate is stored, along with histword, which is most recent (ant)unk k's observed word. if current k is (ant)unk, use w_t as histword.  elseif not coreferent with anything (not corefON), store empty string as histword. else store antecedent's histword.
   
    //cout << "k is: " << k << " and isUnk() is: " << k.isUnk() << endl;
    if (k.isUnk()) {
      histword = W(removeLink(tr.front()).c_str()); //store current unk k's observed word. overrides inherit case above.
    }
    //cout << "saving histword: " << histword << " for word: " << removeLink(tr.front()) << endl;
    antecedentCandidates.emplace_back(trip<Sign,W,K>(aPretrm, histword, k)); //append current prtrm to candidate list for future coref decisions 
  }

  // At unary identity nonpreterminal...
  else if ( tr.size()==1 and getCat(tr)==getCat(tr.front()) ) {
    calcContext( tr.front(), annot2tdisc, antecedentCandidates, tDisc, sentnum, annot2kset, wordnum, isFailTree, excludedIndices, ABLATE_UNARY, s, d, e, l );
  }

  // At unary nonpreterminal...
  else if ( tr.size()==1 ) {
    //// cerr<<"#U"<<getCat(tr)<<" "<<getCat(tr.front())<<endl;
    e = e + getUnaryOp( tr );
    calcContext ( tr.front(), annot2tdisc, antecedentCandidates, tDisc, sentnum, annot2kset, wordnum, isFailTree, excludedIndices, ABLATE_UNARY, s, d, e, l );
  }

  // At binary nonterminal...
  else if ( tr.size()==2 ) {
    //// cerr<<"#B "<<getCat(tr)<<" "<<getCat(tr.front())<<" "<<getCat(tr.back())<<endl;

    if (isFailTree) {
      calcContext ( tr.front(), annot2tdisc, antecedentCandidates, tDisc, sentnum, annot2kset, wordnum, isFailTree, excludedIndices, ABLATE_UNARY, 0, d+s );
      calcContext ( tr.back(),  annot2tdisc, antecedentCandidates, tDisc, sentnum, annot2kset, wordnum, isFailTree, excludedIndices, ABLATE_UNARY, 1, d );
      return;
    }

    // Traverse left child...
    calcContext ( tr.front(), annot2tdisc, antecedentCandidates, tDisc, sentnum, annot2kset, wordnum, isFailTree, excludedIndices, ABLATE_UNARY, 0, d+s );

    J j          = s;
    cout << "~~~~ " << q.back().apex() << endl;
    q = StoreState( q, f );
    const Sign& aLchild = q.getApex();
    e            = e + getUnaryOp( tr );
    O oL         = getOp ( removeLink(tr.front()), removeLink(tr.back()),  removeLink(tr) );
    O oR         = getOp ( removeLink(tr.back()),  removeLink(tr.front()), removeLink(tr) );

    // Print binary / join-phase predictors...
    JPredictorVec ljp( modJ, f, eF.c_str(), aLchild, q ); 
    cout << "==== " << q.getApex() << "   " << removeLink(tr) << " -> " << removeLink(tr.front()) << " " << removeLink(tr.back()) << endl;
#ifdef DENSE_VECTORS
//    cout << "J " << pair<const JModel&,const JPredictorVec&>(modJ,ljp) << " : j" << j << "&" << e << "&" << oL << "&" << oR << endl;  modJ.getResponseIndex(j,e.c_str(),oL,oR);
    cout << "J " << ljp << " " << j << "&" << e << "&" << oL << "&" << oR << endl;  // modJ.getResponseIndex(j,e.c_str(),oL,oR);
#else
    cout << "J " << pair<const JModel&,const JPredictorVec&>(modJ,ljp) << " : j" << j << "&" << e << "&" << oL << "&" << oR << endl;  modJ.getResponseIndex(j,e.c_str(),oL,oR);
#endif
    cout << "A " << APredictorVec(f,j,eF.c_str(),e.c_str(),oL,aLchild,q)                << " : " << getCat(removeLink(l))          << endl;
    cout << "B " << BPredictorVec(f,j,eF.c_str(),e.c_str(),oL,oR,getCat(l),aLchild,q)   << " : " << getCat(removeLink(tr.back()))  << endl;

    // Update storestate...
    q = StoreState ( q, j, e.c_str(), oL, oR, getCat(removeLink(l)), getCat(removeLink(tr.back())) );

    // Traverse right child...
    calcContext ( tr.back(), annot2tdisc, antecedentCandidates, tDisc, sentnum, annot2kset, wordnum, isFailTree, excludedIndices, ABLATE_UNARY, 1, d );
  }

  // At abrupt terminal (e.g. 'T' discourse)...
  else if ( tr.size()==0 );

  else cerr<<"ERROR: non-binary non-unary-preterminal: " << tr << endl;
}

////////////////////////////////////////////////////////////////////////////////

int main ( int nArgs, char* argv[] ) {

  list<DelimitedPair<psX,Delimited<double>,psSpace,L,psX>> lLC;

  // For each command-line flag or model file...
  for ( int a=1; a<nArgs; a++ ) {
    if(      '-'==argv[a][0] && 'f'==argv[a][1] ) FEATCONFIG   = atoi( argv[a]+2 );
    else if( '-'==argv[a][0] && 'u'==argv[a][1] ) MINCOUNTS    = atoi( argv[a]+2 );
    else if( '-'==argv[a][0] && 'c'==argv[a][1] ) COREF_WINDOW = atoi( argv[a]+2 );
    else if( '-'==argv[a][0] && 'a'==argv[a][1] ) ABLATE_UNARY = true;
    else {
      cerr << "Loading model " << argv[a] << "..." << endl;
      // Open file...
      ifstream fin (argv[a], ios::in );
      // Read model lists...
      int linenum = 0;
      while ( fin && EOF!=fin.peek() ) {
//      new changes
        if ( fin.peek()=='E' ) matE = EMat( fin );
        if ( fin.peek()=='O' ) funcO = OFunc( fin );
        else fin >> *lLC.emplace(lLC.end()) >> "\n";
        if ( ++linenum%1000000==0 ) cerr << "  " << linenum << " items loaded..." << endl;
      }
      cerr << "Model " << argv[a] << " loaded." << endl;
    }
  }
  for( auto& l : lLC ) mldLemmaCounts[l.second] = l.first;
//  cout << matE << endl;
  int linenum = 0;  int discourselinenum = 0; //increments on sentence in discourse/article
  map<string,HVec> annot2kset;
  int tDisc = 0; //increments on word in discourse/article
  vector<trip<Sign,W,K>> antecedentCandidates;
  map<string,int> annot2tdisc;
  std::set<int> excludedIndices; //init indices of positive coreference to exclude.  prevents negative examples in training data when they're really positive coreference.
  while ( cin && EOF!=cin.peek() ) {
    linenum++;
    discourselinenum++;
    if( linenum%1000==0 ) cerr<<"line "<<linenum<<"..."<<endl;

    if ( cin.peek() != '\n' ) {
      Tree<L> t("T"); t.emplace_back(); t.emplace_back("T");
      cin >> t.front() >> "\n";
      cout.flush();
      cout << "TREE " << linenum << ": " << t << "\n";
      if ( t.front().size() > 0 and removeLink(t.front().front()) == "!ARTICLE") {
        cerr<<"resetting discourse info..."<<endl;
        discourselinenum=0;
        annot2kset.clear();
        tDisc=0;
        antecedentCandidates.clear();
        annot2tdisc.clear();
        excludedIndices.clear();
      }
      else {
	int wordnum = 0;
        bool isFailTree = (removeLink(t.front()) == "FAIL") ? true : false;
        if( t.front().size() > 0 ) calcContext( t, annot2tdisc, antecedentCandidates, tDisc, discourselinenum, annot2kset, wordnum, isFailTree, excludedIndices, ABLATE_UNARY);
      }
    }
    else {cin.get();}
  }

  // cerr << "F TOTALS: " << modF.getNumPredictors() << " predictors, " << modF.getNumResponses() << " responses." << endl;
  // cerr << "J TOTALS: " << modJ.getNumPredictors() << " predictors, " << modJ.getNumResponses() << " responses." << endl;
}



