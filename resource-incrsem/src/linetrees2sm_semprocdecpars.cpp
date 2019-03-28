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
#include <Beam.hpp>
#include <BerkUnkWord.hpp>
#include <Tree.hpp>
#include <ZeroPad.hpp>

char psSpcColonSpc[]  = " : ";
char psSpcEqualsSpc[] = " = ";

////////////////////////////////////////////////////////////////////////////////

class LVU : public trip<L,arma::rowvec,arma::vec> {
 public:
  // Constructor methods...
  LVU ( )                { }
  LVU ( const char* ps ) { first() = ps; }
  L removeLink ( ) const {
    if (string::npos != first().find("-n")) {
      std::smatch sm;
      std::regex re ("(.*)-n([0-9]+).*"); //get consecutive numbers after a "-n"
      if (std::regex_search(first(), sm, re) && sm.size() > 2) { return(sm.str(1)); }
    } 
    return(first());
  }
  // Accessor methods...
  operator       const L ( ) const { return removeLink(); }
  const arma::rowvec&  v ( ) const { return second(); }
  const arma::vec&     u ( ) const { return third(); }
  arma::rowvec&        v ( )       { return second(); }
  arma::vec&           u ( )       { return third(); }
  L                    getL ( )    { return removeLink(); } //return unlinked label
  operator             L ( )       { return removeLink(); } 
  L              getLink ( ) const { if (string::npos != first().find("-n")) {
                                       std::smatch sm;
                                       std::regex re("(.*)-n([0-9]+).*"); //get consecutive numbers after a "-n"
                                       if (std::regex_search(first(), sm, re) && sm.size() > 2) { 
                                         return(sm.str(2)); 
                                       }
                                     }
                                     return(""); } 
  // Input / output methods  ---  NOTE: only reads and writes label, not vectors...
  friend pair<istream&,LVU&> operator>> ( istream&            is,  LVU& t              )                 { return pair<istream&,LVU&>(is,t); }
  friend istream&            operator>> ( pair<istream&,LVU&> ist, const char* psDelim )                 { return pair<istream&,L&>(ist.first,ist.second.first())>>psDelim; }
  friend bool                operator>> ( pair<istream&,LVU&> ist, const vector<const char*>& vpsDelim ) { return pair<istream&,L&>(ist.first,ist.second.first())>>vpsDelim; }
  friend ostream&            operator<< ( ostream&            os,  const LVU&  t       )                 { return os<<t.first(); }
};

////////////////////////////////////////////////////////////////////////////////

map<L,double> mldLemmaCounts;
int MINCOUNTS = 100;
//int MINCOUNTS = 0;
map<trip<T,T,T>,arma::mat> mtttmG;
map<pair<T,W>,arma::vec> mtwvL;
int iMaxNums = 0;
arma::mat eye3;
arma::mat mFail;
arma::mat mIdent;
arma::vec vOnes;
arma::vec vFirstHot;

map<pair<vector<FPredictor>,FResponse>,arma::vec> mvfrv; //map pair to vec.  pair is (vector<FPredictor>, FResponse). arma::vec is latent scores across different latent versions of the same category. first.first is the list of predictors.  second is the arma:vec, first.second is the FResponse.
map<pair<PPredictor,T>,arma::mat> mpppmP;
map<quad<EVar,K,T,W>,arma::vec> mektwvW;
map<pair<vector<JPredictor>,JResponse>,arma::mat> mvjrm;
map<pair<APredictor,T>,arma::mat> mapamA;
map<pair<BPredictor,T>,arma::mat> mbpbmB;
//map<vector<NPredictor>,bool> mvnb; //Antecedent N model, where NPredictors are usually pair<K,K>, but can be others

////////////////////////////////////////////////////////////////////////////////

const arma::mat normalize( const arma::mat& M, int n ) { double denom = arma::norm(M,n); return ( denom>0.0 ) ? M/denom : M; }
void normalize( arma::mat& M, int n, int dir ) {    //return M / (M * arma::ones(M.n_cols)); }
//cerr<<M<<endl;
  for( uint i=0; i<M.n_rows; i++ ) if( arma::accu(M.row(i))>0.0 ) M.row(i) /= arma::accu(M.row(i));
//cerr<<M<<endl;
}

////////////////////////////////////////////////////////////////////////////////

inline string regex_escape(const string& string_to_escape) {
    return regex_replace( string_to_escape, regex("([.^$|()\\[\\]{}*+?\\\\])"), "\\$1" );
}

////////////////////////////////////////////////////////////////////////////////

/*
int getArityGivenLabel ( const L& l ) {
  int depth = 0;
  int arity = 0;
  if ( l[0]=='N' ) arity++;
  for ( uint i=0; i<l.size(); i++ ) {
    if ( l[i]=='{' ) depth++;
    if ( l[i]=='}' ) depth--;
    if ( l[i]=='-' && l[i+1]>='a' && l[i+1]<='d' && depth==0 ) arity++;
  }
if( arity>7 ) cerr<<"i think that "<<l<<" has "<<arity<<" args."<<endl;
  return arity;
}
*/

////////////////////////////////////////////////////////////////////////////////

//// T T_COLON ( "Pk" );                       // must be changed to avoid confusion with ":" delimiter in K's (where type occurs individually).
//// T T_CONTAINS_COMMA ( "!containscomma!" ); // must be changed to avoid confusion with "," delimiter in F,J params.

T getType ( const L& l ) {
////  if ( l[0]==':' )                 return T_COLON;
////  if ( l.find(',')!=string::npos ) return T_CONTAINS_COMMA;
  return regex_replace( regex_replace( l, regex("-x[^} ][^ |]*[|][^- ]*"), string("") ), regex("-l."), string("") ).c_str();
//  return regex_replace( regex_replace( l, regex("%[^ %|]*[|]"), string("") ), regex("-[xl](?:(?!-[a-z])[^ }])*"), string("") ).c_str();
//  return string( string( l, 0, l.find("-l") ), 0, l.find("-x") ).c_str();
}

////////////////////////////////////////////////////////////////////////////////

O getOp ( const L& l, const L& lSibling, const L& lParent ) {
  if( string::npos != l.find("-lN") or string::npos != l.find("-lG") or string::npos != l.find("-lH") or string::npos != l.find("-lR") ) return 'N';
  if( string::npos != l.find("-lV") ) return 'V';
  if( string::npos != lSibling.find("-lU") ) return ( getType(l).getArity()==1 ) ? 'U' : 'u';
  if( string::npos != l.find("-lC") ) return 'C';
  if( string::npos == l.find("-l")  or string::npos != l.find("-lS") or string::npos != l.find("-lU") ) return 'I';
  if( string::npos != l.find("-lM") or string::npos != l.find("-lQ") ) return 'M';
  if( (string::npos != l.find("-lA") or string::npos != l.find("-lI")) and string::npos != lParent.find("\\") ) return '0'+getType( string(lParent,lParent.find("\\")+1).c_str() ).getArity();
  if( (string::npos != l.find("-lA") or string::npos != l.find("-lI")) and string::npos == lParent.find('\\') ) return '0'+getType( lSibling ).getArity();
  cerr << "WARNING: unhandled -l tag in label \"" << l << "\"" << " -- assuming identity."<<endl;
  return 'I';
}

////////////////////////////////////////////////////////////////////////////////

string getCorefId ( Tree<LVU>& tr ) { 
  //given a tree, get the current "-n[0-9]+" coreference annotation. this id corresponds to a KSet that includes all K contexts up to and including that annotation. e.g., "The Lord said he-n0102 did wash his-n0104 shoes", "-n0102" includes "Lord_", "-n0104" includes "Lord_" and "he_".
  L l = L(tr);
  if (string::npos != l.find("-n")) {
          string target = string(l);
          std::smatch sm;
          std::regex re (".*-n([0-9]+).*"); //get consecutive numbers after a "-n"
          if (std::regex_search(target, sm, re) && sm.size() > 1) {return sm.str(1);} 
  }
  return "";
}

////////////////////////////////////////////////////////////////////////////////

string getUnaryOp ( const Tree<LVU>& tr ) {
//if( FEATCONFIG & 16 ) return 'N';
  //cerr << tr << endl;
  if( string::npos != L(tr.front()).find("-lV") ) return "V";
  if( string::npos != L(tr.front()).find("-lQ") ) return "O";
  N n =  T(L(tr).c_str()).getLastNonlocal();
  if( n == N_NONE ) return "";
  if( (tr.front().size()==0 || tr.front().front().size()==0) && n == N("-rN") ) return "0";
  if( string::npos != L(tr.front()).find("-lE") )
    return ( T(L(tr.front()).c_str()).getArity() > T(L(tr).c_str()).getArity() ) ? (string(1,'0'+T(L(tr.front()).c_str()).getArity())) : "M";
  else return "";
/*
  N n =  T(L(tr).c_str()).getLastNonlocal();
cout<<" . . . "<<" n="<<n<<" for "<<tr<<endl<<(T(L(tr.front()).c_str()).getLastNonlocal()==n)<<" "<<(T(L(tr.back()).c_str()).getLastNonlocal()==n)<<endl;
  if ( n == N_NONE ) return 'N';
  if ( (tr.front().size()==0 || tr.front().front().size()==0) && n == N("-rN") ) return '0';
  if ( T(L(tr.front()).c_str()).getLastNonlocal()==n || T(L(tr.back()).c_str()).getLastNonlocal()==n ) return 'N';
  return ( n.isArg() ) ? T(L(tr).c_str()).getArity()+'1' : 'M';
*/
}

////////////////////////////////////////////////////////////////////////////////

pair<K,T> getPred ( const L& lP, const L& lW ) {
  T t = getType ( lP );   //string( lP, 0, lP.find("-l") ).c_str();

  // If punct, but not special !-delimited label...
  if ( ispunct(lW[0]) && ('!'!=lW[0] || lW.size()==1) ) return pair<K,T>(K::kBot,t);

//if( FEATCONFIG & 16)
//  return pair<K,T>( ( (lP[0]=='N') ? string("N:y0_1") : string("B:y0_0") ).c_str(), t );

cout<<"reducing "<<lP<<" now "<<t;
  string sLemma = lW;  transform(sLemma.begin(), sLemma.end(), sLemma.begin(), [](unsigned char c) { return std::tolower(c); });
//  string sType = t.getString();  regex_replace( sType, regex("-x-x(?:(?!-[a-z])[^ }])*"), string("") );
  string sType = t.getString();  // regex_replace( sType, regex("-x[^| ]*[|](?:(?!-[a-z])[^ }])*"), string("") );
  string sPred = sType + ':' + sLemma;
cout<<" to "<<sType<<endl;

//  smatch m; for( string s=lP; regex_match(s,m,regex("^(.*?)-x((?:(?!-[a-z])[^ }])*)(.*?)$")); s=m[3] ) {
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

if( FEATCONFIG & 16)
  if( sPred[0]=='Y' ) sPred[0] = ( (lP[0]=='N') ? 'N' : 'B' );

  int iSplit = sPred.find( ":", 1 );
  sType  = sPred.substr( 0, iSplit );
  sLemma = sPred.substr( iSplit+1 );
  if ( mldLemmaCounts.find(sLemma)==mldLemmaCounts.end() || mldLemmaCounts[sLemma]<MINCOUNTS ) sLemma = "!unk!";
  if ( isdigit(lW[0]) )                                                                        sLemma = "!num!";

  return pair<K,T>( ( sType + ':' + sLemma + '_' + ((lP[0]=='N') ? '1' : '0') ).c_str(), t );
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

const arma::mat& getG ( T t, T t0, T t1 ) {
  if( mtttmG.end() == mtttmG.find( trip<T,T,T>(t,t0,t1) ) ) { cerr<<"WARNING: Failed to find "<<t<<" -> "<<t0<<" "<<t1<<".  Aborting tree."<<endl; return mFail; }
  return mtttmG[ trip<T,T,T>(t,t0,t1) ];
}

////////////////////////////////////////////////////////////////////////////////

arma::vec& setBackwardMessages ( Tree<LVU>& tr ) {
  // At abrupt terminal (e.g. 'T' discourse)...
  if ( tr.size()==0 ) tr.u() = vFirstHot;
  // At unary preterminal...
  else if( tr.size()==1 && tr.front().size()==0 ) {
    auto itwv = mtwvL.find( pair<T,W>( getType(tr), L(tr.front()).c_str() ) );
    if( itwv == mtwvL.end() ) itwv = mtwvL.find( pair<T,W>( getType(tr), unkWordBerk( L(tr.front()).c_str() ) ) );
    if( itwv == mtwvL.end() ) { cerr<<"WARNING: No UNK for "<<L(tr)<<" -> "<<L(tr.front())<<".  Aborting tree."<<endl; tr.u() = arma::zeros( iMaxNums ); }
    else                      tr.u() = normalize( itwv->second, 1 );
//    tr.u() = normalize( ( itwv != mtwvL.end() ) ? itwv->second : mtwvL[ pair<T,W>( getType(tr), unkWordBerk( L(tr.front()).c_str() ) ) ], 1 );
  }
  // At unary identity nonpreterminal...
  else if ( tr.size()==1 and getType(tr)==getType(tr.front()) ) {
    tr.u() = setBackwardMessages( tr.front() );
  }
  // At unary nonpreterminal...
  else if ( tr.size()==1 ) {
    arma::vec u0 = setBackwardMessages( tr.front() );
    tr.u() = getG( getType(tr), getType(tr.front()), "-" ) * arma::kron( u0, vFirstHot );
  }
  // At binary nonterminal...
  else if ( tr.size()==2 ) {
    arma::vec u0 = setBackwardMessages( tr.front() );
    arma::vec u1 = setBackwardMessages( tr.back()  );
    tr.u() = getG( getType(tr), getType(tr.front()), getType(tr.back()) ) * arma::kron( u0, u1 );
  }
  return tr.u();
}

////////////////////////////////////////////////////////////////////////////////

void setForwardMessages ( Tree<LVU>& tr, const arma::rowvec v ) {
  tr.v() = v;
  // At unary preterminal...
  if( tr.size()==1 && tr.front().size()==0 ) {
  }
  // At unary identity nonpreterminal...
  else if ( tr.size()==1 and getType(tr)==getType(tr.front()) ) {
    setForwardMessages( tr.front(), v );
  }
  // At unary nonpreterminal...
  else if ( tr.size()==1 ) {
    setForwardMessages( tr.front(), v * getG( getType(tr), getType(tr.front()), "-" ) * kron( mIdent, vFirstHot ) );
  }
  // At binary nonterminal...
  else if ( tr.size()==2 ) {
    setForwardMessages( tr.front(), v * getG( getType(tr), getType(tr.front()), getType(tr.back()) ) * kron( mIdent, tr.back().u()  ) );
    setForwardMessages( tr.back(),  v * getG( getType(tr), getType(tr.front()), getType(tr.back()) ) * kron( tr.front().u(), mIdent ) );
  }
}

////////////////////////////////////////////////////////////////////////////////
void calcContext ( Tree<LVU>& tr, const arma::mat& D, const arma::mat& U, map<string,int>& annot2tdisc, vector<Sign>& antecedentCandidates, int& tDisc, const int sentnum, map<string,KSet>& annot2kset, int& wordnum, bool failtree, std::set<int>& excludedIndices, int s=1, int d=0, string e="", L l=L() ) { 
  static F          f;
  static string     eF;
  static Sign       aPretrm;
  static StoreState q;
  static arma::mat eye3( iMaxNums, iMaxNums*iMaxNums, arma::fill::zeros );  if( eye3(0,0)==0.0 ) for( int i=0; i<iMaxNums; i++ ) eye3(i,i*iMaxNums+i)=1.0;    // Init 3D diag.

  if( l==L() ) l = L(tr);

  // At unary preterminal...
  if ( tr.size()==1 && tr.front().size()==0 ) {
    wordnum++; //increment word index at terminal (sentence-level) one-indexing
    tDisc++; //increment discourse-level word index. one-indexing
    string annot = tr.getLink();
    f               = 1 - s;
//    eF = e = ( e!='N' ) ? e : getUnaryOp ( tr );
    eF              = e + getUnaryOp( tr );
    pair<K,T> kt    = getPred ( L(tr), L(tr.front()) );
    K k             = (FEATCONFIG & 8 && kt.first.getString()[2]!='y') ? K::kBot : kt.first;
    aPretrm         = (not failtree) ? Sign( k, getType(l), S_A ) : Sign() ;
    bool validIntra = false;

    std::string annotSentIdx = annot.substr(0,annot.size()-2); //get all but last two...
    if (annotSentIdx == std::to_string(sentnum)) validIntra = true;
    if (INTERSENTENTIAL == true) validIntra = true;
    const KSet& ksAnt = validIntra == true ? annot2kset[annot] : KSet(K::kTop);
    bool nullAnt = (ksAnt.empty()) ? true : false;
    const string currentloc = std::to_string(sentnum) + ZeroPadNumber(2, wordnum); // be careful about where wordnum get initialized and incremented - starts at 1 in main, so get it before incrementing below with "wordnum++"
    if (annot != "")  {
      annot2kset[currentloc] = ksAnt;
      //cerr << "found antecedent " << ksAnt << " from link: " << annot << endl;
    }
    annot2kset[currentloc].push_back(k); //add current k 
    //cerr << "adding k " << k << " to annot2kset at loc: " << currentloc << endl;
    for (auto& ant : ksAnt) {
      if (ksAnt != ksTop) aPretrm.first().emplace_back(ant); //add antecedent ks to aPretrm
    }
    cerr << "adding tDisc: " << tDisc << " as annot2tDisc for currentloc: " << currentloc << endl;
    annot2tdisc[currentloc] = tDisc; //map current sent,word index to discourse word counter
    if (not failtree) {
      // Print preterminal / fork-phase predictors...
      DelimitedList<psX,FPredictor,psComma,psX> lfp;  
      q.calcForkPredictors(lfp, ksAnt, nullAnt);//additional kset argument is set of all antecedents in referent cluster. requires global map from annotation to ksets, as in {"0204":Kset ["Lord_"], "0207": KSet ["Lord_", "he_"]}, etc.
      auto fpCat = lfp.front( );
      lfp.pop_front( );        // remove first element before sorting, then add back, bc later code assumes first element is category.
      lfp.sort( );             // sort to shorten mlr input
      lfp.push_front( fpCat );
      cout<<"----"<<q<<endl;
      cout << "note: F "; for ( auto& fp : lfp ) { if ( &fp!=&lfp.front() ) cout<<","; cout<<fp<<"=1"; }  cout << " : " << FResponse(f,e.c_str(),k) << endl;
      cout << "note: P " << q.calcPretrmTypeCondition(f,e.c_str(),k) << " : " << aPretrm.getType() /*getType(l)*/     << endl;
      cout << "note: W " << e << " " << k << " " << aPretrm.getType() /*getType(l)*/           << " : " << L(tr.front())  << endl;

      for ( int i = tDisc; i > 0 ; i--) { 
        if (excludedIndices.find(i) != excludedIndices.end()) {  //skip indices which have already been found as coreference indices.  this prevents negative examples for non most recent corefs.
          cerr << "encountered excluded index i:" << i << " skipping..." << endl;
          continue; 
        }
        else {
          Sign candidate;
          int isCoref = 0;
          if (i < tDisc) {
            candidate = antecedentCandidates[i-1]; //there are one fewer candidates than tDisc value.  e.g., second word only has one previous candidate.
          }
          else {
            candidate = Sign(KSet(K::kTop), "NONE", "/"); //null antecedent generated at first iteration, where i=tDisc. Sign consists of: kset, type (syncat), side (A/B)
            
            if (annot == "") isCoref = 1; //null antecedent is correct choice, "1" when no annotation TODO fix logic for filtering intra/inter?
          }
          
          //check for non-null coreference 
          if ((i == annot2tdisc[annot]) and (annot != "")) {
            isCoref = 1;
            excludedIndices.insert(annot2tdisc[annot]); //add blocking index here once find true, annotated coref. e.g. word 10 is coref with word 5. add annot2tdisc[annot] (5) to list of excluded.
            for (auto it=excludedIndices.begin(); it != excludedIndices.end(); ++it) 
                      cerr << ' ' << *it;
            cerr << endl;
          }

          bool corefON = ((i==tDisc) ? 0 : 1); //whether current antecedent is non-null or not
          //DelimitedList<psX,NPredictor,psComma,psX> npreds;  
          NPredictorSet nps;// = NPredictorSet( tDisc - i, npreds); //generate NPredictorSet with distance feature - just use "i"
          q.calcNPredictors(nps, candidate, corefON, tDisc - i); //populate npreds with kxk pairs for q vs. candidate q. 
          nps.PrintOut();
          cout << " : " << isCoref << endl; //i-1 because that's candidate index 
        } //single candidate output
      } //all previous antecedent candidates output

      arma::vec& vF = mvfrv[ pair<vector<FPredictor>,FResponse>( vector<FPredictor>( lfp.begin(), lfp.end() ), FResponse(f,e.c_str(),k) ) ];
      arma::mat& mP = mpppmP[ pair<PPredictor,T>(q.calcPretrmTypeCondition(f,e.c_str(),k),aPretrm.getType()) ];
      arma::vec& vW = mektwvW[ quad<EVar,K,T,W>(e.c_str(),k,aPretrm.getType(),L(tr.front()).c_str()) ];
      vF = ((vF.n_elem) ? vF : arma::zeros(iMaxNums))          + normalize( D * U * tr.u(),                     1 );
      mP = ((mP.n_elem) ? mP : arma::zeros(iMaxNums,iMaxNums)) + normalize( D * arma::diagmat(U * tr.u()),      1 );
      vW = ((vW.n_elem) ? vW : arma::zeros(iMaxNums))          + normalize( (vOnes.t() * D).t() % (U * tr.u()), 1 );
      //if( q.calcPretrmTypeCondition(f,e,k).fourth()==T("N-aD") and getType(tr)==T("R-aN") ) 
      //cout<<"D for "<<tr<<" with b="<<q.calcPretrmTypeCondition(f,e,k).fourth()<<endl<<D<<endl<<tr.u()<<endl;
    }

    //cout << "adding aPretrm: " << aPretrm << " to list of antecedentCandidates" << endl;
    antecedentCandidates.emplace_back(aPretrm); //append current prtrm to candidate list for future coref decisions 
    //cout << "new size of antecedentCandidates: " << antecedentCandidates.size() << endl;
  }

  // At unary identity nonpreterminal...
  else if ( tr.size()==1 and getType(tr)==getType(tr.front()) ) {
    calcContext( tr.front(), D, U, annot2tdisc, antecedentCandidates, tDisc, sentnum, annot2kset, wordnum, failtree, excludedIndices, s, d, e, l );
  }

  // At unary nonpreterminal...
  else if ( tr.size()==1 ) {
    //// cerr<<"#U"<<getType(tr)<<" "<<getType(tr.front())<<endl;
    //e = ( e!='N' ) ? e : getUnaryOp ( tr );
    e = e + getUnaryOp( tr );
    calcContext ( tr.front(), D, (getType(tr)==getType(tr.front())) ? U : U * getG(getType(tr),getType(tr.front()),"-") * arma::kron(mIdent,vOnes), annot2tdisc, antecedentCandidates, tDisc, sentnum, annot2kset, wordnum, failtree, excludedIndices, s, d, e, l );
//cout<<"unary at "<<L(tr)<<endl<<mtttmG[trip<T,T,T>(getType(tr),getType(tr.front()),"-")]<<endl;
  }

  // At binary nonterminal...
  else if ( tr.size()==2 ) {
    //// cerr<<"#B "<<getType(tr)<<" "<<getType(tr.front())<<" "<<getType(tr.back())<<endl;

    if (failtree) {
      calcContext ( tr.front(), D * U * getG( getType(tr), getType(tr.front()), getType(tr.back()) ) * arma::kron( mIdent, tr.back().u() ), mIdent, annot2tdisc, antecedentCandidates, tDisc, sentnum, annot2kset, wordnum, failtree, excludedIndices, 0, d+s );
      calcContext ( tr.back(), arma::diagmat( tr.back().v() ), mIdent, annot2tdisc, antecedentCandidates, tDisc, sentnum, annot2kset, wordnum, failtree, excludedIndices, 1, d );
      return;
    }

    // Traverse left child...
    calcContext ( tr.front(), D * U * getG( getType(tr), getType(tr.front()), getType(tr.back()) ) * arma::kron( mIdent, tr.back().u() ), mIdent, annot2tdisc, antecedentCandidates, tDisc, sentnum, annot2kset, wordnum, failtree, excludedIndices, 0, d+s );

    J j          = s;
    LeftChildSign aLchild ( q, f, eF.c_str(), aPretrm );
//    e            = ( e!='N' ) ? e : getUnaryOp ( tr ) ;
    e            = e + getUnaryOp( tr );
    O oL         = getOp ( L(tr.front()), L(tr.back()),  L(tr) );
    O oR         = getOp ( L(tr.back()),  L(tr.front()), L(tr) );

    // Print binary / join-phase predictors...
    DelimitedList<psX,JPredictor,psComma,psX> ljp;  q.calcJoinPredictors(ljp,f,eF.c_str(),aLchild);
    auto jpCat = ljp.front( );
    ljp.pop_front( );        // remove first element before sorting, then add back, bc later code assumes first elemenet is category.
    ljp.sort( );             // sort to shorten mlr input
    ljp.push_front( jpCat );
    cout << "==== " << aLchild << "   " << L(tr) << " -> " << L(tr.front()) << " " << L(tr.back()) << endl;
    cout << "note: J ";  for ( auto& jp : ljp ) { if ( &jp!=&ljp.front() ) cout<<","; cout<<jp<<"=1"; }  cout << " : " << JResponse(j,e.c_str(),oL,oR)  << endl;
    cout << "note: A " << q.calcApexTypeCondition(f,j,eF.c_str(),e.c_str(),oL,aLchild)                  << " : " << getType(l)          << endl;
    cout << "note: B " << q.calcBrinkTypeCondition(f,j,eF.c_str(),e.c_str(),oL,oR,getType(l),aLchild)   << " : " << getType(tr.back())  << endl;
    APredictor apred = q.calcApexTypeCondition(f,j,eF.c_str(),e.c_str(),oL,aLchild);
    BPredictor bpred = q.calcBrinkTypeCondition(f,j,eF.c_str(),e.c_str(),oL,oR,getType(l),aLchild);

    // Update storestate...
    q = StoreState ( q, f, j, eF.c_str(), e.c_str(), oL, oR, getType(l), getType(tr.back()), aPretrm, aLchild );

    // Traverse right child...
    calcContext ( tr.back(), arma::diagmat( tr.back().v() ), mIdent, annot2tdisc, antecedentCandidates, tDisc, sentnum, annot2kset, wordnum, failtree, excludedIndices, 1, d );
//    calcContext ( tr.back(), arma::diagmat( tr.v() * getG( getType(tr), getType(tr.front()), getType(tr.back()) ) * arma::kron( vOnes /*tr.front().u()*/, mIdent ) ), 1, d );

    arma::mat& mJ = mvjrm [ pair<vector<JPredictor>,JResponse>( vector<JPredictor>( ljp.begin(), ljp.end() ), JResponse(j,e.c_str(),oL,oR) ) ];
    arma::mat& mA = mapamA[ pair<APredictor,T>(apred,getType(l)) ];
    arma::mat& mB = mbpbmB[ pair<BPredictor,T>(bpred,getType(tr.back())) ];
    arma::mat tmp; if( j ) tmp = tr.front().u()*vFirstHot.t(); else tmp = arma::diagmat(tr.front().u());
    mJ = ((mJ.n_elem) ? mJ : arma::zeros(iMaxNums,iMaxNums))          + normalize( D * U * getG(getType(tr),getType(tr.front()),getType(tr.back())) * arma::kron(mIdent,tr.back().u()),                                                   1 );
    mA = ((mA.n_elem) ? mA : arma::zeros(iMaxNums,iMaxNums*iMaxNums)) + normalize( eye3 * arma::kron( D.t(), U * getG(getType(tr),getType(tr.front()),getType(tr.back())) * arma::kron(tmp,tr.back().u()) ),                              1 );
    mB = ((mB.n_elem) ? mB : arma::zeros(iMaxNums,iMaxNums*iMaxNums)) + normalize( arma::diagmat(vOnes.t() * D) * U * getG(getType(tr),getType(tr.front()),getType(tr.back())) * arma::diagmat(arma::kron(tr.front().u(),tr.back().u())), 1 );
if( arma::accu(mJ) != arma::accu(mJ) ) { cerr<<"J failed at "<<tr<<endl; exit(0); }
  }

  // At abrupt terminal (e.g. 'T' discourse)...
  else if ( tr.size()==0 );

  else cerr<<"ERROR: non-binary non-unary-preterminal: " << tr << endl;
}

////////////////////////////////////////////////////////////////////////////////

int main ( int nArgs, char* argv[] ) {

  list<DelimitedPair<psX,Delimited<double>,psSpace,L,psX>> lLC;
  list<DelimitedQuad<psX,Delimited<T>,psSpcColonSpc,Delimited<T>,psSpace,Delimited<T>,psSpcEqualsSpc,Delimited<double>,psX>> lCC;
  list<DelimitedTrip<psX,Delimited<T>,psSpcColonSpc,W,psSpcEqualsSpc,Delimited<double>,psX>> lX;

  // For each command-line flag or model file...
  for ( int a=1; a<nArgs; a++ ) {
    //if ( 0==strcmp(argv[a],"t") ) STORESTATE_TYPE = true;
    if(      '-'==argv[a][0] && 'f'==argv[a][1] ) FEATCONFIG = atoi( argv[a]+2 );
    else if( '-'==argv[a][0] && 'u'==argv[a][1] ) MINCOUNTS  = atoi( argv[a]+2 );
    else {
      cerr << "Loading model " << argv[a] << "..." << endl;
      // Open file...
      ifstream fin (argv[a], ios::in );
      // Read model lists...
      int linenum = 0;
      while ( fin && EOF!=fin.peek() ) {
        if      ( fin.peek()=='C' ) fin >> "CC " >> *lCC.emplace(lCC.end()) >> "\n";
        else if ( fin.peek()=='X' ) fin >> "X "  >> *lX.emplace (lX.end() ) >> "\n";
        else                        fin >> *lLC.emplace(lLC.end()) >> "\n";
        if ( ++linenum%1000000==0 ) cerr << "  " << linenum << " items loaded..." << endl;
      }
      cerr << "Model " << argv[a] << " loaded." << endl;
    }
  }
  for( auto& l : lLC ) mldLemmaCounts[l.second] = l.first;
  for( auto& i : lCC ) if( i.first().getNums()>=iMaxNums ) iMaxNums = i.first().getNums()+1;
  cerr << "Detected " << iMaxNums << " latent variable types." << endl;
  if( iMaxNums==0 ) iMaxNums = 1;
  for( auto& i : lCC ) { arma::mat& m = mtttmG[ trip<T,T,T>( i.first().getLets(), i.second().getLets(), i.third().getLets() ) ]; if( m.size()==0 ) m.zeros(iMaxNums,iMaxNums*iMaxNums); m( i.first().getNums(), i.second().getNums() * iMaxNums + i.third().getNums() ) = i.fourth(); }
//  for( auto& tttm : mtttmG ) { normalize( tttm.second, 1, 1 ); }
  for( auto& i : lX  ) { arma::vec& v = mtwvL[ pair<T,W>( i.first().getLets(), i.second() ) ]; if( v.size()==0 ) v.zeros(iMaxNums); v( i.first().getNums() ) = i.third(); }
//cout<<( mtttmG[ trip<T,T,T>( "A-aN", "R-aN", "A-aN-PRTRM" ) ] )<<endl;
//cout<<( mtttmG[ trip<T,T,T>( "A-aN", "R-aN", "A-aN-PRTRM" ) ] * arma::kron( mIdent, vOnes ) )<<endl;

  mFail     = arma::zeros( iMaxNums, iMaxNums*iMaxNums );
  mIdent    = arma::eye( iMaxNums, iMaxNums );
  vOnes     = arma::ones( iMaxNums );
  vFirstHot = arma::zeros( iMaxNums );  vFirstHot(0)=1.0;

  int linenum = 0;
  int discourselinenum = 0; //increments on sentence in discourse/article
  map<string,KSet> annot2kset;
  int tDisc = 0; //increments on word in discourse/article
  vector<Sign> antecedentCandidates; 
  map<string,int> annot2tdisc;
  std::set<int> excludedIndices; //init indices of positive coreference to exclude.  prevents negative examples in training data when they're really positive coreference.
  while ( cin && EOF!=cin.peek() ) {
    linenum++;
    discourselinenum++;
    if( linenum%1000==0 ) cerr<<"line "<<linenum<<"..."<<endl;

    if ( cin.peek() != '\n' ) {
      Tree<LVU> t("T"); t.emplace_back(); t.emplace_back("T");
      cin >> t.front() >> "\n";
      cout.flush();
      cout << "TREE " << linenum << ": " << t << "\n";
      if ( t.front().size() > 0 and L(t.front().front()) == "!ARTICLE") {
        cerr<<"resetting discourse info..."<<endl;
        discourselinenum=0;
        annot2kset.clear();
        tDisc=0;
        antecedentCandidates.clear();
        annot2tdisc.clear();
        excludedIndices.clear();      
      } 
      else {
        setBackwardMessages( t );
        setForwardMessages( t, vFirstHot.t() );
        int wordnum = 0;
        bool failtree = (L(t.front()) == "FAIL") ? true : false;
        //cerr << "t.front: " << t.front() << "L(t.front()): " << L(t.front()) << endl;
        if( t.front().size() > 0 ) calcContext( t, arma::diagmat(vFirstHot), mIdent, annot2tdisc, antecedentCandidates, tDisc, discourselinenum, annot2kset, wordnum, failtree, excludedIndices);
      }
    }
    else {cin.get();}
  }

  cerr << "F TOTALS: " << FPredictor::getDomainSize() << " predictors, " << FResponse::getDomain().getSize() << " responses." << endl;
  cerr << "J TOTALS: " << JPredictor::getDomainSize() << " predictors, " << JResponse::getDomain().getSize() << " responses." << endl;

  for( auto& vfrv : mvfrv ) for( int i=0; i<iMaxNums; i++ ) if( vfrv.second(i) != 0.0 )
    { cout << "F " << vfrv.first.first[0].addNum(i) << "=1"; for( uint n=1; n<vfrv.first.first.size(); n++ ) cout << "," << vfrv.first.first[n] << "=1"; cout << " : " << vfrv.first.second << " = " << vfrv.second(i) << endl; }
  for( auto& pppm : mpppmP ) for( int i=0; i<iMaxNums; i++ ) for( int j=0; j<iMaxNums; j++ ) if( pppm.second(i,j) != 0.0 )
    cout << "P " << pppm.first.first.first() << " " << pppm.first.first.second() << " " << pppm.first.first.third() << " " << pppm.first.first.fourth().addNum(i) << " " << pppm.first.first.fifth() << " : " << pppm.first.second.addNum(j) << " = " << pppm.second(i,j) << endl;
  for( auto& ektwv : mektwvW ) for( int i=0; i<iMaxNums; i++ ) if( ektwv.second(i) != 0.0 )
    cout << "W " << ektwv.first.first() << " " << ektwv.first.second() << " " << ektwv.first.third().addNum(i) << " : " << ektwv.first.fourth() << " = " << ektwv.second(i) << endl;
  for( auto& vjrm : mvjrm ) for( int i=0; i<iMaxNums; i++ ) for( int j=0; j<iMaxNums; j++ ) if( vjrm.second(i,j) )
    { cout << "J " << vjrm.first.first[0].addNums(i,j) << "=1"; for( uint n=1; n<vjrm.first.first.size(); n++ ) cout << "," << vjrm.first.first[n] << "=1"; cout << " : " << vjrm.first.second << " = " << vjrm.second(i,j) << endl; }
  for( auto& apam : mapamA ) for( int i=0; i<iMaxNums; i++ ) for( int j=0; j<iMaxNums; j++ ) for( int k=0; k<iMaxNums; k++ ) if( apam.second(i,j*iMaxNums+k) != 0.0 )
    cout << "A " << apam.first.first.first() << " " << apam.first.first.second() << " " << apam.first.first.third() << " " << apam.first.first.fourth() << " " << apam.first.first.fifth() << " " << apam.first.first.sixth().addNum(j) << " " << apam.first.first.seventh().addNum(k) << " : " << apam.first.second.addNum(i) << " = " << apam.second(i,j*iMaxNums+k) << endl;
  for( auto& bpbm : mbpbmB ) for( int i=0; i<iMaxNums; i++ ) for( int j=0; j<iMaxNums; j++ ) for( int k=0; k<iMaxNums; k++ ) if( bpbm.second(i,j*iMaxNums+k) != 0.0 )
    cout << "B " << bpbm.first.first.first() << " " << bpbm.first.first.second() << " " << bpbm.first.first.third() << " " << bpbm.first.first.fourth() << " " << bpbm.first.first.fifth() << " " << bpbm.first.first.sixth() << " " << bpbm.first.first.seventh().addNum(i) << " " << bpbm.first.first.eighth().addNum(j) << " : " << bpbm.first.second.addNum(k) << " = " << bpbm.second(i,j*iMaxNums+k) << endl;
}


