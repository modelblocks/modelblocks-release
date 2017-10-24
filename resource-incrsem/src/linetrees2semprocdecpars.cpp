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
#include <StoreState.hpp>
#include <BerkUnkWord.hpp>
#include <Tree.hpp>

char psSpcColonSpc[]  = " : ";
char psSpcEqualsSpc[] = " = ";

////////////////////////////////////////////////////////////////////////////////

class LVU : public trip<L,arma::rowvec,arma::vec> {
 public:
  // Constructor methods...
  LVU ( )                { }
  LVU ( const char* ps ) { first() = ps; }
  // Accessor methods...
  operator       const L ( ) const { return first(); }
  const arma::rowvec&  v ( ) const { return second(); }
  const arma::vec&     u ( ) const { return third(); }
  arma::rowvec&        v ( )       { return second(); }
  arma::vec&           u ( )       { return third(); }
  // Input / output methods  ---  NOTE: only reads and writes label, not vectors...
  friend pair<istream&,LVU&> operator>> ( istream&            is,  LVU& t              )                 { return pair<istream&,LVU&>(is,t); }
  friend istream&            operator>> ( pair<istream&,LVU&> ist, const char* psDelim )                 { return pair<istream&,L&>(ist.first,ist.second.first())>>psDelim; }
  friend bool                operator>> ( pair<istream&,LVU&> ist, const vector<const char*>& vpsDelim ) { return pair<istream&,L&>(ist.first,ist.second.first())>>vpsDelim; }
  friend ostream&            operator<< ( ostream&            os,  const LVU&  t       )                 { return os<<t.first(); }
};

////////////////////////////////////////////////////////////////////////////////

map<L,double> mldLemmaCounts;
int MINCOUNTS = 100;
map<trip<T,T,T>,arma::mat> mtttmG;
map<pair<T,W>,arma::vec> mtwvL;
int iMaxNums = 0;
arma::mat eye3;
arma::mat mFail;
arma::mat mIdent;
arma::vec vOnes;
arma::vec vFirstHot;

map<pair<vector<FPredictor>,FResponse>,arma::vec> mvfrv;
map<pair<PPredictor,T>,arma::mat> mpppmP;
map<trip<K,T,W>,arma::vec> mktwvW;
map<pair<vector<JPredictor>,JResponse>,arma::mat> mvjrm;
map<pair<APredictor,T>,arma::mat> mapamA;
map<pair<BPredictor,T>,arma::mat> mbpbmB;

////////////////////////////////////////////////////////////////////////////////

const arma::mat normalize( const arma::mat& M, int n ) { double denom = arma::norm(M,n); return ( denom>0.0 ) ? M/denom : M; }
void normalize( arma::mat& M, int n, int dir ) {    //return M / (M * arma::ones(M.n_cols)); }
//cerr<<M<<endl;
  for( uint i=0; i<M.n_rows; i++ ) if( arma::accu(M.row(i))>0.0 ) M.row(i) /= arma::accu(M.row(i));
//cerr<<M<<endl;
}

////////////////////////////////////////////////////////////////////////////////

inline string regex_escape(const string& string_to_escape) {
    return regex_replace( string_to_escape, regex("[.^$|()\\[\\]{}*+?\\\\]"), string("\\\\$1") );
}

////////////////////////////////////////////////////////////////////////////////

int getArityGivenLabel ( const L& l ) {
  int depth = 0;
  int arity = 0;
  if ( l[0]=='N' ) arity++;
  for ( uint i=0; i<l.size(); i++ ) {
    if ( l[i]=='{' ) depth++;
    if ( l[i]=='}' ) depth--;
    if ( l[i]=='-' && l[i+1]>='a' && l[i+1]<='d' && depth==0 ) arity++;
  }
  return arity;
}

////////////////////////////////////////////////////////////////////////////////

O getOp ( const L& l, const L& lSibling, const L& lParent ) {
  if ( string::npos != l.find("-lN") || string::npos != l.find("-lG") || string::npos != l.find("-lH") || string::npos != l.find("-lR") ) return 'N';
  if ( string::npos != l.find("-lV") ) return 'V';
  if ( string::npos == l.find("-l")  || string::npos != l.find("-lS") || string::npos != l.find("-lC") ) return 'I';
  if ( string::npos != l.find("-lM") || string::npos != l.find("-lQ") || string::npos != l.find("-lU") ) return 'M';
  if ( (string::npos != l.find("-lA") || string::npos != l.find("-lI")) && string::npos != lParent.find("\\") ) return '0'+getArityGivenLabel( string(lParent,lParent.find("\\")+1) );
  if ( (string::npos != l.find("-lA") || string::npos != l.find("-lI")) && string::npos == lParent.find('\\') ) return '0'+getArityGivenLabel( lSibling );
  cerr << "ERROR: unhandled -l tag in label \"" << l << "\"" << endl;
  return O();
}

////////////////////////////////////////////////////////////////////////////////

E getExtr ( const Tree<LVU>& tr ) {
  N n =  T(L(tr).c_str()).getLastNonlocal();
  if ( n == N_NONE ) return 'N';
  if ( (tr.front().size()==0 || tr.front().front().size()==0) && n == N("-rN") ) return '0';
  if ( T(L(tr.front()).c_str()).getLastNonlocal()==n || T(L(tr.back()).c_str()).getLastNonlocal()==n ) return 'N';
  return ( n.isArg() ) ? T(L(tr).c_str()).getArity()+'1' : 'M';
}

////////////////////////////////////////////////////////////////////////////////

T T_COLON ( "Pk" );                       // must be changed to avoid confusion with ":" delimiter in K's (where type occurs individually).
T T_CONTAINS_COMMA ( "!containscomma!" ); // must be changed to avoid confusion with "," delimiter in F,J params.

T getType ( const L& l ) {
  if ( l[0]==':' )                 return T_COLON;
  if ( l.find(',')!=string::npos ) return T_CONTAINS_COMMA;
  return regex_replace( l, regex("-[xl](?:(?!-[a-z])[^ }])*"), string("") ).c_str();
//  return string( string( l, 0, l.find("-l") ), 0, l.find("-x") ).c_str();
}

////////////////////////////////////////////////////////////////////////////////

pair<K,T> getPred ( const L& lP, const L& lW ) {
  T t = getType ( lP );   //string( lP, 0, lP.find("-l") ).c_str();

  // If punct, but not special !-delimited label...
  if ( ispunct(lW[0]) && ('!'!=lW[0] || lW.size()==1) ) return pair<K,T>(K::kBot,t);

  string sLemma = lW;  transform(sLemma.begin(), sLemma.end(), sLemma.begin(), [](unsigned char c) { return std::tolower(c); });
  string sType = t.getString();  regex_replace( sType, regex("-x-x(?:(?!-[a-z])[^ }])*"), string("") );
  string sPred = sType + ':' + sLemma;

  smatch m; for( string s=lP; regex_match(s,m,regex("^(.*?)-x((?:(?!-[a-z])[^ }])*)(.*?)$")); s=m[3] ) {
    string sX = m[2];
    smatch mX;
    if( regex_match( sX, mX, regex("^(.*)%(.*)%(.*)[|](.*)%(.*)%(.*)$") ) )        // transfix (prefix+infix+suffix) rule application
      sPred = regex_replace( sPred, regex("^"+regex_escape(mX[1])+"(.*)"+regex_escape(mX[2])+"(.*)"+regex_escape(mX[3])+"$"), string(mX[4])+"$1"+string(mX[5])+"$2"+string(mX[6]) );
    if( regex_match( sX, mX, regex("^(.*)[%](.*)[|](.*)[%](.*)$") ) )              // circumfix (prefix+suffix) rule application
      sPred = regex_replace( sPred, regex("^"+regex_escape(mX[1])+"(.*)"+regex_escape(mX[2])+"$"), string(mX[3])+"$1"+string(mX[4]) );
  }

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

void calcContext ( const Tree<LVU>& tr, const arma::mat& D, const arma::mat& U, int s=1, int d=0, E e='N', L l=L() ) {
  static F          f;
  static E          eF;
  static Sign       aPretrm;
  static StoreState q;
  static arma::mat eye3( iMaxNums, iMaxNums*iMaxNums, arma::fill::zeros );  if( eye3(0,0)==0.0 ) for( int i=0; i<iMaxNums; i++ ) eye3(i,i*iMaxNums+i)=1.0;    // Init 3D diag.

  if( l==L() ) l = L(tr);

  // At unary preterminal...
  if ( tr.size()==1 && tr.front().size()==0 ) {
    //// cerr<<"#T "<<getType(tr)<<" "<<L(tr.front())<<endl;

    f               = 1 - s;
    eF = e = ( e!='N' ) ? e : getExtr ( tr );
    pair<K,T> kt    = getPred ( L(tr), L(tr.front()) );
    K k             = (FEATCONFIG & 2) ? K::kBot : kt.first;
    aPretrm         = Sign( k, getType(l), S_A );

    // Print preterminal / fork-phase predictors...
    DelimitedList<psX,FPredictor,psComma,psX> lfp;  q.calcForkPredictors(lfp);
    cout<<"----"<<q<<endl;
////    cout << "F "; for ( auto& fp : lfp ) { if ( &fp!=&lfp.front() ) cout<<","; cout<<fp<<"=1"; }  cout << " : " << FResponse(f,e,k) << endl;
////    cout << "P " << q.calcPretrmTypeCondition(f,e,k) << " : " << aPretrm.getType() /*getType(l)*/     << endl;
////    cout << "W " << k << " " << aPretrm.getType() /*getType(l)*/           << " : " << L(tr.front())  << endl;

    arma::vec& vF = mvfrv[ pair<vector<FPredictor>,FResponse>( vector<FPredictor>( lfp.begin(), lfp.end() ), FResponse(f,e,k) ) ];
    arma::mat& mP = mpppmP[ pair<PPredictor,T>(q.calcPretrmTypeCondition(f,e,k),aPretrm.getType()) ];
    arma::vec& vW = mktwvW[ trip<K,T,W>(k,aPretrm.getType(),L(tr.front()).c_str()) ];
    vF = ((vF.n_elem) ? vF : arma::zeros(iMaxNums))          + normalize( D * U * tr.u(),                     1 );
    mP = ((mP.n_elem) ? mP : arma::zeros(iMaxNums,iMaxNums)) + normalize( D * arma::diagmat(U * tr.u()),      1 );
    vW = ((vW.n_elem) ? vW : arma::zeros(iMaxNums))          + normalize( (vOnes.t() * D).t() % (U * tr.u()), 1 );
//if( q.calcPretrmTypeCondition(f,e,k).fourth()==T("N-aD") and getType(tr)==T("R-aN") ) 
//cout<<"D for "<<tr<<" with b="<<q.calcPretrmTypeCondition(f,e,k).fourth()<<endl<<D<<endl<<tr.u()<<endl;
  }

  // At unary nonpreterminal...
  else if ( tr.size()==1 ) {
    //// cerr<<"#U"<<getType(tr)<<" "<<getType(tr.front())<<endl;
    e = ( e!='N' ) ? e : getExtr ( tr );
    calcContext ( tr.front(), D, (getType(tr)==getType(tr.front())) ? U : U * mtttmG[trip<T,T,T>(getType(tr),getType(tr.front()),"-")] * arma::kron(mIdent,vOnes), s, d, e, l );
//cout<<"unary at "<<L(tr)<<endl<<mtttmG[trip<T,T,T>(getType(tr),getType(tr.front()),"-")]<<endl;
  }

  // At binary nonterminal...
  else if ( tr.size()==2 ) {
    //// cerr<<"#B "<<getType(tr)<<" "<<getType(tr.front())<<" "<<getType(tr.back())<<endl;

    // Traverse left child...
    calcContext ( tr.front(), D * U * getG( getType(tr), getType(tr.front()), getType(tr.back()) ) * arma::kron( mIdent, tr.back().u() ), mIdent, 0, d+s );

    J j          = s;
    LeftChildSign aLchild ( q, f, eF, aPretrm );
    e            = ( e!='N' ) ? e : getExtr ( tr ) ;
    O oL         = getOp ( L(tr.front()), L(tr.back()),  L(tr) );
    O oR         = getOp ( L(tr.back()),  L(tr.front()), L(tr) );

    // Print binary / join-phase predictors...
    DelimitedList<psX,JPredictor,psComma,psX> ljp;  q.calcJoinPredictors(ljp,f,eF,aLchild);
    cout << "==== " << aLchild << "   " << L(tr) << " -> " << L(tr.front()) << " " << L(tr.back()) << endl;
////    cout << "J ";  for ( auto& jp : ljp ) { if ( &jp!=&ljp.front() ) cout<<","; cout<<jp<<"=1"; }  cout << " : " << JResponse(j,e,oL,oR)  << endl;
////    cout << "A " << q.calcApexTypeCondition(f,j,eF,e,oL,aLchild)                  << " : " << getType(l)          << endl;
////    cout << "B " << q.calcBrinkTypeCondition(f,j,eF,e,oL,oR,getType(l),aLchild)   << " : " << getType(tr.back())  << endl;
    APredictor apred = q.calcApexTypeCondition(f,j,eF,e,oL,aLchild);
    BPredictor bpred = q.calcBrinkTypeCondition(f,j,eF,e,oL,oR,getType(l),aLchild);

    // Update storestate...
    q = StoreState ( q, f, j, eF, e, oL, oR, getType(l), getType(tr.back()), aPretrm, aLchild );

    // Traverse right child...
    calcContext ( tr.back(), arma::diagmat( tr.back().v() ), mIdent, 1, d );
//    calcContext ( tr.back(), arma::diagmat( tr.v() * getG( getType(tr), getType(tr.front()), getType(tr.back()) ) * arma::kron( vOnes /*tr.front().u()*/, mIdent ) ), 1, d );

    arma::mat& mJ = mvjrm [ pair<vector<JPredictor>,JResponse>( vector<JPredictor>( ljp.begin(), ljp.end() ), JResponse(j,e,oL,oR) ) ];
    arma::mat& mA = mapamA[ pair<APredictor,T>(apred,getType(l)) ];
    arma::mat& mB = mbpbmB[ pair<BPredictor,T>(bpred,getType(tr.back())) ];
    arma::mat tmp; if( j ) tmp = tr.front().u()*vFirstHot.t(); else tmp = arma::diagmat(tr.front().u());
    mJ = ((mJ.n_elem) ? mJ : arma::zeros(iMaxNums,iMaxNums))          + normalize( D * U * getG(getType(tr),getType(tr.front()),getType(tr.back())) * arma::kron(mIdent,tr.back().u()),                                                   1 );
    mA = ((mA.n_elem) ? mA : arma::zeros(iMaxNums,iMaxNums*iMaxNums)) + normalize( eye3 * arma::kron( D.t(), U * getG(getType(tr),getType(tr.front()),getType(tr.back())) * arma::kron(tmp,tr.back().u()) ),                              1 );
    mB = ((mB.n_elem) ? mB : arma::zeros(iMaxNums,iMaxNums*iMaxNums)) + normalize( arma::diagmat(vOnes.t() * D) * U * getG(getType(tr),getType(tr.front()),getType(tr.back())) * arma::diagmat(arma::kron(tr.front().u(),tr.back().u())), 1 );
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
  while ( cin && EOF!=cin.peek() ) {
    linenum++;
    Tree<LVU> t("T"); t.emplace_back(); t.emplace_back("T");
    cin >> t.front() >> "\n";
    cout.flush();
    cout << "TREE " << linenum << ": " << t << "\n";
    setBackwardMessages( t );
    setForwardMessages( t, vFirstHot.t() );
    if( t.front().size() > 0 ) calcContext( t, arma::diagmat(vFirstHot), mIdent );
  }

  cerr << "F TOTALS: " << FPredictor::getDomainSize() << " predictors, " << FResponse::getDomain().getSize() << " responses." << endl;
  cerr << "J TOTALS: " << JPredictor::getDomainSize() << " predictors, " << JResponse::getDomain().getSize() << " responses." << endl;

  for( auto& vfrv : mvfrv ) for( int i=0; i<iMaxNums; i++ ) if( vfrv.second(i) != 0.0 )
    { cout << "F " << vfrv.first.first[0].addNum(i) << "=1"; for( uint n=1; n<vfrv.first.first.size(); n++ ) cout << "," << vfrv.first.first[n] << "=1"; cout << " : " << vfrv.first.second << " = " << vfrv.second(i) << endl; }
  for( auto& pppm : mpppmP ) for( int i=0; i<iMaxNums; i++ ) for( int j=0; j<iMaxNums; j++ ) if( pppm.second(i,j) != 0.0 )
    cout << "P " << pppm.first.first.first() << " " << pppm.first.first.second() << " " << pppm.first.first.third() << " " << pppm.first.first.fourth().addNum(i) << " " << pppm.first.first.fifth() << " : " << pppm.first.second.addNum(j) << " = " << pppm.second(i,j) << endl;
  for( auto& ktwv : mktwvW ) for( int i=0; i<iMaxNums; i++ ) if( ktwv.second(i) != 0.0 )
    cout << "W " << ktwv.first.first() << " " << ktwv.first.second().addNum(i) << " : " << ktwv.first.third() << " = " << ktwv.second(i) << endl;
  for( auto& vjrm : mvjrm ) for( int i=0; i<iMaxNums; i++ ) for( int j=0; j<iMaxNums; j++ ) if( vjrm.second(i,j) )
    { cout << "J " << vjrm.first.first[0].addNums(i,j) << "=1"; for( uint n=1; n<vjrm.first.first.size(); n++ ) cout << "," << vjrm.first.first[n] << "=1"; cout << " : " << vjrm.first.second << " = " << vjrm.second(i,j) << endl; }
  for( auto& apam : mapamA ) for( int i=0; i<iMaxNums; i++ ) for( int j=0; j<iMaxNums; j++ ) for( int k=0; k<iMaxNums; k++ ) if( apam.second(i,j*iMaxNums+k) != 0.0 )
    cout << "A " << apam.first.first.first() << " " << apam.first.first.second() << " " << apam.first.first.third() << " " << apam.first.first.fourth() << " " << apam.first.first.fifth() << " " << apam.first.first.sixth().addNum(j) << " " << apam.first.first.seventh().addNum(k) << " : " << apam.first.second.addNum(i) << " = " << apam.second(i,j*iMaxNums+k) << endl;
  for( auto& bpbm : mbpbmB ) for( int i=0; i<iMaxNums; i++ ) for( int j=0; j<iMaxNums; j++ ) for( int k=0; k<iMaxNums; k++ ) if( bpbm.second(i,j*iMaxNums+k) != 0.0 )
    cout << "B " << bpbm.first.first.first() << " " << bpbm.first.first.second() << " " << bpbm.first.first.third() << " " << bpbm.first.first.fourth() << " " << bpbm.first.first.fifth() << " " << bpbm.first.first.sixth() << " " << bpbm.first.first.seventh().addNum(i) << " " << bpbm.first.first.eighth().addNum(j) << " : " << bpbm.first.second.addNum(k) << " = " << bpbm.second(i,j*iMaxNums+k) << endl;
}

