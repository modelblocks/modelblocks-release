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

#include <typeinfo>
#include <regex>

const uint SEM_SIZE = 20;
const uint SYN_SIZE = 20;

// XModel E, K, P, Char, RNN hidden sizes
const uint X_E_SIZE = 20;
const uint X_K_SIZE = 400;
const uint X_P_SIZE = 20;
const uint X_C_SIZE = 20;
const uint X_H_SIZE = 460;

// MModel E, P, LCat, Char, RNN hidden sizes
const uint M_E_SIZE = 20;
const uint M_P_SIZE = 20;
const uint M_L_SIZE = 400;
const uint M_C_SIZE = 20;
const uint M_H_SIZE = 460;

vector<string> PUNCT = { "-LCB-", "-LRB-", "-RCB-", "-RRB-" };

arma::mat relu( const arma::mat& km ) {
  if ( km.max() < 0 ) return zeros( size(km) );
  else return clamp(km, 0, km.max());
}

// for semantic ablation
class TVec : public DelimitedCol<psLBrack, double, psComma, psRBrack> {
  public:
    TVec ( )                       : DelimitedCol<psLBrack, double, psComma, psRBrack>(SEM_SIZE) { }
    TVec ( const Col<double>& kv ) : DelimitedCol<psLBrack, double, psComma, psRBrack>(kv)       { }
    TVec& add( const TVec& kv ) { *this += kv; return *this; }
};

const TVec foo   ( arma::zeros<Col<double>>(SEM_SIZE) );

class NPredictorVec {
//need to be able to output real-valued distance integers, NPreds
//TODO maybe try quadratic distance
  private:

     int mdist;
     list<unsigned int> mnpreds;

  public:

    //constructor
    template<class LM>
    NPredictorVec( LM& lm, const Sign& candidate, bool bcorefON, int antdist, const StoreState& ss, bool ABLATE_UNARY) : mdist(antdist), mnpreds() {
//      //probably will look like Join model feature generation.ancestor is a sign, sign has T and Kset.
//      //TODO add dependence to P model.  P category should be informed by which antecedent category was chosen here
//
// //     mdist = antdist;
//      mnpreds.emplace_back( lm.getPredictorIndex( "bias" ) ); //add bias term
//
//      const HVec& hvB = ss.at(ss.size()-1).getHVec(); //contexts of lowest b (bdbar)
//      for( unsigned int iA=0; iA<candidate.getHVec().size(); iA++ )  for( auto& antk : candidate.getHVec()[iA] ) {
//        mnpreds.emplace_back( lm.getPredictorIndex( antk.project(-iA), kNil ) ); //add unary antecedent k feat, using kxk template
//        for( unsigned int iB=0; iB<hvB.size(); iB++)  for( auto& currk : hvB[iB] ) {
//          mnpreds.emplace_back( lm.getPredictorIndex( antk.project(-iA), currk.project(-iB) ) ); //pairwise kxk feat
//        }
//      }
//      for( unsigned int iB=0; iB<hvB.size(); iB++ )  for( auto& currk : hvB[iB] ) {
//        mnpreds.emplace_back( lm.getPredictorIndex( kNil, currk.project(-iB) ) ); //unary ancestor k feat
//      }
//
//      mnpreds.emplace_back( lm.getPredictorIndex( candidate.getCat(), N_NONE                      ) ); // antecedent CVar
//      mnpreds.emplace_back( lm.getPredictorIndex( N_NONE,             ss.at(ss.size()-1).getCat() ) ); // ancestor CVar
//      mnpreds.emplace_back( lm.getPredictorIndex( candidate.getCat(), ss.at(ss.size()-1).getCat() ) ); // pairwise T
//
//      //corefON feature
//      if (bcorefON == true) {
//        mnpreds.emplace_back( lm.getPredictorIndex( "corefON" ) );
//      }
    }

    const list<unsigned int>& getList    ( ) const { return mnpreds; }
    int                       getAntDist ( ) const { return mdist;   }
};

////////////////////////////////////////////////////////////////////////////////

class NModel {

  private:

    arma::mat matN;                              // matrix itself

    unsigned int iNextPredictor = 0;             // predictor and response next-pointers
    unsigned int iNextResponse  = 0;

    map<unsigned int,string>    mis;
    map<string,unsigned int>    msi;

    map<pair<K,K>,unsigned int>       mkki;
    map<unsigned int,pair<K,K>>       mikk;

    map<pair<CVar,CVar>,unsigned int> mcci; //pairwise CVarCVar? probably too sparse...
    map<unsigned int,pair<CVar,CVar>> micc;

  public:

    NModel( ) { }
    NModel( istream& is ) {
      list< trip< unsigned int, unsigned int, double > > l;    // store elements on list until we know dimensions of matrix
      while( is.peek()=='N' ) {
        auto& prw = *l.emplace( l.end() );
        is >> "N ";
        if( is.peek()=='a' )   { Delimited<string> s;   is >> "a" >> s >> " : ";                 prw.first()  = getPredictorIndex( s );      }
        else{
          if( is.peek()=='t' ) { Delimited<CVar> cA,cB; is >> "t" >> cA >> "&t" >> cB >> " : ";  prw.first()  = getPredictorIndex( cA, cB ); }
          else                 { Delimited<K>    kA,kB; is >> kA >> "&" >> kB >> " : ";          prw.first()  = getPredictorIndex( kA, kB ); }
        }
        Delimited<int> n;                               is >> n >> " = ";                        prw.second() = n;
        Delimited<double> w;                            is >> w >> "\n";                         prw.third()  = w;
      }

      if( l.size()==0 ) cerr << "ERROR: No N items found." << endl;
      matN.zeros ( 2, iNextPredictor );
      for( auto& prw : l ) { matN( prw.second(), prw.first() ) = prw.third(); }
    }

    unsigned int getPredictorIndex( const string& s ) {
      const auto& it = msi.find( s );  if( it != msi.end() ) return( it->second );
      msi[ s ] = iNextPredictor;  mis[ iNextPredictor ] = s;  return( iNextPredictor++ );
    }
    unsigned int getPredictorIndex( const string& s ) const {                  // const version with closed predictor domain
      const auto& it = msi.find( s );  return( ( it != msi.end() ) ? it->second : 0 );
    }

    unsigned int getPredictorIndex( K kA, K kB ) {
      const auto& it = mkki.find( pair<K,K>(kA,kB) );  if( it != mkki.end() ) return( it->second );
      mkki[ pair<K,K>(kA,kB) ] = iNextPredictor;  mikk[ iNextPredictor ] = pair<K,K>(kA,kB);  return( iNextPredictor++ );
    }
    unsigned int getPredictorIndex( K kA, K kB ) const {                       // const version with closed predictor domain
      const auto& it = mkki.find( pair<K,K>(kA,kB) );  return( ( it != mkki.end() ) ? it->second : 0 );
    }

    unsigned int getPredictorIndex( CVar cA, CVar cB ) {
      const auto& it = mcci.find( pair<CVar,CVar>(cA,cB) );  if( it != mcci.end() ) return( it->second );
      mcci[ pair<CVar,CVar>(cA,cB) ] = iNextPredictor;  micc[ iNextPredictor ] = pair<CVar,CVar>(cA,cB);  return( iNextPredictor++ );
    }
    unsigned int getPredictorIndex( CVar cA, CVar cB ) const {                 // const version with closed predictor domain
      const auto& it = mcci.find( pair<CVar,CVar>(cA,cB) );  return( ( it != mcci.end() ) ? it->second : 0 );
    }

    arma::vec calcLogResponses( const NPredictorVec& npv ) const {
      arma::vec nlogresponses = arma::ones( 2 );
//      nlogresponses += npv.getAntDist() * matN.col(getPredictorIndex("ntdist"));
//      for ( auto& npredr : npv.getList() ) {
//        if ( npredr < matN.n_cols ) {
//          nlogresponses += matN.col( npredr );
//        }
//      }
      return nlogresponses;
    }

    friend ostream& operator<<( ostream& os, const pair< const NModel&, const NPredictorVec& >& mv ) {
      os << "antdist=" << mv.second.getAntDist();
      for( const auto& i : mv.second.getList() ) {
        // if( &i != &mv.second.getList().front() )
        os << ",";
        const auto& itK = mv.first.mikk.find(i);
       	if( itK != mv.first.mikk.end() ) { os << itK->second.first << "&" << itK->second.second << "=1"; continue; }
        const auto& itC = mv.first.micc.find(i);
        if( itC != mv.first.micc.end() ) { os << "t" << itC->second.first << "&t" << itC->second.second << "=1"; continue; }
        const auto& itS = mv.first.mis.find(i);
        if( itS != mv.first.mis.end()  ) { os << "a" << itS->second << "=1"; }
      }
      return os;
    }

    unsigned int getNumPredictors( ) { return iNextPredictor; }
    unsigned int getNumResponses(  ) { return iNextResponse;  }
};

////////////////////////////////////////////////////////////////////////////////

class FPredictorVec {

  private:
    int d;
    int iCarrier;
    const HVec& hvB;
    const HVec& hvF;
    CVar catBase;

    public:
    template<class FM>  // J model is template variable to allow same behavior for const and non-const up until getting predictor indices
    FPredictorVec( FM& fm, const HVec& hvAnt, bool nullAnt, const StoreState& ss ) : hvB (( ss.getBase().getHVec().size() > 0 ) ? ss.getBase().getHVec() : hvBot), hvF (( ss.getBase().getCat().getNoloArity() && ss.getNoloBack().getHVec().size() != 0 ) ? ss.getNoloBack().getHVec() : hvBot){
      d = (FEATCONFIG & 1) ? 0 : ss.getDepth();
      catBase = ss.getBase().getCat();
    }

    int getD() {
        return d;
    }
    const HVec& getHvB() {
        return hvB;
    }
    const HVec& getHvF() {
        return hvF;
    }
    CVar getCatBase() {
        return catBase;
    }

    friend ostream& operator<< ( ostream& os, const FPredictorVec& fpv ) {
      os << fpv.d << " " << fpv.catBase << " " << fpv.hvB << " " << fpv.hvF;
      return os;
    }
};

////////////////////////////////////////////////////////////////////////////////

class FModel {

  typedef DelimitedTrip<psX,F,psAmpersand,Delimited<EVar>,psAmpersand,Delimited<K>,psX> FEK;
  typedef DelimitedCol<psLBrack, double, psComma, psRBrack> DenseVec;

  private:

    map<CVar,vec> mcbv;                        // map between syntactic category and embeds
    map<KVec,vec> mkbdv;                       // map between KVec and embeds
    map<KVec,vec> mkfdv;

    map<FEK,unsigned int> mfeki;               // response indices
    map<unsigned int,FEK> mifek;

    unsigned int iNextResponse  = 0;

//    7+2*sem+syn
    DelimitedVector<psX, double, psComma, psX> fwf;  // weights for F model
    DelimitedVector<psX, double, psComma, psX> fws;
    DelimitedVector<psX, double, psComma, psX> fbf;  // biases for F model
    DelimitedVector<psX, double, psComma, psX> fbs;
    mat fwfm;
    mat fwsm;
    vec fbfv;
    vec fbsv;

  public:

    FModel( ) { }
    FModel( istream& is ) {
      while ( is.peek()=='F' ) {
        Delimited<char> c;
        is >> "F " >> c >> " ";
        if (c == 'F') is >> fwf >> "\n";
        if (c == 'f') is >> fbf >> "\n";
        if (c == 'S') is >> fws >> "\n";
        if (c == 's') is >> fbs >> "\n";
      }
      while ( is.peek()=='C' ) {
        Delimited<char> c;
        Delimited<CVar> cv;
        DenseVec dv = DenseVec(SYN_SIZE);
        is >> "C " >> c >> " " >> cv >> " " >> dv >> "\n";
        if (c == 'B') mcbv.try_emplace(cv, vec(dv));
      }
      while ( is.peek()=='K' ) {
        Delimited<char> c;
        Delimited<K> k;
        DenseVec dv = DenseVec(SEM_SIZE);
        is >> "K " >> c >> " " >> k >> " " >> dv >> "\n";
        if (c == 'B') mkbdv.try_emplace(k, vec(dv));
        else if (c == 'F') mkfdv.try_emplace(k, vec(dv));
      }
      while ( is.peek()=='f' ) {
        unsigned int i;
        is >> "f " >> i >> " ";
        is >> mifek[i] >> "\n";
        mfeki[mifek[i]] = i;
      }
      fwfm = fwf;
      fwsm = fws;
      fbfv = fbf;
      fbsv = fbs;
      fwfm.reshape(fwf.size()/(7 + 2*SEM_SIZE + SYN_SIZE), 7 + 2*SEM_SIZE + SYN_SIZE);
      fwsm.reshape(fws.size()/(fwf.size()/(7 + 2*SEM_SIZE + SYN_SIZE)), (fwf.size()/(7 + 2*SEM_SIZE + SYN_SIZE)));
    }

    const FEK& getFEK( unsigned int i ) const {
      auto it = mifek.find( i );
      assert( it != mifek.end() );
      return it->second;
    }

    const vec getCatEmbed( CVar i, Delimited<char> c) const {
      if (c == 'B') {
        auto it = mcbv.find( i );
        return ( ( it != mcbv.end() ) ? it->second : arma::zeros(SYN_SIZE) );
      }
      cerr << "ERROR: F model CVar position misspecified." << endl;
      return arma::zeros(SYN_SIZE);
    }

    const vec getKVecEmbed( HVec hv, Delimited<char> c ) const {
      vec KVecEmbed = arma::zeros(SEM_SIZE);
      if (c == 'B') {
        for ( auto& kV : hv.at(0) ) {
          if ( kV == K::kTop) {
            KVecEmbed += arma::ones(SEM_SIZE);
            continue;
          }
          auto it = mkbdv.find( kV );
          if ( it == mkbdv.end() ) {
            continue;
          } else {
            KVecEmbed += it->second;
          }
        }
      }
      else if (c == 'F') {
        for ( auto& kV : hv.at(0) ) {
          if ( kV == K::kTop) {
            KVecEmbed += arma::ones(SEM_SIZE);
            continue;
          }
          auto it = mkfdv.find( kV );
          if ( it == mkfdv.end() ) {
            continue;
          } else {
            KVecEmbed += it->second;
          }
        }
      }
      else cerr << "ERROR: F model KVec position misspecified." << endl;
      return KVecEmbed;
    }

    unsigned int getResponseIndex( F f, EVar e, K k ) {
      const auto& it = mfeki.find( FEK(f,e,k) );  if( it != mfeki.end() ) return( it->second );
      mfeki[ FEK(f,e,k) ] = iNextResponse;  mifek[ iNextResponse ] = FEK(f,e,k);  return( iNextResponse++ );
    }

    unsigned int getResponseIndex( F f, EVar e, K k ) const {                  // const version with closed predictor domain
      const auto& it = mfeki.find( FEK(f,e,k) );
      return ( ( it != mfeki.end() ) ? it->second : uint(-1) );
    }

    arma::vec calcResponses( FPredictorVec& lfpredictors ) const {
// return distribution over FEK indices
// vectorize predictors: one-hot for depth, two hvecs, one cat-embed
      CVar catB = lfpredictors.getCatBase();
      const HVec& hvB = lfpredictors.getHvB();
      const HVec& hvF = lfpredictors.getHvF();
      int d = lfpredictors.getD();

      const vec& catBEmb = getCatEmbed(catB, 'B');
      const vec& hvBEmb = getKVecEmbed(hvB, 'B');
      const vec& hvFEmb = getKVecEmbed(hvF, 'F');

// populate predictor vector
      arma::vec flogresponses = join_cols(join_cols(join_cols(catBEmb, hvBEmb), hvFEmb), arma::zeros(7));
      flogresponses(2*SEM_SIZE + SYN_SIZE + d) = 1;

// implementation of MLP
      arma::vec flogscores = fwsm * relu(fwfm*flogresponses + fbfv) + fbsv;
      arma::vec fscores = arma::exp(flogscores);
      double fnorm = arma::accu(fscores);
      return fscores/fnorm;
    }

  arma::vec testCalcResponses( arma::vec testvec ) const {
      arma::vec flogscores = fwsm * relu(fwfm*testvec + fbfv) + fbsv;
      arma::vec fscores = arma::exp(flogscores);
      double fnorm = arma::accu(fscores);
      return fscores/fnorm;
    }
};

////////////////////////////////////////////////////////////////////////////////

class WPredictor : public DelimitedTrip<psX,Delimited<EVar>,psSpace,Delimited<K>,psSpace,Delimited<CVar>,psX> {
  public:
    WPredictor ( ) : DelimitedTrip<psX,Delimited<EVar>,psSpace,Delimited<K>,psSpace,Delimited<CVar>,psX>(){}
    WPredictor ( EVar e, K k, CVar c ) : DelimitedTrip<psX,Delimited<EVar>,psSpace,Delimited<K>,psSpace,Delimited<CVar>,psX>(e,k,c){}
};


class WModel {

  typedef DelimitedTrip<psX,Delimited<EVar>,psSlash,Delimited<K>,psSlash,Delimited<CVar>,psX> WPredictor;
  typedef DelimitedTrip<psX,Delimited<EVar>,psSlash,Delimited<CVar>,psSlash,Delimited<string>,psX> MPredictor;
  typedef DelimitedCol<psLBrack, double, psComma, psRBrack> DenseVec;

  private:

    // map from pair<lemma, primcat> to list of compatible WPredictors and MPredictors (read in from WModel)
    map<pair<string,string>,DelimitedList<psLBrack,WPredictor,psSpace,psRBrack>> mxwp;
    map<pair<string,string>,DelimitedList<psLBrack,MPredictor,psSpace,psRBrack>> mxmp;

    // map from predictor components to dense vectors
    // XModel (E, K, P, char to dense vector)
    map<EVar,vec> mxev;
    map<K,vec> mxkv;
    map<CVar,vec> mxpv;
    map<string,vec> mxcv;
    // MModel (E, P, SK, char to dense vector)
    map<EVar,vec> mmev;
    map<CVar,vec> mmpv;
    map<string,vec> mmlv;
    map<string,vec> mmcv;

    map<string,unsigned int> mci; // map from character to index (required for indexing probabilities)
    map<string,unsigned int> mmi; // map from morph rule to index (required for indexing probabilities)

    // weights and biases for XModel
    DelimitedVector<psX, double, psComma, psX> xihw;  // SRN i2h
    DelimitedVector<psX, double, psComma, psX> xhhw;  // SRN h2h
    DelimitedVector<psX, double, psComma, psX> xfcw;  // FC classifier
    DelimitedVector<psX, double, psComma, psX> xihb;  // SRN i2h bias
    DelimitedVector<psX, double, psComma, psX> xhhb;  // SRN h2h bias
    DelimitedVector<psX, double, psComma, psX> xfcb;  // FC classifier bias
    mat xihwm;
    mat xhhwm;
    mat xfcwm;
    vec xihbv;
    vec xhhbv;
    vec xfcbv;

    // weights and biases for MModel
    DelimitedVector<psX, double, psComma, psX> mihw;  // SRN i2h
    DelimitedVector<psX, double, psComma, psX> mhhw;  // SRN h2h
    DelimitedVector<psX, double, psComma, psX> mfcw;  // FC classifier
    DelimitedVector<psX, double, psComma, psX> mihb;  // SRN i2h bias
    DelimitedVector<psX, double, psComma, psX> mhhb;  // SRN h2h bias
    DelimitedVector<psX, double, psComma, psX> mfcb;  // FC classifier bias
    mat mihwm;
    mat mhhwm;
    mat mfcwm;
    vec mihbv;
    vec mhhbv;
    vec mfcbv;

  public:

    // map from W to map from WPredictor to P(W | WPredictor)
    typedef map<WPredictor,double> WPPMap;
    typedef map<W,WPPMap> WWPPMap;
    // map from pair<lemma, primcat> to vector of P(lemma | WPredictor)
    typedef map<pair<string,string>,rowvec> XPMap;
    // map from pair<lemma, primcat> to matrix of P(M | WPredictor lemma)
    typedef map<pair<string,string>,mat> MPMap;

    WModel ( ) { }
    WModel ( istream& is ) {
      while( is.peek()=='W' ) {
        Delimited<char> i;
        Delimited<char> j;
        is >> "W " >> i >> " " >> j >> " ";
        if (i == 'X') {
          if (j == 'I') is >> xihw >> "\n";
          if (j == 'i') is >> xihb >> "\n";
          if (j == 'H') is >> xhhw >> "\n";
          if (j == 'h') is >> xhhb >> "\n";
          if (j == 'F') is >> xfcw >> "\n";
          if (j == 'f') is >> xfcb >> "\n";
        }
        else if (i == 'M') {
          if (j == 'I') is >> mihw >> "\n";
          if (j == 'i') is >> mihb >> "\n";
          if (j == 'H') is >> mhhw >> "\n";
          if (j == 'h') is >> mhhb >> "\n";
          if (j == 'F') is >> mfcw >> "\n";
          if (j == 'f') is >> mfcb >> "\n";
        }
      }
      while ( is.peek()=='E' ) {
        Delimited<char> i;
        Delimited<EVar> e;
        is >> "E " >> i >> " " >> e >> " ";
        if (i == 'X') {
          DenseVec dv = DenseVec(X_E_SIZE);
          is >> dv >> "\n";
          mxev.try_emplace(e, vec(dv));
        }
        else if (i == 'M') {
          DenseVec dv = DenseVec(M_E_SIZE);
          is >> dv >> "\n";
          mmev.try_emplace(e, vec(dv));
        }
      }
      while ( is.peek()=='K' ) {
        Delimited<K> k;
        DenseVec dv = DenseVec(X_K_SIZE);
        is >> "K " >> k >> " " >> dv >> "\n";
        mxkv.try_emplace(k, vec(dv));
      }
      while ( is.peek()=='P' ) {
        Delimited<char> i;
        Delimited<CVar> c;
        is >> "P " >> i >> " " >> c >> " ";
        if (i == 'X') {
          DenseVec dv = DenseVec(X_P_SIZE);
          is >> dv >> "\n";
          mxpv.try_emplace(c, vec(dv));
        }
        else if (i == 'M') {
          DenseVec dv = DenseVec(M_P_SIZE);
          is >> dv >> "\n";
          mmpv.try_emplace(c, vec(dv));
        }
      }
      while ( is.peek()=='L' ) {
        string l;
        DenseVec dv = DenseVec(M_L_SIZE);
        is >> "L " >> l >> " " >> dv >> "\n";
        mmlv.try_emplace(l, vec(dv));
      }
      while ( is.peek()=='C' ) {
        Delimited<char> i;
        string c;
        is >> "C " >> i >> " " >> c >> " ";
        if (i == 'X') {
          DenseVec dv = DenseVec(X_C_SIZE);
          is >> dv >> "\n";
          mxcv.try_emplace(c, vec(dv));
        }
        else if (i == 'M') {
          DenseVec dv = DenseVec(M_C_SIZE);
          is >> dv >> "\n";
          mmcv.try_emplace(c, vec(dv));
        }
        else if (i == 'I') {
          is >> mci[c] >> "\n";
        }
      }
      while ( is.peek()=='R' ) {
        string x;
        is >> "R " >> x >> " ";
        is >> mmi[x] >> "\n";
      }
      while ( is.peek()=='X' ) {
        string x;
        string p;
        DelimitedList<psLBrack,WPredictor,psSpace,psRBrack> wp;
        is >> "X " >> x >> " " >> p >> " " >> wp >> "\n";
        pair<string,string> xppair (x,p);
        mxwp.try_emplace(xppair, wp);
      }
      while ( is.peek()=='M' ) {
        string x;
        string p;
        DelimitedList<psLBrack,MPredictor,psSpace,psRBrack> mp;
        is >> "M " >> x >> " " >> p >> " " >> mp >> "\n";
        pair<string,string> xppair (x,p);
        mxmp.try_emplace(xppair, mp);
      }

      // initialize armadillo mat/vecs
      xihwm = xihw;
      xhhwm = xhhw;
      xfcwm = xfcw;
      xihbv = xihb;
      xhhbv = xhhb;
      xfcbv = xfcb;
      xihwm.reshape(X_H_SIZE, X_E_SIZE + X_K_SIZE + X_P_SIZE + X_C_SIZE);
      xhhwm.reshape(X_H_SIZE, X_H_SIZE);
      xfcwm.reshape(xfcw.size()/X_H_SIZE, X_H_SIZE);

      mihwm = mihw;
      mhhwm = mhhw;
      mfcwm = mfcw;
      mihbv = mihb;
      mhhbv = mhhb;
      mfcbv = mfcb;
      mihwm.reshape(M_H_SIZE, M_E_SIZE + M_P_SIZE + M_L_SIZE + M_C_SIZE);
      mhhwm.reshape(M_H_SIZE, M_H_SIZE);
      mfcwm.reshape(mfcw.size()/M_H_SIZE, M_H_SIZE);

    }

    // XModel: index input character embedding
    const mat getXCharMat( string a, unsigned int i ) const {
      auto it = mxcv.find( a );
      assert ( it != mxcv.end() );
      return repmat(it->second, 1, i);
    }

    // XModel: index list of compatible WPredictors given pair<lemma, primcat>
    const DelimitedList<psLBrack,WPredictor,psSpace,psRBrack> getWPredictorList( pair<string,string> xsp ) const {
      if ( isdigit(xsp.first.at(0)) ) {
        pair<string,string> numpair ("NUM", "All");
        return mxwp.find(numpair)->second;
      } else {
        auto it = mxwp.find( xsp );
        pair<string,string> unkpair ("UNK", xsp.second);
        auto unklist = mxwp.find(unkpair)->second;
        if ( it != mxwp.end() ) {
          auto predlist = it->second;
          unklist.splice(unklist.begin(), predlist);
          return unklist;
        } else {
          return unklist;
        }
      }
    }

    // XModel: index input WPredictor embedding given list of WPredictors
    const mat getWPredictorMat( DelimitedList<psLBrack,WPredictor,psSpace,psRBrack> lwp ) const {
      unsigned int idx = 0;
      mat wpmat = mat(X_E_SIZE + X_K_SIZE + X_P_SIZE, lwp.size());
      for ( auto& wp : lwp ) {
        auto ite = mxev.find( wp.first() );
        assert ( ite != mxev.end() );
        auto itk = mxkv.find( wp.second() );
        assert ( itk != mxkv.end() );
        auto itp = mxpv.find( wp.third() );
        assert ( itp != mxpv.end() );
        wpmat.col(idx) = join_cols(join_cols(ite->second, itk->second), itp->second);
        idx ++;
      }
      return wpmat;
    }

    // XModel: index input character index
    const unsigned int getXCharIndex( string a ) const {
      auto it = mci.find( a );
      assert ( it != mci.end() );
      return it->second;
    }

    // MModel: index input character embedding
    const mat getMCharMat( string a, unsigned int i ) const {
      auto it = mmcv.find( a );
      assert ( it != mmcv.end() );
      return repmat(it->second, 1, i);
    }

    // MModel: index list of compatible MPredictors given pair<lemma, primcat>
    const DelimitedList<psLBrack,MPredictor,psSpace,psRBrack> getMPredictorList( pair<string,string> xsp ) const {
      if ( isdigit(xsp.first.at(0)) ) {
        pair<string,string> numpair ("NUM", "All");
        return mxmp.find(numpair)->second;
      } else {
        auto it = mxmp.find( xsp );
        pair<string,string> unkpair ("UNK", xsp.second);
        auto unklist = mxmp.find(unkpair)->second;
        if ( it != mxmp.end() ) {
          auto predlist = it->second;
          unklist.splice(unklist.begin(), predlist);
          return unklist;
        } else {
          return unklist;
        }
      }
    }

    // MModel: index input MPredictor embedding given list of MPredictors
    const mat getMPredictorMat( DelimitedList<psLBrack,MPredictor,psSpace,psRBrack> lmp ) const {
      unsigned int idx = 0;
      mat mpmat = mat(M_E_SIZE + M_P_SIZE + M_L_SIZE, lmp.size());
      for ( auto& mp : lmp ) {
        auto ite = mmev.find( mp.first() );
        assert ( ite != mmev.end() );
        auto itp = mmpv.find( mp.second() );
        assert ( itp != mmpv.end() );
        auto itl = mmlv.find( mp.third() );
        assert ( itp != mmlv.end() );
        mpmat.col(idx) = join_cols(join_cols(ite->second, itp->second), itl->second);
        idx ++;
      }
      return mpmat;
    }

    // MModel: index morph rule index
    const unsigned int getMRuleIndex( string a ) const {
      auto it = mmi.find( a );
      assert ( it != mmi.end() );
      return it->second;
    }

    // takes input word and iterates over morph rules that are read in as part of the WModel
    // if morph rule can apply, generates <<lemma, primcat>, rule> and appends it to list
    const list<pair<pair<string,string>,string>> applyMorphRules ( const W& w_t ) const {
      list<pair<pair<string,string>,string>> lxmp;
      string sW = w_t.getString().c_str();

      // do not lowercase word if special punctuation token ("-LCB-", "-LRB-", "-RCB-", "-RRB-")
      if ( find( PUNCT.begin(), PUNCT.end(), sW ) == PUNCT.end() ) transform(sW.begin(), sW.end(), sW.begin(), [](unsigned char c) { return std::tolower(c); });

      // loop over morph rules
      for ( const auto& mi : mmi ) {
        smatch mM;
        string sX;
        string sP;

        // for identity or annihilator rules, return the word itself as lemma
        if ( mi.first == "%|%" || mi.first == "%|" ) {
          sX = sW;
          sP = "All";
          lxmp.push_back(pair<pair<string,string>,string>(pair<string,string>(sX,sP),mi.first));
        } else {
          // otherwise, apply morph rule for lemma and primcat
          if ( regex_match( mi.first, mM, regex("^(.*)[%](.*)[|](.*)[%](.*)$") ) ) {
            smatch mW;
            if ( regex_match(sW, mW, regex("^(.*)"+string(mM[2])+"$")) ) {
              sX = string(mW[1])+string(mM[4]);
              sP = string(mM[3]);
              lxmp.push_back(pair<pair<string,string>,string>(pair<string,string>(sX,sP),mi.first));
            }
          }
        }
      }
      return lxmp;
    }

    // XModel: calculate P(lemma | WPredictor)
    // takes input pair<lemma, primcat> and calculates RNN probabilities
    rowvec calcLemmaLikelihoods( const pair<string,string>& xsp, XPMap& xpmap ) const {
      auto it = xpmap.find( xsp );
      rowvec seqlogprobs;
      if ( it == xpmap.end() ) {
        // index list of compatible WPredictors
        auto wplist = getWPredictorList( xsp );
        string x_t = xsp.first;
        seqlogprobs = zeros<mat>(1, wplist.size());
        mat xihbm = repmat(xihbv, 1, wplist.size());
        mat xhhbm = repmat(xhhbv, 1, wplist.size());
        mat xfcbm = repmat(xfcbv, 1, wplist.size());
        mat wpmat = getWPredictorMat(wplist);
        // calculate first hidden state with start character <S>
        mat ht = relu(xihwm * join_cols(wpmat, getXCharMat("<S>", wplist.size())) + xihbm + xhhbm);
        mat st_scores = exp(xfcwm * ht + xfcbm);
        rowvec st_norm = sum(st_scores, 0);
        mat st_logprobs = log(st_scores.each_row() / st_norm);
        rowvec st1_logprobs;

        if ( find( PUNCT.begin(), PUNCT.end(), x_t ) != PUNCT.end() ) {
          // if lemma is special punctuation token ("-LCB-", "-LRB-", "-RCB-", "-RRB-"), index probability using the token itself
          seqlogprobs += st_logprobs.row( getXCharIndex( x_t ) );
          mat ht1 = relu(xihwm * join_cols(wpmat, getXCharMat(x_t, wplist.size())) + xihbm + xhhwm * ht + xhhbm);
          mat st1_scores = exp(xfcwm * ht1 + xfcbm);
          rowvec st1_norm = sum(st1_scores, 0);
          st1_logprobs = log(st1_scores.row(getXCharIndex( "<E>" )) / st1_norm);
          seqlogprobs += st1_logprobs;
        } else {
          string c0(1, x_t[0]);
          // index probability for first character
          seqlogprobs += st_logprobs.row( getXCharIndex( c0.c_str() ));

          for ( unsigned i = 0; i < x_t.length(); ++i ){
            string ct(1, x_t[i]);
            string ct1(1, x_t[i+1]);
            mat ht1 = relu(xihwm * join_cols(wpmat, getXCharMat(ct.c_str(), wplist.size())) + xihbm + xhhwm * ht + xhhbm);
            mat st1_scores = exp(xfcwm * ht1 + xfcbm);
            rowvec st1_norm = sum(st1_scores, 0);

            if (i != x_t.length()-1) {
              st1_logprobs = log(st1_scores.row(getXCharIndex( ct1.c_str() )) / st1_norm);
            } else {
              st1_logprobs = log(st1_scores.row(getXCharIndex( "<E>" )) / st1_norm);
            }
            seqlogprobs += st1_logprobs;
            ht = ht1;
          }
        }
        xpmap.try_emplace(xsp, seqlogprobs);
      }
      else {
        seqlogprobs = it->second;
      }
      return seqlogprobs;
    }

    // MModel: calculate P(M | WPredictor lemma)
    // takes input pair<lemma, primcat> and calculates RNN probabilities
    mat calcRuleLikelihoods( const pair<string,string>& xsp, MPMap& mpmap ) const {
      auto it = mpmap.find( xsp );
      mat rulelogprobs;
      if ( it == mpmap.end() ) {
        // index list of compatible MPredictors
        auto mplist = getMPredictorList( xsp );
        string x_t = xsp.first;
        mat mihbm = repmat(mihbv, 1, mplist.size());
        mat mhhbm = repmat(mhhbv, 1, mplist.size());
        mat mfcbm = repmat(mfcbv, 1, mplist.size());
        mat mpmat = getMPredictorMat(mplist);

        if ( find( PUNCT.begin(), PUNCT.end(), x_t ) != PUNCT.end() ) {
          // if lemma is special punctuation token ("-LCB-", "-LRB-", "-RCB-", "-RRB-"), calculate probability using the token itself
          mat ht = relu(mihwm * join_cols(mpmat, getMCharMat(x_t, mplist.size())) + mihbm + mhhbm);
          mat st_scores = exp(mfcwm * ht + mfcbm);
          rowvec st_norm = sum(st_scores, 0);
          rulelogprobs = log(st_scores.each_row() / st_norm);
        } else {
          string c0(1, x_t[0]);
          // calculate probability using first character
//          cerr << "word " << x_t << " c0 " << c0 << endl;
          mat ht = relu(mihwm * join_cols(mpmat, getMCharMat(c0.c_str(), mplist.size())) + mihbm + mhhbm);

          for ( unsigned i = 1; i < x_t.length(); ++i ){
            string ct(1, x_t[i]);
//            cerr << "word " << x_t << " c" << i << " " << ct << endl;
            mat ht1 = relu(mihwm * join_cols(mpmat, getMCharMat(ct.c_str(), mplist.size())) + mihbm + mhhwm * ht + mhhbm);
            ht = ht1;
          }
          mat st_scores = exp(mfcwm * ht + mfcbm);
          rowvec st_norm = sum(st_scores, 0);
          rulelogprobs = log(st_scores.each_row() / st_norm);
        }
        mpmap.try_emplace(xsp, rulelogprobs);
      }
      else {
        rulelogprobs = it->second;
      }
      return rulelogprobs;
    }

    void calcPredictorLikelihoods( const W& w_t, const WWPPMap& wwppmap, XPMap& xpmap, MPMap& mpmap, WPPMap& wppmap ) const {
      auto it = wwppmap.find( w_t );
      if ( it == wwppmap.end() ) {
        // generate list of <<lemma, primcat>, rule>
        list<pair<pair<string,string>,string>> lxmp = applyMorphRules(w_t);
        // loop over <<lemma, primcat>, rule>
        for ( const auto& xmp : lxmp ) {
          cerr << "generated word " << w_t << " from lemma " << xmp.first.first << ", primcat " << xmp.first.second << ", rule " << xmp.second << endl;
          DelimitedList<psLBrack,WPredictor,psSpace,psRBrack> lwp = getWPredictorList(xmp.first);
          rowvec xll = calcLemmaLikelihoods(xmp.first, xpmap);
          mat mllall = calcRuleLikelihoods(xmp.first, mpmap);
          rowvec mll = mllall.row(getMRuleIndex(xmp.second));
          rowvec wprobs = exp(xll + mll);
          unsigned int idx = 0;
          for ( const auto& wp : lwp ) {
            wppmap[wp] += wprobs(idx);
            idx ++;
          }
        }
      } else {
        wppmap = it->second;
      }
    }
};

////////////////////////////////////////////////////////////////////////////////

class JPredictorVec {

  private:
    int d;
    const Sign& aAncstr;
    const HVec& hvAncstr;
    const HVec& hvFiller;
    const HVec hvLchild;
    CVar catAncstr;
    CVar catLchild;

  public:
    template<class JM>  // J model is template variable to allow same behavior for const and non-const up until getting predictor indices
    JPredictorVec( JM& jm, F f, EVar eF, const LeftChildSign& aLchild, const StoreState& ss ) : aAncstr(ss.getBase()),
    hvAncstr (( aAncstr.getHVec().size()==0 ) ? hvBot : aAncstr.getHVec()),
    hvFiller (( ss.getBase().getCat().getNoloArity() && ss.getNoloBack().getHVec().size() != 0 ) ? ss.getNoloBack().getHVec() : hvBot),
    hvLchild (( aLchild.getHVec().size()==0 ) ? hvBot : aLchild.getHVec()){
      d = (FEATCONFIG & 1) ? 0 : ss.getDepth();
      catAncstr = ( aAncstr.getHVec().size()==0 ) ? cBot : aAncstr.getCat();
      catLchild = ( aLchild.getHVec().size()==0 ) ? cBot : aLchild.getCat();
    }

    int getD() {
        return d;
    }
    const HVec& getHvAncstr() {
        return hvAncstr;
    }
    const HVec& getHvFiller() {
        return hvFiller;
    }
    const HVec& getHvLchild() {
        return hvLchild;
    }
    CVar getCatAncstr() {
        return catAncstr;
    }
    CVar getCatLchild() {
        return catLchild;
    }

    friend ostream& operator<< ( ostream& os, const JPredictorVec& jpv ) {
      os << jpv.d << " " << jpv.catAncstr << " " << jpv.hvAncstr << " " << jpv.hvFiller << " " << jpv.catLchild << " " << jpv.hvLchild;
      return os;
    }
};


////////////////////////////////////////////////////////////////////////////////

class JModel {

  typedef DelimitedQuad<psX,J,psAmpersand,Delimited<EVar>,psAmpersand,O,psAmpersand,O,psX> JEOO;
  typedef DelimitedCol<psLBrack, double, psComma, psRBrack> DenseVec;
  unsigned int jr0;
  unsigned int jr1;

  private:

    map<CVar,vec> mcav;                        // map between syntactic category and embeds
    map<CVar,vec> mclv;
    map<KVec,vec> mkadv;                  // map between KVec and embeds
    map<KVec,vec> mkfdv;
    map<KVec,vec> mkldv;

    map<JEOO,unsigned int> mjeooi;               // response indices
    map<unsigned int,JEOO> mijeoo;

    unsigned int iNextResponse  = 0;

//    7+3*sem+2*syn
    DelimitedVector<psX, double, psComma, psX> jwf;  // weights for J model
    DelimitedVector<psX, double, psComma, psX> jws;
    DelimitedVector<psX, double, psComma, psX> jbf;  // biases for J model
    DelimitedVector<psX, double, psComma, psX> jbs;
    mat jwfm;
    mat jwsm;
    vec jbfv;
    vec jbsv;

  public:

    JModel() {
      jr0 = getResponseIndex( 0, EVar::eNil, 'N', O_I );
      jr1 = getResponseIndex( 1, EVar::eNil, 'N', O_I );
    }
    // read in weights, embeddings, and JEOOs
    JModel(istream& is) {
      while ( is.peek()=='J' ) {
        Delimited<char> c;
        is >> "J " >> c >> " ";
        if (c == 'F') is >> jwf >> "\n";
        if (c == 'f') is >> jbf >> "\n";
        if (c == 'S') is >> jws >> "\n";
        if (c == 's') is >> jbs >> "\n";
      }
      while ( is.peek()=='C' ) {
        Delimited<char> c;
        Delimited<CVar> cv;
        DenseVec dv = DenseVec(SYN_SIZE);
        is >> "C " >> c >> " " >> cv >> " " >> dv >> "\n";
        if (c == 'A') mcav.try_emplace(cv, vec(dv));
        else if (c == 'L') mclv.try_emplace(cv, vec(dv));
      }
      while ( is.peek()=='K' ) {
        Delimited<char> c;
        Delimited<K> k;
        DenseVec dv = DenseVec(SYN_SIZE);
        is >> "K " >> c >> " " >> k >> " " >> dv >> "\n";
        if (c == 'A') mkadv.try_emplace(k, vec(dv));
        else if (c == 'F') mkfdv.try_emplace(k, vec(dv));
        else if (c == 'L') mkldv.try_emplace(k, vec(dv));
      }
      while ( is.peek()=='j' ) {
        Delimited<int> k;
        is >> "j " >> k >> " ";
        is >> mijeoo[k] >> "\n";
        mjeooi[mijeoo[k]] = k;
      }
      jr0 = getResponseIndex( 0, EVar::eNil, 'N', O_I );
      jr1 = getResponseIndex( 1, EVar::eNil, 'N', O_I );
      jwfm = jwf;
      jwsm = jws;
      jbfv = jbf;
      jbsv = jbs;
      jwfm.reshape(jwf.size()/(7 + 3*SEM_SIZE + 2*SYN_SIZE), (7 + 3*SEM_SIZE + 2*SYN_SIZE));
      jwsm.reshape(jws.size()/(jwf.size()/(7 + 3*SEM_SIZE + 2*SYN_SIZE)), (jwf.size()/(7 + 3*SEM_SIZE + 2*SYN_SIZE)));
    }

    const JEOO& getJEOO( unsigned int i ) const {
      auto it = mijeoo.find( i );
      assert( it != mijeoo.end() );
      return it->second;
    }

    const vec getCatEmbed( CVar i, Delimited<char> c ) const {
      if (c == 'A') {
        auto it = mcav.find( i );
        return ( ( it != mcav.end() ) ? it->second : arma::zeros(SYN_SIZE) );
      }
      else if (c == 'L') {
        auto it = mclv.find( i );
        return ( ( it != mclv.end() ) ? it->second : arma::zeros(SYN_SIZE) );
      }
      cerr << "ERROR: J model CVar position misspecified." << endl;
      return arma::zeros(SYN_SIZE);
    }

    const vec getKVecEmbed( HVec hv, Delimited<char> c ) const {
      vec KVecEmbed = arma::zeros(SEM_SIZE);
      if (c == 'A') {
        for ( auto& kV : hv.at(0) ) {
          if ( kV == K::kTop) {
            KVecEmbed += arma::ones(SEM_SIZE);
            continue;
          }
          auto it = mkadv.find( kV );
          if ( it == mkadv.end() ) {
            continue;
          } else {
            KVecEmbed += it->second;
          }
        }
      }
      else if (c == 'F') {
        for ( auto& kV : hv.at(0) ) {
          if ( kV == K::kTop) {
            KVecEmbed += arma::ones(SEM_SIZE);
            continue;
          }
          auto it = mkfdv.find( kV );
          if ( it == mkfdv.end() ) {
            continue;
          } else {
            KVecEmbed += it->second;
          }
        }
      }
      else if (c == 'L') {
        for ( auto& kV : hv.at(0) ) {
          if ( kV == K::kTop) {
            KVecEmbed += arma::ones(SEM_SIZE);
            continue;
          }
          auto it = mkldv.find( kV );
          if ( it == mkldv.end() ) {
            continue;
          } else {
            KVecEmbed += it->second;
          }
        }
      }
      else cerr << "ERROR: J model KVec position misspecified." << endl;
      return KVecEmbed;
    }

    unsigned int getResponse0( ) const { return jr0; }
    unsigned int getResponse1( ) const { return jr1; }

    unsigned int getResponseIndex( J j, EVar e, O oL, O oR ) {
      const auto& it = mjeooi.find( JEOO(j,e,oL,oR) );  if( it != mjeooi.end() ) return( it->second );
      mjeooi[ JEOO(j,e,oL,oR) ] = iNextResponse;  mijeoo[ iNextResponse ] = JEOO(j,e,oL,oR);  return( iNextResponse++ );
    }

    unsigned int getResponseIndex( J j, EVar e, O oL, O oR ) const {           // const version with closed predictor domain
      const auto& it = mjeooi.find( JEOO(j,e,oL,oR) );  assert( it != mjeooi.end() );  return( ( it != mjeooi.end() ) ? it->second : uint(-1) );
    }

    arma::vec calcResponses( JPredictorVec& ljpredictors ) const {
// return distribution over JEOO indices
// vectorize predictors: one-hot for depth, three hvecs, two cat-embeds
      CVar catA = ljpredictors.getCatAncstr();
      const HVec& hvA = ljpredictors.getHvAncstr();
      const HVec& hvF = ljpredictors.getHvFiller();
      CVar catL = ljpredictors.getCatLchild();
      const HVec& hvL = ljpredictors.getHvLchild();
      int d = ljpredictors.getD();

      const vec& catAEmb = getCatEmbed(catA, 'A');
      const vec& catLEmb = getCatEmbed(catL, 'L');
      const vec& hvAEmb = getKVecEmbed(hvA, 'A');
      const vec& hvFEmb = getKVecEmbed(hvF, 'F');
      const vec& hvLEmb = getKVecEmbed(hvL, 'L');

// populate predictor vector
      arma::vec jlogresponses = join_cols(join_cols(join_cols(join_cols(join_cols(catAEmb, hvAEmb), hvFEmb), catLEmb), hvLEmb), arma::zeros(7));
      jlogresponses(3*SEM_SIZE + 2*SYN_SIZE + d) = 1;

// implementation of MLP
      arma::vec jlogscores = jwsm * relu(jwfm*jlogresponses + jbfv) + jbsv;
      arma::vec jscores = arma::exp(jlogscores);
      double jnorm = arma::accu(jscores);
      return jscores/jnorm;
    }

  arma::vec testCalcResponses( arma::vec testvec ) const {
      arma::vec jlogscores = jwsm * relu(jwfm*testvec + jbfv) + jbsv;
      arma::vec jscores = arma::exp(jlogscores);
      double jnorm = arma::accu(jscores);
      return jscores/jnorm;
    }
};

////////////////////////////////////////////////////////////////////////////////
