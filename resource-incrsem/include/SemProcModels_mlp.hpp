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

const uint SEM_SIZE = 20;
const uint SYN_SIZE = 20;
const uint WPRED_SIZE = 40;
const uint CHAR_SIZE = 20;
const uint RNNH_SIZE = 60;

vector<string> PUNCT = { "-LCB-", "-LRB-", "-RCB-", "-RRB-" };

//arma::mat relu( const arma::mat& km ) {
//  arma::mat A(km.n_rows, 1);
//  for ( unsigned int c = 0; c<km.n_rows; c++ ) {
//    if ( km(c,0) <= 0 ) {A(c,0)=(0.0);}
//    else A(c,0) = (km(c));
//  }
//  return A;
//}

arma::mat relu( const arma::mat& km ) {
  return clamp(km, 0, km.max());
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
        DenseVec dv = DenseVec(SYN_SIZE);
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

// map from W to WPredictors
class WModel {

  typedef DelimitedCol<psLBrack, double, psComma, psRBrack> DenseVec;

  private:

    map<string,mat> mcm;
    map<WPredictor,unsigned int> mwpi;
    map<string,unsigned int> mci;
//    map<W,list<DelimitedPair<psX,WPredictor,psSpace,Delimited<double>,psX>>> mymap;

    DelimitedVector<psX, double, psComma, psX> ihw;  // weights for SRN
    DelimitedVector<psX, double, psComma, psX> hhw;
    DelimitedVector<psX, double, psComma, psX> fcw;  // weights for FC layer
    DelimitedVector<psX, double, psComma, psX> ihb;  // biases for SRN
    DelimitedVector<psX, double, psComma, psX> hhb;
    DelimitedVector<psX, double, psComma, psX> fcb;  // biases for FC layer
    DelimitedVector<psX, double, psComma, psX> wpw;
    mat ihwm;
    mat hhwm;
    mat fcwm;
    mat wpwm;
    vec ihbv;
    vec hhbv;
    vec fcbv;

    mat ihbm;
    mat hhbm;
    mat fcbm;
    mat h1;
    mat s1_scores;
    rowvec s1_norm;
    mat s1_logprobs;

  public:

    // map between W and vector of P(W | WPredictor), each row represents a unique WPredictor
    typedef map<W,list<DelimitedPair<psX,WPredictor,psSpace,Delimited<double>,psX>>> MapWP;
    WModel ( ) { }
    WModel ( istream& is ) {
      while( is.peek()=='W' ) {
        Delimited<char> c;
        is >> "W " >> c >> " ";
        if (c == 'I') is >> ihw >> "\n";
        if (c == 'i') is >> ihb >> "\n";
        if (c == 'H') is >> hhw >> "\n";
        if (c == 'h') is >> hhb >> "\n";
        if (c == 'F') is >> fcw >> "\n";
        if (c == 'f') is >> fcb >> "\n";
        if (c == 'P') is >> wpw >> "\n";
      }
      while ( is.peek()=='p' ) {
        WPredictor wp;
        is >> "p " >> wp >> " ";
        is >> mwpi[wp] >> "\n";
      }
      while ( is.peek()=='A' ) {
        string a;
        DenseVec dv = DenseVec(CHAR_SIZE);
        is >> "A " >> a >> " " >> dv >> "\n";
        mcm.try_emplace(a, repmat(vec(dv), 1, mwpi.size()));
      }
      while ( is.peek()=='a' ) {
        string a;
        is >> "a " >> a >> " ";
        is >> mci[a] >> "\n";
      }
      ihwm = ihw;
      hhwm = hhw;
      fcwm = fcw;
      wpwm = wpw;
      ihbv = ihb;
      hhbv = hhb;
      fcbv = fcb;
      ihwm.reshape(RNNH_SIZE, WPRED_SIZE + CHAR_SIZE);
      hhwm.reshape(RNNH_SIZE, RNNH_SIZE);
      fcwm.reshape(fcw.size()/RNNH_SIZE, RNNH_SIZE);
      wpwm.reshape(WPRED_SIZE, mwpi.size());

//      for ( const auto &myPair : mci ) {
//        std::cerr << myPair.first << "\n";
//     }

      // duplicated vectors for RNN batch processing
      ihbm = repmat(ihbv, 1, mwpi.size());
      hhbm = repmat(hhbv, 1, mwpi.size());
      fcbm = repmat(fcbv, 1, mwpi.size());
      h1 = relu(ihwm * join_cols(wpwm, mcm.find("<S>")->second) + ihbm);
      s1_scores = exp(fcwm * h1 + fcbm);
      s1_norm = sum(s1_scores, 0);
      s1_logprobs = log(s1_scores.each_row() / s1_norm);
    }

//    const vec getCharEmbed( string a ) const {
//      auto it = mcv.find( a );
//      assert ( ( it != mcv.end() );
//      return it->second;
//    }

    const mat getCharMat( string a ) const {
      auto it = mcm.find( a );
      assert ( it != mcm.end() );
      return it->second;
    }

    const unsigned int getCharIndex( string a ) const {
      auto it = mci.find( a );
      assert ( it != mci.end() );
      return it->second;
    }

    const unsigned int getWPredictorIndex( WPredictor wp ) const {
      auto it = mwpi.find( wp );
      assert ( it != mwpi.end() );
      return it->second;
    }

//    list<DelimitedPair<psX,WPredictor,psSpace,Delimited<double>,psX>> calcPredictorLikelihoods( const W w_t ) const {
    list<DelimitedPair<psX,WPredictor,psSpace,Delimited<double>,psX>> calcPredictorLikelihoods( const W w_t, MapWP mymap ) const {
      auto it = mymap.find( w_t );
//      cerr << "mymap size " << mymap.size() << endl;
      list<DelimitedPair<psX,WPredictor,psSpace,Delimited<double>,psX>> results;
      if ( it == mymap.end() ) {
        rowvec seqlogprobs = zeros<mat>(1, mwpi.size());
        mat ht = h1;
        rowvec st1_logprobs;
        // if w_t in punct: word = w_t, else {
        if ( find( PUNCT.begin(), PUNCT.end(), w_t.getString().c_str() ) != PUNCT.end() ) {
          cerr << "PUNCT found" << endl;
          seqlogprobs += s1_logprobs.row( getCharIndex( w_t.getString().c_str() ) );
          mat ht1 = relu(ihwm * join_cols(wpwm, getCharMat( w_t.getString().c_str() )) + ihbm + hhwm * ht + hhbm);
          mat st1_scores = exp(fcwm * ht1 + fcbm);
          rowvec st1_norm = sum(st1_scores, 0);
          st1_logprobs = log(st1_scores.row(getCharIndex( "<E>" )) / st1_norm);
          seqlogprobs += st1_logprobs;
        } else {
          string c0(1, w_t.getString()[0]);
          seqlogprobs += s1_logprobs.row( getCharIndex( c0.c_str() ));
//          cerr << strlen(w_t.getString().c_str()) << endl;
          for ( unsigned i = 1; i < strlen(w_t.getString().c_str()); ++i ){
            string ct(1, w_t.getString()[i]);
            string ct1(1, w_t.getString()[i+1]);
//            cerr << w_t.getString().c_str() << endl;
//            cerr << ct.c_str() << endl;
//            cerr << ct1.c_str() << endl;
            mat ht1 = relu(ihwm * join_cols(wpwm, getCharMat( ct.c_str() )) + ihbm + hhwm * ht + hhbm);
            mat st1_scores = exp(fcwm * ht1 + fcbm);
            rowvec st1_norm = sum(st1_scores, 0);
            if (i != strlen(w_t.getString().c_str())-1) {
              st1_logprobs = log(st1_scores.row(getCharIndex( ct1.c_str() )) / st1_norm);
            }
            else {
//              cerr << "EOS reached" << endl;
              st1_logprobs = log(st1_scores.row(getCharIndex( "<E>" )) / st1_norm);
            }
            seqlogprobs += st1_logprobs;
            ht = ht1;
          }
        }
        rowvec seqprobs = exp(seqlogprobs);
        for ( const auto &it: mwpi ) {
          results.emplace_back(it.first,seqprobs(it.second));
        }
//        cerr << "emplacing results " << mymap.size() << endl;
//        mymap[w_t] = results;
//        mymap.emplace(w_t, results);
//        cerr << "emplaced results " << mymap.size() << endl;
      }
      else {
        cerr << "getting results" << endl;
        results = it->second;
        cerr << "got results" << endl;
      }
      return results;
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
