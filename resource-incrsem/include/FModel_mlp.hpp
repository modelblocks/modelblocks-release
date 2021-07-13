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

//#include <typeinfo>
//#include <regex>
//#include <algorithm>

////////////////////////////////////////////////////////////////////////////////

class FPredictorVec {

  private:
    int d;
    int iCarrier;
    const HVec& hvB;
    const HVec& hvF;
    CVar catBase;
    const HVec& hvA;
    bool nullA;

    public:
    template<class FM>  // J model is template variable to allow same behavior for const and non-const up until getting predictor indices
    FPredictorVec( FM& fm, const HVec& hvAnt, bool nullAnt, const StoreState& ss ) : hvB (( ss.getBase().getHVec().size() > 0 ) ? ss.getBase().getHVec() : hvBot), hvF (( ss.getBase().getCat().getNoloArity() && ss.getNoloBack().getHVec().size() != 0 ) ? ss.getNoloBack().getHVec() : hvBot), hvA ((hvAnt.size() > 0) ? hvAnt : hvBot), nullA (nullAnt){
      d = (FEATCONFIG & 1) ? 0 : ss.getDepth();
      catBase = ss.getBase().getCat();
    }

    bool getNullA() {
      return nullA;
    }
    const HVec& getHvA() {
      return hvA; //antecedent
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
      os << fpv.d << " " << fpv.catBase << " " << fpv.hvB << " " << fpv.hvF << " " << fpv.hvA << " " << fpv.nullA;
      return os;
    }
};

////////////////////////////////////////////////////////////////////////////////

class FModel {

  typedef DelimitedTrip<psX,F,psAmpersand,Delimited<EVar>,psAmpersand,Delimited<K>,psX> FEK;
  typedef DelimitedCol<psLBrack, double, psComma, psRBrack> CVec;
  typedef DelimitedCol<psLBrack, double, psComma, psRBrack> KDenseVec;
  uint FSEM_SIZE = 20; 
  uint FSYN_SIZE = 20;
  uint FANT_SIZE = 20;
  uint FFULL_WIDTH = 13;
  private:

    map<CVar,vec> mcbv;                        // map between syntactic category and embeds
    map<KVec,vec> mkbdv;                  // map between KVec and embeds
    map<KVec,vec> mkfdv;
    map<KVec,vec> mkadv;

    map<FEK,unsigned int> mfeki;                // response indices
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
    vec zeroCatEmb;

  public:

    //FModel( ) : zeroCatEmb(SYN_SIZE) { }
    //FModel( istream& is ) : zeroCatEmb(SYN_SIZE) {
    FModel( ) : zeroCatEmb(13) { }
    FModel( istream& is ) : zeroCatEmb(13) {
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
        DelimitedVector<psX, double, psComma, psX> vtemp;  
        is >> "C " >> c >> " " >> cv >> " ";
        is >> vtemp >> "\n";
        if (c == 'B') mcbv.try_emplace(cv,vtemp);
        FSYN_SIZE=vtemp.size();
      }
//      zeroCatEmb=CVec(FSYN_SIZE);
      zeroCatEmb=arma::zeros(FSYN_SIZE);
      while ( is.peek()=='K' ) {
        Delimited<char> c;
        Delimited<K> k;
        DelimitedVector<psX, double, psComma, psX> vtemp;
        is >> "K " >> c >> " " >> k >> " ";
        is >> vtemp >> "\n";
        if (c == 'B') { 
          mkbdv.try_emplace(k, vtemp);
          FSEM_SIZE=vtemp.size();
        }
        else if (c == 'F') { 
          mkfdv.try_emplace(k, vtemp);
          FSEM_SIZE=vtemp.size();
        }
        else if (c == 'A') {
          mkadv.try_emplace(k, vtemp);
          FANT_SIZE=vtemp.size();
        }
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
      cerr << "FSEM: " << FSEM_SIZE << " FSYN: " << FSYN_SIZE << " FANT: " << FANT_SIZE << endl;
      FFULL_WIDTH = 8 + 2*FSEM_SIZE + FSYN_SIZE + FANT_SIZE;
      fwfm.reshape(fwf.size()/(FFULL_WIDTH), FFULL_WIDTH);
      fwsm.reshape(fws.size()/(fwf.size()/(FFULL_WIDTH)), (fwf.size()/(FFULL_WIDTH)));
    }

    const FEK& getFEK( unsigned int i ) const {
      auto it = mifek.find( i );
      assert( it != mifek.end() );
      if (it == mifek.end()) { 
        cerr << "ERROR: FEK not defined in fmodel: no value found for: " << i << endl; 
      }
      return it->second;
    }

    const vec getCatEmbed( CVar i, Delimited<char> c) const {
//    const CVec& getCatEmbed( CVar i, Delimited<char> c) const {
      if (c == 'B') {
        auto it = mcbv.find( i );
        return ( ( it != mcbv.end() ) ? it->second : zeroCatEmb );
      }
      cerr << "ERROR: F model CVar position misspecified." << endl;
      return zeroCatEmb;
    }

    const vec getKVecEmbed( HVec hv, Delimited<char> c ) const {
      vec KVecEmbed;// = arma::zeros(FSEM_SIZE);
      if (c == 'B') {
        KVecEmbed = arma::zeros(FSEM_SIZE);
        for ( auto& kV : hv.at(0) ) {
          if ( kV == K::kTop) {
            KVecEmbed += arma::ones(FSEM_SIZE);
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
        KVecEmbed = arma::zeros(FSEM_SIZE);
        for ( auto& kV : hv.at(0) ) {
          if ( kV == K::kTop) {
            KVecEmbed += arma::ones(FSEM_SIZE);
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
      else if (c == 'A') {
        KVecEmbed = arma::zeros(FANT_SIZE);
        for ( auto& kV : hv.at(0) ) {
          if ( kV == K::kTop) {
            KVecEmbed += arma::ones(FANT_SIZE);
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
// vectorize predictors: one-hot for depth(7), three hvecs, one cat-embed - also 1bit for nullant
      CVar catB = lfpredictors.getCatBase();
      const HVec& hvB = lfpredictors.getHvB();
      const HVec& hvF = lfpredictors.getHvF();
      int d = lfpredictors.getD();
      const HVec& hvA = lfpredictors.getHvA();
      bool nullA = lfpredictors.getNullA();

      const vec& catBEmb = getCatEmbed(catB, 'B');
      const vec& hvBEmb = getKVecEmbed(hvB, 'B');
      const vec& hvFEmb = getKVecEmbed(hvF, 'F');
      const vec& hvAEmb = getKVecEmbed(hvA, 'A');

// populate predictor vector
      arma::vec flogresponses = join_cols(join_cols(join_cols(join_cols(catBEmb, hvBEmb), hvFEmb), hvAEmb), arma::zeros(8));
      if (nullA) flogresponses(2*FSEM_SIZE + FSYN_SIZE + FANT_SIZE) = 1;
      flogresponses(2*FSEM_SIZE + FSYN_SIZE + FANT_SIZE + 1 + d) = 1;

// implementation of MLP
      arma::vec flogscores = fwsm * relu(fwfm*flogresponses + fbfv) + fbsv;
      arma::vec fscores = arma::exp(flogscores);
      double fnorm = arma::accu(fscores);
      return fscores/fnorm;
    }
};

