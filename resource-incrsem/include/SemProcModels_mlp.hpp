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

// maybe include as part of WModel
vector<string> PUNCT = { "-LCB-", "-LRB-", "-RCB-", "-RRB-" };

// ReLU function
arma::mat relu( const arma::mat& km ) {
  if ( km.max() < 0 ) return zeros( arma::size(km) );
  else return clamp(km, 0, km.max());
}

////////////////////////////////////////////////////////////////////////////////

class NPredictorVec {
  private:
     int mdist;
     const HVec& basesem;
     const HVec& antecedentsem;
     CVar        basec;
     CVar        antecedentc;
     bool corefON;

  public:
    //constructor
    template<class LM>
    NPredictorVec( LM& lm, const Sign& candidate, bool bcorefON, int antdist, const StoreState& ss, bool ABLATE_UNARY ) :
      basesem (( ss.getBase().getHVec().size() > 0 ) ? ss.getBase().getHVec() : hvBot ), 
      antecedentsem ((candidate.getHVec().size() > 0 ) ? candidate.getHVec() : hvBot ) { 
      mdist = antdist;
      antecedentc = candidate.getCat();
      basec = ss.getBase().getCat();
      corefON = bcorefON;
      //ABLATE_UNARY not defined for dense semproc since NN implicitly captures/can capture joint feature relations
    }

    //accessors
    int   getAntDist()   const { return mdist;   }
    int   getAntDistSq() const { return mdist * mdist; }
    const HVec& getBaseSem()   const { return basesem; }
    const HVec& getAnteSem()   const { return antecedentsem; }
    CVar  getBaseC()     const { return basec; }
    CVar  getAnteC()     const { return antecedentc; }
    bool  getCorefOn()   const { return corefON; }

    friend ostream& operator<< ( ostream& os, const NPredictorVec& mv ) {
      os << " " << mv.getBaseC() << " " << mv.getAnteC() << " " << mv.getBaseSem() << " " << mv.getAnteSem() << " " << mv.getAntDist() << " " << mv.getAntDistSq() << " " << mv.getCorefOn();
      return os;
    }
};

////////////////////////////////////////////////////////////////////////////////

class NModel {
  typedef DelimitedCol<psLBrack, double, psComma, psRBrack> KDenseVec;
  typedef DelimitedCol<psLBrack, double, psComma, psRBrack> CVec;

  uint NSYN_SIZE = 10; //these will all be overwritten
  uint NSEM_SIZE = 20;
  uint NFULL_WIDTH = 13; 

  private:
    mat nwm;
    mat nwsm;
    vec nwbv;
    vec nwsbv;
    DelimitedVector<psX, double, psComma, psX> nwb; // n model weights (first layer), biases
    DelimitedVector<psX, double, psComma, psX> nwsb; // n model weights (second layer), biases
    DelimitedVector<psX, double, psComma, psX> nw;                              // n model weights. square for now, doesn't have to be
    DelimitedVector<psX, double, psComma, psX> nws; // n model weights (second layer), vector for reading in, to be unsqueezed before use
    map<CVar,CVec> mcv; //map between cat and 10d embeddings
    CVec zeroCatEmb;
    map<KVec,KDenseVec> mkdv;                  // map between KVec and embeds

  public:
    NModel( ) : zeroCatEmb(NSYN_SIZE) { }
    NModel( istream& is ) : zeroCatEmb(NSYN_SIZE) {
      //DelimitedMat<psX, double, psComma, NPREDDIM, NPREDDIM, psX> nw;                              // n model weights. square for now, doesn't have to be

      //rewrite how to read in N model learned weights, crib off of J model
      while( is.peek()=='N' ) {
        Delimited<char> c;
        is >> "N " >> c >> " ";
        if (c == 'F') is >> nw >> "\n";
        if (c == 'f') is >> nwb >> "\n";
        if (c == 'S') is >> nws >> "\n";
        if (c == 's') is >> nwsb >> "\n";
      }
      while ( is.peek()=='C' ) {
        Delimited<CVar> c;
        DelimitedVector<psX, double, psComma, psX> vtemp;  
        is >> "C " >> c >> " ";
        is >> vtemp >> "\n";
        mcv.try_emplace(c,vtemp);
        NSYN_SIZE=vtemp.size();
      }
      zeroCatEmb=CVec(NSYN_SIZE);
      while ( is.peek()=='K' ) {
        Delimited<K> k;
        DelimitedVector<psX, double, psComma, psX> vtemp;
        is >> "K " >> k >> " ";
        is >> vtemp >> "\n";
        mkdv.try_emplace(k,vtemp);
        //is >> mkdv.try_emplace(k,NSEM_SIZE).first->second >> "\n";
        NSEM_SIZE=vtemp.size();
      }
      NFULL_WIDTH = 2*NSEM_SIZE+2*NSYN_SIZE+3;
      //Reshape read-in model params to correct dimensions, save to private members
      //first layer
      nwm = nw;
      nwm.reshape(nwm.size()/NFULL_WIDTH, NFULL_WIDTH); //hidden x input. converts input dim to hidden dim
      //cerr << "nwm reshaped n_rows, ncols, size: " << nwm.n_rows << " " << nwm.n_cols << " " << nwm.size() << endl;
      nwbv = nwb;
      //cerr << "nwbv size: " << nwbv.size() << endl;

      //second layer
      nwsm = nws;
      nwsm.reshape(nwsm.size()/(nwm.size()/NFULL_WIDTH), nwm.size()/NFULL_WIDTH); //unsqueeze vector to matrix.  //outputdim, hidden. converts hidden dim to outputdim
      //cerr << "nwsm reshaped n_rows, n_cols, size: " << nwsm.n_rows << " " << nwsm.n_cols << " " << nwsm.size() << endl;
      nwsbv = nwsb;
      //cerr << "nwsbv size: " << nwsbv.size() << endl;
      
      //print out first row
      //cerr << "first nwm matrix col: " << endl;
      //for (uint i = 0; i < nwm.n_rows; i++) { cerr << nwm(i,0) << ","; }
      //cerr << endl;
    }

    const CVec& getCatEmbed( CVar i ) const {
      //cerr << "attempting to find cat with index i: " << i << endl;
      //cerr << "mcv size: " << mcv.size() << endl;
      auto it = mcv.find( i );
      if (it == mcv.end()) { 
        cerr << "ERROR: CVar not defined in nmodel: no embedding found for: " << i << endl; 
        return zeroCatEmb;
      }
      assert( it != mcv.end() );
      return it->second;
      //}
    }
  
    const KDenseVec getKVecEmbed( HVec hv ) const {
      KDenseVec KVecEmbed = KDenseVec(arma::zeros(NSEM_SIZE));
      for ( auto& kV : hv.at(0) ) {
        if ( kV == K::kTop) {
          KVecEmbed += KDenseVec(arma::ones(NSEM_SIZE));
          continue;
        }
        auto it = mkdv.find( kV );
        if ( it == mkdv.end() ) {
          continue;
        } else {
          KVecEmbed += it->second;
        }
      }
      return KVecEmbed;
    }

    arma::vec calcResponses( const NPredictorVec& npv ) const {
      //cerr << "entering calcLogResponses..." << endl;
      arma::vec nlogresponses = arma::zeros( NFULL_WIDTH); 
      
      //cerr << "getting cat indices, hvecs, ad-hoc feat values..." << endl;
      CVar catB = npv.getBaseC();
      CVar catA = npv.getAnteC();
      const HVec& hvB = npv.getBaseSem();
      const HVec& hvA = npv.getAnteSem();
      int antdist = npv.getAntDist();
      int antdistsq = npv.getAntDistSq();
      bool corefon = npv.getCorefOn();

      //cerr << "getting cat embeds..." << endl;
      const CVec& catBEmb = getCatEmbed(catB);
      const CVec& catAEmb = getCatEmbed(catA);
      const KDenseVec& hvBEmb = getKVecEmbed(hvB);
      const KDenseVec& hvAEmb = getKVecEmbed(hvA);

      //populate predictor vec by catting vecs or else filling in nlogresponses' values
      //cerr << "populating predictor vec with feats..." << endl;
      //arma::Col<double> feats = std::vector<double> {antdist, antdistsq, corefon}; 
      for(unsigned int i = 0; i < catBEmb.n_elem; i++){ nlogresponses(i) = catBEmb(i); }
      for(unsigned int i = 0; i < catAEmb.n_elem; i++){ nlogresponses(catBEmb.n_elem+i) = catAEmb(i); }
      //cerr << "inserting hvec feats into predictor vec ..." << endl;
      for(unsigned int i = 0; i < hvBEmb.n_elem; i++){ nlogresponses(catBEmb.n_elem+catAEmb.n_elem+i) = hvBEmb(i); }
      for(unsigned int i = 0; i < hvAEmb.n_elem; i++){ nlogresponses(catBEmb.n_elem+catAEmb.n_elem+hvBEmb.n_elem+i) = hvAEmb(i); }
      int denseendind = catBEmb.n_elem+catAEmb.n_elem+hvBEmb.n_elem+hvAEmb.n_elem; 
      //cerr << "inserting ad-hoc feats into predictor vec ..." << endl;
      nlogresponses(denseendind+0) = antdist;
      nlogresponses(denseendind+1) = antdistsq;
      nlogresponses(denseendind+2) = corefon;
      //arma::vec nlogresponses = arma::join_cols(catBEmb, catAEmb, hvBEmb, hvAEmb); //, feats);
      ////later potentially add gender animacy etc features, if continue to have inconsistency issues with number, gender, etc.
      //cerr << "catB: " << catB << " catA: " << catA << " hvB: " << hvB << " hvA: " << hvA << " antdist: " << antdist << " sqantdist: " << antdistsq << " corefon: " << corefon << endl;
      //cerr << "catBEmb: catBEmb << catAEmb << hvBEmb << hvAEmb << endl;
      //cerr << nlogresponses << endl;
      //cerr << "feature multiplication by weights..." << endl;
      //push through weight matrix, norm and return score
      //cout << "NPREDDIM: " << NPREDDIM << endl;
      //cerr << "nlogresponses size: " << nlogresponses.size() << endl;
      //cerr << "nwm row,cols: " << nwm.n_rows << "," << nwm.n_cols << endl;
      //cerr << "nwsm row,cols: " << nwsm.n_rows << "," << nwsm.n_cols << endl;
      arma::vec nlogscores = nwsm * relu(nwm*nlogresponses+nwbv) + nwsbv;
      arma::vec nscores = arma::exp(nlogscores);
      double nnorm = arma::accu(nscores);

      return nscores/nnorm;
    }
};

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
  uint FSYN_SIZE = 10;
  uint FFULL_WIDTH = 13;
  private:

//    map<CVar,CVec> mcbv;                        // map between syntactic category and embeds
//    map<KVec,KDenseVec> mkbdv;                  // map between KVec and embeds
//    map<KVec,KDenseVec> mkfdv;
//    map<KVec,KDenseVec> mkadv;

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
        if (c == 'B') mkbdv.try_emplace(k, vtemp);
        else if (c == 'F') mkfdv.try_emplace(k, vtemp);
        else if (c == 'A') mkadv.try_emplace(k, vtemp);
        FSEM_SIZE=vtemp.size();
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
      FFULL_WIDTH = 8 + 3*FSEM_SIZE + FSYN_SIZE;
      fwfm.reshape(fwf.size()/(FFULL_WIDTH), FFULL_WIDTH);
      //fwfm.reshape(fwf.size()/(8 + 3*FSEM_SIZE + FSYN_SIZE), 8 + 3*FSEM_SIZE + FSYN_SIZE);
      fwsm.reshape(fws.size()/(fwf.size()/(FFULL_WIDTH)), (fwf.size()/(FFULL_WIDTH)));
      //fwsm.reshape(fws.size()/(fwf.size()/(8 + 3*FSEM_SIZE + FSYN_SIZE)), (fwf.size()/(8 + 3*FSEM_SIZE + FSYN_SIZE)));
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

//    const KDenseVec getKVecEmbed( HVec hv, Delimited<char> c ) const {
//      KDenseVec KVecEmbed = KDenseVec(arma::zeros(FSEM_SIZE));
    const vec getKVecEmbed( HVec hv, Delimited<char> c ) const {
      vec KVecEmbed = arma::zeros(FSEM_SIZE);
      if (c == 'B') {
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
        for ( auto& kV : hv.at(0) ) {
          if ( kV == K::kTop) {
            KVecEmbed += arma::ones(FSEM_SIZE);
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
      if (nullA) flogresponses(3*FSEM_SIZE + FSYN_SIZE + 1) = 1;
      flogresponses(3*FSEM_SIZE + FSYN_SIZE + 1 + d) = 1;

//      for(unsigned int i = 0; i < catBEmb.n_elem; i++){
//        flogresponses(i) = catBEmb(i);
//      }
//      for(unsigned int i = 0; i < hvBEmb.n_elem; i++){
//        flogresponses(catBEmb.n_elem+i) = hvBEmb(i);
//      }
//      for(unsigned int i = 0; i < hvFEmb.n_elem; i++){
//        flogresponses(catBEmb.n_elem+hvBEmb.n_elem+i) = hvFEmb(i);
//      }
//      for(unsigned int i = 0; i < hvAEmb.n_elem; i++){
//        flogresponses(catBEmb.n_elem+hvBEmb.n_elem+hvFEmb.n_elem+i) = hvAEmb(i);
//      }
//      if (nullA) flogresponses(catBEmb.n_elem+hvBEmb.n_elem+hvFEmb.n_elem+hvAEmb.n_elem) = 1;
//      flogresponses(catBEmb.n_elem+hvBEmb.n_elem+hvFEmb.n_elem+hvAEmb.n_elem+1+d) = 1;

// implementation of MLP
//      cout << "trying f model matmul..." << endl;
      arma::vec flogscores = fwsm * relu(fwfm*flogresponses + fbfv) + fbsv;
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
  uint JSYN_SIZE = 10; //placeholders - these will be overwritten when reading in the model
  uint JSEM_SIZE = 20;
  uint JFULL_WIDTH = 13;

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
    vec zeroCatEmb;

  public:

    JModel() : zeroCatEmb(13) {
      //jr0 = getResponseIndex( 0, EVar::eNil, 'N', O_I );
      //jr1 = getResponseIndex( 1, EVar::eNil, 'N', O_I );
      jr0 = -1;
      jr1 = -1;
    }
    // read in weights, embeddings, and JEOOs
    JModel(istream& is) : zeroCatEmb(13) {
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
        DelimitedVector<psX, double, psComma, psX> vtemp;
        is >> "C " >> c >> " " >> cv >> " " >> vtemp >> "\n";
        if (c == 'A') mcav.try_emplace(cv,vtemp);
        else if (c == 'L') mclv.try_emplace(cv,vtemp);
        JSYN_SIZE = vtemp.size();
        //is >> mcv.try_emplace(c,SYN_SIZE).first->second >> "\n";
      }
      zeroCatEmb=arma::zeros(JSYN_SIZE);
      while ( is.peek()=='K' ) {
        Delimited<char> c;
        Delimited<K> k;
        DelimitedVector<psX, double, psComma, psX> vtemp;
        is >> "K " >> c >> " " >> k >> " " >> vtemp >> "\n";
        if (c == 'A') mkadv.try_emplace(k,vtemp);
        else if (c == 'F') mkfdv.try_emplace(k,vtemp);
        else if (c == 'L') mkldv.try_emplace(k,vtemp);
        JSEM_SIZE = vtemp.size();
        //is >> mkdv.try_emplace(k,SEM_SIZE).first->second >> "\n";
      }
      while ( is.peek()=='j' ) {
        Delimited<int> k;
        is >> "j " >> k >> " ";
        is >> mijeoo[k] >> "\n";
        mjeooi[mijeoo[k]] = k;
        iNextResponse = k+1; //code review WS this should be handled more elegantly, since inextresponse is legacy
      }
      //cout << "finished reading in J model..." << endl;
      jr0 = getResponseIndex( 0, EVar::eNil, 'N', O_I );
      jr1 = getResponseIndex( 1, EVar::eNil, 'N', O_I );
      jwfm = jwf;
      jwsm = jws;
      jbfv = jbf;
      jbsv = jbs;
      JFULL_WIDTH = 7 + 3*JSEM_SIZE + 2*JSYN_SIZE;
      //jwfm.reshape(jwf.size()/(7 + 3*JSEM_SIZE + 2*JSYN_SIZE), (7 + 3*JSEM_SIZE + 2*JSYN_SIZE));
      //jwsm.reshape(jws.size()/(jwf.size()/(7 + 3*JSEM_SIZE + 2*JSYN_SIZE)), (jwf.size()/(7 + 3*JSEM_SIZE + 2*JSYN_SIZE)));
      jwfm.reshape(jwf.size()/(JFULL_WIDTH), (JFULL_WIDTH));
      jwsm.reshape(jws.size()/(jwf.size()/(JFULL_WIDTH)), (jwf.size()/(JFULL_WIDTH)));
    }

    const JEOO& getJEOO( unsigned int i ) const {
      auto it = mijeoo.find( i );
      if (it == mijeoo.end()) {
        cerr << "ERROR: no jeoo for " << i << endl;
      }
      assert( it != mijeoo.end() );
      return it->second;
    }

    const vec getCatEmbed( CVar i, Delimited<char> c ) const {
      if (c == 'A') {
        auto it = mcav.find( i );
        return ( ( it != mcav.end() ) ? it->second : zeroCatEmb );
      }
      else if (c == 'L') {
        auto it = mclv.find( i );
        return ( ( it != mclv.end() ) ? it->second : zeroCatEmb );
      }
      cerr << "ERROR: J model CVar position misspecified." << endl;
      return zeroCatEmb;
    }

    const vec getKVecEmbed( HVec hv, Delimited<char> c ) const {
      vec KVecEmbed = arma::zeros(JSEM_SIZE);
      if (c == 'A') {
        for ( auto& kV : hv.at(0) ) {
          if ( kV == K::kTop) {
            KVecEmbed += arma::ones(JSEM_SIZE);
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
            KVecEmbed += arma::ones(JSEM_SIZE);
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
            KVecEmbed += arma::ones(JSEM_SIZE);
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
      //arma::vec jlogresponses = arma::zeros( 7 + 3*JSEM_SIZE + 2*JSYN_SIZE );
//      arma::vec jlogresponses = arma::zeros( JFULL_WIDTH );
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
      jlogresponses(3*JSEM_SIZE + 2*JSYN_SIZE + d) = 1;

// implementation of MLP
      arma::vec jlogscores = jwsm * relu(jwfm*jlogresponses + jbfv) + jbsv;
      arma::vec jscores = arma::exp(jlogscores);
      double jnorm = arma::accu(jscores);
      return jscores/jnorm;
    }
};

////////////////////////////////////////////////////////////////////////////////
