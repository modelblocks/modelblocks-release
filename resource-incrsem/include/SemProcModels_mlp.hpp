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
const uint SYN_SIZE = 10;
const uint NPREDDIM = 2*SEM_SIZE+2*SYN_SIZE+3;

arma::mat relu( const arma::mat& km ) {
  arma::mat A(km.n_rows, 1);
  for ( unsigned int c = 0; c<km.n_rows; c++ ) {
    if ( km(c,0) <= 0 ) {A(c,0)=(0.0);}
    else A(c,0) = (km(c));
  }
  return A;
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
    NModel( ) : zeroCatEmb(SYN_SIZE) { }
    NModel( istream& is ) : zeroCatEmb(SYN_SIZE) {
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
        is >> "C " >> c >> " ";
        is >> mcv.try_emplace(c,SYN_SIZE).first->second >> "\n"; //mcv[c]
      }
      while ( is.peek()=='K' ) {
        Delimited<K> k;
        is >> "K " >> k >> " ";
        is >> mkdv.try_emplace(k,SEM_SIZE).first->second >> "\n";
      }
      //Reshape read-in model params to correct dimensions, save to private members
      //first layer
      nwm = nw;
      nwm.reshape(nwm.size()/NPREDDIM, NPREDDIM); //hidden x input. converts input dim to hidden dim
      cerr << "nwm reshaped n_rows, ncols, size: " << nwm.n_rows << " " << nwm.n_cols << " " << nwm.size() << endl;
      nwbv = nwb;
      cerr << "nwbv size: " << nwbv.size() << endl;

      //second layer
      nwsm = nws;
      nwsm.reshape(nwsm.size()/(nwm.size()/NPREDDIM), nwm.size()/NPREDDIM); //unsqueeze vector to matrix.  //outputdim, hidden. converts hidden dim to outputdim
      cerr << "nwsm reshaped n_rows, n_cols, size: " << nwsm.n_rows << " " << nwsm.n_cols << " " << nwsm.size() << endl;
      nwsbv = nwsb;
      cerr << "nwsbv size: " << nwsbv.size() << endl;
      
      //print out first row
      cerr << "first nwm matrix col: " << endl;
      for (uint i = 0; i < nwm.n_rows; i++) { cerr << nwm(i,0) << ","; }
      cerr << endl;
    }

    const CVec& getCatEmbed( CVar i ) const {
      //cerr << "attempting to find cat with index i: " << i << endl;
      //cerr << "mcv size: " << mcv.size() << endl;
      auto it = mcv.find( i );
      if (it == mcv.end()) { 
        cerr << "ERROR: CVar not defined in model: no embedding found for: " << i << endl; 
        return zeroCatEmb;
      }
      assert( it != mcv.end() );
      return it->second;
      //}
    }
  
    const KDenseVec getKVecEmbed( HVec hv ) const {
      KDenseVec KVecEmbed = KDenseVec(arma::zeros(SEM_SIZE));
      for ( auto& kV : hv.at(0) ) {
        if ( kV == K::kTop) {
          KVecEmbed += KDenseVec(arma::ones(SEM_SIZE));
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
      arma::vec nlogresponses = arma::zeros( NPREDDIM ); 
      
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

  private:

    map<CVar,CVec> mcv;                        // map between syntactic category and embeds
    map<KVec,KDenseVec> mkdv;                  // map between KVec and embeds

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
    CVec zeroCatEmb;

  public:

    FModel( ) : zeroCatEmb(SYN_SIZE) { }
    FModel( istream& is ) : zeroCatEmb(SYN_SIZE) {
      while ( is.peek()=='F' ) {
        Delimited<char> c;
        is >> "F " >> c >> " ";
        if (c == 'F') is >> fwf >> "\n";
        if (c == 'f') is >> fbf >> "\n";
        if (c == 'S') is >> fws >> "\n";
        if (c == 's') is >> fbs >> "\n";
      }
      while ( is.peek()=='C' ) {
        Delimited<CVar> c;
        is >> "C " >> c >> " ";
        is >> mcv.try_emplace(c,SYN_SIZE).first->second >> "\n";
      }
      while ( is.peek()=='K' ) {
        Delimited<K> k;
        is >> "K " >> k >> " ";
        is >> mkdv.try_emplace(k,SEM_SIZE).first->second >> "\n";
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
      fwfm.reshape(fwf.size()/(8 + 3*SEM_SIZE + SYN_SIZE), 8 + 3*SEM_SIZE + SYN_SIZE);
      fwsm.reshape(fws.size()/(fwf.size()/(8 + 3*SEM_SIZE + SYN_SIZE)), (fwf.size()/(8 + 3*SEM_SIZE + SYN_SIZE)));
    }

    const FEK& getFEK( unsigned int i ) const {
      auto it = mifek.find( i );
      assert( it != mifek.end() );
      if (it == mifek.end()) { 
        cerr << "ERROR: FEK not defined in model: no value found for: " << i << endl; 
      }
      return it->second;
    }

    const CVec getCatEmbed( CVar i ) const { //TODO change this to reference to avoid unnecessary copying
      auto it = mcv.find( i );
      assert( it != mcv.end() );
      if (it == mcv.end()) { 
        cerr << "ERROR: CVar not defined in model: no embedding found for: " << i << endl; 
        return zeroCatEmb;
      }
      return it->second;
    }

    const KDenseVec getKVecEmbed( HVec hv ) const {
      KDenseVec KVecEmbed = KDenseVec(arma::zeros(SEM_SIZE));
      for ( auto& kV : hv.at(0) ) {
        if ( kV == K::kTop) {
          KVecEmbed += KDenseVec(arma::ones(SEM_SIZE));
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

    // const auto& it = mxv.find( x ); return ( it == mxv.end() ) ? KVec() : it->second;

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
      arma::vec flogresponses = arma::zeros( 8 + 3*SEM_SIZE + SYN_SIZE );
      CVar catB = lfpredictors.getCatBase();
      const HVec& hvB = lfpredictors.getHvB();
      const HVec& hvF = lfpredictors.getHvF();
      int d = lfpredictors.getD();
      const HVec& hvA = lfpredictors.getHvA();
      bool nullA = lfpredictors.getNullA();

      const CVec& catBEmb = getCatEmbed(catB);
      const KDenseVec& hvBEmb = getKVecEmbed(hvB);
      const KDenseVec& hvFEmb = getKVecEmbed(hvF);
      const KDenseVec& hvAEmb = getKVecEmbed(hvA);

// populate predictor vector
      for(unsigned int i = 0; i < catBEmb.n_elem; i++){
        flogresponses(i) = catBEmb(i);
      }
      for(unsigned int i = 0; i < hvBEmb.n_elem; i++){
        flogresponses(catBEmb.n_elem+i) = hvBEmb(i);
      }
      for(unsigned int i = 0; i < hvFEmb.n_elem; i++){
        flogresponses(catBEmb.n_elem+hvBEmb.n_elem+i) = hvFEmb(i);
      }
      for(unsigned int i = 0; i < hvAEmb.n_elem; i++){
        flogresponses(catBEmb.n_elem+hvBEmb.n_elem+hvFEmb.n_elem+i) = hvAEmb(i);
      }
      if (nullA) flogresponses(catBEmb.n_elem+hvBEmb.n_elem+hvFEmb.n_elem+hvAEmb.n_elem) = 1;
      flogresponses(catBEmb.n_elem+hvBEmb.n_elem+hvFEmb.n_elem+hvAEmb.n_elem+1+d) = 1;

// implementation of MLP
      //cout << "trying f model matmul..." << endl;
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
  typedef DelimitedCol<psLBrack, double, psComma, psRBrack> CVec;
  typedef DelimitedCol<psLBrack, double, psComma, psRBrack> KDenseVec;
  unsigned int jr0;
  unsigned int jr1;

  private:

    map<CVar,CVec> mcv;                        // map between syntactic category and embeds
    map<KVec,KDenseVec> mkdv;                  // map between KVec and embeds

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
    CVec zeroCatEmb;

  public:

    JModel() : zeroCatEmb(SYN_SIZE) {
      //jr0 = getResponseIndex( 0, EVar::eNil, 'N', O_I );
      //jr1 = getResponseIndex( 1, EVar::eNil, 'N', O_I );
      jr0 = -1;
      jr1 = -1;
    }
    // read in weights, embeddings, and JEOOs
    JModel(istream& is) : zeroCatEmb(SYN_SIZE) {
      while ( is.peek()=='J' ) {
        Delimited<char> c;
        is >> "J " >> c >> " ";
        if (c == 'F') is >> jwf >> "\n";
        if (c == 'f') is >> jbf >> "\n";
        if (c == 'S') is >> jws >> "\n";
        if (c == 's') is >> jbs >> "\n";
      }
      while ( is.peek()=='C' ) {
        Delimited<CVar> c;
        is >> "C " >> c >> " ";
        is >> mcv.try_emplace(c,SYN_SIZE).first->second >> "\n";
      }
      while ( is.peek()=='K' ) {
        Delimited<K> k;
        is >> "K " >> k >> " ";
        is >> mkdv.try_emplace(k,SEM_SIZE).first->second >> "\n";
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
      jwfm.reshape(jwf.size()/(7 + 3*SEM_SIZE + 2*SYN_SIZE), (7 + 3*SEM_SIZE + 2*SYN_SIZE));
      jwsm.reshape(jws.size()/(jwf.size()/(7 + 3*SEM_SIZE + 2*SYN_SIZE)), (jwf.size()/(7 + 3*SEM_SIZE + 2*SYN_SIZE)));
    }

    const JEOO& getJEOO( unsigned int i ) const {
      auto it = mijeoo.find( i );
      if (it == mijeoo.end()) {
        cerr << "ERROR: no jeoo for " << i << endl;
      }
      assert( it != mijeoo.end() );
      return it->second;
    }

    const CVec getCatEmbed( CVar i ) const { //TODO same copy issue as above
      auto it = mcv.find( i );
      if (it == mcv.end()) { 
        cerr << "ERROR: CVar not defined in model: no embedding found for: " << i << endl; 
        return zeroCatEmb;
      }
      assert( it != mcv.end() );
      return it->second;
    }

    const KDenseVec getKVecEmbed( HVec hv ) const {
      KDenseVec KVecEmbed = KDenseVec(arma::zeros(SEM_SIZE));
      for ( auto& kV : hv.at(0) ) {
        if ( kV == K::kTop) {
          KVecEmbed += KDenseVec(arma::ones(SEM_SIZE));
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
      arma::vec jlogresponses = arma::zeros( 7 + 3*SEM_SIZE + 2*SYN_SIZE );
      CVar catA = ljpredictors.getCatAncstr();
      const HVec& hvA = ljpredictors.getHvAncstr();
      const HVec& hvF = ljpredictors.getHvFiller();
      CVar catL = ljpredictors.getCatLchild();
      const HVec& hvL = ljpredictors.getHvLchild();
      int d = ljpredictors.getD();

      const CVec& catAEmb = getCatEmbed(catA);
      const CVec& catLEmb = getCatEmbed(catL);
      const KDenseVec& hvAEmb = getKVecEmbed(hvA);
      const KDenseVec& hvFEmb = getKVecEmbed(hvF);
      const KDenseVec& hvLEmb = getKVecEmbed(hvL);

// populate predictor vector
      for(unsigned int i = 0; i < catAEmb.n_elem; i++){
        jlogresponses(i) = catAEmb(i);
      }
      for(unsigned int i = 0; i < hvAEmb.n_elem; i++){
        jlogresponses(catAEmb.n_elem+i) = hvAEmb(i);
      }
      for(unsigned int i = 0; i < hvFEmb.n_elem; i++){
        jlogresponses(catAEmb.n_elem+hvAEmb.n_elem+i) = hvFEmb(i);
      }
      for(unsigned int i = 0; i < catLEmb.n_elem; i++){
        jlogresponses(catAEmb.n_elem+hvAEmb.n_elem+hvFEmb.n_elem+i) = catLEmb(i);
      }
      for(unsigned int i = 0; i < hvLEmb.n_elem; i++){
        jlogresponses(catAEmb.n_elem+hvAEmb.n_elem+hvFEmb.n_elem+catLEmb.n_elem+i) = hvLEmb(i);
      }
      jlogresponses(catAEmb.n_elem+hvAEmb.n_elem+hvFEmb.n_elem+catLEmb.n_elem+hvLEmb.n_elem+d) = 1;

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
