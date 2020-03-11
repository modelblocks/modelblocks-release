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
const int CVECDIM = 10;
const int KVECDIM = 20;
const int NPREDDIM = 2*KVECDIM+2*CVECDIM+3; //ancestor and antecedent sem and syn, and antdist(1) + antdistsq(1) + corefON(1)

const uint SEM_SIZE = 20;
const uint SYN_SIZE = 10;

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
    CVar  getBaseC()           { return basec; }
    CVar  getAnteC()           { return antecedentc; }
    bool  getCorefOn()         { return corefON; }

    friend ostream& operator<< ( ostream& os, const NPredictorVec& mv ) {
      os << " " << mv.getBaseC() << " " << mv.getAnteC() << " " << mv.getBaseSem() << " " << mv.getAnteSem() << " " << mv.getAntDist() << " " << mv.getAntDistSq() << " " << mv.getCorefOn();
      return os;
    }
};

////////////////////////////////////////////////////////////////////////////////

class NModel {

  typedef DelimitedCol<psLBrack, double, psComma, CVECDIM, psRBrack> CVec;

  private:

    DelimitedMat<psX, double, psComma, NPREDDIM, NPREDDIM, psX> nw;                              // n model weights. square for now, doesn't have to be
    DelimitedVector<psX, double, psComma, psX> nws; // n model weights (second layer), vector for reading in, to be unsqueezed before use

    map<CVar,CVec> mcv; //map between cat and 10d embeddings

    unsigned int iNextPredictor = 0;             // predictor and response next-pointers

    //map<unsigned int,string>    mis;
    //map<string,unsigned int>    msi;


  public:

    NModel( ) { }
    NModel( istream& is ) {
      //rewrite how to read in N model learned weights, crib off of J model
      list< trip< unsigned int, unsigned int, double > > l;    // store elements on list until we know dimensions of matrix
      while( is.peek()=='N' ) {
        Delimited<char> c;
        is >> "N " >> c >> " ";
        if (c == 'F') is >> nw >> "\n";
        if (c == 'S') is >> nws >> "\n";
      }
      while ( is.peek()=='C' ) {
        Delimited<CVar> c;
        is >> "C " >> c >> " ";
        is >> mcv[c] >> "\n";
      }
    }

      //if( l.size()==0 ) cerr << "ERROR: No N items found." << endl;
      //matN.zeros ( 2, iNextPredictor );
      //for( auto& prw : l ) { matN( prw.second(), prw.first() ) = prw.third(); }

    //unsigned int getPredictorIndex( const string& s ) {
     // const auto& it = msi.find( s );  if( it != msi.end() ) return( it->second );
     // msi[ s ] = iNextPredictor;  mis[ iNextPredictor ] = s;  return( iNextPredictor++ );
   // }
    //unsigned int getPredictorIndex( const string& s ) const {                  // const version with closed predictor domain
    // const auto& it = msi.find( s );  return( ( it != msi.end() ) ? it->second : 0 );
   // }

    //unsigned int getPredictorIndex( K kA, K kB ) {
    //  const auto& it = mkki.find( pair<K,K>(kA,kB) );  if( it != mkki.end() ) return( it->second );
    //  mkki[ pair<K,K>(kA,kB) ] = iNextPredictor;  mikk[ iNextPredictor ] = pair<K,K>(kA,kB);  return( iNextPredictor++ );
   // }
   // unsigned int getPredictorIndex( K kA, K kB ) const {                       // const version with closed predictor domain
   //   const auto& it = mkki.find( pair<K,K>(kA,kB) );  return( ( it != mkki.end() ) ? it->second : 0 );
   // }

   // unsigned int getPredictorIndex( CVar cA, CVar cB ) {
   //   const auto& it = mcci.find( pair<CVar,CVar>(cA,cB) );  if( it != mcci.end() ) return( it->second );
   //   mcci[ pair<CVar,CVar>(cA,cB) ] = iNextPredictor;  micc[ iNextPredictor ] = pair<CVar,CVar>(cA,cB);  return( iNextPredictor++ );
   // }
   // unsigned int getPredictorIndex( CVar cA, CVar cB ) const {                 // const version with closed predictor domain
   //   const auto& it = mcci.find( pair<CVar,CVar>(cA,cB) );  return( ( it != mcci.end() ) ? it->second : 0 );
   // }

    const CVec& getCatEmbed( CVar i ) const {
      auto it = mcv.find( i );
      assert( it != mcv.end() );
      return it->second;
    }

    arma::vec calcLogResponses( const NPredictorVec& npv ) const {
      arma::vec nlogresponses = arma::zeros( NPREDDIM ); 
      
      CVar catB = npv.getBaseC();
      CVar catA = npv.getAnteC();
      const HVec& hvB = npv.getBaseSem();
      const HVec& hvA = npv.getAnteSem();
      int antdist = npv.getAntDist();
      int antdistsq = npv.getAntDistSq();
      bool corefon = npv.getCorefOn();

      const CVec& catBEmb = getCatEmbed(catB);
      const CVec& catAEmb = getCatEmbed(catA);

      //populate predictor vec by catting vecs or else filling in nlogresponses' values
      for(unsigned int i = 0; i < catBEmb.n_elem; i++){ nlogresponses(i) = catBEmb(i); }
      for(unsigned int i = 0; i < catAEmb.n_elem; i++){ nlogresponses(catBEmb.n_elem+i) = catAEmb(i); }
      for(unsigned int i = 0; i < hvB.at(0).n_elem; i++){ nlogresponses(catBEmb.n_elem+catAEmb.n_elem+i) = hvB.at(0)(i); }
      for(unsigned int i = 0; i < hvA.at(0).n_elem; i++){ nlogresponses(catBEmb.n_elem+catAEmb.n_elem+hvB.at(0).n_elem+i) = hvA.at(0)(i); }
      int denseendind = catBEmb.n_elem+catAEmb.n_elem+hvB.at(0).n_elem+hvA.at(0).n_elem+1; 
      nlogresponses(denseendind+1) = antdist;
      nlogresponses(denseendind+2) = antdistsq;
      nlogresponses(denseendind+3) = corefon;
      ////TODO later potentially add gender animacy etc features, if continue to have inconsistency issues with number, gender, etc.

      //push through weight matrix, norm and return score
      mat nwsm(nws);
      nwsm.reshape(nws.size()/NPREDDIM, NPREDDIM); //unsqueeze vector to matrix
      arma::vec nlogscores = nwsm * relu(Mat<double>(nw)*nlogresponses);
      arma::vec nscores = arma::exp(nlogscores);
      double nnorm = arma::accu(nscores);

      return nscores/nnorm;
    }

    

    unsigned int getNumPredictors( ) { return iNextPredictor; }
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
//      os << fpv.d << " " << fpv.catBase << " " << foo << " " << foo;
      return os;
    }
};

////////////////////////////////////////////////////////////////////////////////

class FModel {

  typedef DelimitedTrip<psX,F,psAmpersand,Delimited<EVar>,psAmpersand,Delimited<K>,psX> FEK;
  typedef DelimitedCol<psLBrack, double, psComma, CVECDIM, psRBrack> CVec;
  //typedef DelimitedCol<psLBrack, double, psComma, psRBrack> CVec;
//  typedef DelimitedCol<psLBrack, double, psComma, 20, psRBrack> CVec;


  private:

    map<CVar,CVec> mcv;                     // map between cat and embeds

    map<FEK,unsigned int> mfeki;               // response indices
    map<unsigned int,FEK> mifek;

    unsigned int iNextResponse  = 0;

    // Matrix dimensions could be different; how to accommodate for this?
//    7+2*sem+syn
    DelimitedVector<psX, double, psComma, psX> fwf;  // weights for F model
    DelimitedVector<psX, double, psComma, psX> fws;
    DelimitedVector<psX, double, psComma, psX> fbf;  // biases for F model
    DelimitedVector<psX, double, psComma, psX> fbs;
//    DelimitedVector<psX, double, psComma, psX> fwt;

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
//        if (c == 'T') is >> fwt >> "\n";
      }
      while ( is.peek()=='C' ) {
        Delimited<CVar> c;
        is >> "C " >> c >> " ";
        is >> mcv.try_emplace(c,SYN_SIZE).first->second >> "\n";
      }
      while ( is.peek()=='f' ) {
        unsigned int i;
        is >> "f " >> i >> " ";
        is >> mifek[i] >> "\n";
        mfeki[mifek[i]] = i;
      }
    }

    const FEK& getFEK( unsigned int i ) const {
      auto it = mifek.find( i );
      assert( it != mifek.end() );
      return it->second;
    }

    const CVec& getCatEmbed( CVar i ) const {
      auto it = mcv.find( i );
      assert( it != mcv.end() );
      return it->second;
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
      arma::vec flogresponses = arma::zeros( 7 + 2*SEM_SIZE + SYN_SIZE );
      CVar catB = lfpredictors.getCatBase();
      const HVec& hvB = lfpredictors.getHvB();
      const HVec& hvF = lfpredictors.getHvF();
      int d = lfpredictors.getD();

      const CVec& catBEmb = getCatEmbed(catB);

// populate predictor vector
      for(unsigned int i = 0; i < catBEmb.n_elem; i++){
        flogresponses(i) = catBEmb(i);
      }
      for(unsigned int i = 0; i < hvB.at(0).n_elem; i++){
        flogresponses(catBEmb.n_elem+i) = hvB.at(0)(i);
      }
      for(unsigned int i = 0; i < hvF.at(0).n_elem; i++){
        flogresponses(catBEmb.n_elem+hvB.at(0).n_elem+i) = hvF.at(0)(i);
      }
      flogresponses(catBEmb.n_elem+hvB.at(0).n_elem+hvF.at(0).n_elem+d) = 1;

// implementation of MLP
      mat fwfm(fwf);
      mat fwsm(fws);
      vec fbfv(fbf);
      vec fbsv(fbs);
      fwfm.reshape(fwf.size()/(7 + 2*SEM_SIZE + SYN_SIZE), 7 + 2*SEM_SIZE + SYN_SIZE);
      fwsm.reshape(fws.size()/(fwf.size()/(7 + 2*SEM_SIZE + SYN_SIZE)), (fwf.size()/(7 + 2*SEM_SIZE + SYN_SIZE)));
      arma::vec flogscores = fwsm * relu(fwfm*flogresponses + fbfv) + fbsv;
      arma::vec fscores = arma::exp(flogscores);
      double fnorm = arma::accu(fscores);
      return fscores/fnorm;
    }

  arma::vec testCalcResponses( arma::vec testvec ) const {
      mat fwfm(fwf);
      mat fwsm(fws);
      vec fbfv(fbf);
      vec fbsv(fbs);
      fwfm.reshape(fwf.size()/(7 + 2*SEM_SIZE + SYN_SIZE), 7 + 2*SEM_SIZE + SYN_SIZE);
      fwsm.reshape(fws.size()/(fwf.size()/(7 + 2*SEM_SIZE + SYN_SIZE)), (fwf.size()/(7 + 2*SEM_SIZE + SYN_SIZE)));
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
//      os << jpv.d << " " << jpv.catAncstr << " " << foo << " " << foo << " " << jpv.catLchild << " " << foo;
      return os;
    }
};


////////////////////////////////////////////////////////////////////////////////

class JModel {

  typedef DelimitedQuad<psX,J,psAmpersand,Delimited<EVar>,psAmpersand,O,psAmpersand,O,psX> JEOO;
  typedef DelimitedCol<psLBrack, double, psComma, psRBrack> CVec;
  unsigned int jr0;
  unsigned int jr1;

  private:

    map<CVar,CVec> mcv;                     // map between cat and 10-dim embeds

    map<JEOO,unsigned int> mjeooi;               // response indices
    map<unsigned int,JEOO> mijeoo;

    unsigned int iNextResponse  = 0;

    // Matrix dimensions could be different; how to accommodate for this?
//    7+3*sem+2*syn
    DelimitedVector<psX, double, psComma, psX> jwf;  // weights for J model
    DelimitedVector<psX, double, psComma, psX> jws;
    DelimitedVector<psX, double, psComma, psX> jbf;  // biases for J model
    DelimitedVector<psX, double, psComma, psX> jbs;

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
        Delimited<CVar> c;
        is >> "C " >> c >> " ";
        is >> mcv.try_emplace(c,SYN_SIZE).first->second >> "\n";
      }
      while ( is.peek()=='j' ) {
        Delimited<int> k;
        is >> "j " >> k >> " ";
        is >> mijeoo[k] >> "\n";
        mjeooi[mijeoo[k]] = k;
      }
      jr0 = getResponseIndex( 0, EVar::eNil, 'N', O_I );
      jr1 = getResponseIndex( 1, EVar::eNil, 'N', O_I );
    }

    const JEOO& getJEOO( unsigned int i ) const {
      auto it = mijeoo.find( i );
      assert( it != mijeoo.end() );
      return it->second;
    }

    const CVec& getCatEmbed( CVar i ) const {
      auto it = mcv.find( i );
      assert( it != mcv.end() );
      return it->second;
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
//      arma::vec jlogresponses = arma::zeros( 107 );
//      arma::vec jlogresponses = arma::zeros( 147 );
      CVar catA = ljpredictors.getCatAncstr();
      const HVec& hvA = ljpredictors.getHvAncstr();
      const HVec& hvF = ljpredictors.getHvFiller();
      CVar catL = ljpredictors.getCatLchild();
      const HVec& hvL = ljpredictors.getHvLchild();
      int d = ljpredictors.getD();

      const CVec& catAEmb = getCatEmbed(catA);
      const CVec& catLEmb = getCatEmbed(catL);

// populate predictor vector
      for(unsigned int i = 0; i < catAEmb.n_elem; i++){
        jlogresponses(i) = catAEmb(i);
      }
      for(unsigned int i = 0; i < hvA.at(0).n_elem; i++){
        jlogresponses(catAEmb.n_elem+i) = hvA.at(0)(i);
      }
      for(unsigned int i = 0; i < hvF.at(0).n_elem; i++){
        jlogresponses(catAEmb.n_elem+hvA.at(0).n_elem+i) = hvF.at(0)(i);
      }
      for(unsigned int i = 0; i < catLEmb.n_elem; i++){
        jlogresponses(catAEmb.n_elem+hvA.at(0).n_elem+hvF.at(0).n_elem+i) = catLEmb(i);
      }
      for(unsigned int i = 0; i < hvL.at(0).n_elem; i++){
        jlogresponses(catAEmb.n_elem+hvA.at(0).n_elem+hvF.at(0).n_elem+catLEmb.n_elem+i) = hvL.at(0)(i);
      }
      jlogresponses(catAEmb.n_elem+hvA.at(0).n_elem+hvF.at(0).n_elem+catLEmb.n_elem+hvL.at(0).n_elem+d) = 1;

// implementation of MLP
      mat jwfm(jwf);
      mat jwsm(jws);
      vec jbfv(jbf);
      vec jbsv(jbs);
      jwfm.reshape(jwf.size()/(7 + 3*SEM_SIZE + 2*SYN_SIZE), (7 + 3*SEM_SIZE + 2*SYN_SIZE));
      jwsm.reshape(jws.size()/(jwf.size()/(7 + 3*SEM_SIZE + 2*SYN_SIZE)), (jwf.size()/(7 + 3*SEM_SIZE + 2*SYN_SIZE)));
      arma::vec jlogscores = jwsm * relu(jwfm*jlogresponses + jbfv) + jbsv;
      arma::vec jscores = arma::exp(jlogscores);
      double jnorm = arma::accu(jscores);
      return jscores/jnorm;
    }

  arma::vec testCalcResponses( arma::vec testvec ) const {
      mat jwfm(jwf);
      mat jwsm(jws);
      vec jbfv(jbf);
      vec jbsv(jbs);
      jwfm.reshape(jwf.size()/(7 + 3*SEM_SIZE + 2*SYN_SIZE), (7 + 3*SEM_SIZE + 2*SYN_SIZE));
      jwsm.reshape(jws.size()/(jwf.size()/(7 + 3*SEM_SIZE + 2*SYN_SIZE)), (jwf.size()/(7 + 3*SEM_SIZE + 2*SYN_SIZE)));
      arma::vec jlogscores = jwsm * relu(jwfm*testvec + jbfv) + jbsv;
      arma::vec jscores = arma::exp(jlogscores);
      double jnorm = arma::accu(jscores);
      return jscores/jnorm;
    }
};

////////////////////////////////////////////////////////////////////////////////

