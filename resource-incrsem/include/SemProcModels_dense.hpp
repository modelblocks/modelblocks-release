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

class NPredictorVec {
//need to be able to output real-valued distance integers, NPreds
//TODO maybe try quadratic distance
  private:

     int mdist;
     list<unsigned int> mnpreds;

  public:

    //constructor
    template<class LM>
    NPredictorVec( LM& lm, const Sign& candidate, bool bcorefON, int antdist, const StoreState& ss ) : mdist(antdist), mnpreds() {
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
    FPredictorVec( FM& fm, const HVec& hvAnt, bool nullAnt, const StoreState& ss ) : hvB (( ss.getBase().getHVec().size() > 0 ) ? ss.getBase().getHVec() : hvBot), hvF (( ss.getBase().getCat().getNoloArity() ) ? ss.getNoloBack().getHVec() : hvBot){
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
      os << fpv.d << "|" << fpv.catBase << "|" << fpv.hvB << "|" << fpv.hvF;
      return os;
    }
};

////////////////////////////////////////////////////////////////////////////////

class FModel {

  typedef DelimitedTrip<psX,F,psAmpersand,Delimited<EVar>,psAmpersand,Delimited<K>,psX> FEK;
  typedef DelimitedCol<psLBrack, double, psComma, 10, psRBrack> CVec;

  private:

    map<CVar,CVec> mcv;                     // map between cat and 10-dim embeds

    map<FEK,unsigned int> mfeki;               // response indices
    map<unsigned int,FEK> mifek;

    unsigned int iNextResponse  = 0;

    // Matrix dimensions could be different; how to accommodate for this?
    DelimitedMat<psX, double, psComma, 56, 56, psX> fwf;  // weights for J model
    DelimitedMat<psX, double, psComma, 606, 56, psX> fws;

  public:
    FModel( ) { }
    FModel( istream& is ) {
      while ( is.peek()=='F' ) {
        Delimited<char> c;
        is >> "F " >> c >> " ";
        if (c == 'F') is >> fwf >> "\n";
        if (c == 'S') is >> fws >> "\n";
      }
      while ( is.peek()=='C' ) {
        Delimited<CVar> c;
        is >> "C " >> c >> " ";
        is >> mcv[c] >> "\n";
      }
      while ( is.peek()=='f' ) {
        unsigned int i;
        is >> "f " >> i >> " ";
        is >> mifek[i] >> " " >> "\n";
        mfeki[mifek[i]] = i;
//        F f; Delimited<EVar> e; Delimited<K> k;
//        is >> f >> "&" >> e >> "&" >> k >> "\n";
//        mifek[i] = FEK(f,e,k);
//        mfeki[FEK(f,e,k)] = i;
      }
    }

    void checkMap() {
      for(auto it = mfeki.cbegin(); it != mfeki.cend(); ++it) {
        std::cout << it->first << " " << it->second << "\n";
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
// return distribution over JEOO indices
// vectorize predictors: one-hot for depth, three hvecs, two cat-embeds
      arma::vec flogresponses = arma::zeros( fwf.n_cols );
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
      arma::vec flogscores = Mat<double>(fws) * relu(Mat<double>(fwf)*flogresponses);
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
    hvFiller (( ss.getBase().getCat().getNoloArity() ) ? ss.getNoloBack().getHVec() : hvBot),
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
      os << jpv.d << "|" << jpv.catAncstr << "|" << jpv.hvAncstr << "|" << jpv.hvFiller << "|" << jpv.catLchild << "|" << jpv.hvLchild;
      return os;
    }
};


////////////////////////////////////////////////////////////////////////////////

class JModel {

  typedef DelimitedQuad<psX,J,psAmpersand,Delimited<EVar>,psAmpersand,O,psAmpersand,O,psX> JEOO;
  typedef DelimitedCol<psLBrack, double, psComma, 10, psRBrack> CVec;
  unsigned int jr0;
  unsigned int jr1;

  private:

    map<CVar,CVec> mcv;                     // map between cat and 10-dim embeds

    map<JEOO,unsigned int> mjeooi;               // response indices
    map<unsigned int,JEOO> mijeoo;

    unsigned int iNextResponse  = 0;

    // Matrix dimensions could be different; how to accommodate for this?
    DelimitedMat<psX, double, psComma, 87, 87, psX> jwf;  // weights for J model
    DelimitedMat<psX, double, psComma, 50, 87, psX> jws;

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
        if (c == 'S') is >> jws >> "\n";
      }
      while ( is.peek()=='C' ) {
        Delimited<CVar> c;
        is >> "C " >> c >> " ";
        is >> mcv[c] >> "\n";
      }
      while ( is.peek()=='j' ) {
        Delimited<int> k;
        is >> "j " >> k >> " ";
        is >> mijeoo[k] >> " " >> "\n";
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
      arma::vec jlogresponses = arma::zeros( jwf.n_cols );
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
      cout << "cataemb done" << endl;
      for(unsigned int i = 0; i < hvA.at(0).n_elem; i++){
        jlogresponses(catAEmb.n_elem+i) = hvA.at(0)(i);
      }
      cout << "hva done" << endl;
      for(unsigned int i = 0; i < hvF.at(0).n_elem; i++){
        jlogresponses(catAEmb.n_elem+hvA.at(0).n_elem+i) = hvF.at(0)(i);
      }
      cout << "hvf done" << endl;
      for(unsigned int i = 0; i < catLEmb.n_elem; i++){
        jlogresponses(catAEmb.n_elem+hvA.at(0).n_elem+hvF.at(0).n_elem+i) = catLEmb(i);
      }
      cout << "catlemb done" << endl;
      for(unsigned int i = 0; i < hvL.at(0).n_elem; i++){
        jlogresponses(catAEmb.n_elem+hvA.at(0).n_elem+hvF.at(0).n_elem+catLEmb.n_elem+i) = hvL.at(0)(i);
      }
      cout << "hvl done" << endl;
      jlogresponses(catAEmb.n_elem+hvA.at(0).n_elem+hvF.at(0).n_elem+catLEmb.n_elem+hvL.at(0).n_elem+d) = 1;
      cout << "depth done" << endl;

// implementation of MLP
      arma::vec jlogscores = Mat<double>(jws) * relu(Mat<double>(jwf)*jlogresponses);
      arma::vec jscores = arma::exp(jlogscores);
      double jnorm = arma::accu(jscores);

      return jscores/jnorm;
    }
};

////////////////////////////////////////////////////////////////////////////////

