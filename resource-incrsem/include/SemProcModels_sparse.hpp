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

#include<Delimited.hpp>

////////////////////////////////////////////////////////////////////////////////

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
      //probably will look like Join model feature generation.ancestor is a sign, sign has T and Kset.
      //TODO add dependence to P model.  P category should be informed by which antecedent category was chosen here

//      mdist = antdist;
      mnpreds.emplace_back( lm.getPredictorIndex( "bias" ) ); //add bias term

      const HVec& hvA = ( antdist ) ? candidate.getHVec() : hvBot;
#ifdef SIMPLE_STORE
      const HVec& hvB = ss.getBase().getHVec(); //contexts of lowest b (bdbar)
#else
      const HVec& hvB = ss.at(ss.size()-1).getHVec(); //contexts of lowest b (bdbar)
#endif
      for( unsigned int iA=0; iA<hvA.size(); iA++ )  for( auto& antk : hvA[iA] ) {
        mnpreds.emplace_back( lm.getPredictorIndex( antk.project(-iA), kNil ) ); //add unary antecedent k feat, using kxk template
        for( unsigned int iB=0; iB<hvB.size(); iB++)  for( auto& currk : hvB[iB] ) {
          mnpreds.emplace_back( lm.getPredictorIndex( antk.project(-iA), currk.project(-iB) ) ); //pairwise kxk feat
        }
      }
      for( unsigned int iB=0; iB<hvB.size(); iB++ )  for( auto& currk : hvB[iB] ) {
        mnpreds.emplace_back( lm.getPredictorIndex( kNil, currk.project(-iB) ) ); //unary ancestor k feat
      }

      CVar cAnt = ( antdist ) ? candidate.getCat() : cNone;
      mnpreds.emplace_back( lm.getPredictorIndex( cAnt,   N_NONE                      ) ); // antecedent CVar
#ifdef SIMPLE_STORE
      mnpreds.emplace_back( lm.getPredictorIndex( N_NONE, ss.getBase().getCat() ) ); // ancestor CVar
      mnpreds.emplace_back( lm.getPredictorIndex( cAnt,   ss.getBase().getCat() ) ); // pairwise T
#else
      mnpreds.emplace_back( lm.getPredictorIndex( N_NONE, ss.at(ss.size()-1).getCat() ) ); // ancestor CVar
      mnpreds.emplace_back( lm.getPredictorIndex( cAnt,   ss.at(ss.size()-1).getCat() ) ); // pairwise T
#endif

      //corefON feature
      if (bcorefON == true) {
        mnpreds.emplace_back( lm.getPredictorIndex( "corefON" ) );
      }
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
      arma::vec nlogresponses = arma::zeros( matN.n_rows );
      nlogresponses += npv.getAntDist() * matN.col(getPredictorIndex("ntdist"));
      for ( auto& npredr : npv.getList() ) {
        if ( npredr < matN.n_cols ) { 
          nlogresponses += matN.col( npredr ); 
        }
      }
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

class FPredictorVec : public list<unsigned int> {

  public:

    template<class FM>  // J model is template variable to allow same behavior for const and non-const up until getting predictor indices
    FPredictorVec( FM& fm, const HVec& hvAnt, bool nullAnt, const StoreState& ss ) {
      int d = (FEATCONFIG & 1) ? 0 : ss.getDepth(); // max used depth - (dbar)
#ifdef SIMPLE_STORE
      const HVec& hvB = ( ss.getBase().getHVec().size() > 0 ) ? ss.getBase().getHVec() : hvBot; //contexts of lowest b (bdbar)
      const HVec& hvF = ( ss.getBase().getCat().getNoloArity() ) ? ss.getNoloBack().getHVec() : HVec();
#else
      const HVec& hvB = ( ss.at(ss.size()-1).getHVec().size() > 0 ) ? ss.at(ss.size()-1).getHVec() : hvBot; //contexts of lowest b (bdbar)
      int iCarrier = ss.getAncestorBCarrierIndex( 1 ); // get lowest nonlocal above bdbar
      const HVec& hvF = ( iCarrier >= 0 ) ? ss.at(iCarrier).getHVec() : HVec();
#endif
      emplace_back( fm.getPredictorIndex( "Bias" ) );  // add bias
#ifdef SIMPLE_STORE
      if( STORESTATE_TYPE ) emplace_back( fm.getPredictorIndex( d, ss.getBase().getCat() ) ); 
#else
      if( STORESTATE_TYPE ) emplace_back( fm.getPredictorIndex( d, ss.at(ss.size()-1).getCat() ) ); 
#endif
      if( !(FEATCONFIG & 2) ) {
        for( uint iB=0; iB<hvB.size();   iB++ )  for( auto& kB : hvB[iB] )   emplace_back( fm.getPredictorIndex( d, kNil,            kB.project(-iB), kNil ) );
        for( uint iF=0; iF<hvF.size();   iF++ )  for( auto& kF : hvF[iF] )   emplace_back( fm.getPredictorIndex( d, kF.project(-iF), kNil,            kNil ) );
        for( uint iA=0; iA<hvAnt.size(); iA++ )  for( auto& kA : hvAnt[iA] ) emplace_back( fm.getPredictorIndex( d, kNil,            kNil,            kA.project(-iA) ) );
      }
      if( nullAnt ) emplace_back( fm.getPredictorIndex( "corefOFF" ) );
      else          emplace_back( fm.getPredictorIndex( "corefON"  ) ); 
    }
};

////////////////////////////////////////////////////////////////////////////////

class FModel {

  typedef DelimitedTrip<psX,F,psAmpersand,Delimited<EVar>,psAmpersand,Delimited<K>,psX> FEK;

  private:

    arma::mat matF;                              // matrix itself

    unsigned int iNextPredictor = 0;             // predictor and response next-pointers
    unsigned int iNextResponse  = 0;

    map<string,unsigned int> msi;                // predictor indices for ad-hoc feature
    map<unsigned int,string> mis;
    map<quad<D,K,K,K>,unsigned int> mdkkki;      // predictor indices for k-context tuples
    map<unsigned int,quad<D,K,K,K>> midkkk;
    map<pair<D,CVar>,unsigned int> mdci;         // predictor indices for category tuples
    map<unsigned int,pair<D,CVar>> midc;

    map<FEK,unsigned int> mfeki;                 // response indices
    map<unsigned int,FEK> mifek;

  public:

    FModel( )             { }
    FModel( istream& is ) {
      list< trip< unsigned int, unsigned int, double > > l;    // store elements on list until we know dimensions of matrix
      while( is.peek()=='F' ) {
        auto& prw = *l.emplace( l.end() );
        is >> "F ";
        if( is.peek()=='a' )   { Delimited<string> s;   is >> "a" >> s >> " : ";                          prw.first()  = getPredictorIndex( s );             }
        else{
          D d;                                          is >> "d" >> d >> "&";
          if( is.peek()=='t' ) { Delimited<CVar> c;     is >> "t" >> c >> " : ";                          prw.first()  = getPredictorIndex( d, c );          }
          else                 { Delimited<K> kN,kF,kA; is >> kN >> "&" >> kF >> "&" >> kA >> " : ";      prw.first()  = getPredictorIndex( d, kN, kF, kA ); }
        }
        F f; Delimited<EVar> e; Delimited<K> k;         is >> "f" >> f >> "&" >> e >> "&" >> k >> " = ";  prw.second() = getResponseIndex( f, e, k );
        Delimited<double> w;                            is >> w >> "\n";                                  prw.third()  = w;
      }

      if( l.size()==0 ) cerr << "ERROR: No F items found." << endl;
      matF.zeros ( mifek.size(), iNextPredictor );
      for( auto& prw : l ) { matF( prw.second(), prw.first() ) = prw.third(); }
    }

    unsigned int getPredictorIndex( const string& s ) {
      const auto& it = msi.find( s );  if( it != msi.end() ) return( it->second );
      msi[ s ] = iNextPredictor;  mis[ iNextPredictor ] = s;  return( iNextPredictor++ );
    }
    unsigned int getPredictorIndex( const string& s ) const {                  // const version with closed predictor domain
      const auto& it = msi.find( s );  return( ( it != msi.end() ) ? it->second : 0 );
    }

    unsigned int getPredictorIndex( D d, K kF, K kA, K kL ) {
      const auto& it = mdkkki.find( quad<D,K,K,K>(d,kF,kA,kL) );  if( it != mdkkki.end() ) return( it->second );
      mdkkki[ quad<D,K,K,K>(d,kF,kA,kL) ] = iNextPredictor;  midkkk[ iNextPredictor ] = quad<D,K,K,K>(d,kF,kA,kL);  return( iNextPredictor++ );
    }
    unsigned int getPredictorIndex( D d, K kF, K kA, K kL ) const {            // const version with closed predictor domain
      const auto& it = mdkkki.find( quad<D,K,K,K>(d,kF,kA,kL) );  return( ( it != mdkkki.end() ) ? it->second : 0 );
    }

    unsigned int getPredictorIndex( D d, CVar c ) {
      const auto& it = mdci.find( pair<D,CVar>(d,c) );  if( it != mdci.end() ) return( it->second );
      mdci[ pair<D,CVar>(d,c) ] = iNextPredictor;  midc[ iNextPredictor ] = pair<D,CVar>(d,c);  return( iNextPredictor++ );
    }
    unsigned int getPredictorIndex( D d, CVar c ) const {                      // const version with closed predictor domain
      const auto& it = mdci.find( pair<D,CVar>(d,c) );  return( ( it != mdci.end() ) ? it->second : 0 );
    }

    unsigned int getResponseIndex( F f, EVar e, K k ) {
      const auto& it = mfeki.find( FEK(f,e,k) );  if( it != mfeki.end() ) return( it->second );
      mfeki[ FEK(f,e,k) ] = iNextResponse;  mifek[ iNextResponse ] = FEK(f,e,k);  return( iNextResponse++ );
    }
    unsigned int getResponseIndex( F f, EVar e, K k ) const {                  // const version with closed predictor domain
      const auto& it = mfeki.find( FEK(f,e,k) );  return( ( it != mfeki.end() ) ? it->second : uint(-1) );
    }

    const FEK& getFEK( unsigned int i ) const {
      auto it = mifek.find( i );
      assert( it != mifek.end() );
      return it->second;
    }

    arma::vec calcResponses( const FPredictorVec& lfpredictors ) const {
      arma::vec flogresponses = arma::zeros( matF.n_rows );
      for ( auto& fpredr : lfpredictors ) if ( fpredr < matF.n_cols ) flogresponses += matF.col( fpredr );
      arma::vec fresponses = arma::exp( flogresponses );
      double fnorm = arma::accu( fresponses );                                 // fork normalization term (denominator)

      // Replace overflowing distribs by max...
      if( fnorm == 1.0/0.0 ) {
        uint ind_max=0; for( uint i=0; i<flogresponses.size(); i++ ) if( flogresponses(i)>flogresponses(ind_max) ) ind_max=i;
        flogresponses -= flogresponses( ind_max );
        fresponses = arma::exp( flogresponses );
        fnorm = arma::accu( fresponses ); //accumulate is sum over elements
      } //closes if fnorm
      return fresponses / fnorm;
    }

    friend ostream& operator<<( ostream& os, const pair< const FModel&, const FPredictorVec& >& mv ) {
      for( const auto& i : mv.second ) {
        if( &i != &mv.second.front() ) os << ",";
        const auto& itK = mv.first.midkkk.find(i);
       	if( itK != mv.first.midkkk.end() ) { os << "d" << itK->second.first() << "&" << itK->second.second() << "&" << itK->second.third() << "&" << itK->second.fourth() << "=1"; continue; }
        const auto& itC = mv.first.midc.find(i);
        if( itC != mv.first.midc.end()   ) { os << "d" << itC->second.first << "&t" << itC->second.second << "=1"; continue; }
        const auto& itS = mv.first.mis.find(i);
        if( itS != mv.first.mis.end()    ) { os << "a" << itS->second << "=1"; }
      }
      return os;
    }

    unsigned int getNumPredictors( ) { return iNextPredictor; }
    unsigned int getNumResponses(  ) { return iNextResponse;  }
};

////////////////////////////////////////////////////////////////////////////////

class JPredictorVec : public list<unsigned int> {

  public:

    template<class JM>  // J model is template variable to allow same behavior for const and non-const up until getting predictor indices
    JPredictorVec( JM& jm, F f, EVar eF, const LeftChildSign& aLchild, const StoreState& ss ) {
#ifdef SIMPLE_STORE
      int d = (FEATCONFIG & 1) ? 0 : ss.getDepth();
      const Sign& aAncstr  = ss.getBase();
#else
      int d = (FEATCONFIG & 1) ? 0 : ss.getDepth()+f;
      const Sign& aAncstr  = ss.at( ss.getAncestorBIndex(f) );
#endif
      const HVec& hvAncstr = ( aAncstr.getHVec().size()==0 ) ? hvBot : aAncstr.getHVec();
#ifdef SIMPLE_STORE
      const HVec& hvFiller = ( ss.getBase().getCat().getNoloArity() ) ? ss.getNoloBack().getHVec() : hvBot; //HVec();
#else
      int iCarrierB = ss.getAncestorBCarrierIndex( f );
      const HVec& hvFiller = ( iCarrierB<0                 ) ? hvBot : ss.at( iCarrierB ).getHVec();
#endif
      const HVec& hvLchild = ( aLchild.getHVec().size()==0 ) ? hvBot : aLchild.getHVec() ;
      emplace_back( jm.getPredictorIndex( "Bias" ) );  // add bias
      if( STORESTATE_TYPE ) emplace_back( jm.getPredictorIndex( d, aAncstr.getCat(), aLchild. getCat() ) );
      if( !(FEATCONFIG & 32) ) {
        for( uint iA=0; iA<hvAncstr.size(); iA++ ) for( auto& kA : hvAncstr[iA] )
          for( uint iL=0; iL<hvLchild.size(); iL++ ) for( auto& kL : hvLchild[iL] ) emplace_back( jm.getPredictorIndex( d, kNil, kA.project(-iA), kL.project(-iL) ) );
        for( uint iF=0; iF<hvFiller.size(); iF++ ) for( auto& kF : hvFiller[iF] )
          for( uint iA=0; iA<hvAncstr.size(); iA++ ) for( auto& kA : hvAncstr[iA] ) emplace_back( jm.getPredictorIndex( d, kF.project(-iF), kA.project(-iA), kNil ) );
        for( uint iF=0; iF<hvFiller.size(); iF++ ) for( auto& kF : hvFiller[iF] )
          for( uint iL=0; iL<hvLchild.size(); iL++ ) for( auto& kL : hvLchild[iL] ) emplace_back( jm.getPredictorIndex( d, kF.project(-iF), kNil, kL.project(-iL) ) );
      }
    }
};

////////////////////////////////////////////////////////////////////////////////

class JModel {

  typedef DelimitedQuad<psX,J,psAmpersand,Delimited<EVar>,psAmpersand,O,psAmpersand,O,psX> JEOO;

  private:

    arma::mat matJ;                              // matrix itself

    unsigned int iNextPredictor = 0;             // predictor and response next-pointers
    unsigned int iNextResponse  = 0;

    map<string,unsigned int> msi;                // predictor indices for ad-hoc feature
    map<unsigned int,string> mis;
    map<quad<D,K,K,K>,unsigned int> mdkkki;      // predictor indices for k-context tuples
    map<unsigned int,quad<D,K,K,K>> midkkk;
    map<trip<D,CVar,CVar>,unsigned int> mdcci;   // predictor indices for category tuples
    map<unsigned int,trip<D,CVar,CVar>> midcc;

    map<JEOO,unsigned int> mjeooi;               // response indices
    map<unsigned int,JEOO> mijeoo;

    unsigned int jr0;
    unsigned int jr1;

  public:

    JModel( )             : jr0(getResponseIndex(0,EVar::eNil,O_N,O_I)), jr1(getResponseIndex(1,EVar::eNil,O_N,O_I)) { }
    JModel( istream& is ) : jr0(getResponseIndex(0,EVar::eNil,O_N,O_I)), jr1(getResponseIndex(1,EVar::eNil,O_N,O_I)) {
      list< trip< unsigned int, unsigned int, double > > l;    // store elements on list until we know dimensions of matrix
      while( is.peek()=='J' ) {
        auto& prw = *l.emplace( l.end() );
	is >> "J ";
	if( is.peek()=='a' )   { Delimited<string> s;   is >> "a" >> s >> " : ";                                        prw.first()  = getPredictorIndex( s );             }
        else{
          D d;                                          is >> "d" >> d >> "&";
          if( is.peek()=='t' ) { Delimited<CVar> cA,cL; is >> "t" >> cA >> "&t" >> cL >> " : ";                         prw.first()  = getPredictorIndex( d, cA, cL );     }
          else                 { Delimited<K> kF,kA,kL; is >> kF >> "&" >> kA >> "&" >> kL >> " : ";                    prw.first()  = getPredictorIndex( d, kF, kA, kL ); }
        }
        J j; Delimited<EVar> e; O oL,oR;                is >> "j" >> j >> "&" >> e >> "&" >> oL >> "&" >> oR >> " = ";  prw.second() = getResponseIndex( j, e, oL, oR );
        Delimited<double> w;                            is >> w >> "\n";                                                prw.third()  = w;
      }

      if( l.size()==0 ) cerr << "ERROR: No J items found." << endl;
      matJ.zeros ( mijeoo.size(), iNextPredictor );
      for( auto& prw : l ) { matJ( prw.second(), prw.first() ) = prw.third(); }

      // Ensure JResponses exist...
      jr0 = getResponseIndex( 0, EVar::eNil, 'N', 'I' ); 
      jr1 = getResponseIndex( 1, EVar::eNil, 'N', 'I' ); 
    }

    unsigned int getResponse0( ) const { return jr0; }
    unsigned int getResponse1( ) const { return jr1; }

    unsigned int getPredictorIndex( const string& s ) {
      const auto& it = msi.find( s );  if( it != msi.end() ) return( it->second );
      msi[ s ] = iNextPredictor;  mis[ iNextPredictor ] = s;  return( iNextPredictor++ );
    }
    unsigned int getPredictorIndex( const string& s ) const {                  // const version with closed predictor domain
      const auto& it = msi.find( s );  return( ( it != msi.end() ) ? it->second : 0 );
    }

    unsigned int getPredictorIndex( D d, K kF, K kA, K kL ) {
      const auto& it = mdkkki.find( quad<D,K,K,K>(d,kF,kA,kL) );  if( it != mdkkki.end() ) return( it->second );
      mdkkki[ quad<D,K,K,K>(d,kF,kA,kL) ] = iNextPredictor;  midkkk[ iNextPredictor ] = quad<D,K,K,K>(d,kF,kA,kL);  return( iNextPredictor++ );
    }
    unsigned int getPredictorIndex( D d, K kF, K kA, K kL ) const {            // const version with closed predictor domain
      const auto& it = mdkkki.find( quad<D,K,K,K>(d,kF,kA,kL) );  return( ( it != mdkkki.end() ) ? it->second : 0 );
    }

    unsigned int getPredictorIndex( D d, CVar cA, CVar cL ) {
      const auto& it = mdcci.find( trip<D,CVar,CVar>(d,cA,cL) );  if( it != mdcci.end() ) return( it->second );
      mdcci[ trip<D,CVar,CVar>(d,cA,cL) ] = iNextPredictor;  midcc[ iNextPredictor ] = trip<D,CVar,CVar>(d,cA,cL);  return( iNextPredictor++ );
    }
    unsigned int getPredictorIndex( D d, CVar cA, CVar cL ) const {            // const version with closed predictor domain
      const auto& it = mdcci.find( trip<D,CVar,CVar>(d,cA,cL) );  return( ( it != mdcci.end() ) ? it->second : 0 );
    }

    unsigned int getResponseIndex( J j, EVar e, O oL, O oR ) {
      const auto& it = mjeooi.find( JEOO(j,e,oL,oR) );  if( it != mjeooi.end() ) return( it->second );
      mjeooi[ JEOO(j,e,oL,oR) ] = iNextResponse;  mijeoo[ iNextResponse ] = JEOO(j,e,oL,oR);  return( iNextResponse++ );
    }
    unsigned int getResponseIndex( J j, EVar e, O oL, O oR ) const {           // const version with closed predictor domain
      const auto& it = mjeooi.find( JEOO(j,e,oL,oR) );  assert( it != mjeooi.end() );  return( ( it != mjeooi.end() ) ? it->second : uint(-1) );
    }

    const JEOO& getJEOO( unsigned int i ) const {
      auto it = mijeoo.find( i );
      assert( it != mijeoo.end() );
      return it->second;
    }

    arma::vec calcResponses( const JPredictorVec& ljpredictors ) const {
      arma::vec jlogresponses = arma::zeros( matJ.n_rows );
      for ( auto& jpredr : ljpredictors ) if ( jpredr < matJ.n_cols ) jlogresponses += matJ.col( jpredr );
      arma::vec jresponses = arma::exp( jlogresponses );
      double jnorm = arma::accu( jresponses );                                 // join normalization term (denominator)

      // Replace overflowing distribs by max...
      if( jnorm == 1.0/0.0 ) {
        uint ind_max=0; for( uint i=0; i<jlogresponses.size(); i++ ) if( jlogresponses(i)>jlogresponses(ind_max) ) ind_max=i;
        jlogresponses -= jlogresponses( ind_max );
        jresponses = arma::exp( jlogresponses );
        jnorm = arma::accu( jresponses ); //accumulate is sum over elements
      } //closes if jnorm
      return jresponses / jnorm;
    }

    friend ostream& operator<<( ostream& os, const pair< const JModel&, const JPredictorVec& >& mv ) {
      for( const auto& i : mv.second ) {
        if( &i != &mv.second.front() ) os << ",";
        const auto& itK = mv.first.midkkk.find(i);
       	if( itK != mv.first.midkkk.end() ) { os << "d" << itK->second.first() << "&" << itK->second.second() << "&" << itK->second.third() << "&" << itK->second.fourth() << "=1"; continue; }
        const auto& itC = mv.first.midcc.find(i);
        if( itC != mv.first.midcc.end()  ) { os << "d" << itC->second.first() << "&t" << itC->second.second() << "&t" << itC->second.third() << "=1"; continue; }
        const auto& itS = mv.first.mis.find(i);
        if( itS != mv.first.mis.end()    ) { os << "a" << itS->second << "=1"; }
      }
      return os;
    }

    unsigned int getNumPredictors( ) { return iNextPredictor; }
    unsigned int getNumResponses(  ) { return iNextResponse;  }
};

////////////////////////////////////////////////////////////////////////////////

