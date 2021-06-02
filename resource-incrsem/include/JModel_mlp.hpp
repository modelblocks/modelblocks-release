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
  uint JSYN_SIZE = 20; //placeholders - these will be overwritten when reading in the model
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
        //cerr << "found a J" << endl;
        Delimited<char> c;
        is >> "J " >> c >> " ";
        if (c == 'F') is >> jwf >> "\n";
        if (c == 'f') is >> jbf >> "\n";
        if (c == 'S') is >> jws >> "\n";
        if (c == 's') is >> jbs >> "\n";
      }
     //cerr << "loaded J weights" << endl;
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
      //cerr << "loaded J Cs" << endl;
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
      //cerr << "loaded J Ks" << endl;
      while ( is.peek()=='j' ) {
        Delimited<int> k;
        is >> "j " >> k >> " ";
        is >> mijeoo[k] >> "\n";
        mjeooi[mijeoo[k]] = k;
        iNextResponse = k+1; //code review WS this should be handled more elegantly, since inextresponse is legacy
      }
      //cerr << "loaded Jresps" << endl;
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

