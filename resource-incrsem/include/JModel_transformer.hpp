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

const HVec getHvLchild( const BeamElement<HiddState>& be ) {
  HVec hvLchild;
  // If we don't fork (i.e., lexical match decision = 1), the left child
  // is the apex of the deepest derivation fragment from the previous time
  // step
  if ( be.getHidd().getF() == 0 ) {
    StoreState ssPrev = be.getBack().getHidd().getStoreState();
    hvLchild = (( ssPrev.getApex().getHVec().size() > 0 ) ? ssPrev.getApex().getHVec() : hvBot);

  }
  // If we do fork (i.e., lexical match decision = 0), the left child is
  // the predicted preterminal node
  else {
    hvLchild = be.getHidd().getPrtrm().getHVec();
  }
  return hvLchild;
}


const CVar getCatLchild( const BeamElement<HiddState>& be ) {
  CVar catLchild;

  // Same logic as getHvLchild()
  if ( be.getHidd().getF() == 0 ) {
    catLchild = be.getBack().getHidd().getStoreState().getBase().getCat();
  }
  else {
    catLchild = be.getHidd().getPrtrm().getCat();
  }
  return catLchild;
}


const HVec getHvAncestor( const BeamElement<HiddState>& be ) {
  HVec hvAnc;
  // If we don't fork (i.e., lexical match decision = 1), the ancestor is
  // the base of the second deepest derivation fragment from the previous
  // time step
  if ( be.getHidd().getF() == 0 ) {
    StoreState ssPrev = be.getBack().getHidd().getStoreState();
    // getBase(1) retrieves the base of the second deepest derivation fragment
    hvAnc = (( ssPrev.getBase(1).getHVec().size() > 0 ) ? ssPrev.getBase(1).getHVec() : hvBot);

  }
  // If we do fork (i.e., lexical match decision = 0), the ancestor is
  // the base of the deepest derivation fragment from the previous time step
  else {
    StoreState ssPrev = be.getBack().getHidd().getStoreState();
    hvAnc = (( ssPrev.getBase().getHVec().size() > 0 ) ? ssPrev.getBase().getHVec() : hvBot);
  }
  return hvAnc;
}


const CVar getCatAncestor( const BeamElement<HiddState>& be ) {
  CVar catAnc;

  // Same logic as getHvAncestor()
  if ( be.getHidd().getF() == 0 ) {
    catAnc = be.getBack().getHidd().getStoreState().getBase(1).getCat();
  }
  else {
    catAnc = be.getBack().getHidd().getStoreState().getBase().getCat();
  }
  return catAnc;
}


class JPredictorVec {

  private:
    const BeamElement<HiddState>& be;
    //const HVec hvLchild;
    //CVar catLchild;

  public:
//    JPredictorVec( const BeamElement<HiddState>& belement, const LeftChildSign& aLchild )
//      : be (belement),
//    {
//      hvLchild  = ( aLchild.getHVec().size()==0 ) ? hvBot : aLchild.getHVec();
//      catLchild = ( aLchild.getHVec().size()==0 ) ? cBot : aLchild.getCat();
//    }
    JPredictorVec( const BeamElement<HiddState>& belement )
      : be (belement) { }

    const BeamElement<HiddState>& getBeamElement() const {
        return be;
    }

//    const HVec getHvLchild() {
//      return hvLchild;
//    }
//
//    CVar getCatLchild() {
//        return catLchild;
//    }

    friend ostream& operator<< ( ostream& os, const JPredictorVec& jpv ) {
      const int d = getDepth(jpv.be);
      const CVar catAncstr = getCatAncestor(jpv.be);
      const HVec hvAncstr = getHvAncestor(jpv.be);
      const HVec hvFiller = getHvF(jpv.be);
      const CVar catLchild = getCatLchild(jpv.be);
      const HVec hvLchild = getHvLchild(jpv.be);
        
      os << d << " " << catAncstr << " " << hvAncstr << " " << hvFiller << " " << catLchild << " " << hvLchild;
      return os;
    }
};



////////////////////////////////////////////////////////////////////////////////
class JModel {

  typedef DelimitedQuad<psX,J,psAmpersand,Delimited<EVar>,psAmpersand,O,psAmpersand,O,psX> JEOO;
  unsigned int jr0;
  unsigned int jr1;

  private:
    static const uint JSEM_DIM_DEFAULT = 20;
    static const uint JSYN_DIM_DEFAULT = 20;
    uint JSEM_DIM;
    uint JSYN_DIM;

    map<CVar,vec> mcav; // map between ancestor syntactic category and embeds
    map<CVar,vec> mclv; // map between left-child syntactic category and embeds
    map<KVec,vec> mkadv; // map between ancestor KVec and embeds
    map<KVec,vec> mkfdv; // map between filler KVec and embeds
    map<KVec,vec> mkldv; // map between left-child KVec and embeds

    map<JEOO,unsigned int> mjeooi;               // response indices
    map<unsigned int,JEOO> mijeoo;

    unsigned int iNextResponse  = 0;

    // weights
    /*
    DelimitedVector<psX, double, psComma, psX> jwpq; // pre-attention query projection
    DelimitedVector<psX, double, psComma, psX> jwpkv; // pre-attention key/value projection
    */
    DelimitedVector<psX, double, psComma, psX> jwp; // pre-attention projection
    DelimitedVector<psX, double, psComma, psX> jwi; // attention input projection -- concatenation of query, key, and value matrices
    DelimitedVector<psX, double, psComma, psX> jwo; // attention output projection
    DelimitedVector<psX, double, psComma, psX> jwf; // first feedforward
    DelimitedVector<psX, double, psComma, psX> jws; // second feedforward

    //mat jwpqm;
    //mat jwpkvm;
    mat jwpm;
    mat jwim;
    mat jwqm; // query
    mat jwkm; // key
    mat jwvm; // value
    mat jwom;
    mat jwfm;
    mat jwsm;

    // biases
    //DelimitedVector<psX, double, psComma, psX> jbpq; // pre-attention query projection
    //DelimitedVector<psX, double, psComma, psX> jbpkv; // pre-attention key/value projection
    DelimitedVector<psX, double, psComma, psX> jbp; // pre-attention projection
    DelimitedVector<psX, double, psComma, psX> jbi; // attention input projection
    DelimitedVector<psX, double, psComma, psX> jbo; // attention output projection
    DelimitedVector<psX, double, psComma, psX> jbf; // first feedforward
    DelimitedVector<psX, double, psComma, psX> jbs; // second feedforward

    //vec jbpqv;
    //vec jbpkvv;
    vec jbpv;
    vec jbiv;
    vec jbqv; // query
    vec jbkv; // key
    vec jbvv; // value
    vec jbov;
    vec jbfv;
    vec jbsv;

    vec computeResult( vector<vec> attnInput, uint wordIndex, bool verbose ) const;

  public:

    JModel( ) { 
      jr0 = -1;
      jr1 = -1;
    }

    // read in weights, embeddings, and JEOOs
    JModel( istream& is ) {
      while ( is.peek()=='J' ) {
        Delimited<char> c;
        is >> "J " >> c >> " ";
        /*
        if (c == 'Q') is >> jwpq >> "\n";
        if (c == 'q') is >> jbpq >> "\n"; 
        if (c == 'K') is >> jwpkv >> "\n";
        if (c == 'k') is >> jbpkv >> "\n"; 
        */
        if (c == 'P') is >> jwp >> "\n";
        if (c == 'p') is >> jbp >> "\n"; 
        if (c == 'I') is >> jwi >> "\n";
        if (c == 'i') is >> jbi >> "\n"; 
        if (c == 'O') is >> jwo >> "\n";
        if (c == 'o') is >> jbo >> "\n"; 
        if (c == 'F') is >> jwf >> "\n";
        if (c == 'f') is >> jbf >> "\n"; 
        if (c == 'S') is >> jws >> "\n";
        if (c == 's') is >> jbs >> "\n"; 
      }

      JSYN_DIM = JSYN_DIM_DEFAULT;
      while ( is.peek()=='C' ) {
        Delimited<char> c;
        Delimited<CVar> cv;
        DelimitedVector<psX, double, psComma, psX> vtemp;  
        is >> "C " >> c >> " " >> cv >> " ";
        is >> vtemp >> "\n";
        if (c == 'A') mcav.try_emplace(cv,vtemp);
        else {
          assert (c == 'L');
          mclv.try_emplace(cv,vtemp);
        }
        JSYN_DIM=vtemp.size();
      }

      JSEM_DIM = JSEM_DIM_DEFAULT;
      while ( is.peek()=='K' ) {
        Delimited<char> c;
        Delimited<K> k;
        DelimitedVector<psX, double, psComma, psX> vtemp;
        is >> "K " >> c >> " " >> k >> " " >> vtemp >> "\n";
        if (c == 'A') mkadv.try_emplace(k, vtemp);
        else if (c == 'F') mkfdv.try_emplace(k, vtemp);
        else {
          assert (c == 'L');
          mkldv.try_emplace(k, vtemp);
        }
        JSEM_DIM=vtemp.size();
      }

      while ( is.peek()=='j' ) {
        Delimited<int> k;
        is >> "j " >> k >> " ";
        is >> mijeoo[k] >> "\n";
        mjeooi[mijeoo[k]] = k;
        iNextResponse = k+1; //code review WS this should be handled more elegantly, since inextresponse is legacy
      }

      jr0 = getResponseIndex( 0, EVar::eNil, 'N', O_I );
      jr1 = getResponseIndex( 1, EVar::eNil, 'N', O_I );

      //jwpqm = jwpq;
      //jwpkvm = jwpkv;
      jwpm = jwp;
      jwim = jwi;
      jwom = jwo;
      jwfm = jwf;
      jwsm = jws;

      //jbpqv = jbpq;
      //jbpkvv = jbpkv;
      jbpv = jbp;
      jbiv = jbi;
      jbov = jbo;
      jbfv = jbf;
      jbsv = jbs;

      // reshape weight matrices
      //uint pre_attn_q_dim = 7 + 2*JSYN_DIM + 3*JSEM_DIM;
      //uint pre_attn_kv_dim = 7 + JSYN_DIM + 2*JSEM_DIM;
      uint pre_attn_dim = 7 + JSYN_DIM + 2*JSEM_DIM;
      uint attn_dim = jwp.size()/pre_attn_dim;
      // output of attn layer is concatenated with catLchild and hvLchild
      uint post_attn_dim = pre_attn_dim + JSYN_DIM + JSEM_DIM;
      uint hidden_dim = jwf.size()/post_attn_dim;
      uint output_dim = jws.size()/hidden_dim;

      //jwpqm.reshape(attn_dim, pre_attn_q_dim);
      //jwpkvm.reshape(attn_dim, pre_attn_kv_dim);
      jwpm.reshape(attn_dim, pre_attn_dim);

      // fwim contains query, key, and value projection matrices,
      // each of dimension attn_dim x attn_dim
      jwim.reshape(3*attn_dim, attn_dim);
      jwqm = jwim.rows(0, attn_dim-1);
      jwkm = jwim.rows(attn_dim, 2*attn_dim-1);
      jwvm = jwim.rows(2*attn_dim, 3*attn_dim-1);

      jwom.reshape(attn_dim, attn_dim);
      jwfm.reshape(hidden_dim, post_attn_dim);
      jwsm.reshape(output_dim, hidden_dim);

      // fbiv contains biases vectors for query, key, and value
      jbqv = jbiv(span(0, attn_dim-1));
      jbkv = jbiv(span(attn_dim, 2*attn_dim-1));
      jbvv = jbiv(span(2*attn_dim, 3*attn_dim-1));
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
      vec zeroCatEmb = zeros(JSYN_DIM);
      if (c == 'A') {
        auto it = mcav.find( i );
        return ( ( it != mcav.end() ) ? it->second : zeroCatEmb );
      }
      else {
        assert ( c == 'L' );
        auto it = mclv.find( i );
        return ( ( it != mclv.end() ) ? it->second : zeroCatEmb );
      }
      cerr << "ERROR: J model CVar position misspecified." << endl;
      return zeroCatEmb;
    }


    const vec getKVecEmbed( HVec hv, Delimited<char> c ) const {
      vec KVecEmbed = arma::zeros(JSEM_DIM);
      if (c == 'A') {
        for ( auto& kV : hv.at(0) ) {
          if ( kV == K::kTop) {
            KVecEmbed += arma::ones(JSEM_DIM);
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
            KVecEmbed += arma::ones(JSEM_DIM);
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
            KVecEmbed += arma::ones(JSEM_DIM);
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

    vec calcResponses( JPredictorVec& ljpredictors, int wordIndex ) const;

    void testCalcResponses() const;
};

vec JModel::calcResponses( JPredictorVec& ljpredictors, int wordIndex ) const {

//  const HVec hvLchild = ljpredictors.getHvLchild();
//  const vec hvLchildEmb = getKVecEmbed(hvLchild, 'A');
//  const CVar catLchild = ljpredictors.getCatLchild();
//  const vec catLchildEmb = getCatEmbed(hvLchild, 'A');
//  vec lchildVec = join_cols(catLchildEmb, hvLchildEmb);
  const BeamElement<HiddState> be = ljpredictors.getBeamElement();

  vector<vec> attnInput;
  
  //uint MAX_WINDOW_SIZE = 20;
  uint MAX_WINDOW_SIZE = 10;
  uint wordOffset = 0;
  // this moves backwards in time, starting with the word for which the
  // response is being calculated. wordOffset tracks how many
  // words back we've moved
  for (const BeamElement<HiddState>* curr = &be; ( (curr != &BeamElement<HiddState>::beStableDummy) && (wordOffset < MAX_WINDOW_SIZE) ); curr=&curr->getBack(), wordOffset++) {
    CVar catAnc = getCatAncestor(*curr);
    HVec hvAnc = getHvAncestor(*curr);
    HVec hvFiller = getHvF(*curr);
    CVar catLchild = getCatLchild(*curr);
    HVec hvLchild = getHvLchild(*curr);
    uint d = getDepth(*curr);
    
    vec catAncEmb = getCatEmbed(catAnc, 'A');
    vec hvAncEmb = getKVecEmbed(hvAnc, 'A');
    vec hvFillerEmb = getKVecEmbed(hvFiller, 'F');
    vec catLchildEmb = getCatEmbed(catLchild, 'L');
    vec hvLchildEmb = getKVecEmbed(hvLchild, 'L');

    vec currAttnInput = join_cols(join_cols(join_cols(join_cols(join_cols(catAncEmb, hvAncEmb), hvFillerEmb), catLchildEmb), hvLchildEmb), zeros(7)); 
    currAttnInput(3*JSEM_DIM + 2*JSYN_DIM + d) = 1;
    // vector<> doesn't have an emplace_front method
    attnInput.emplace_back(currAttnInput);
  }

  // reverse attnInput so that the last item is the most recent word
  reverse(attnInput.begin(), attnInput.end());
  
  return computeResult(attnInput, wordIndex, 0);
}


// TODO continue here (stuff below is from F transformer)

// return distribution over JEOO indices
// attnInput contains the embeddings for previous words up to the current word
// attnInput.back() is the current word that we are making an J decision for
vec JModel::computeResult( vector<vec> attnInput, uint wordIndex, bool verbose ) const {
  bool usePositionalEncoding = false;
  vec last = attnInput.back();
  if ( verbose ) {
    //cerr << "F last" << last << endl;

//    cerr << "F fwpm" << endl;
//    cerr << "num rows: " << fwpm.n_rows << endl;
//    cerr << "num cols: " << fwpm.n_cols << endl;
//    for ( uint j=0; j<fwpm.n_cols; j++ ) {
//      for ( uint i=0; i<fwpm.n_rows; i++ ) {
//        cerr << fwpm(i, j) << endl;
//      }
//    }
//    cerr << "F fbpv" << fbpv << endl;
  }
  vec proj = jwpm*last + jbpv;
  if ( usePositionalEncoding ) {
    proj = proj + getPositionalEncoding(jbpv.size(), wordIndex);
  }
  if ( verbose ) {
    cerr << "J proj" << proj << endl;
  }
  const vec query = jwqm*proj + jbqv;
  // used to scale the attention softmax (see section 3.2.1 of Attention
  // Is All You Need)
  const double scalingFactor = sqrt(jbqv.size());

  vector<vec> values;
  vector<double> scaledDotProds;
  int currIndex = wordIndex - attnInput.size() + 1;
  for ( vec curr : attnInput ) { 
    proj = jwpm*curr + jbpv;
    if ( usePositionalEncoding ) {
      proj = proj + getPositionalEncoding(jbpv.size(), currIndex++);
    }
    vec key = jwkm*proj + jbkv;
    vec value = jwvm*proj + jbvv;
    values.emplace_back(value);
    scaledDotProds.emplace_back(dot(query, key)/scalingFactor);
  }

  vec sdp = vec(scaledDotProds.size());
  for ( uint i=0; (i < scaledDotProds.size()); i++ ) {
    sdp(i) = scaledDotProds[i];
  }

  vec sdpExp = exp(sdp);
  double norm = accu(sdpExp);
  vec sdpSoftmax = sdpExp/norm;

  // calculate scaled_softmax(QK)*V
  vec attnResult = zeros<vec>(jbvv.size());

  for ( uint i=0; (i < values.size()); i++ ) {
    double weight = sdpSoftmax(i);
    vec val = values[i];
    attnResult += weight*val;
  }

  vec attnOutput = jwom*attnResult + jbov;

  if ( verbose ) {
    cerr << "J attnOutput" << attnOutput << endl;
  }
  //vec hiddenInput = join_cols(attnOutput, corefVec);
  vec logScores = jwsm * relu(jwfm*attnOutput + jbfv) + jbsv;
  vec scores = exp(logScores);
  double outputNorm = accu(scores);
  vec result = scores/outputNorm;
  if ( verbose ) {
    cerr << "J result" << result << endl;
  }
  return result;
} 


// pass vectors of 1s as inputs as a sanity check (for comparison with
// Python training code)
void JModel::testCalcResponses() const {
  //vec corefVec = ones<vec>(FANT_DIM+1);
  vec currAttnInput;
  vector<vec> attnInput;
  uint attnInputDim = 3*JSEM_DIM + 2*JSYN_DIM + 7;
  uint seqLength = 5;
  // the ith dummy input will be a vector of 0.1*(i+1)
  // [0.1 0.1 0.1 ...] [0.2 0.2 0.2 ...] ...
  for ( uint i=0; (i<seqLength); i++ ) {
    cerr << "J ==== output for word " << i << " ====" << endl;
    currAttnInput = vec(attnInputDim);
    currAttnInput.fill(0.1 * (i+1));
    attnInput.emplace_back(currAttnInput);
    computeResult(attnInput, i, 1);
  }
}
