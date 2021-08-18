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

void print_vec(vec vector, uint maxlen=10) {
  cerr << "Printing first " << maxlen << " items..." << endl;
  uint len = vector.size();
  for ( uint i=0; (i<maxlen) && (i<len); i++ ) {
    cerr << vector(i) << endl;
  }
}

// TODO probably okay to pass in actual BeamElement, not pointer
//const HVec getHvLchild( const BeamElement<HiddState>& be ) {
//const HVec getHvLchild( const BeamElement<HiddState>* pbe, const EMat& matE, const OFunc& funcO ) {
const HVec getHvLchild( const BeamElement<HiddState>* pbe ) {
  return pbe->getHidd().getLchild().getHVec();
//  // If we don't fork (i.e., lexical match decision = 1), the left child
//  // is the apex of the deepest derivation fragment from the previous time
//  // step
//  if ( pbe->getHidd().getF() == 0 ) {
//    // TODO if hvLChild here is ["], need to retrieve semantics of base of fragment?
//    StoreState ssPrev = pbe->getBack().getHidd().getStoreState();
//    hvLchild = (( ssPrev.getApex().getHVec().size() > 0 ) ? ssPrev.getApex().getHVec() : hvBot);
//
//    // Add in semantics predicted by F step
//    //EVar e = pbe->getHidd().getForkE();
//    K k = pbe->getHidd().getForkK();
//    HVec hvF = HVec( k, matE, funcO );
//    hvLchild.add( hvF );
//    // TODO may need to add something like this from StoreState line 531
//    //applyUnariesBotUp( back().apex(), evJ );                   // Calc apex contexts.
//  }
//  // If we do fork (i.e., lexical match decision = 0), the left child is
//  // the predicted preterminal node
//  else {
//    hvLchild = pbe->getHidd().getPrtrm().getHVec();
//  }
//  return hvLchild;
}


//const CVar getCatLchild( const BeamElement<HiddState>& be ) {
const CVar getCatLchild( const BeamElement<HiddState>* pbe ) {
  return pbe->getHidd().getLchild().getCat();
//  CVar catLchild;
//
//  if ( pbe->getHidd().getF() == 0 ) {
//    catLchild = pbe->getBack().getHidd().getStoreState().getApex().getCat();
//  }
//  else {
//    catLchild = pbe->getHidd().getPrtrm().getCat();
//  }
//  return catLchild;
}


//const HVec getHvAncestor( const BeamElement<HiddState>& be ) {
const HVec getHvAncestor( const BeamElement<HiddState>* pbe ) {
  HVec hvAnc;
  StoreState ssPrev = pbe->getBack().getHidd().getStoreState();
  // If we don't fork (i.e., lexical match decision = 1), the ancestor is
  // the base of the second deepest derivation fragment from the previous
  // time step
  if ( pbe->getHidd().getF() == 0 ) {
    // getBase(1) retrieves the base of the second deepest derivation fragment
    hvAnc = (( ssPrev.getBase(1).getHVec().size() > 0 ) ? ssPrev.getBase(1).getHVec() : hvBot);

  }
  // If we do fork (i.e., lexical match decision = 0), the ancestor is
  // the base of the deepest derivation fragment from the previous time step
  else {
    hvAnc = (( ssPrev.getBase().getHVec().size() > 0 ) ? ssPrev.getBase().getHVec() : hvBot);
  }
  return hvAnc;
}


//const CVar getCatAncestor( const BeamElement<HiddState>& be ) {
const CVar getCatAncestor( const BeamElement<HiddState>* pbe ) {
  CVar catAnc;

  // Same logic as getHvAncestor()
  if ( pbe->getHidd().getF() == 0 ) {
    catAnc = pbe->getBack().getHidd().getStoreState().getBase(1).getCat();
  }
  else {
    catAnc = pbe->getBack().getHidd().getStoreState().getBase().getCat();
  }
  return catAnc;
}

// similar to getHvF in transformer.hpp but uses a pointer as input
HVec getHvFiller( const BeamElement<HiddState>* pbe ) {
    StoreState ss = pbe->getHidd().getStoreState();
    return (( ss.getBase().getCat().getNoloArity() && ss.getNoloBack().getHVec().size() != 0 ) ? ss.getNoloBack().getHVec() : hvBot);
}


uint getD( const BeamElement<HiddState>* pbe ) {
  uint depth = pbe->getBack().getHidd().getStoreState().getDepth();
  if ( pbe->getHidd().getF() == 1 ) depth++;
  return depth;
}


class JPredictorVec {

  private:
    const BeamElement<HiddState>& be;
    const F f;
    const HVec hvLchild;
    const CVar catLchild;

  public:
    JPredictorVec( const BeamElement<HiddState>& belement, const F eff, const Sign& aLchild )
      : be (belement), 
      f (eff),
      hvLchild  (( aLchild.getHVec().size()==0 ) ? hvBot : aLchild.getHVec()),
      catLchild (( aLchild.getHVec().size()==0 ) ? cBot : aLchild.getCat())
    {}

    const F getF() const {
      return f;
    }

    const BeamElement<HiddState>& getBeamElement() const {
      return be;
    }

    const CVar getCatAncestor() const {
      uint backOffset = 0;
      if ( f == 0 ) backOffset = 1;
      StoreState ss = be.getHidd().getStoreState();
      return ss.getBase(backOffset).getCat();
    }

    const HVec getHvAncestor() const {
      uint backOffset = 0;
      if ( f == 0 ) backOffset = 1;
      StoreState ss = be.getHidd().getStoreState();
      return (( ss.getBase(backOffset).getHVec().size() > 0 ) ? ss.getBase(backOffset).getHVec() : hvBot);
    }

    const HVec getHvFiller() const {
      uint backOffset = 0;
      if ( f == 0 ) backOffset = 1;
      StoreState ss = be.getHidd().getStoreState();
      return (( ss.getBase(backOffset).getCat().getNoloArity() && ss.getNoloBack(backOffset).getHVec().size() != 0 ) ? ss.getNoloBack(backOffset).getHVec() : hvBot);
    }

    const CVar getCatLchild() const {
      return catLchild;
    }

    const HVec getHvLchild() const {
      return hvLchild;
    }

    const uint getDepth() const {
      uint depth = be.getHidd().getStoreState().getDepth();
      if ( f == 1 ) depth++;
      return depth;
    }

//    const EMat& getMatE() const {
//      return matE;
//    }
//
//    const OFunc& getFuncO() const {
//      return funcO;
//    }

    friend ostream& operator<< ( ostream& os, const JPredictorVec& jpv ) {
      const int d = jpv.getDepth();
      const CVar catAncstr = jpv.getCatAncestor();
      const HVec hvAncstr = jpv.getHvAncestor();
      const HVec hvFiller = jpv.getHvFiller();
      const CVar catLchild = jpv.getCatLchild();
      const HVec hvLchild = jpv.getHvLchild();
        
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
    uint num_heads;
    uint head_dim;

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
        if (c == 'H') {
          Delimited<int> h;
          is >> h >> "\n";
          num_heads = h;
        }
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
      uint pre_attn_dim = 7 + 2*JSYN_DIM + 3*JSEM_DIM;
      uint attn_dim = jwp.size()/pre_attn_dim;
      //// output of attn layer is concatenated with catLchild and hvLchild
      //uint post_attn_dim = pre_attn_dim + JSYN_DIM + JSEM_DIM;
      uint hidden_dim = jwf.size()/attn_dim;
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
      jwfm.reshape(hidden_dim, attn_dim);
      jwsm.reshape(output_dim, hidden_dim);

      // fbiv contains biases vectors for query, key, and value
      jbqv = jbiv(span(0, attn_dim-1));
      jbkv = jbiv(span(attn_dim, 2*attn_dim-1));
      jbvv = jbiv(span(2*attn_dim, 3*attn_dim-1));
      
      head_dim = jbqv.size()/num_heads;
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
//          if ( kV == K::kTop) {
//            KVecEmbed += arma::ones(JSEM_DIM);
//            continue;
//          }
          if ( kV == K::kBot) {
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
//          if ( kV == K::kTop) {
//            KVecEmbed += arma::ones(JSEM_DIM);
//            continue;
//          }
          if ( kV == K::kBot) {
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
//          if ( kV == K::kTop) {
//            KVecEmbed += arma::ones(JSEM_DIM);
//            continue;
//          }
          if ( kV == K::kBot) {
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

    vec calcResponses( JPredictorVec& ljpredictors, int wordIndex, bool verbose ) const;

    void testCalcResponses() const;
};

vec JModel::calcResponses( JPredictorVec& ljpredictors, int wordIndex, bool verbose ) const {

  //const BeamElement<HiddState> be = ljpredictors.getBeamElement();
  const BeamElement<HiddState>* pbe = &ljpredictors.getBeamElement();
  //const EMat matE = ljpredictors.getMatE();
  //const OFunc funcO = ljpredictors.getFuncO();

  vector<vec> attnInput;

  // find attention input for the word for which the j response is being
  // calculated
  
  // If no-fork, then the ancestor is the second deepest fragment
  // in the store state. If yes-fork, it is the deepest fragment
  StoreState ss = pbe->getHidd().getStoreState();
  F f = ljpredictors.getF();
  uint backOffset = 0;
  if ( f == 0 ) backOffset = 1;
  CVar catAnc = ss.getBase(backOffset).getCat();
  HVec hvAnc = (( ss.getBase(backOffset).getHVec().size() > 0 ) ? ss.getBase(backOffset).getHVec() : hvBot);
  HVec hvFiller = (( ss.getBase(backOffset).getCat().getNoloArity() && ss.getNoloBack(backOffset).getHVec().size() != 0 ) ? ss.getNoloBack(backOffset).getHVec() : hvBot);
  CVar catLchild = ljpredictors.getCatLchild();
  HVec hvLchild = ljpredictors.getHvLchild();

  vec catAncEmb = getCatEmbed(catAnc, 'A');
  vec hvAncEmb = getKVecEmbed(hvAnc, 'A');
  vec hvFillerEmb = getKVecEmbed(hvFiller, 'F');
  vec catLchildEmb = getCatEmbed(catLchild, 'L');
  vec hvLchildEmb = getKVecEmbed(hvLchild, 'L');

  if ( verbose ) {
    cerr << "\nJ catAnc of current word: " << catAnc << endl;
    cerr << "\nJ catAncEmb of current word:" << endl;
    print_vec(catAncEmb);
    // TODO switch later stuff to print_vec
    cerr << "\nJ hvAnc of current word: " << hvAnc << endl;
    cerr << "\nJ hvAncEmb of current word\n" << hvAncEmb << endl;
    cerr << "\nJ hvFiller of current word: " << hvFiller << endl;
    cerr << "\nJ hvFillerEmb of current word\n" << hvFillerEmb << endl;
    cerr << "\nJ catLchild of current word: " << catLchild << endl;
    cerr << "\nJ catLchildEmb of current word\n" << catLchildEmb << endl;
    cerr << "\nJ hvLchild of current word: " << hvLchild << endl;
    cerr << "\nJ hvLchildEmb of current word\n" << hvLchildEmb << endl;
    cerr << "\nJ latest word hidd\n" << pbe->getHidd() << endl;
  }

  uint depth = pbe->getHidd().getStoreState().getDepth();
  //cerr << "\nJ latest word depth pre f adjustment\n" << depth << endl;
  if ( f == 1 ) depth++;

  vec currAttnInput = join_cols(join_cols(join_cols(join_cols(join_cols(catAncEmb, hvAncEmb), hvFillerEmb), catLchildEmb), hvLchildEmb), zeros(7)); 
  currAttnInput(3*JSEM_DIM + 2*JSYN_DIM + depth) = 1;
  if ( verbose ) {
    cerr << "\nJ latest word depth\n" << depth << endl;
    cerr << "\nJ latest word attn input\n" << currAttnInput << endl;
  }
  // vector<> doesn't have an emplace_front method
  //attnInput.emplace_back(currAttnInput);
  attnInput.push_back(currAttnInput);

  uint MAX_WINDOW_SIZE = 13;
  uint wordOffset = 1;

  // find attention input for previous words up to MAX_WINDOW_SIZE
  // before the new word
  for (const BeamElement<HiddState>* curr = pbe; ( (&curr->getBack() != &BeamElement<HiddState>::beStableDummy) && (wordOffset < MAX_WINDOW_SIZE) ); curr=&curr->getBack(), wordOffset++) {


    catAnc = getCatAncestor(curr);
    hvAnc = getHvAncestor(curr);
    hvFiller = getHvFiller(curr);
    catLchild = getCatLchild(curr);
    //hvLchild = getHvLchild(curr, matE, funcO);
    hvLchild = getHvLchild(curr);
    depth = getD(curr);
    if ( verbose ) {
      cerr << "\nJ curr attn hvLchild " << wordOffset << " words back\n" << hvLchild << endl;
      cerr << "\nJ curr attn depth " << wordOffset << " words back\n" << depth << endl;
      cerr << "\nJ curr hidd" << wordOffset << " words back\n" << curr->getHidd() << endl;
    }
 
    catAncEmb = getCatEmbed(catAnc, 'A');
    hvAncEmb = getKVecEmbed(hvAnc, 'A');
    hvFillerEmb = getKVecEmbed(hvFiller, 'F');
    catLchildEmb = getCatEmbed(catLchild, 'L');
    hvLchildEmb = getKVecEmbed(hvLchild, 'L');
 
    currAttnInput = join_cols(join_cols(join_cols(join_cols(join_cols(catAncEmb, hvAncEmb), hvFillerEmb), catLchildEmb), hvLchildEmb), zeros(7)); 
    currAttnInput(3*JSEM_DIM + 2*JSYN_DIM + depth) = 1;
    if ( verbose ) {
      cerr << "\nJ curr attn input " << wordOffset << " words back\n" << currAttnInput << endl;
    }
    // vector<> doesn't have an emplace_front method
    //attnInput.emplace_back(currAttnInput);
    attnInput.push_back(currAttnInput);
  }

  // reverse attnInput so that the last item is the most recent word
  reverse(attnInput.begin(), attnInput.end());
  
  return computeResult(attnInput, wordIndex, verbose);
}


// return distribution over JEOO indices
// attnInput contains the embeddings for previous words up to the current word
// attnInput.back() is the current word that we are making an J decision for
vec JModel::computeResult( vector<vec> attnInput, uint wordIndex, bool verbose ) const {
  bool usePositionalEncoding = false;
  vec last = attnInput.back();
  if ( verbose ) {
    cerr << "J length of attnInput: " << attnInput.size() << endl;
    cerr << "J attn input of current word\n" << last << endl;
  }
  vec proj = jwpm*last + jbpv;
  if ( usePositionalEncoding ) {
    proj = proj + getPositionalEncoding(jbpv.size(), wordIndex);
  }
  if ( verbose ) {
    cerr << "J proj\n" << proj << endl;
  }
  const vec query = jwqm*proj + jbqv;
  if ( verbose ) {
    cerr << "J query after linear\n" << query << endl;
  }
  // used to scale the attention softmax (see section 3.2.1 of Attention
  // Is All You Need)
  const double scalingFactor = sqrt(head_dim);

//  vector<vec> values;
//  vector<double> scaledDotProds;

//  vector<vec>[num_heads] values;

  // we only care about the latest word's query, hence vec instead of mat
  vec queries[num_heads];
  // each matrix is the list of q/k/v for a single attn head
  mat keys[num_heads];
  mat values[num_heads];

  for ( uint i=0; i<num_heads; i++ ) {
    vec q_i = query(span( i*head_dim, (i+1)*head_dim-1 ));
    queries[i] = q_i/scalingFactor;
    //queries[i] = query(span( i*head_dim, (i+1)*head_dim-1 )/scalingFactor);
    // set dimensions for key and value matrices
    keys[i] = mat(attnInput.size(), head_dim);
    values[i] = mat(attnInput.size(), head_dim);
  }
  
//  vector<double>[num_heads] scaledDotProds;
//  int currIndex = wordIndex - attnInput.size() + 1;
//  for ( vec curr : attnInput ) { 
  vec curr;
  for ( uint j=0; j<attnInput.size(); j++ ) { 
    vec curr = attnInput[j];
//    if ( verbose ) {
//      cerr << "J attnInput (progresses from first to last)\n" << curr << endl;
//    }
    proj = jwpm*curr + jbpv;
//    if ( usePositionalEncoding ) {
//      proj = proj + getPositionalEncoding(jbpv.size(), currIndex++);
//    }
    vec key = jwkm*proj + jbkv;
//    if ( verbose ) {
//      cerr << "J key (progresses from first to last)\n" << key << endl;
//    }
    vec value = jwvm*proj + jbvv;
    // iterate over attention heads
    for ( uint i=0; i < num_heads; i++ ) {
      keys[i].row(j) = key(span( i*head_dim, (i+1)*head_dim-1 ));
      values[i].row(j) = value(span( i*head_dim, (i+1)*head_dim-1 ));
     // scaledDotProds[i].emplace_back(dot(query_i, key_i)/scalingFactor);
    }
//    values.emplace_back(value);
//    scaledDotProds.emplace_back(dot(query, key)/scalingFactor);
  }

//  vec sdp = vec(scaledDotProds.size());
//  for ( uint i=0; (i < scaledDotProds.size()); i++ ) {
//    sdp(i) = scaledDotProds[i];
//  }

  vec attnResult = vec(jbqv.size());
  for ( uint i=0; i<num_heads; i++ ) {
    // find softmax. sdp = scaled dot product
    vec sdp = keys[i] * queries[i];
    vec sdpExp = exp(sdp);
    double norm = accu(sdpExp);
    vec sdpSoftmax = sdpExp/norm;
    // (head_dim x seq_length) x (seq_legth x 1) => (head_dim x 1)
    vec perHeadAttnResult = values[i].t() * sdpSoftmax;
    attnResult(span( i*head_dim, (i+1)*head_dim-1 )) = perHeadAttnResult;
  }

  

//  if ( verbose ) {
//    cerr << "J pre-softmax attn output weights\n" << sdp << endl;
//  }


//  if ( verbose ) {
//    cerr << "J scaled dot product softmax\n" << sdpSoftmax << endl;
//  }

//  // calculate scaled_softmax(QK)*V
//  vec attnResult = zeros<vec>(jbvv.size());
//
//  for ( uint i=0; (i < values.size()); i++ ) {
//    double weight = sdpSoftmax(i);
//    vec val = values[i];
//    attnResult += weight*val;
//  }

  if ( verbose ) {
    cerr << "J attn result\n" << attnResult << endl;
  }

  vec attnOutput = jwom*attnResult + jbov;

  if ( verbose ) {
    cerr << "J attnOutput\n" << attnOutput << endl;
  }
  //vec hiddenInput = join_cols(attnOutput, corefVec);
  vec logScores = jwsm * relu(jwfm*attnOutput + jbfv) + jbsv;
  vec scores = exp(logScores);
  double outputNorm = accu(scores);
  vec result = scores/outputNorm;
  if ( verbose ) {
    cerr << "J result\n" << result << endl;
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
    cerr << "J ==== word " << i << " ====" << endl;
    currAttnInput = vec(attnInputDim);
    currAttnInput.fill(0.1 * (i+1));
    attnInput.emplace_back(currAttnInput);
    computeResult(attnInput, i, 1);
  }
}
