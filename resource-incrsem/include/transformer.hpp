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

// This file contains functions used by the *Model_transformer.hpp classes.

// For debugging
void print_vec(vec vector, uint maxlen=10) {
  cerr << "Printing first " << maxlen << " items..." << endl;
  uint len = vector.size();
  for ( uint i=0; (i<maxlen) && (i<len); i++ ) {
    cerr << vector(i) << endl;
  }
  cerr << endl;
}

// ReLU function
arma::mat relu( const arma::mat& km ) {
  if ( km.max() < 0 ) return zeros( arma::size(km) );
  else return clamp(km, 0, km.max());
}


int getDepth( const BeamElement<HiddState>& be ) {
    return be.getHidd().getStoreState().getDepth();
}


CVar getCatBase( const BeamElement<HiddState>& be ) {
    return be.getHidd().getStoreState().getBase().getCat();
}


HVec getHvB( const BeamElement<HiddState>& be ) {
    StoreState ss = be.getHidd().getStoreState();
    return (( ss.getBase().getHVec().size() > 0 ) ? ss.getBase().getHVec() : hvBot);
}


HVec getHvF( const BeamElement<HiddState>& be ) {
    StoreState ss = be.getHidd().getStoreState();
    return (( ss.getBase().getCat().getNoloArity() && ss.getNoloBack().getHVec().size() != 0 ) ? ss.getNoloBack().getHVec() : hvBot);
}


// returns a sinusoidal positional encoding for an embedding of dimensionality 
// dim, at position wordIndex in sequence
vec getPositionalEncoding( uint dim, uint wordIndex ) {
  vec encoding = vec(dim);
  for ( uint i=0; i<dim; i++ ) {
    if ( i % 2 == 0 ) {
      encoding[i] = sin(wordIndex * exp(i * -log(10000) / dim));
    }
    else {
      encoding[i] = cos(wordIndex * exp((i-1) * -log(10000) / dim));
    }
  }
  return encoding;
}


// TODO this only works if numHeads is a power of 2. Can make it work for
// other values by copying this:
// https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py#L742
// Get slopes for each attention head for Alibi positional representations
// (see https://arxiv.org/pdf/2108.12409.pdf )
vec getAlibiSlopes( uint numHeads ) {
//  start = (2**(-2**-(math.log2(n)-3)))
//  ratio = start
//  return [start*ratio**i for i in range(n)]
  float curr = pow( 2, -pow(2, -(log2(numHeads)-3)) );
  float ratio = curr;
  vec slopes = vec(numHeads);
  for ( uint i=0; i<numHeads; i++ ) {
    slopes[i] = curr;
    curr = curr * ratio;
  }
  return slopes;
}


vector<mat> getAlibiMatrices( uint numHeads, uint seqLength ) {
  vec perHeadSlopes = getAlibiSlopes(numHeads);
  vector<mat> alibiMats; //= mat(numHeads, seqLength);
  for ( uint i=0; i<numHeads; i++ ) {
    float slope = perHeadSlopes[i];
    mat alibiMat(seqLength, seqLength);
    vec alibiVec(seqLength);
    for ( uint j=0; j<seqLength; j++ ) {
      alibiVec(j) = j*slope;
    }
    for ( uint j=0; j<seqLength; j++ ) {
      alibiMat.row(j) = alibiVec.t();
    }
    alibiMats.push_back(alibiMat);
  }
  return alibiMats;
}
