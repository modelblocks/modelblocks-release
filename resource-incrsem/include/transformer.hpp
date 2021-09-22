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


// returns a positional encoding for an embedding of dimensionality dim,
// at position wordIndex in sequence
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
