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
#include<sstream>

bool NODEP = false;

////////////////////////////////////////////////////////////////////////////////

char psLBrack[] = "[";
char psRBrack[] = "]";

////////////////////////////////////////////////////////////////////////////////

typedef Delimited<int>  D;  // depth
typedef Delimited<int>  F;  // fork decision
typedef Delimited<int>  J;  // join decision

////////////////////////////////////////////////////////////////////////////////

DiscreteDomain<int> domT;
class T : public Delimited<DiscreteDomainRV<int,domT>> {
 public:
  T ( )                : Delimited<DiscreteDomainRV<int,domT>> ( )    { }
  T ( int i )          : Delimited<DiscreteDomainRV<int,domT>> ( i )  { }
  T ( const char* ps ) : Delimited<DiscreteDomainRV<int,domT>> ( ps ) { }
};
T tTop("T");
T tBot("-");

////////////////////////////////////////////////////////////////////////////////

class FPredictor : public DelimitedPair<psX,D,psSpace,T,psX> {
 public:
  FPredictor ( )           : DelimitedPair<psX,D,psSpace,T,psX> ( )       { }
  FPredictor ( D d, T tP ) : DelimitedPair<psX,D,psSpace,T,psX> ( d, tP ) { }
};

class PPredictor : public DelimitedTrip<psX,D,psSpace,F,psSpace,T,psX> {
 public:
  PPredictor ( )                : DelimitedTrip<psX,D,psSpace,F,psSpace,T,psX> ( )          { }
  PPredictor ( D d, F f, T tB ) : DelimitedTrip<psX,D,psSpace,F,psSpace,T,psX> ( d, f, tB ) { }
};

class WPredictor : public T { };

class JPredictor : public DelimitedTrip<psX,D,psSpace,T,psSpace,T,psX> {
 public:
  JPredictor ( )                 : DelimitedTrip<psX,D,psSpace,T,psSpace,T,psX> ( )           { }
  JPredictor ( D d, T tB, T tP ) : DelimitedTrip<psX,D,psSpace,T,psSpace,T,psX> ( d, tB, tP ) { }
};

class APredictor : public DelimitedQuad<psX,D,psSpace,J,psSpace,T,psSpace,T,psX> {
 public:
  APredictor ( )                      : DelimitedQuad<psX,D,psSpace,J,psSpace,T,psSpace,T,psX> ( )              { }
  APredictor ( D d, J j, T tB, T tL ) : DelimitedQuad<psX,D,psSpace,J,psSpace,T,psSpace,T,psX> ( d, j, tB, tL ) { }
};

class BPredictor : public DelimitedQuad<psX,D,psSpace,J,psSpace,T,psSpace,T,psX> {
 public:
  BPredictor ( )                      : DelimitedQuad<psX,D,psSpace,J,psSpace,T,psSpace,T,psX> ( )              { }
  BPredictor ( D d, J j, T tP, T tL ) : DelimitedQuad<psX,D,psSpace,J,psSpace,T,psSpace,T,psX> ( d, j, tP, tL ) { }
};

////////////////////////////////////////////////////////////////////////////////

typedef T Sign;

////////////////////////////////////////////////////////////////////////////////

class IncompleteSign : public DelimitedPair<psX,Sign,psSlash,Sign,psX> {
 public:
  IncompleteSign ( )                                  : DelimitedPair<psX,Sign,psSlash,Sign,psX> ( )          { }
  IncompleteSign ( const Sign& sA1, const Sign& sB1 ) : DelimitedPair<psX,Sign,psSlash,Sign,psX> ( sA1, sB1 ) { }
  Sign&       setA ( )       { return first;  }
  Sign&       setB ( )       { return second; }
  const Sign& getA ( ) const { return first;  }
  const Sign& getB ( ) const { return second; }
};

////////////////////////////////////////////////////////////////////////////////

class StoreState : public DelimitedVector<psX,IncompleteSign,psSemi,psX> {
 public:
  static Sign qTop;
  StoreState ( ) : DelimitedVector<psX,IncompleteSign,psSemi,psX> ( ) { }
  StoreState ( const StoreState& qPrev, F f, J j, T tA, T tB, const Sign& aPretrm ) {

    // If end of sentence, don't create incomplete sign at discourse depth, just bail...
    if ( qPrev.size()+f-j <= 0 ) return;

    // Copy from previous and add new IS at end based on f,j...
    reserve( qPrev.size()+f-j );                           // Ensure new store state is correct size to avoid reallocation.
    insert( end(), qPrev.begin(), qPrev.end()-(1-f+j) );   // Copy unmodified portion of storestate sequence from previous time step based on fork and join.
    IncompleteSign& isNew = *emplace( end() );             // Add new incomplete sign at end.

    // Get the incomplete sign that is to be joined...
    const IncompleteSign& isToJoin = qPrev[qPrev.size()+f-j-1];

    isNew = IncompleteSign( (j==0) ? tA : isToJoin.getA(), tB );
  }

  const Sign& getAncstr ( F f ) const {
    return (int(size())-2+f>=0) ? operator[](size()-2+f).getB() :
                                  qTop;
  }

  const Sign& getLchild ( Sign& aLchildTmp, F f, const Sign& aPretrm ) const {
    return (f==1)      ? aPretrm :
           (size()==0) ? StoreState::qTop :   // NOTE: should not happen.
                         back().getA();
  }

  FPredictor calcForkTypeCondition ( ) const {
    return FPredictor( (NODEP) ? 0 : size(), getAncstr(1) );
  }

  PPredictor calcPretrmTypeCondition ( F f ) const {
    return PPredictor( (NODEP) ? 0 : size(), f, (size()>0) ? back().getB() : tTop );
  }

  JPredictor calcJoinTypeCondition ( F f, const Sign& aPretrm ) const {
    Sign aLchildTmp;
    return JPredictor( (NODEP) ? 0 : size()+f, getAncstr(f), getLchild(aLchildTmp,f,aPretrm) );
  }

  APredictor calcApexTypeCondition ( F f, J j, const Sign& aPretrm ) const {
    Sign aLchildTmp;
    return APredictor( (NODEP) ? 0 : size()+f-j, j, getAncstr(f), (j==0) ? getLchild(aLchildTmp,f,aPretrm) : tBot );
  }

  BPredictor calcBrinkTypeCondition ( F f, J j, T tParent, const Sign& aPretrm ) const {
    Sign aLchildTmp;
    return BPredictor( (NODEP) ? 0 : size()+f-j, j, tParent, getLchild(aLchildTmp,f,aPretrm) );
  }
};
Sign StoreState::qTop;

