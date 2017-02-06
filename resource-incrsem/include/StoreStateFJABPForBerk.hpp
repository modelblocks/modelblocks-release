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

////////////////////////////////////////////////////////////////////////////////

char psLBrack[] = "[";
char psRBrack[] = "]";

////////////////////////////////////////////////////////////////////////////////

typedef Delimited<int>  D;  // depth
typedef Delimited<int>  F;  // fork decision
typedef Delimited<int>  J;  // join decision

////////////////////////////////////////////////////////////////////////////////

DiscreteDomain<int> domW;
class W : public Delimited<DiscreteDomainRV<int,domW>> {
 public:
  W ( )                : Delimited<DiscreteDomainRV<int,domW>> ( )    { }
  W ( int i )          : Delimited<DiscreteDomainRV<int,domW>> ( i )  { }
  W ( const char* ps ) : Delimited<DiscreteDomainRV<int,domW>> ( ps ) { }
};
typedef W ObsWord;

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

class PPredictor : public T {
 public:
  PPredictor ( )           : T ( )       { }
  PPredictor ( T tB ) : T ( tB ) { }
};

class WPredictor : public T { };

class FPredictor : public DelimitedTrip<psX,D,psSpace,T,psSpace,T,psX> {
 public:
  FPredictor ( )                 : DelimitedTrip<psX,D,psSpace,T,psSpace,T,psX> ( )           { }
  FPredictor ( D d, T tB, T tP ) : DelimitedTrip<psX,D,psSpace,T,psSpace,T,psX> ( d, tB, tP ) { }
};

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

  PPredictor calcPretrmTypeCondition ( ) const {
    return PPredictor( (size()>0) ? back().getB() : tTop );
  }

  FPredictor calcForkTypeCondition ( const Sign& aPretrm ) const {
    return FPredictor( size(), getAncstr(1), aPretrm );
  }

  JPredictor calcJoinTypeCondition ( F f, const Sign& aPretrm ) const {
    Sign aLchildTmp;
    return JPredictor( size()+f, getAncstr(f), getLchild(aLchildTmp,f,aPretrm) );
  }

  APredictor calcApexTypeCondition ( F f, J j, const Sign& aPretrm ) const {
    Sign aLchildTmp;
    return APredictor( size()+f-j, j, getAncstr(f), (j==0) ? getLchild(aLchildTmp,f,aPretrm) : tBot );
  }

  BPredictor calcBrinkTypeCondition ( F f, J j, T tParent, const Sign& aPretrm ) const {
    Sign aLchildTmp;
    return BPredictor( size()+f-j, j, tParent, getLchild(aLchildTmp,f,aPretrm) );
  }
};
Sign StoreState::qTop;

W unkWord (const char* ps ) {
  // the following code ripped from berkeley sophisticated lexicon
    string s = string("UNK");
    string word = std::string(ps); //.getString();
    size_t wlen = word.length();
    int numCaps = 0;
    bool hasDigit = false;
    bool hasDash = false;
    bool hasLower = false;
    for(size_t i = 0; i < wlen; i++) {
      char ch = word[i];
      if(isdigit(ch)) {
	hasDigit = true;
      } else if(ch == '-') {
	hasDash = true;
      } else if(isalpha(ch)) {
	if(islower(ch)) {
	  hasLower = true;
	} else {
	  numCaps++;
	}
      }
    }
    char ch = word[0];
    string lowered = std::string(ps); //.getString();
    for(size_t i = 0; i < wlen; i++) {
      lowered[i] = tolower(lowered[i]);
    }
    if(isupper(ch)) {
      if(numCaps == 1) {
	s += "-INITC";
	//X low = X(lowered.c_str());
  //	if(this->get(f).contains(low)) {
	//  s += "-KNOWNLC";
	//}
      } else {
	s += "-CAPS";
      }
    } else if(!isalpha(ch) && numCaps > 0) {
      s += "-CAPS";
    } else if(hasLower) {
      s += "-LC";
    }
    if(hasDigit) {
      s += "-NUM";
    }
    if(hasDash) {
      s += "-DASH";
    }
    if(lowered.rfind("s") == (wlen-1) && wlen >= 3) {
      char ch2 = lowered[wlen-2];
      if(ch2 != 's' && ch2 != 'i' && ch2 != 'u') {
	s += "-s";
      }
    } else if(wlen >= 5 && !hasDash && !(hasDigit && numCaps > 0)) {
      if(lowered.rfind("ed") == (wlen-2)) {
	s += "-ed";
      } else if(lowered.rfind("ing") == (wlen-3)) {
	s += "-ing";
      } else if(lowered.rfind("ion") == (wlen-3)) {
	s += "-ion";
      } else if(lowered.rfind("er") == (wlen-2)) {
	s += "-er";
      } else if(lowered.rfind("est") == (wlen-3)) {
	s += "-est";
      } else if(lowered.rfind("ly") == (wlen-2)) {
	s += "-ly";
      } else if(lowered.rfind("ity") == (wlen-3)) {
	s += "-ity";
      } else if(lowered.rfind("y") == (wlen-1)) {
	s += "-y";
      } else if(lowered.rfind("al") == (wlen-2)) {
	s += "-al";
      }
    }
    // cerr << "Converting " << word << " to " << s << endl;
    return W(s.c_str()); //.c_str();
  }
//  return ( 0==strcmp(ps+strlen(ps)-strlen("ing"), "ing") ) ? W("!unk!ing") :
//         ( 0==strcmp(ps+strlen(ps)-strlen("ed"),  "ed" ) ) ? W("!unk!ed") :
//         ( 0==strcmp(ps+strlen(ps)-strlen("s"),   "s"  ) ) ? W("!unk!s") :
//         ( 0==strcmp(ps+strlen(ps)-strlen("ion"), "ion") ) ? W("!unk!ion") :
//         ( 0==strcmp(ps+strlen(ps)-strlen("er"),  "er" ) ) ? W("!unk!er") :
//         ( 0==strcmp(ps+strlen(ps)-strlen("est"), "est") ) ? W("!unk!est") :
//         ( 0==strcmp(ps+strlen(ps)-strlen("ly"),  "ly" ) ) ? W("!unk!ly") : 
//         ( 0==strcmp(ps+strlen(ps)-strlen("ity"), "ity") ) ? W("!unk!ity") : 
//         ( 0==strcmp(ps+strlen(ps)-strlen("y"),   "y"  ) ) ? W("!unk!y") : 
//         ( 0==strcmp(ps+strlen(ps)-strlen("al"),  "al" ) ) ? W("!unk!al") :
//         ( ps[0]>='A' && ps[0]<='Z'                      ) ? W("!unk!cap") :
//         ( ps[0]>='0' && ps[0]<='9'                      ) ? W("!unk!num") :
//                                                             W("!unk!");
//}

