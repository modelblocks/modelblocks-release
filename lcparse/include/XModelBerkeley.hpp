///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// This file is part of ModelBlocks. Copyright 2009, ModelBlocks developers. //
//                                                                           //
//    ModelBlocks is free software: you can redistribute it and/or modify    //
//    it under the terms of the GNU General Public License as published by   //
//    the Free Software Foundation, either version 3 of the License, or      //
//    (at your option) any later version.                                    //
//                                                                           //
//    ModelBlocks is distributed in the hope that it will be useful,         //
//    but WITHOUT ANY WARRANTY; without even the implied warranty of         //
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          //
//    GNU General Public License for more details.                           //
//                                                                           //
//    You should have received a copy of the GNU General Public License      //
//    along with ModelBlocks.  If not, see <http://www.gnu.org/licenses/>.   //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include <Model.hpp>


////////////////////////////////////////////////////////////////////////////////

DiscreteDomain<int> domX;
class X : public DiscreteDomainRV<int,domX> {
 public:
  X ( )                : DiscreteDomainRV<int,domX> ( )    { }
  X ( const char* ps ) : DiscreteDomainRV<int,domX> ( ps ) { }
};
const X X_NIL("");


////////////////////////////////////////////////////////////////////////////////

template<class F>
class XModel : public UnorderedModel<F,X,LogProb> {

 private:

  SimpleMap<X,X> mU;

 public:

  // Constructor methods...
  XModel(const char* ps) : UnorderedModel<F,X,LogProb>(ps) { }
  LogProb getProb(const F& f, const X& x) const {
    if (!mU.contains(x)) const_cast<SimpleMap<X,X>&>(mU).set(x) = getSig(f,x);
    X unkx = mU.get(x);
    if(!get(f).contains(x)) {
      return get(f).get(unkx);
    } else {
      return get(f).get(x) + get(f).get(unkx);
    }
  }

 private:

  const char* getSig(F f, X x) const {
    // the following code ripped from berkeley sophisticated lexicon
    string s = string("UNK");
    string word = x.getString();
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
    string lowered = x.getString();
    for(size_t i = 0; i < wlen; i++) {
      lowered[i] = tolower(lowered[i]);
    }
    if(isupper(ch)) {
      if(numCaps == 1) {
	s += "-INITC";
	X low = X(lowered.c_str());
	if(get(f).contains(low)) {
	  s += "-KNOWNLC";
	}
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
    return s.c_str();
  }
};

