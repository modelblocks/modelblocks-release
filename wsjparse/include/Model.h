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

#ifndef _MODEL__
#define _MODEL__

//#include "nl-hash.h"
//#include "nl-array.h"

template<class C,class V,class P>
class OrderedModel : public SimpleHash<C,Array<pair<V,P> > > {
//class OrderedModel : public SimpleMap<C,Array<pair<V,P> > > {   // much slower!
 private:
  String sLbl;
 public:
  // Constructor methods...
  OrderedModel(const char* ps) : sLbl(ps) { }
  // Extractor methods...
  const Array<pair<V,P> >& getDistrib ( const C& c ) const { return get(c); }
  const String&            getLbl     ( )            const { return sLbl;   }
  // I/O methods...
  friend pair<StringInput,OrderedModel<C,V,P>*> operator>> ( const StringInput ps, OrderedModel<C,V,P>& m ) { return pair<StringInput,OrderedModel<C,V,P>*>(ps,&m); }
  friend StringInput operator>>( pair<StringInput,OrderedModel<C,V,P>*> si_m, const char* psD) {
    StringInput          si =  si_m.first;
    OrderedModel<C,V,P>& m  = *si_m.second;
    if (si == StringInput()) return si;
    C c;  V v;  P pr;
    si = si >> m.sLbl.c_array() >> " " >> c >> " : " >> v >> " = " >> pr >> "\0";
    if ( si != StringInput() ) m.set(c).add() = pair<V,P>(v,pr);
    return si;
  }
};

template<class C,class V,class P>
class UnorderedModel : public SimpleHash<C,SimpleHash<V,P> > {
//class UnorderedModel : public SimpleMap<C,SimpleMap<V,P> > {   // much slower!
 private:
  String sLbl;
 public:
  // Constructor methods...
  UnorderedModel(const char* ps) : sLbl(ps) { }
  // Extractor methods...
  const SimpleHash<V,P>& getDistrib ( const C& c ) const { return get(c); }
  //const SimpleMap<V,P>&  getDistrib ( const C& c ) const { return get(c); }   // much slower!
  const String&          getLbl     ( )            const { return sLbl;   }
  // I/O methods...
  friend pair<StringInput,UnorderedModel<C,V,P>*> operator>> ( const StringInput ps, UnorderedModel<C,V,P>& m ) { return pair<StringInput,UnorderedModel<C,V,P>*>(ps,&m); }
  friend StringInput operator>>( pair<StringInput,UnorderedModel<C,V,P>*> si_m, const char* psD) {
    StringInput            si =  si_m.first;
    UnorderedModel<C,V,P>& m  = *si_m.second;
    if (si == StringInput()) return si;
    C c;  V v;  P pr;
    si = si >> m.sLbl.c_array() >> " " >> c >> " : " >> v >> " = " >> pr >> "\0";
    if ( si != StringInput() ) m.set(c).set(v) = pr;
    return si;
  }
};

#endif //_MODEL__
