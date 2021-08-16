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

#ifndef _DELIMITED__
#define _DELIMITED__

#include <iostream>
#include <fstream>
#include <list>
using namespace std;
#include <nl-string.h>

//// the below is only for Delimited<Col>...
#include <armadillo>
using namespace arma;
//#include <vector>

////////////////////////////////////////////////////////////////////////////////

char psX[] = "";
char psSpace[] = " ";
char psComma[] = ",";
char psSemi[] = ";";
char psEquals[] = "=";
char psColon[] = ":";
char psSlash[] = "/";
char psPipe[] = "|";
char psLine[] = "\n";

////////////////////////////////////////////////////////////////////////////////

istream isDummy(NULL);

template<class T>
class Delimited : public T {
 public:
  Delimited<T> ( )                : T ( )    { }
  Delimited<T> ( int i )          : T ( i )  { }
  Delimited<T> ( const T& t )     : T ( t )  { }
  Delimited<T> ( const char* ps ) : T ( ps ) { }
};

template<>
class Delimited<double> {
 private:
  double val;
 public:
  Delimited<double> ( )                 : val()         { }
  Delimited<double> ( const double& t ) : val(t)        { }
  Delimited<double> ( const char* ps )  : val(stod(ps)) { }
  operator double() const { return val; }
};

template<>
class Delimited<int> {
 private:
  int val;
 public:
  Delimited<int> ( )                : val()         { }
  Delimited<int> ( const int& t )   : val(t)        { }
  Delimited<int> ( const char* ps ) : val(stoi(ps)) { }
  operator int() const { return val; }
  //bool operator< ( const Delimited<int>& i ) { return int(*this)<int(i); }
};

template<>
class Delimited<char> {
 private:
  int val;
 public:
  Delimited<char> ( )                : val()      { }
  Delimited<char> ( char c )         : val(c)     { }
//  Delimited<char> ( const int& t )   : val(t)  { }
  Delimited<char> ( const char* ps ) : val(ps[0]) { }
  operator char() const { return val; }
};

template<class T>
inline pair<istream&,Delimited<T>&> operator>> ( istream& is, Delimited<T>& f ) {
  return pair<istream&,Delimited<T>&>(is,f);
}

template<class T>
inline istream& operator>> ( pair<istream&,Delimited<T>&> isps, const char* psDelim ) {
  if ( &isDummy==&isps.first ) return isDummy;
  String sBuff(100);
  int i = 0;
  int j = 0;
  do {
    sBuff[i++] = isps.first.get();
    j = ( psDelim[j] == sBuff[i-1] ) ? j+1 : 0;
    if ( psDelim[j] == '\0' ) {
      sBuff[i-j] = '\0';
      isps.second = sBuff.c_array();
      return isps.first;
    }
  } while ( EOF != sBuff[i-1] );
  return isDummy;
}

template<class T>
inline bool operator>> ( pair<istream&,Delimited<T>&> isps, const vector<const char*>& vpsDelim ) {
  if ( &isDummy==&isps.first ) return false;
  String sBuff(100);
  int i = 0;
  vector<int> vj(vpsDelim.size(),0);
  do {
    sBuff[i++] = isps.first.get();
    for ( size_t k=0; k<vpsDelim.size(); k++ ) {
      vj[k] = ( vpsDelim[k][vj[k]] == sBuff[i-1] ) ? vj[k]+1 : 0;
      if ( vpsDelim[k][vj[k]] == '\0' ) {
        sBuff[i-vj[k]] = '\0';
        isps.second = sBuff.c_array();
        return k==0;
      }
    }
  } while ( EOF != sBuff[i-1] );
  return false;
}

inline istream& operator>> ( istream& is, const char* ps ) {
  for ( uint i=0; i<strlen(ps); i++ ) { char c = is.get(); if( c!= ps[i]) cerr<<"ERROR: expected delimiter '"<<ps[i]<<"' encountered '"<<c<<"'"<<endl; assert( c == ps[i] ); }
  return is;
}

inline bool operator>> ( istream& is, const vector<const char*> vpsDelim ) {
  for ( uint i=0; ; i++ ) {
    // NOTE: cannot contain nulls!
    char c = is.get();
    for ( uint j=0; j<vpsDelim.size(); j++ )  if ( i+1==strlen(vpsDelim[j]) && c==vpsDelim[j][i] )  return (j==0);
  }
}

////////////////////////////////////////////////////////////////////////////////
//
//  delimited containers
//
////////////////////////////////////////////////////////////////////////////////

template<const char* psD1,class T,const char* psD2,const char* psD3>
class DelimitedList : public list<T> {
 public:
  friend pair<istream&,DelimitedList<psD1,T,psD2,psD3>&> operator>> ( istream& is, DelimitedList<psD1,T,psD2,psD3>& t ) {
    return pair<istream&,DelimitedList<psD1,T,psD2,psD3>&>(is,t);
  }
  friend istream& operator>> ( pair<istream&,DelimitedList<psD1,T,psD2,psD3>&> ist, const char* psDelim ) {
    if (psX!=psD1) ist.first >> psD1;
    if (psX==psD3) {
      if ( ist.first.peek()==psDelim[0] ) ist.first >> psDelim; // NOTE: only works for unit-length delim!
      else {
        while ( ist.first >> *ist.second.emplace(ist.second.end()) >> vector<const char*>{psD2,psDelim} );
      }
    }
    else {
      if ( ist.first.peek()==psD3[0] ) ist.first >> psD3;
      else {
        while ( ist.first >> *ist.second.emplace(ist.second.end()) >> vector<const char*>{psD2,psD3} );
        ist.first >> psDelim;
      }
    }
    return ist.first;
  }
  friend bool operator>> ( pair<istream&,DelimitedList<psD1,T,psD2,psD3>&> ist, const vector<const char*>& vpsDelim ) {
    if (psX!=psD1) ist.first >> psD1;
//    if (psX==psD3) { cerr<<"ERROR: nested lists without final delimiters."<<endl; return false; }
    if (psX==psD3) {
      vector<const char*> vpsJointDelim( vpsDelim );  vpsJointDelim.push_back( psD2 );
      while ( ist.first >> *ist.second.emplace(ist.second.end()) >> vpsJointDelim );
    }
    while ( ist.first >> *ist.second.emplace(ist.second.end()) >> vector<const char*>{psD2,psD3} );
    return ist.first >> vpsDelim;
  }
  friend ostream& operator<< ( ostream& os, const DelimitedList<psD1,T,psD2,psD3>& vt ) {
    os<<psD1; for ( auto& t : vt ) os << t << ((&t==&vt.back())?"":psD2); os<<psD3;
    return os;
  }
};

////////////////////////////////////////////////////////////////////////////////

template<const char* psD1,class T,const char* psD2,const char* psD3>
class DelimitedVector : public vector<T> {
 public:
  DelimitedVector<psD1,T,psD2,psD3> ( )                                             : vector<T> ( )    { }
  DelimitedVector<psD1,T,psD2,psD3> ( int i )                                       : vector<T> ( i )  { }
  DelimitedVector<psD1,T,psD2,psD3> ( const DelimitedVector<psD1,T,psD2,psD3>& vt ) : vector<T> ( vt ) { }
  friend pair<istream&,DelimitedVector<psD1,T,psD2,psD3>&> operator>> ( istream& is, DelimitedVector<psD1,T,psD2,psD3>& t ) {
    return pair<istream&,DelimitedVector<psD1,T,psD2,psD3>&>(is,t);
  }
  friend istream& operator>> ( pair<istream&,DelimitedVector<psD1,T,psD2,psD3>&> ist, const char* psDelim ) {
    while ( ist.first >> *ist.second.emplace(ist.second.end()) >> vector<const char*>{psD2,psDelim} );
    return ist.first;
  }
  friend ostream& operator<< ( ostream& os, const DelimitedVector<psD1,T,psD2,psD3>& vt ) {
    os<<psD1; for ( auto& t : vt ) os << t << ((&t==&vt.back())?"":psD2); os<<psD3;
    return os;
  }
};

////////////////////////////////////////////////////////////////////////////////
//
//  Delimited column vector -- statically sized
//
////////////////////////////////////////////////////////////////////////////////

#ifdef STATIC_DELIM_VEC
template<const char* psD1,class T,const char* psD2,size_t iSize,const char* psD3>
class DelimitedCol : public Col<T> {
 public:
  DelimitedCol<psD1,T,psD2,iSize,psD3> ( ) : Col<T>(iSize,fill::zeros) { }
  DelimitedCol<psD1,T,psD2,iSize,psD3> ( const Col<T>& ct ) : Col<T>(ct) { }
  //operator Col<T>() { return *this; }
  bool operator<( const DelimitedCol<psD1,T,psD2,iSize,psD3>& c1 ) const {
    for( size_t i=0; i<iSize; i++ ) {
      if( this->at(i) < c1.at(i) ) return true;
      if( this->at(i) > c1.at(i) ) return false;
    }
    return false;
  }
  bool operator==(const DelimitedCol<psD1,T,psD2,iSize,psD3>& c1) const {
    for( size_t i=0; i<iSize; i++ ) if( this->at(i)!=c1.at(i) ) return false;
    return true;
//    return approx_equal(*this, c1, "absdiff", 0.002);
//    return Col<T>(*this)==Col<T>(c1);
//    return (*this).Col<T>::operator==(c1);
  }
  friend pair<istream&,DelimitedCol<psD1,T,psD2,iSize,psD3>&> operator>> ( istream& is, DelimitedCol<psD1,T,psD2,iSize,psD3>& t ) {
    return pair<istream&,DelimitedCol<psD1,T,psD2,iSize,psD3>&>(is,t);
  }
  friend istream& operator>> ( pair<istream&,DelimitedCol<psD1,T,psD2,iSize,psD3>&> ist, const char* psDelim ) {
    if ( psD1[0] != '\0' ) ist.first >> psD1;
    if ( psD3[0] == '\0' ) for ( size_t t=0; t<iSize && (ist.first >> reinterpret_cast<Delimited<T>&>(ist.second(t)) >> vector<const char*>{psD2,psDelim}); t++ );
    else for ( size_t t=0; t<iSize && (ist.first >> reinterpret_cast<Delimited<T>&>(ist.second(t)) >> vector<const char*>{psD2,psD3}); t++ );
    if ( psD3[0] != '\0' ) ist.first >> psDelim;
    return ist.first;
  }
  friend ostream& operator<< ( ostream& os, const DelimitedCol<psD1,T,psD2,iSize,psD3>& vt ) {
    os<<psD1; for ( size_t t=0; t<iSize; t++ ) os << vt(t) << ((t==iSize-1) ? "":psD2); os<<psD3;
    return os;
  }
};
#endif

////////////////////////////////////////////////////////////////////////////////
//
//  Delimited column vector -- dynamically sized
//
////////////////////////////////////////////////////////////////////////////////

template<const char* psD1,class T,const char* psD2,const char* psD3>
class DelimitedCol : public Col<T> {
 public:
  DelimitedCol<psD1,T,psD2,psD3> ( size_t iSize     ) : Col<T>(iSize,fill::zeros) { }
  DelimitedCol<psD1,T,psD2,psD3> ( const Col<T>& ct ) : Col<T>(ct) { }
  //operator Col<T>() { return *this; }
  bool operator<( const DelimitedCol<psD1,T,psD2,psD3>& c1 ) const {
    for( size_t i=0; i<c1.n_elem; i++ ) {
      if( this->at(i) < c1.at(i) ) return true;
      if( this->at(i) > c1.at(i) ) return false;
    }
    return false;
  }
  bool operator==(const DelimitedCol<psD1,T,psD2,psD3>& c1) const {
    for( size_t i=0; i<c1.n_elem; i++ ) if( this->at(i)!=c1.at(i) ) return false;
    return true;
//    return approx_equal(*this, c1, "absdiff", 0.002);
//    return Col<T>(*this)==Col<T>(c1);
//    return (*this).Col<T>::operator==(c1);
  }
  friend pair<istream&,DelimitedCol<psD1,T,psD2,psD3>&> operator>> ( istream& is, DelimitedCol<psD1,T,psD2,psD3>& t ) {
    return pair<istream&,DelimitedCol<psD1,T,psD2,psD3>&>(is,t);
  }
  friend istream& operator>> ( pair<istream&,DelimitedCol<psD1,T,psD2,psD3>&> ist, const char* psDelim ) {
    if ( psD1[0] != '\0' ) ist.first >> psD1;
    if ( psD3[0] == '\0' ) for ( size_t t=0; t<ist.second.n_elem && (ist.first >> reinterpret_cast<Delimited<T>&>(ist.second(t)) >> vector<const char*>{psD2,psDelim}); t++ );
    else for ( size_t t=0; t<ist.second.n_elem && (ist.first >> reinterpret_cast<Delimited<T>&>(ist.second(t)) >> vector<const char*>{psD2,psD3}); t++ );
    if ( psD3[0] != '\0' ) ist.first >> psDelim;
    return ist.first;
  }
  friend ostream& operator<< ( ostream& os, const DelimitedCol<psD1,T,psD2,psD3>& vt ) {
    os<<psD1; for ( size_t t=0; t<vt.n_elem; t++ ) os << vt(t) << ((t==vt.n_elem-1) ? "":psD2); os<<psD3;
    return os;
  }
};

////////////////////////////////////////////////////////////////////////////////
//
//  Delimited matrix -- statically sized
//
////////////////////////////////////////////////////////////////////////////////

#ifdef STATIC_DELIM_VEC
template<const char* psD1,class T,const char* psD2,size_t rSize,size_t cSize,const char* psD3>
class DelimitedMat : public Mat<T> {
public:
    DelimitedMat<psD1,T,psD2,rSize,cSize,psD3> ( ) : Mat<T>(rSize, cSize) { }
    DelimitedMat<psD1,T,psD2,rSize,cSize,psD3> ( const Mat<T>& ct ) : Mat<T>(ct) { }
    //operator Col<T>() { return *this; }
//    bool operator==(const DelimitedCol<psD1,T,psD2,iSize,psD3>& c1) const {
//      return approx_equal(*this, c1, "absdiff", 0.002);
//    return Col<T>(*this)==Col<T>(c1);
//    return (*this).Col<T>::operator==(c1);
//    }
    friend pair<istream&,DelimitedMat<psD1,T,psD2,rSize,cSize,psD3>&> operator>> ( istream& is, DelimitedMat<psD1,T,psD2,rSize,cSize,psD3>& t ) {
      return pair<istream&,DelimitedMat<psD1,T,psD2,rSize,cSize,psD3>&>(is,t);
    }
    friend istream& operator>> ( pair<istream&,DelimitedMat<psD1,T,psD2,rSize,cSize,psD3>&> ist, const char* psDelim ) {
      if ( psD1[0] != '\0' ) ist.first >> psD1;
      bool b = true;
      for ( size_t r=0; r<rSize; r++ )
        for ( size_t c=0; c<cSize && b; c++ ){
          if ( psD3[0] != '\0' ) b = ist.first >> reinterpret_cast<Delimited<T>&>(ist.second(r,c)) >> vector<const char*>{psD2,psD3};
          else b = ist.first >> reinterpret_cast<Delimited<T>&>(ist.second(r,c)) >> vector<const char*>{psD2,psDelim};
        }
      if ( psD3[0] != '\0' ) ist.first >> psD3;
      return ist.first;
    }
    friend ostream& operator<< ( ostream& os, const DelimitedMat<psD1,T,psD2,rSize,cSize,psD3>& vt ) {
      os<<psD1;
      for ( size_t r=0; r<rSize; r++ )
        for ( size_t c=0; c<cSize; c++ ) os << vt(r,c) << ((r==rSize-1 && c==cSize-1) ? "":psD2);
      os<<psD3;
      return os;
    }
};
#endif

////////////////////////////////////////////////////////////////////////////////
//
//  Delimited matrix -- dynamically sized
//
////////////////////////////////////////////////////////////////////////////////

template<const char* psD1,class T,const char* psD2,const char* psD3>
class DelimitedMat : public Mat<T> {
public:
    DelimitedMat<psD1,T,psD2,psD3> ( size_t rSize, size_t cSize ) : Mat<T>(rSize, cSize) { }
    DelimitedMat<psD1,T,psD2,psD3> ( const Mat<T>& ct )           : Mat<T>(ct)           { }
    //operator Col<T>() { return *this; }
//    bool operator==(const DelimitedCol<psD1,T,psD2,iSize,psD3>& c1) const {
//      return approx_equal(*this, c1, "absdiff", 0.002);
//    return Col<T>(*this)==Col<T>(c1);
//    return (*this).Col<T>::operator==(c1);
//    }
    friend pair<istream&,DelimitedMat<psD1,T,psD2,psD3>&> operator>> ( istream& is, DelimitedMat<psD1,T,psD2,psD3>& t ) {
      return pair<istream&,DelimitedMat<psD1,T,psD2,psD3>&>(is,t);
    }
    friend istream& operator>> ( pair<istream&,DelimitedMat<psD1,T,psD2,psD3>&> ist, const char* psDelim ) {
      if ( psD1[0] != '\0' ) ist.first >> psD1;
      bool b = true;
      for ( size_t r=0; r<ist.second.n_rows; r++ )
        for ( size_t c=0; c<ist.second.n_cols && b; c++ ){
          if ( psD3[0] != '\0' ) b = ist.first >> reinterpret_cast<Delimited<T>&>(ist.second(r,c)) >> vector<const char*>{psD2,psD3};
          else b = ist.first >> reinterpret_cast<Delimited<T>&>(ist.second(r,c)) >> vector<const char*>{psD2,psDelim};
        }
      if ( psD3[0] != '\0' ) ist.first >> psD3;
      return ist.first;
    }
    friend ostream& operator<< ( ostream& os, const DelimitedMat<psD1,T,psD2,psD3>& vt ) {
      os<<psD1;
      for ( size_t r=0; r<vt.n_rows; r++ )
        for ( size_t c=0; c<vt.n_cols; c++ ) os << vt(r,c) << ((r==vt.n_rows-1 && c==vt.n_cols-1) ? "":psD2);
      os<<psD3;
      return os;
    }
};

////////////////////////////////////////////////////////////////////////////////
//
//  tuples
//
////////////////////////////////////////////////////////////////////////////////

template< class T, class U, class V >
class trip : public pair<T,pair<U,V>> {
 public:
  trip( ) { }
  trip( const T& t, const U& u, const V& v ) { this->first()=t; this->second()=u; this->third()=v; }
  T&       first  ( )       { return pair<T,pair<U,V>>::first;         }
  U&       second ( )       { return pair<T,pair<U,V>>::second.first;  }
  V&       third  ( )       { return pair<T,pair<U,V>>::second.second; }
  const T& first  ( ) const { return pair<T,pair<U,V>>::first;         }
  const U& second ( ) const { return pair<T,pair<U,V>>::second.first;  }
  const V& third  ( ) const { return pair<T,pair<U,V>>::second.second; }
};

template< class T, class U, class V, class W >
class quad : public trip<T,U,pair<V,W>> {
 public:
  quad( ) { }
  quad( const T& t, const U& u, const V& v, const W& w ) { this->first()=t; this->second()=u; this->third()=v; this->fourth()=w; }
  V&       third  ( )       { return trip<T,U,pair<V,W>>::third().first;  }
  W&       fourth ( )       { return trip<T,U,pair<V,W>>::third().second; }
  const V& third  ( ) const { return trip<T,U,pair<V,W>>::third().first;  }
  const W& fourth ( ) const { return trip<T,U,pair<V,W>>::third().second; }
};

template< class T, class U, class V, class W, class X >
class quint : public quad<T,U,V,pair<W,X>> {
 public:
  quint( ) { }
  quint( const T& t, const U& u, const V& v, const W& w, const X& x ) { this->first()=t; this->second()=u; this->third()=v; this->fourth()=w; this->fifth()=x; }
  W&       fourth ( )       { return quad<T,U,V,pair<W,X>>::fourth().first;  }
  X&       fifth  ( )       { return quad<T,U,V,pair<W,X>>::fourth().second; }
  const W& fourth ( ) const { return quad<T,U,V,pair<W,X>>::fourth().first;  }
  const X& fifth  ( ) const { return quad<T,U,V,pair<W,X>>::fourth().second; }
};

template< class T, class U, class V, class W, class X, class Y >
class sext : public quint<T,U,V,W,pair<X,Y>> {
 public:
  sext( ) { }
  sext( const T& t, const U& u, const V& v, const W& w, const X& x, const Y& y ) { this->first()=t; this->second()=u; this->third()=v; this->fourth()=w; this->fifth()=x; this->sixth()=y; }
  X&       fifth ( )       { return quint<T,U,V,W,pair<X,Y>>::fifth().first;  }
  Y&       sixth ( )       { return quint<T,U,V,W,pair<X,Y>>::fifth().second; }
  const X& fifth ( ) const { return quint<T,U,V,W,pair<X,Y>>::fifth().first;  }
  const Y& sixth ( ) const { return quint<T,U,V,W,pair<X,Y>>::fifth().second; }
};

template< class T, class U, class V, class W, class X, class Y, class Z >
class sept : public sext<T,U,V,W,X,pair<Y,Z>> {
 public:
  sept( ) { }
  sept( const T& t, const U& u, const V& v, const W& w, const X& x, const Y& y, const Z& z ) { this->first()=t; this->second()=u; this->third()=v; this->fourth()=w; this->fifth()=x; this->sixth()=y; this->seventh()=z; }
  Y&       sixth   ( )       { return sext<T,U,V,W,X,pair<Y,Z>>::sixth().first;  }
  Z&       seventh ( )       { return sext<T,U,V,W,X,pair<Y,Z>>::sixth().second; }
  const Y& sixth   ( ) const { return sext<T,U,V,W,X,pair<Y,Z>>::sixth().first;  }
  const Z& seventh ( ) const { return sext<T,U,V,W,X,pair<Y,Z>>::sixth().second; }
};

template< class S, class T, class U, class V, class W, class X, class Y, class Z >
class okt : public sept<S,T,U,V,W,X,pair<Y,Z>> {
 public:
  okt( ) { }
  okt( const S& s, const T& t, const U& u, const V& v, const W& w, const X& x, const Y& y, const Z& z ) { this->first()=s; this->second()=t; this->third()=u; this->fourth()=v; this->fifth()=w; this->sixth()=x; this->seventh()=y; this->eighth()=z; }
  Y&       seventh ( )       { return sept<S,T,U,V,W,X,pair<Y,Z>>::seventh().first;  }
  Z&       eighth  ( )       { return sept<S,T,U,V,W,X,pair<Y,Z>>::seventh().second; }
  const Y& seventh ( ) const { return sept<S,T,U,V,W,X,pair<Y,Z>>::seventh().first;  }
  const Z& eighth  ( ) const { return sept<S,T,U,V,W,X,pair<Y,Z>>::seventh().second; }
};

template< class R, class S, class T, class U, class V, class W, class X, class Y, class Z >
class non : public okt<R,S,T,U,V,W,X,pair<Y,Z>> {
 public:
  non( ) { }
  non( const R& r, const S& s, const T& t, const U& u, const V& v, const W& w, const X& x, const Y& y, const Z& z ) { this->first()=r; this->second()=s; this->third()=t; this->fourth()=u; this->fifth()=v; this->sixth()=w; this->seventh()=x; this->eighth()=y; this->ninth()=z; }
  Y&       eighth  ( )       { return okt<R,S,T,U,V,W,X,pair<Y,Z>>::eighth().first;  }
  Z&       ninth   ( )       { return okt<R,S,T,U,V,W,X,pair<Y,Z>>::eighth().second; }
  const Y& eighth  ( ) const { return okt<R,S,T,U,V,W,X,pair<Y,Z>>::eighth().first;  }
  const Z& ninth   ( ) const { return okt<R,S,T,U,V,W,X,pair<Y,Z>>::eighth().second; }
};

////////////////////////////////////////////////////////////////////////////////
//
//  delimited tuples
//
////////////////////////////////////////////////////////////////////////////////

template<const char* psD1,class T,const char* psD2,class U,const char* psD3>
class DelimitedPair : public pair<T,U> {
 public:
  DelimitedPair ( )                        : pair<T,U>()    { }
  DelimitedPair ( const T& t, const U& u ) : pair<T,U>(t,u) { }
  friend pair<istream&,DelimitedPair<psD1,T,psD2,U,psD3>&> operator>> ( istream& is, DelimitedPair<psD1,T,psD2,U,psD3>& t ) {
    return pair<istream&,DelimitedPair<psD1,T,psD2,U,psD3>&>(is,t);
  }
  friend istream& operator>> ( pair<istream&,DelimitedPair<psD1,T,psD2,U,psD3>&> ist, const char* psDelim ) {
    if (psD3==psX) return ist.first >> psD1 >> ist.second.first >> psD2 >> ist.second.second >> psDelim;
    else           return ist.first >> psD1 >> ist.second.first >> psD2 >> ist.second.second >> psD3 >> psDelim;
  }
  friend bool operator>> ( pair<istream&,DelimitedPair<psD1,T,psD2,U,psD3>&> ist, const vector<const char*>& vpsDelim ) {
    if (psD3==psX) return ist.first >> psD1 >> ist.second.first >> psD2 >> ist.second.second >> vpsDelim;
    else           return ist.first >> psD1 >> ist.second.first >> psD2 >> ist.second.second >> psD3 >> vpsDelim;
  }
  friend ostream& operator<< ( ostream& os, const DelimitedPair<psD1,T,psD2,U,psD3>& t ) {
    return os << psD1 << t.first << psD2 << t.second << psD3;
  }
};

////////////////////////////////////////////////////////////////////////////////

template< const char* psD1,class T,const char* psD2,class U,const char* psD3,class V, const char* psD4 >
class DelimitedTrip : public trip<T,U,V> {
  typedef DelimitedTrip<psD1,T,psD2,U,psD3,V,psD4> TUPLE;
 public:
  DelimitedTrip ( )                                    : trip<T,U,V>()      { }
  DelimitedTrip ( const T& t, const U& u, const V& v ) : trip<T,U,V>(t,u,v) { }
  friend pair<istream&,TUPLE&> operator>> ( istream& is, TUPLE& t ) {
    return pair<istream&,TUPLE&>(is,t);
  }
  friend istream& operator>> ( pair<istream&, TUPLE&> ist, const char* psDelim ) {
    return ist.first >> ist.second.first() >> psD2 >> ist.second.second() >> psD3 >> ist.second.third() >> psDelim;
  }
  friend bool operator>> ( pair<istream&,TUPLE&> ist, const vector<const char*>& vpsDelim ) {
    return ist.first >> ist.second.first() >> psD2 >> ist.second.second() >> psD3 >> ist.second.third() >> vpsDelim;
  }
  friend ostream& operator<< ( ostream& os, const TUPLE& t ) {
    return os << psD1 << t.first() << psD2 << t.second() << psD3 << t.third() << psD4;
  }
};

////////////////////////////////////////////////////////////////////////////////

template< const char* psD1,class T,const char* psD2,class U,const char* psD3,class V,const char* psD4,class W,const char* psD5 >
class DelimitedQuad : public quad<T,U,V,W> {
  typedef DelimitedQuad<psD1,T,psD2,U,psD3,V,psD4,W,psD5> TUPLE;
 public:
  DelimitedQuad ( )                                                : quad<T,U,V,W>()        { }
  DelimitedQuad ( const T& t, const U& u, const V& v, const W& w ) : quad<T,U,V,W>(t,u,v,w) { }
  friend pair<istream&,TUPLE&> operator>> ( istream& is, TUPLE& t ) {
    return pair<istream&,TUPLE&>(is,t);
  }
  friend istream& operator>> ( pair<istream&, TUPLE&> ist, const char* psDelim ) {
    return ist.first >> ist.second.first() >> psD2 >> ist.second.second() >> psD3 >> ist.second.third() >> psD4 >> ist.second.fourth() >> psDelim;
  }
  friend bool operator>> ( pair<istream&,TUPLE&> ist, const vector<const char*>& vpsDelim ) {
    return ist.first >> ist.second.first() >> psD2 >> ist.second.second() >> psD3 >> ist.second.third() >> psD4 >> ist.second.fourth() >> vpsDelim;
  }
  friend ostream& operator<< ( ostream& os, const TUPLE& t ) {
    return os << psD1 << t.first() << psD2 << t.second() << psD3 << t.third() << psD4 << t.fourth() << psD5;
  }
};

////////////////////////////////////////////////////////////////////////////////

template< const char* psD1,class T,const char* psD2,class U,const char* psD3,class V,const char* psD4,class W,const char* psD5,class X,const char* psD6 >
class DelimitedQuint : public quint<T,U,V,W,X> {
  typedef DelimitedQuint<psD1,T,psD2,U,psD3,V,psD4,W,psD5,X,psD6> TUPLE;
 public:
  DelimitedQuint ( )                                                            : quint<T,U,V,W,X>()          { }
  DelimitedQuint ( const T& t, const U& u, const V& v, const W& w, const X& x ) : quint<T,U,V,W,X>(t,u,v,w,x) { }
  friend pair<istream&,TUPLE&> operator>> ( istream& is, TUPLE& t ) {
    return pair<istream&,TUPLE&>(is,t);
  }
  friend istream& operator>> ( pair<istream&, TUPLE&> ist, const char* psDelim ) {
    return ist.first >> ist.second.first() >> psD2 >> ist.second.second() >> psD3 >> ist.second.third() >> psD4 >> ist.second.fourth() >> psD5 >> ist.second.fifth() >> psDelim;
  }
  friend bool operator>> ( pair<istream&,TUPLE&> ist, const vector<const char*>& vpsDelim ) {
    return ist.first >> ist.second.first() >> psD2 >> ist.second.second() >> psD3 >> ist.second.third() >> psD4 >> ist.second.fourth() >> psD5 >> ist.second.fifth() >> vpsDelim;
  }
  friend ostream& operator<< ( ostream& os, const TUPLE& t ) {
    return os << psD1 << t.first() << psD2 << t.second() << psD3 << t.third() << psD4 << t.fourth() << psD5 << t.fifth() << psD6;
  }
};

////////////////////////////////////////////////////////////////////////////////

template< const char* psD1,class T,const char* psD2,class U,const char* psD3,class V,const char* psD4,class W,const char* psD5,class X,const char* psD6,class Y,const char* psD7 >
class DelimitedSext : public sext<T,U,V,W,X,Y> {
  typedef DelimitedSext<psD1,T,psD2,U,psD3,V,psD4,W,psD5,X,psD6,Y,psD7> TUPLE;
 public:
  DelimitedSext ( )                                                                        : sext<T,U,V,W,X,Y>()            { }
  DelimitedSext ( const T& t, const U& u, const V& v, const W& w, const X& x, const Y& y ) : sext<T,U,V,W,X,Y>(t,u,v,w,x,y) { }
  friend pair<istream&,TUPLE&> operator>> ( istream& is, TUPLE& t ) {
    return pair<istream&,TUPLE&>(is,t);
  }
  friend istream& operator>> ( pair<istream&, TUPLE&> ist, const char* psDelim ) {
    return ist.first >> ist.second.first() >> psD2 >> ist.second.second() >> psD3 >> ist.second.third() >> psD4 >> ist.second.fourth() >> psD5 >> ist.second.fifth() >> psD6 >> ist.second.sixth() >> psDelim;
  }
  friend bool operator>> ( pair<istream&,TUPLE&> ist, const vector<const char*>& vpsDelim ) {
    return ist.first >> ist.second.first() >> psD2 >> ist.second.second() >> psD3 >> ist.second.third() >> psD4 >> ist.second.fourth() >> psD5 >> ist.second.fifth() >> psD6 >> ist.second.sixth() >> vpsDelim;
  }
  friend ostream& operator<< ( ostream& os, const TUPLE& t ) {
    return os << psD1 << t.first() << psD2 << t.second() << psD3 << t.third() << psD4 << t.fourth() << psD5 << t.fifth() << psD6 << t.sixth() << psD7;
  }
};

////////////////////////////////////////////////////////////////////////////////

template< const char* psD1,class T,const char* psD2,class U,const char* psD3,class V,const char* psD4,class W,const char* psD5,class X,const char* psD6,class Y,const char* psD7,class Z,const char* psD8 >
class DelimitedSept : public sept<T,U,V,W,X,Y,Z> {
  typedef DelimitedSept<psD1,T,psD2,U,psD3,V,psD4,W,psD5,X,psD6,Y,psD7,Z,psD8> TUPLE;
 public:
  DelimitedSept ( )                                                                                    : sept<T,U,V,W,X,Y,Z>()              { }
  DelimitedSept ( const T& t, const U& u, const V& v, const W& w, const X& x, const Y& y, const Z& z ) : sept<T,U,V,W,X,Y,Z>(t,u,v,w,x,y,z) { }
  friend pair<istream&,TUPLE&> operator>> ( istream& is, TUPLE& t ) {
    return pair<istream&,TUPLE&>(is,t);
  }
  friend istream& operator>> ( pair<istream&, TUPLE&> ist, const char* psDelim ) {
    return ist.first >> ist.second.first() >> psD2 >> ist.second.second() >> psD3 >> ist.second.third() >> psD4 >> ist.second.fourth() >> psD5 >> ist.second.fifth() >> psD6 >> ist.second.sixth() >> psD7 >> ist.second.seventh() >> psDelim;
  }
  friend bool operator>> ( pair<istream&,TUPLE&> ist, const vector<const char*>& vpsDelim ) {
    return ist.first >> ist.second.first() >> psD2 >> ist.second.second() >> psD3 >> ist.second.third() >> psD4 >> ist.second.fourth() >> psD5 >> ist.second.fifth() >> psD6 >> ist.second.sixth() >> psD7 >> ist.second.seventh() >> vpsDelim;
  }
  friend ostream& operator<< ( ostream& os, const TUPLE& t ) {
    return os << psD1 << t.first() << psD2 << t.second() << psD3 << t.third() << psD4 << t.fourth() << psD5 << t.fifth() << psD6 << t.sixth() << psD7 << t.seventh() << psD8;
  }
};

////////////////////////////////////////////////////////////////////////////////

template< const char* psD1,class S,const char* psD2,class T,const char* psD3,class U,const char* psD4,class V,const char* psD5,class W,const char* psD6,class X,const char* psD7,class Y,const char* psD8,class Z,const char* psD9 >
class DelimitedOkt : public okt<S,T,U,V,W,X,Y,Z> {
  typedef DelimitedOkt<psD1,S,psD2,T,psD3,U,psD4,V,psD5,W,psD6,X,psD7,Y,psD8,Z,psD9> TUPLE;
 public:
  DelimitedOkt ( )                                                                                                : okt<S,T,U,V,W,X,Y,Z>()                { }
  DelimitedOkt ( const S& s, const T& t, const U& u, const V& v, const W& w, const X& x, const Y& y, const Z& z ) : okt<S,T,U,V,W,X,Y,Z>(s,t,u,v,w,x,y,z) { }
  friend pair<istream&,TUPLE&> operator>> ( istream& is, TUPLE& t ) {
    return pair<istream&,TUPLE&>(is,t);
  }
  friend istream& operator>> ( pair<istream&, TUPLE&> ist, const char* psDelim ) {
    return ist.first >> ist.second.first() >> psD2 >> ist.second.second() >> psD3 >> ist.second.third() >> psD4 >> ist.second.fourth() >> psD5 >> ist.second.fifth() >> psD6 >> ist.second.sixth() >> psD7 >> ist.second.seventh() >> psD8 >> ist.second.eighth() >> psDelim;
  }
  friend bool operator>> ( pair<istream&,TUPLE&> ist, const vector<const char*>& vpsDelim ) {
    return ist.first >> ist.second.first() >> psD2 >> ist.second.second() >> psD3 >> ist.second.third() >> psD4 >> ist.second.fourth() >> psD5 >> ist.second.fifth() >> psD6 >> ist.second.sixth() >> psD7 >> ist.second.seventh() >> psD8 >> ist.second.eighth() >> vpsDelim;
  }
  friend ostream& operator<< ( ostream& os, const TUPLE& t ) {
    return os << psD1 << t.first() << psD2 << t.second() << psD3 << t.third() << psD4 << t.fourth() << psD5 << t.fifth() << psD6 << t.sixth() << psD7 << t.seventh() << psD8 << t.eighth() << psD9;
  }
};

////////////////////////////////////////////////////////////////////////////////

template< const char* psD1,class R,const char* psD2,class S,const char* psD3,class T,const char* psD4,class U,const char* psD5,class V,const char* psD6,class W,const char* psD7,class X,const char* psD8,class Y,const char* psD9,class Z,const char* psD10 >
class DelimitedNon : public non<R,S,T,U,V,W,X,Y,Z> {
  typedef DelimitedNon<psD1,R,psD2,S,psD3,T,psD4,U,psD5,V,psD6,W,psD7,X,psD8,Y,psD9,Z,psD10> TUPLE;
 public:
  DelimitedNon ( )                                                                                                : non<R,S,T,U,V,W,X,Y,Z>()                { }
  DelimitedNon ( const R& r, const S& s, const T& t, const U& u, const V& v, const W& w, const X& x, const Y& y, const Z& z ) : non<R,S,T,U,V,W,X,Y,Z>(r,s,t,u,v,w,x,y,z) { }
  friend pair<istream&,TUPLE&> operator>> ( istream& is, TUPLE& t ) {
    return pair<istream&,TUPLE&>(is,t);
  }
  friend istream& operator>> ( pair<istream&, TUPLE&> ist, const char* psDelim ) {
    return ist.first >> ist.second.first() >> psD2 >> ist.second.second() >> psD3 >> ist.second.third() >> psD4 >> ist.second.fourth() >> psD5 >> ist.second.fifth() >> psD6 >> ist.second.sixth() >> psD7 >> ist.second.seventh() >> psD8 >> ist.second.eighth() >> psD9 >> ist.second.ninth() >> psDelim;
  }
  friend bool operator>> ( pair<istream&,TUPLE&> ist, const vector<const char*>& vpsDelim ) {
    return ist.first >> ist.second.first() >> psD2 >> ist.second.second() >> psD3 >> ist.second.third() >> psD4 >> ist.second.fourth() >> psD5 >> ist.second.fifth() >> psD6 >> ist.second.sixth() >> psD7 >> ist.second.seventh() >> psD8 >> ist.second.eighth() >> psD9 >> ist.second.ninth() >> vpsDelim;
  }
  friend ostream& operator<< ( ostream& os, const TUPLE& t ) {
    return os << psD1 << t.first() << psD2 << t.second() << psD3 << t.third() << psD4 << t.fourth() << psD5 << t.fifth() << psD6 << t.sixth() << psD7 << t.seventh() << psD8 << t.eighth() << psD9 << t.ninth() << psD10;
  }
};

#endif //_DELIMTED__
