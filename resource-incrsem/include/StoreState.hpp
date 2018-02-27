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
typedef Delimited<char> O;  // composition operation
typedef Delimited<char> E;  // extraction operation
typedef Delimited<char> S;  // side (A,B)
const S S_A("/");
const S S_B(";");

////////////////////////////////////////////////////////////////////////////////

#ifndef W__
#define W__
DiscreteDomain<int> domW;
class W : public Delimited<DiscreteDomainRV<int,domW>> {
 public:
  W ( )                : Delimited<DiscreteDomainRV<int,domW>> ( )    { }
  W ( int i )          : Delimited<DiscreteDomainRV<int,domW>> ( i )  { }
  W ( const char* ps ) : Delimited<DiscreteDomainRV<int,domW>> ( ps ) { }
};
typedef W ObsWord;
#endif

////////////////////////////////////////////////////////////////////////////////

class T;
typedef T N;

////////////////////////////////////////////////////////////////////////////////

DiscreteDomain<int> domT;
class T : public DiscreteDomainRV<int,domT> {
 private:
  static map<N,bool>         mnbArg;
  static map<T,int>          mtiArity;
  static map<T,bool>         mtbIsCarry;
  static map<T,N>            mtnLastNol;
  static map<pair<T,N>,bool> mtnbIn;
  static map<T,T>            mttLets;
  static map<T,int>          mtiNums;
  static map<pair<T,int>,T>  mtitLetNum;
  int getArity ( const char* l ) {
    int depth = 0;
    int ctr   = 0;
    for ( uint i=0; i<strlen(l); i++ ) {
      if ( l[i]=='{' ) depth++;
      if ( l[i]=='}' ) depth--;
      if ( l[i]=='-' && l[i+1]>='a' && l[i+1]<='d' && depth==0 ) ctr++;
    }
    return ('N'==l[0]) ? ctr+1 : ctr;
  }
  N getLastNolo ( const char* l ) {
    int depth = 0;
    uint beg = strlen(l);
    uint end = strlen(l);
    for ( uint i=0; i<strlen(l); i++ ) {
      if ( l[i]=='{' ) depth++;
      if ( l[i]=='}' ) depth--;
      if ( l[i]=='-' && (l[i+1]=='g' || l[i+1]=='h'   // || l[i+1]=='i'
                                                         || l[i+1]=='r' || l[i+1]=='v') && depth==0 ) beg = i;
      if ( beg<i && end>i && depth==0 && (l[i]=='-' || l[i]=='_' || l[i]=='\\' || l[i]=='^') ) end = i;
    }
    return N( string(l,beg,end-beg).c_str() );  // l+strlen(l);
  }
  void calcDetermModels ( const char* ps ) {
    if( mnbArg.end()==mnbArg.find(*this) ) { mnbArg[*this]=( strlen(ps)<=4 ); }
    if( mtiArity.end()  ==mtiArity.  find(*this) ) { mtiArity  [*this]=getArity(ps); }
    if( mtbIsCarry.end()==mtbIsCarry.find(*this) ) { mtbIsCarry[*this]=( ps[0]=='-' && ps[1]>='a' && ps[1]<='z' ); }  //( ps[strlen(ps)-1]=='^' ); }
    if( strlen(ps)>0 && !(ps[0]=='-'&&ps[1]>='a'&&ps[1]<='z') && mtnLastNol.end()==mtnLastNol.find(*this) ) { N& n=mtnLastNol[*this]; n=getLastNolo(ps); }
    if( mttLets.end()==mttLets.find(*this) ) { const char* ps_=strchr(ps,'_');
                                               if( ps_!=NULL ) { mttLets[*this] = string(ps,0,ps_-ps).c_str(); mtiNums[*this] = atoi(ps_+1);
                                                                 mtitLetNum[pair<T,int>(mttLets[*this],mtiNums[*this])]=*this; } }
                                               //else { mttLets[*this]=*this; mtiNums[*this]=0; mtitLetNum[pair<T,int>(*this,0)]=*this; } }
    uint depth = 0;  uint beg = strlen(ps);
    for( uint i=0; i<strlen(ps); i++ ) {
      if ( ps[i]=='{' ) depth++;
      if ( ps[i]=='}' ) depth--;
      if ( depth==0 && ps[i]=='-' && (ps[i+1]=='g' || ps[i+1]=='h'    // || ps[i+1]=='i'
                                                                         || ps[i+1]=='r' || ps[i+1]=='v') ) beg = i;
      if ( depth==0 && beg<i && (ps[i+1]=='-' || ps[i+1]=='_' || ps[i+1]=='\\' || ps[i+1]=='^' || ps[i+1]=='\0') ) { mtnbIn[pair<T,N>(*this,string(ps,beg,i).c_str())]=true;  beg = strlen(ps); }
    }
  }
 public:
  T ( )                : DiscreteDomainRV<int,domT> ( )    { }
  T ( const char* ps ) : DiscreteDomainRV<int,domT> ( ps ) { calcDetermModels(ps); }
  bool isArg           ( )       const { return mnbArg[*this]; }
  int  getArity        ( )       const { return mtiArity  [*this]; }
  bool isCarrier       ( )       const { return mtbIsCarry[*this]; }
  N    getLastNonlocal ( )       const { return mtnLastNol[*this]; }
  bool containsCarrier ( N n )   const { return mtnbIn.find(pair<T,N>(*this,n))!=mtnbIn.end(); }
  T    getLets         ( )       const { const auto& x = mttLets.find(*this); return (x==mttLets.end()) ? *this : x->second; }
  int  getNums         ( )       const { const auto& x = mtiNums.find(*this); return (x==mtiNums.end()) ? 0 : x->second; }
  T    addNum          ( int i ) const { const auto& x = mtitLetNum.find(pair<T,int>(*this,i)); return (x==mtitLetNum.end()) ? *this : x->second; }
};
map<N,bool>         T::mnbArg;
map<T,int>          T::mtiArity;
map<T,bool>         T::mtbIsCarry;
map<T,N>            T::mtnLastNol;
map<pair<T,N>,bool> T::mtnbIn;
map<T,T>            T::mttLets;
map<T,int>          T::mtiNums;
map<pair<T,int>,T>  T::mtitLetNum;
const T tTop("T");
const T tBot("-");
const T tBOT("bot");  // not sure if this really needs to be distinct from tBot
const N N_NONE("");

////////////////////////////////////////////////////////////////////////////////

DiscreteDomain<int> domK;
class K : public DiscreteDomainRV<int,domK> {   // NOTE: can't be subclass of Delimited<...> or string-argument constructor of this class won't get called!
 public:
  static const K kTop;
  static const K kBot;
  private:
  static map<K,T> mkt;
  static map<pair<K,int>,K> mkik;
  void calcDetermModels ( const char* ps ) {
    if( strchr(ps,':')!=NULL ) {
      char cSelf = ('N'==ps[0]) ? '1' : '0';
      // Add associations to label and related K's...  (NOTE: related K's need two-step constructor so as to avoid infinite recursion!)
      if( mkt.end()==mkt.find(*this) ) mkt[*this]=T(string(ps,strchr(ps,':')).c_str());
      if( mkik.end()==mkik.find(pair<K,int>(*this,-3)) && strlen(ps)>2 && ps[strlen(ps)-2]!='-' ) { K& k=mkik[pair<K,int>(*this,-3)]; k=string(ps).append("-3").c_str(); }
      if( mkik.end()==mkik.find(pair<K,int>(*this,-2)) && strlen(ps)>2 && ps[strlen(ps)-2]!='-' ) { K& k=mkik[pair<K,int>(*this,-2)]; k=string(ps).append("-2").c_str(); }
      if( mkik.end()==mkik.find(pair<K,int>(*this,-1)) && strlen(ps)>2 && ps[strlen(ps)-2]!='-' ) { K& k=mkik[pair<K,int>(*this,-1)]; k=string(ps).append("-1").c_str(); }
      if( mkik.end()==mkik.find(pair<K,int>(*this,0))                            ) { K& k=mkik[pair<K,int>(*this,0)]; k=ps; }
      if( mkik.end()==mkik.find(pair<K,int>(*this,1)) && ps[strlen(ps)-2]!='-' && ps[strlen(ps)-1]==cSelf ) { K& k=mkik[pair<K,int>(*this,1)]; k=string(ps,strlen(ps)-1).append("1").c_str(); }
      if( mkik.end()==mkik.find(pair<K,int>(*this,2)) && ps[strlen(ps)-2]!='-' && ps[strlen(ps)-1]==cSelf ) { K& k=mkik[pair<K,int>(*this,2)]; k=string(ps,strlen(ps)-1).append("2").c_str(); }
      if( mkik.end()==mkik.find(pair<K,int>(*this,3)) && ps[strlen(ps)-2]!='-' && ps[strlen(ps)-1]==cSelf ) { K& k=mkik[pair<K,int>(*this,3)]; k=string(ps,strlen(ps)-1).append("3").c_str(); }
      if( mkik.end()==mkik.find(pair<K,int>(*this,1)) && ps[strlen(ps)-2]=='-' && ps[strlen(ps)-1]=='1' ) { K& k=mkik[pair<K,int>(*this,1)]; k=string(ps,strlen(ps)-2).c_str(); }
    }
    else mkt[*this] = (*this==kBot) ? tBOT : (*this==kTop) ? tTop : tBot;
  }
 public:
  K ( )                : DiscreteDomainRV<int,domK> ( )    { }
  K ( const char* ps ) : DiscreteDomainRV<int,domK> ( ps ) { calcDetermModels(ps); }
  T getType ( )       const { auto it = mkt.find(*this); return (it==mkt.end()) ? tBot : it->second; }
  K project ( int n ) const { auto it = mkik.find(pair<K,int>(*this,n)); return (it==mkik.end()) ? kBot : it->second; }
};
map<K,T> K::mkt;
map<pair<K,int>,K> K::mkik;
const K K_DITTO("\"");
const K kNil("");
const K K::kTop("Top");
const K K::kBot("Bot");

////////////////////////////////////////////////////////////////////////////////

class FPredictor {
 private:

  // Data members...
  uint id;

  // Static data members...
  static uint                  nextid;
  static map<uint,D>           mid;
  static map<uint,T>           mit;
  static map<uint,K>           mikF;
  static map<uint,K>           mikA;
  static map<pair<D,T>,uint>   mdti;
  static map<trip<D,K,K>,uint> mdkki;
  static map<pair<K,K>,uint>   mkki;

 public:

  // Construtors...
  FPredictor ( ) : id(0) { }
  FPredictor ( D d, T t ) {
    const auto& it = mdti.find(pair<D,T>(d,t));
    if ( it != mdti.end() ) id = it->second;
    else { id = nextid++;  mid[id] = d;  mit[id] = t;  mdti[pair<D,T>(d,t)] = id; }
    //cout<<"did id "<<id<<"/"<<nextid<<" as "<<*this<<endl;
  }
  FPredictor ( D d, K kF, K kA ) {
    const auto& it = mdkki.find(trip<D,K,K>(d,kF,kA));
    if ( it != mdkki.end() ) id = it->second;
    else { id = nextid++;  mid[id] = d;  mikF[id] = kF;  mikA[id] = kA;  mdkki[trip<D,K,K>(d,kF,kA)] = id; }
    //cout<<"did id "<<id<<"/"<<nextid<<" as "<<*this<<endl;
  }
  FPredictor ( K kF, K kA ) {
    const auto& it = mkki.find(pair<K,K>(kF,kA));
    if ( it != mkki.end() ) id = it->second;
    else { id = nextid++;  mikF[id] = kF;  mikA[id] = kA;  mkki[pair<K,K>(kF,kA)] = id; }
    //cout<<"did id "<<id<<"/"<<nextid<<" as "<<*this<<endl;
  }

  // Accessor methods...
  uint toInt() const { return id; }
  operator uint() const { return id; }
  D getDepth()    const { return mid[id]; }
  T getT()        const { return mit[id]; }
  K getFillerK()  const { return mikF[id]; }
  K getAncstrK()  const { return mikA[id]; }
  static uint getDomainSize() { return nextid; }

  // Ordering operator...
  bool operator< ( FPredictor fp ) { return id<fp.id; }

  // Input / output methods...
  friend pair<istream&,FPredictor&> operator>> ( istream& is, FPredictor& t ) {
    return pair<istream&,FPredictor&>(is,t);
  }
  friend istream& operator>> ( pair<istream&,FPredictor&> ist, const char* psDelim ) {
    if ( ist.first.peek()==psDelim[0] ) { auto& o = ist.first >> psDelim;  ist.second = FPredictor();  return o; }
    if ( ist.first.peek()=='d' ) {
      D d;  ist.first >> "d" >> d >> "&";
      if ( ist.first.peek()=='t' ) { Delimited<T> t;       auto& o = ist.first >> "t" >> t        >> psDelim;  ist.second = FPredictor(d,t);      return o; }
      else                         { Delimited<K> kF, kA;  auto& o = ist.first >> kF >> "&" >> kA >> psDelim;  ist.second = FPredictor(d,kF,kA);  return o; }
    } else { 
                                     Delimited<K> kF, kA;  auto& o = ist.first >> kF >> "&" >> kA >> psDelim;  ist.second = FPredictor(kF,kA);    return o;
    }
  }
  friend bool operator>> ( pair<istream&,FPredictor&> ist, const vector<const char*>& vpsDelim ) {
    D d;  ist.first >> "d" >> d >> "&"; 
    if ( ist.first.peek()=='d' ) { 
      if ( ist.first.peek()=='t' ) { Delimited<T> t;       auto o = ist.first >> "t" >> t        >> vpsDelim;  ist.second = FPredictor(d,t);      return o; }
      else                         { Delimited<K> kF, kA;  auto o = ist.first >> kF >> "&" >> kA >> vpsDelim;  ist.second = FPredictor(d,kF,kA);  return o; }
    } else { 
                                     Delimited<K> kF, kA;  auto o = ist.first >> kF >> "&" >> kA >> vpsDelim;  ist.second = FPredictor(kF,kA);    return o; 
    }
  }
  friend ostream& operator<< ( ostream& os, const FPredictor& t ) {
    if      ( mit.end()  != mit.find(t.id)  ) return os << "d" << mid[t.id] << "&" << "t" << mit[t.id];
    else if ( mid.end()  != mid.find(t.id)  ) return os << "d" << mid[t.id] << "&" << mikF[t.id] << "&" << mikA[t.id];
    else if ( mikA.end() != mikA.find(t.id) ) return os << mikF[t.id] << "&" << mikA[t.id];
    else                                      return os << "NON_STRING_ID_" << t.id;
  }
  static bool exists ( D d, T t )        { return( mdti.end()!=mdti.find(pair<D,T>(d,t)) ); }
  static bool exists ( D d, K kF, K kA ) { return( mdkki.end()!=mdkki.find(trip<D,K,K>(d,kF,kA)) ); }
  static bool exists ( K kF, K kA )      { return( mkki.end()!=mkki.find(pair<K,K>(kF,kA)) ); }
  FPredictor  addNum ( int i ) const     { return( FPredictor( mid[id], mit[id].addNum(i) ) ); }
};
uint                  FPredictor::nextid = 1;   // space for bias "" predictor
map<uint,D>           FPredictor::mid;
map<uint,T>           FPredictor::mit;
map<uint,K>           FPredictor::mikF;
map<uint,K>           FPredictor::mikA;
map<pair<D,T>,uint>   FPredictor::mdti;
map<trip<D,K,K>,uint> FPredictor::mdkki;
map<pair<K,K>,uint>   FPredictor::mkki;

////////////////////////////////////////////////////////////////////////////////

DiscreteDomain<int> domFResponse;
class FResponse : public DiscreteDomainRV<int,domFResponse> {
 private:
  static map<FResponse,F>           mfrf;
  static map<FResponse,E>           mfre;
  static map<FResponse,K>           mfrk;
  static map<trip<F,E,K>,FResponse> mfekfr;
  void calcDetermModels ( const char* ps ) {
    F f = ps[1]-'0';
    E e = ps[3];
    K k = ps+5;
    if( mfrf.end()==mfrf.find(*this) ) mfrf[*this]=f;
    if( mfre.end()==mfre.find(*this) ) mfre[*this]=e;
    if( mfrk.end()==mfrk.find(*this) ) mfrk[*this]=k;
    if( mfekfr.end()==mfekfr.find(trip<F,E,K>(f,e,k)) ) mfekfr[trip<F,E,K>(f,e,k)]=*this;
  }
 public:
  FResponse ( )                : DiscreteDomainRV<int,domFResponse> ( )    { }
  FResponse ( const char* ps ) : DiscreteDomainRV<int,domFResponse> ( ps ) { calcDetermModels(ps); }
  FResponse ( F f, E e, K k )  : DiscreteDomainRV<int,domFResponse> ( )    {
    *this = ( mfekfr.end()==mfekfr.find(trip<F,E,K>(f,e,k)) ) ? ("f" + to_string(f) + "&" + string(1,e) + "&" + k.getString()).c_str()
                                                              : mfekfr[trip<F,E,K>(f,e,k)];
  }
  static bool exists ( F f, E e, K k ) { return( mfekfr.end()!=mfekfr.find(trip<F,E,K>(f,e,k)) ); }

  F getFork ( ) const { return mfrf[*this]; }
  E getE    ( ) const { return mfre[*this]; }
  K getK    ( ) const { return mfrk[*this]; }
};
map<FResponse,F>           FResponse::mfrf;
map<FResponse,E>           FResponse::mfre;
map<FResponse,K>           FResponse::mfrk;
map<trip<F,E,K>,FResponse> FResponse::mfekfr;

////////////////////////////////////////////////////////////////////////////////

class JPredictor {
 private:

  // Data members...
  uint id;

  // Static data members...
  static uint                    nextid;
  static map<uint,D>             mid;
  static map<uint,T>             mitA;
  static map<uint,T>             mitL;
  static map<uint,K>             mikF;
  static map<uint,K>             mikA;
  static map<uint,K>             mikL;
  static map<trip<D,T,T>,uint>   mdtti;
  static map<quad<D,K,K,K>,uint> mdkkki;

 public:

  // Construtors...
  JPredictor ( ) : id(0) { }
  JPredictor ( D d, T tA, T tL ) {
    const auto& it = mdtti.find(trip<D,T,T>(d,tA,tL));
    if ( it != mdtti.end() ) id = it->second;
    else { id = nextid++;  mid[id] = d;  mitA[id] = tA;  mitL[id] = tL;  mdtti[trip<D,T,T>(d,tA,tL)] = id; }
  }
  JPredictor ( D d, K kF, K kA, K kL ) {
    const auto& it = mdkkki.find(quad<D,K,K,K>(d,kF,kA,kL));
    if ( it != mdkkki.end() ) id = it->second;
    else { id = nextid++;  mid[id] = d;  mikF[id] = kF;  mikA[id] = kA;  mikL[id] = kL;  mdkkki[quad<D,K,K,K>(d,kF,kA,kL)] = id; }
  }

  // Accessor methods...
  uint toInt() const { return id; }
  operator uint() const { return id; }
  D getDepth()   const { return mid[id]; }
  T getAncstrT() const { return mitA[id]; }
  T getLchildT() const { return mitL[id]; }
  K getFillerK() const { return mikF[id]; }
  K getAncstrK() const { return mikA[id]; }
  K getLchildK() const { return mikL[id]; }
  static uint getDomainSize() { return nextid; }

  // Ordering operator...
  bool operator< ( JPredictor jp ) { return id<jp.id; }

  // Input / output methods...
  friend pair<istream&,JPredictor&> operator>> ( istream& is, JPredictor& t ) {
    return pair<istream&,JPredictor&>(is,t);
  }
  friend istream& operator>> ( pair<istream&,JPredictor&> ist, const char* psDelim ) {
    if ( ist.first.peek()==psDelim[0] ) { auto& o =  ist.first >> psDelim;  ist.second = JPredictor();  return o; }
    D d;  ist.first >> "d" >> d >> "&";
    if ( ist.first.peek()=='t' ) { Delimited<T>     tA, tL;  auto& o = ist.first >> "t"       >> tA >> "&" >> "t" >> tL >> psDelim;  ist.second = JPredictor(d,tA,tL);     return o; }
    else                         { Delimited<K> kF, kA, kL;  auto& o = ist.first >> kF >> "&" >> kA >> "&" >>        kL >> psDelim;  ist.second = JPredictor(d,kF,kA,kL);  return o; }
  }
  friend bool operator>> ( pair<istream&,JPredictor&> ist, const vector<const char*>& vpsDelim ) {
    D d;  ist.first >> "d" >> d >> "&";
    if ( ist.first.peek()=='t' ) { Delimited<T>     tA, tL;  auto o = ist.first >> "t"       >> tA >> "&" >> "t" >> tL >> vpsDelim;  ist.second = JPredictor(d,tA,tL);     return o; }
    else                         { Delimited<K> kF, kA, kL;  auto o = ist.first >> kF >> "&" >> kA >> "&"        >> kL >> vpsDelim;  ist.second = JPredictor(d,kF,kA,kL);  return o; }
  }
  friend ostream& operator<< ( ostream& os, const JPredictor& t ) {
    if      ( mitA.end() != mitA.find(t.id) ) return os << "d" << mid[t.id] << "&" << "t"               << mitA[t.id] << "&" << "t" << mitL[t.id];
    else if ( mikA.end() != mikA.find(t.id) ) return os << "d" << mid[t.id] << "&" << mikF[t.id] << "&" << mikA[t.id] << "&"        << mikL[t.id];
    else                                      return os << "NON_STRING_ID_" << t.id;
  }
  static bool exists ( D d, T tA, T tL )       { return( mdtti.end()!=mdtti.find(trip<D,T,T>(d,tA,tL)) ); }
  static bool exists ( D d, K kF, K kA, K kL ) { return( mdkkki.end()!=mdkkki.find(quad<D,K,K,K>(d,kF,kA,kL)) ); }
  JPredictor addNums ( int i, int j ) const    { return( JPredictor( mid[id], mitA[id].addNum(i), mitL[id].addNum(j) ) ); }
};
uint                    JPredictor::nextid = 1;  // space for bias "" predictor
map<uint,D>             JPredictor::mid;
map<uint,T>             JPredictor::mitA;
map<uint,T>             JPredictor::mitL;
map<uint,K>             JPredictor::mikF;
map<uint,K>             JPredictor::mikA;
map<uint,K>             JPredictor::mikL;
map<trip<D,T,T>,uint>   JPredictor::mdtti;
map<quad<D,K,K,K>,uint> JPredictor::mdkkki;

////////////////////////////////////////////////////////////////////////////////

DiscreteDomain<int> domJResponse;
class JResponse : public DiscreteDomainRV<int,domJResponse> {
 private:
  static map<JResponse,J> mjrj;
  static map<JResponse,E> mjre;
  static map<JResponse,O> mjroL;
  static map<JResponse,O> mjroR;
  static map<quad<J,E,O,O>,JResponse> mjeoojr;
  void calcDetermModels ( const char* ps ) {
    if( mjrj.end() ==mjrj.find(*this)  ) mjrj[*this]=ps[1]-'0';
    if( mjre.end() ==mjre.find(*this)  ) mjre[*this]=ps[3];
    if( mjroL.end()==mjroL.find(*this) ) mjroL[*this]=ps[5];
    if( mjroR.end()==mjroR.find(*this) ) mjroR[*this]=ps[7];
    if( mjeoojr.end()==mjeoojr.find(quad<J,E,O,O>(ps[1]-'0',ps[3],ps[5],ps[7])) ) mjeoojr[quad<J,E,O,O>(ps[1]-'0',ps[3],ps[5],ps[7])]=*this;
  }
 public:
  JResponse ( )                      : DiscreteDomainRV<int,domJResponse> ( )    { }
  JResponse ( const char* ps )       : DiscreteDomainRV<int,domJResponse> ( ps ) { calcDetermModels(ps); }
  JResponse ( J j, E e, O oL, O oR ) : DiscreteDomainRV<int,domJResponse> ( )    {
    *this = ( mjeoojr.end()==mjeoojr.find(quad<J,E,O,O>(j,e,oL,oR)) ) ? ("j" + to_string(j) + "&" + string(1,e) + "&" + string(1,oL) + "&" + string(1,oR)).c_str()
                                                                      : mjeoojr[quad<J,E,O,O>(j,e,oL,oR)];
  }
  J getJoin ( ) const { return mjrj[*this]; }
  E getE    ( ) const { return mjre[*this]; }
  O getLOp  ( ) const { return mjroL[*this]; }
  O getROp  ( ) const { return mjroR[*this]; }
  static bool exists ( J j, E e, O oL, O oR ) { return( mjeoojr.end()!=mjeoojr.find(quad<J,E,O,O>(j,e,oL,oR)) ); }
};
map<JResponse,J>             JResponse::mjrj;
map<JResponse,E>             JResponse::mjre;
map<JResponse,O>             JResponse::mjroL;
map<JResponse,O>             JResponse::mjroR;
map<quad<J,E,O,O>,JResponse> JResponse::mjeoojr;

////////////////////////////////////////////////////////////////////////////////

typedef Delimited<int> D;

class PPredictor : public DelimitedQuint<psX,D,psSpace,F,psSpace,E,psSpace,Delimited<T>,psSpace,Delimited<T>,psX> {
 public:
  PPredictor ( )                           : DelimitedQuint<psX,D,psSpace,F,psSpace,E,psSpace,Delimited<T>,psSpace,Delimited<T>,psX> ( )                 { }
  PPredictor ( D d, F f, E e, T tB, T tK ) : DelimitedQuint<psX,D,psSpace,F,psSpace,E,psSpace,Delimited<T>,psSpace,Delimited<T>,psX> ( d, f, e, tB, tK ) { }

};

class WPredictor : public DelimitedPair<psX,Delimited<K>,psSpace,Delimited<T>,psX> { };

class APredictor : public DelimitedSept<psX,D,psSpace,F,psSpace,J,psSpace,E,psSpace,O,psSpace,Delimited<T>,psSpace,Delimited<T>,psX> {
 public:
  APredictor ( )                                      : DelimitedSept<psX,D,psSpace,F,psSpace,J,psSpace,E,psSpace,O,psSpace,Delimited<T>,psSpace,Delimited<T>,psX> ( ) { }
  APredictor ( D d, F f, J j, E e, O oL, T tB, T tL ) : DelimitedSept<psX,D,psSpace,F,psSpace,J,psSpace,E,psSpace,O,psSpace,Delimited<T>,psSpace,Delimited<T>,psX> ( d, f, j, e, oL, tB, tL ) { }
};

class BPredictor : public DelimitedOct<psX,D,psSpace,F,psSpace,J,psSpace,E,psSpace,O,psSpace,O,psSpace,Delimited<T>,psSpace,Delimited<T>,psX> {
 public:
  BPredictor ( )                                            : DelimitedOct<psX,D,psSpace,F,psSpace,J,psSpace,E,psSpace,O,psSpace,O,psSpace,Delimited<T>,psSpace,Delimited<T>,psX> ( )                         { }
  BPredictor ( D d, F f, J j, E e, O oL, O oR, T tP, T tL ) : DelimitedOct<psX,D,psSpace,F,psSpace,J,psSpace,E,psSpace,O,psSpace,O,psSpace,Delimited<T>,psSpace,Delimited<T>,psX> ( d, f, j, e, oL, oR, tP, tL ) { }
};

////////////////////////////////////////////////////////////////////////////////

class KSet : public DelimitedVector<psLBrack,Delimited<K>,psComma,psRBrack> {
 public:
  static const KSet ksDummy;
  KSet ( )                                                      : DelimitedVector<psLBrack,Delimited<K>,psComma,psRBrack> ( ) { }
  KSet ( const K& k )                                           : DelimitedVector<psLBrack,Delimited<K>,psComma,psRBrack> ( ) { emplace_back(k); }
  KSet ( const KSet& ks1, const KSet& ks2 )                     : DelimitedVector<psLBrack,Delimited<K>,psComma,psRBrack> ( ) {
    reserve( ks1.size() + ks2.size() );
    insert( end(), ks1.begin(), ks1.end() );
    insert( end(), ks2.begin(), ks2.end() );
  }
  KSet ( const KSet& ks, int iProj, const KSet& ks2 = ksDummy ) : DelimitedVector<psLBrack,Delimited<K>,psComma,psRBrack> ( ) {
    reserve( ks.size() + ks2.size() );
    for( const K& k : ks ) if( k.project(iProj)!=K::kBot ) push_back( k.project(iProj) );
    insert( end(), ks2.begin(), ks2.end() );
  }
  bool isDitto ( ) const { return ( size()>0 && front()==K_DITTO ); }
};
const KSet KSet::ksDummy;
const KSet ksTop = KSet( K::kTop );
const KSet ksBot = KSet( K::kBot );

////////////////////////////////////////////////////////////////////////////////

class Sign : public DelimitedTrip<psX,KSet,psColon,T,psX,S,psX> {
 public:
  Sign ( )                           : DelimitedTrip<psX,KSet,psColon,T,psX,S,psX> ( )           { }
  Sign ( const KSet& ks1, T t, S s ) : DelimitedTrip<psX,KSet,psColon,T,psX,S,psX> ( ks1, t, s ) { }
  Sign ( const KSet& ks1, const KSet& ks2, T t, S s ) {
    first().reserve( ks1.size() + ks2.size() );
    first().insert( first().end(), ks1.begin(), ks1.end() );
    first().insert( first().end(), ks2.begin(), ks2.end() );
    second() = t;
    third()  = s;
  }
  KSet&       setKSet ( )       { return first();  }
  T&          setType ( )       { return second(); }
  S&          setSide ( )       { return third();  }
  const KSet& getKSet ( ) const { return first();  }
  T           getType ( ) const { return second(); }
  S           getSide ( ) const { return third();  }
  bool        isDitto ( ) const { return getKSet().isDitto(); }
};

////////////////////////////////////////////////////////////////////////////////

class StoreState;

////////////////////////////////////////////////////////////////////////////////

class LeftChildSign : public Sign {
 public:
  LeftChildSign ( const Sign& a ) : Sign(a) { }
  LeftChildSign ( const StoreState& qPrev, F f, E eF, const Sign& aPretrm );
};

////////////////////////////////////////////////////////////////////////////////

class StoreState : public DelimitedVector<psX,Sign,psX,psX> {  // NOTE: format can't be read in bc of internal psX delimiter, but we don't need to.
 public:

  static const Sign aTop;
  static const Sign aBot;

  StoreState ( ) : DelimitedVector<psX,Sign,psX,psX> ( ) { }
  StoreState ( const StoreState& qPrev, F f, J j, E eF, E eJ, O opL, O opR, T tA, T tB, const Sign& aPretrm, const LeftChildSign& aLchild ) {

    //// A. FIND STORE LANDMARKS AND EXISTING P,A,B CARRIERS...

    int iAncestorA = qPrev.getAncestorAIndex(f);
    int iAncestorB = qPrev.getAncestorBIndex(f);
    int iLowerA    = (f==1) ? qPrev.size() : qPrev.getAncestorAIndex(1);

    // Find existing nonlocal carriers...
    N nP = aPretrm.getType().getLastNonlocal();  N nA = tA.getLastNonlocal();  N nB = tB.getLastNonlocal();  N nL = aLchild.getType().getLastNonlocal();
    int iCarrierP = -1;                          int iCarrierA = -1;           int iCarrierB = -1;           int iCarrierL = -1;
    // Find preterm nonlocal carrier, traversing up from ancestorB through carriers or noncarriers containing preterminal nonlocal, until carrier found...
    if( nP!=N_NONE ) for( int i=iAncestorB-1; i>=0 && (qPrev[i].getType().isCarrier() || qPrev[i].getType().containsCarrier(nP)); i-- ) if( qPrev[i].getType()==nP ) iCarrierP=i;
    // Find apex nonlocal carrier, traversing up from ancestorB through carriers or noncarriers containing apex nonlocal, until carrier found...
    if( nA!=N_NONE ) for( int i=iLowerA-1;   i>=0 && (qPrev[i].getType().isCarrier() || qPrev[i].getType().containsCarrier(nA)); i-- ) if( qPrev[i].getType()==nA ) iCarrierA=i;
    // Find brink nonlocal carrier, traversing up from ancestorB through carriers or noncarriers containing brink nonlocal, until carrier found...
    if( nB!=N_NONE ) for( int i=iAncestorB-1; i>=0 && (qPrev[i].getType().isCarrier() || qPrev[i].getType().containsCarrier(nB)); i-- ) if( qPrev[i].getType()==nB ) iCarrierB=i;
    // Find lchild nonlocal carrier, traversing up from ancestorB through carriers or noncarriers containing lchild nonlocal, until carrier found...
    if( nL!=N_NONE ) for( int i=iLowerA-1;   i>=0 && (qPrev[i].getType().isCarrier() || qPrev[i].getType().containsCarrier(nL)); i-- ) if( qPrev[i].getType()==nL ) iCarrierL=i;

    // Reserve store big enough for ancestorB + new A and B if no join + any needed carriers...
    reserve( iAncestorB + 1 + ((j==0) ? 2 : 0) + ((nP!=N_NONE && iCarrierP!=-1) ? 1 : 0) + ((nA!=N_NONE && iCarrierA==-1) ? 1 : 0) + ((nB!=N_NONE && iCarrierB==-1) ? 1 : 0) ); 

    //// B. FILL IN NEW PARTS OF NEW STORE...

    const KSet& ksLchild = aLchild.getKSet();
    const KSet  ksParent = (iCarrierA!=-1 && eJ!='N') ? KSet( aLchild.getKSet(), -getDir(opL), KSet( qPrev[iCarrierA].getKSet(), -getDir(eJ), (j==0) ? KSet() : qPrev.at(iAncestorB).getKSet() ) )
                         :                              KSet( aLchild.getKSet(), -getDir(opL),                                                (j==0) ? KSet() : qPrev.at(iAncestorB).getKSet()   );
    const KSet  ksRchild( ksParent, getDir(opR) );

    for( int i=0; i<((f==0&&j==1)?iAncestorB:(f==0&&j==0)?iLowerA:(f==1&&j==1)?iAncestorB:iAncestorB+1); i++ )
      *emplace( end() ) = ( i==iAncestorA && j==1 && qPrev[i].isDitto() && opR!='I' ) ? Sign( ksParent, qPrev[i].getType(), qPrev[i].getSide() )                                      // End of ditto.
                        : ( i==iCarrierP && eF!='N' )                                 ? Sign( KSet(ksLchild,getDir(eF),qPrev[i].getKSet()), qPrev[i].getType(), qPrev[i].getSide() )  // Update to P carrier.
                        : ( i==iCarrierA && eJ!='N' )                                 ? Sign( KSet(ksParent,getDir(eJ),qPrev[i].getKSet()), qPrev[i].getType(), qPrev[i].getSide() )  // Update to A carrier. 
                        :                                                               qPrev[i];                                                                                     // Copy store element.

    if( j==0 && nP!=N_NONE && iCarrierP==-1 )          if( STORESTATE_CHATTY ) cout<<"(adding carrierP for "<<nP<<" bc none above "<<iAncestorB<<")"<<endl;
    if( j==0 && nP!=N_NONE && iCarrierP==-1 )          *emplace( end() ) = Sign( KSet(aPretrm.getKSet(),getDir(eF)), nP, S_B );    // If no join and nonloc P with no existing carrier, add P carrier.
    if( j==0 && nA!=N_NONE && iCarrierA==-1 )          if( STORESTATE_CHATTY ) cout<<"(adding carrierA for "<<nA<<" bc none above "<<iLowerA<<")"<<endl;
    if( j==0 && nA!=N_NONE && iCarrierA==-1 )          *emplace( end() ) = Sign( KSet(ksParent,         getDir(eJ)), nA, S_B );    // If no join and nonloc A with no existing carrier, add A carrier.
    if( j==0 )                                         *emplace( end() ) = Sign( (opR=='I') ? KSet(K_DITTO) : ksParent, tA, S_A ); // If no join, add A sign.
    if( nB!=N_NONE && nB!=nA && iCarrierB==-1 )        if( STORESTATE_CHATTY ) cout<<"(adding carrierB for "<<nB<<" bc none above "<<iAncestorB<<")"<<endl;
    if( nB!=N_NONE && nB!=nA && iCarrierB==-1 )        *emplace( end() ) = Sign( ksLchild, nB, S_A );                              // Add left child kset as A carrier (G rule).
    // WS: SUPPOSED TO BE FOR C-rN EXTRAPOSITION, BUT DOESN'T QUITE WORK...
    // if( nL!=N_NONE && iCarrierL>iAncestorB )    if( STORESTATE_CHATTY ) cout<<"(adding carrierL for "<<nL<<" bc none above "<<iLowerA<<" and below "<<iAncestorB<<")"<<endl;
    // if( nL!=N_NONE && iCarrierL>iAncestorB )    *emplace( end() ) = Sign( qPrev[iCarrierL].getKSet(), nL, S_A );            // Add right child kset as L carrier (H rule).
    if( nL!=N_NONE && nL!=nA && iCarrierL>iAncestorB ) if( STORESTATE_CHATTY ) cout<<"(attaching carrierL for "<<nL<<" above "<<iLowerA<<" and below "<<iAncestorB<<")"<<endl;
    if( nL!=N_NONE && nL!=nA && iCarrierL>iAncestorB ) *emplace( end() ) = Sign( qPrev[iCarrierL].getKSet(), ksRchild, tB, S_B );  // Add right child kset as B (H rule).
    else if( size()>0 )                                *emplace( end() ) = Sign( ksRchild, tB, S_B );                              // Add B sign.

    //    cerr << "            " << qPrev << "  " << aLchild << "  ==(f" << f << ",j" << j << "," << opL << "," << opR << ")=>  " << *this << endl;
  }

  int getDir ( char cOp ) const {
    return (cOp>='0' && cOp<='9') ? cOp-'0' :  // (numbered argument)
           (cOp=='M')             ? -1      :  // (modifier)
           (cOp=='I' || cOp=='V') ? 0       :  // (identity)
                                    -10;       // (will not map)
  }

  const Sign& at ( int i ) const { assert(i<int(size())); return (i<0) ? aTop : operator[](i); }

  int getDepth ( ) const {
    int d = 0; for( int i=size()-1; i>=0; i-- ) if( !operator[](i).getType().isCarrier() && operator[](i).getSide()==S_B ) d++;
    return d;
  }

  int getAncestorBIndex ( F f ) const {
    if( f==1 ) return size()-1;
    for( int i=size()-2; i>=0; i-- ) if( !operator[](i).getType().isCarrier() && operator[](i).getSide()==S_B ) return i;
    return -1;
  }

  int getAncestorAIndex ( F f ) const {
    for( int i=getAncestorBIndex(f)-1; i>=0; i-- ) if( !operator[](i).getType().isCarrier() && operator[](i).getSide()==S_A ) return i;
    return -1;
  }

  int getAncestorBCarrierIndex ( F f ) const {
    int iAncestor = getAncestorBIndex( f );
    N nB = at(iAncestor).getType().getLastNonlocal();
    if( nB!=N_NONE ) for( int i=iAncestor-1; i>=0 && (operator[](i).getType().isCarrier() || operator[](i).getType().containsCarrier(nB)); i-- ) if( operator[](i).getType()==nB ) return i;
    return -1;
  } 

  list<FPredictor>& calcForkPredictors ( list<FPredictor>& lfp, bool bAdd=true ) const {
    int d = (FEATCONFIG & 1) ? 0 : getDepth();
    const KSet& ksB = at(size()-1).getKSet();
    int iCarrier = getAncestorBCarrierIndex( 1 );
    if( STORESTATE_TYPE ) if( bAdd || FPredictor::exists(d,at(size()-1).getType()) ) lfp.emplace_back( d, at(size()-1).getType() );
    if( !(FEATCONFIG & 2) ) {
      for( auto& kA : (ksB.size()==0) ? ksBot  : ksB                    ) if( bAdd || FPredictor::exists(d,kNil,kA) ) lfp.emplace_back( d, kNil, kA );
      for( auto& kF : (iCarrier<0)    ? KSet() : at(iCarrier).getKSet() ) if( bAdd || FPredictor::exists(d,kF,kNil) ) lfp.emplace_back( d, kF, kNil );
//    } else if( FEATCONFIG & 1 ) {
//      for( auto& kA : (ksB.size()==0) ? ksBot  : ksB                    ) if( bAdd || FPredictor::exists(kNil,kA) ) lfp.emplace_back( kNil, kA );
//      for( auto& kF : (iCarrier<0)    ? KSet() : at(iCarrier).getKSet() ) if( bAdd || FPredictor::exists(kF,kNil) ) lfp.emplace_back( kF, kNil );
    }
    return lfp;
  }

  PPredictor calcPretrmTypeCondition ( F f, E e, K k_p_t ) const {
    if( FEATCONFIG & 1 ) return PPredictor( 0, f, (FEATCONFIG & 4) ? E('-') : e, at(size()-1).getType(), (FEATCONFIG & 16384) ? tBot : k_p_t.getType() );
    return             PPredictor( getDepth(), f, (FEATCONFIG & 4) ? E('-') : e, at(size()-1).getType(), (FEATCONFIG & 16384) ? tBot : k_p_t.getType() );
  }

  list<JPredictor>& calcJoinPredictors ( list<JPredictor>& ljp, F f, E eF, const LeftChildSign& aLchild, bool bAdd=true ) const {
    int d = (FEATCONFIG & 1) ? 0 : getDepth()+f;
    int iCarrierB = getAncestorBCarrierIndex( f );
    const Sign& aAncstr  = at( getAncestorBIndex(f) );
    const KSet& ksAncstr = ( aAncstr.getKSet().size()==0 ) ? ksBot : aAncstr.getKSet();
    const KSet& ksFiller = ( iCarrierB<0                 ) ? ksBot : at( iCarrierB ).getKSet();
    const KSet& ksLchild = ( aLchild.getKSet().size()==0 ) ? ksBot : aLchild.getKSet() ;
    if( STORESTATE_TYPE ) if( bAdd || JPredictor::exists(d,aAncstr.getType(),aLchild.getType()) ) ljp.emplace_back( d, aAncstr.getType(), aLchild.getType() );
    if( !(FEATCONFIG & 32) ) {
      for( auto& kA : ksAncstr ) for( auto& kL : ksLchild ) if( bAdd || JPredictor::exists(d,kNil,kA,kL) ) ljp.emplace_back( d, kNil, kA, kL );
      for( auto& kF : ksFiller ) for( auto& kA : ksAncstr ) if( bAdd || JPredictor::exists(d,kF,kA,kNil) ) ljp.emplace_back( d, kF, kA, kNil );
      for( auto& kF : ksFiller ) for( auto& kL : ksLchild ) if( bAdd || JPredictor::exists(d,kF,kNil,kL) ) ljp.emplace_back( d, kF, kNil, kL );
    }
    return ljp;
  }

  APredictor calcApexTypeCondition ( F f, J j, E eF, E eJ, O opL, const LeftChildSign& aLchild ) const {
    if( FEATCONFIG & 1 ) return APredictor( 0, 0, j, (FEATCONFIG & 64) ? E('-') : eJ, (FEATCONFIG & 128) ? O('-') : opL, at(getAncestorBIndex(f)).getType(), (j==0) ? aLchild.getType() : tBot );
    return         APredictor( getDepth()+f-j, f, j, (FEATCONFIG & 64) ? E('-') : eJ, (FEATCONFIG & 128) ? O('-') : opL, at(getAncestorBIndex(f)).getType(), (j==0) ? aLchild.getType() : tBot );
  }

  BPredictor calcBrinkTypeCondition ( F f, J j, E eF, E eJ, O opL, O opR, T tParent, const LeftChildSign& aLchild ) const {
    if( FEATCONFIG & 1 ) return  BPredictor( 0, 0, 0, (FEATCONFIG & 64) ? E('-') : eJ, (FEATCONFIG & 128) ? O('-') : opL, (FEATCONFIG & 128) ? O('-') : opR, tParent, aLchild.getType() );
    return          BPredictor( getDepth()+f-j, f, j, (FEATCONFIG & 64) ? E('-') : eJ, (FEATCONFIG & 128) ? O('-') : opL, (FEATCONFIG & 128) ? O('-') : opR, tParent, aLchild.getType() );
  }
};
const Sign StoreState::aTop( KSet(K::kTop), tTop, S_B );

////////////////////////////////////////////////////////////////////////////////

LeftChildSign::LeftChildSign ( const StoreState& qPrev, F f, E eF, const Sign& aPretrm ) {
    int         iCarrierB  = qPrev.getAncestorBCarrierIndex( 1 );
    const Sign& aAncestorA = qPrev.at( qPrev.getAncestorAIndex(1) );
    const Sign& aAncestorB = qPrev.at( qPrev.getAncestorBIndex(1) );
    const KSet& ksExtrtn   = (iCarrierB<0) ? KSet() : qPrev.at(iCarrierB).getKSet();
    *this = (f==1 && eF!='N')                  ? Sign( KSet(ksExtrtn,-qPrev.getDir(eF),aPretrm.getKSet()), aPretrm.getType(), S_A )
          : (f==1)                             ? aPretrm                             // if fork, lchild is preterm.
          : (qPrev.size()<=0)                  ? StoreState::aTop                    // if no fork and stack empty, lchild is T (NOTE: should not happen).
          : (!aAncestorA.isDitto() && eF!='N') ? Sign( KSet(ksExtrtn,-qPrev.getDir(eF),aAncestorA.getKSet()), aAncestorA.getType(), S_A )
          : (!aAncestorA.isDitto())            ? aAncestorA                          // if no fork and stack exists and last apex context set is not ditto, return last apex.
          :                                      Sign( KSet(ksExtrtn,-qPrev.getDir(eF),aAncestorB.getKSet()), aPretrm.getKSet(), aAncestorA.getType(), S_A );  // otherwise make new context set.
}


////////////////////////////////////////////////////////////////////////////////

W unkWord ( const char* ps ) {
  return ( 0==strcmp(ps+strlen(ps)-strlen("ing"), "ing") ) ? W("!unk!ing") :
         ( 0==strcmp(ps+strlen(ps)-strlen("ed"),  "ed" ) ) ? W("!unk!ed") :
         ( 0==strcmp(ps+strlen(ps)-strlen("s"),   "s"  ) ) ? W("!unk!s") :
         ( 0==strcmp(ps+strlen(ps)-strlen("ion"), "ion") ) ? W("!unk!ion") :
         ( 0==strcmp(ps+strlen(ps)-strlen("er"),  "er" ) ) ? W("!unk!er") :
         ( 0==strcmp(ps+strlen(ps)-strlen("est"), "est") ) ? W("!unk!est") :
         ( 0==strcmp(ps+strlen(ps)-strlen("ly"),  "ly" ) ) ? W("!unk!ly") : 
         ( 0==strcmp(ps+strlen(ps)-strlen("ity"), "ity") ) ? W("!unk!ity") : 
         ( 0==strcmp(ps+strlen(ps)-strlen("y"),   "y"  ) ) ? W("!unk!y") : 
         ( 0==strcmp(ps+strlen(ps)-strlen("al"),  "al" ) ) ? W("!unk!al") :
         ( ps[0]>='A' && ps[0]<='Z'                      ) ? W("!unk!cap") :
         ( ps[0]>='0' && ps[0]<='9'                      ) ? W("!unk!num") :
                                                             W("!unk!");
}

