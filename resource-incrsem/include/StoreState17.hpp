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
S S_A("/");
S S_B(";");

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

DiscreteDomain<int> domN;
class N : public Delimited<DiscreteDomainRV<int,domN>> {
 private:
  static map<N,bool> mnbLft;
  static map<N,bool> mnbArg;
  void calcDetermModels ( const char* ps ) {
    if( mnbArg.end()==mnbArg.find(*this) ) { mnbArg[*this]=( strlen(ps)<=4 ); }
    if( mnbLft.end()==mnbLft.find(*this) ) { mnbLft[*this]=( ps[0]!='\0' && ps[1]!='h' ); }
  }
 public:
  N ( )                : Delimited<DiscreteDomainRV<int,domN>> ( )    { }
  N ( int i )          : Delimited<DiscreteDomainRV<int,domN>> ( i )  { }
  N ( const char* ps ) : Delimited<DiscreteDomainRV<int,domN>> ( ps ) { calcDetermModels(ps); }
  bool isArg ( ) { return mnbArg[*this]; }
  bool isLft ( ) { return mnbLft[*this]; }
};
map<N,bool> N::mnbArg;
map<N,bool> N::mnbLft;
N N_NONE("");

////////////////////////////////////////////////////////////////////////////////

DiscreteDomain<int> domT;
class T : public DiscreteDomainRV<int,domT> {
 private:
  static map<T,int>  mtiArity;
  static map<T,bool> mtbIsCarry;
  static map<T,T>    mttCarrier;
  static map<T,T>    mttNoCarry;
  static map<T,N>    mtnLastNol;
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
      if ( l[i]=='-' && (l[i+1]=='g' || l[i+1]=='h'    /* || l[i+1]=='i' */   || l[i+1]=='r' || l[i+1]=='v') && depth==0 ) beg = i;
      if ( beg<i && end>i && depth==0 && (l[i]=='-' || l[i]=='\\' || l[i]=='^') ) end = i;
    }
//cout<<"!!"<<string(l,beg,end-beg)<<"!!"<<l<<endl;
    return N( string(l,beg,end-beg).c_str() );  // l+strlen(l);
  }
  void calcDetermModels ( const char* ps ) {
    if( mtiArity.end()  ==mtiArity.  find(*this) ) { mtiArity  [*this]=getArity(ps); }
    if( mtbIsCarry.end()==mtbIsCarry.find(*this) ) { mtbIsCarry[*this]=( ps[strlen(ps)-1]=='^' ); }
    if( mttCarrier.end()==mttCarrier.find(*this) ) { T& t=mttCarrier[*this]; t=( (ps[strlen(ps)-1]=='^') ? *this : string(ps).append("^").c_str() ); }
    if( mttNoCarry.end()==mttNoCarry.find(*this) ) { T& t=mttNoCarry[*this]; t=( (ps[strlen(ps)-1]!='^') ? *this : string(ps,strlen(ps)-1).c_str() ); }
    if( mtnLastNol.end()==mtnLastNol.find(*this) ) { mtnLastNol[*this]=getLastNolo(ps); }
  }
 public:
  T ( )                : DiscreteDomainRV<int,domT> ( )    { }
//  T ( int i )          : DiscreteDomainRV<int,domT> ( i )  { }
  T ( const char* ps ) : DiscreteDomainRV<int,domT> ( ps ) { calcDetermModels(ps); }
  int  getArity        ( ) { return mtiArity  [*this]; }
  bool hasCarrierMark  ( ) { return mtbIsCarry[*this]; }
  T    giveCarrierMark ( ) { return mttCarrier[*this]; }
  T    takeCarrierMark ( ) { return mttNoCarry[*this]; }
  N    getLastNonlocal ( ) { return mtnLastNol[*this]; }
};
map<T,int>  T::mtiArity;
map<T,bool> T::mtbIsCarry;
map<T,T>    T::mttCarrier;
map<T,T>    T::mttNoCarry;
map<T,N>    T::mtnLastNol;
T tTop("T");
T tBot("-");

////////////////////////////////////////////////////////////////////////////////

DiscreteDomain<int> domK;
class K : public DiscreteDomainRV<int,domK> {   // NOTE: can't be subclass of Delimited<...> or string-argument constructor of this class won't get called!
 public:
  static K kTop;
  static K kBot;
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
//    // Special case for top...
//    else if ( *this==kTop && mkt.end()==mkt.find(*this) ) mkt[*this]=T("T");
//    else if ( *this==kBot && mkt.end()==mkt.find(*this) ) mkt[*this]=T("bot");
  }
 public:
  K ( )                : DiscreteDomainRV<int,domK> ( )    { }
  K ( const char* ps ) : DiscreteDomainRV<int,domK> ( ps ) { calcDetermModels(ps); }
  T getType ( )       const { return ( *this==kBot ) ? T("bot") : mkt[*this]; }
  K project ( int n ) const { auto it = mkik.find(pair<K,int>(*this,n)); return (it==mkik.end()) ? kBot : it->second; }
};
map<K,T> K::mkt;
map<pair<K,int>,K> K::mkik;
const K K_DITTO("\"");
K kNil("");
K K::kTop("Top");
K K::kBot("Bot");

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
    D d;  ist.first >> "d" >> d >> "&";
    if ( ist.first.peek()=='t' ) { Delimited<T> t;       auto& o = ist.first >> "t" >> t        >> psDelim;  ist.second = FPredictor(d,t);      return o; }
    else                         { Delimited<K> kF, kA;  auto& o = ist.first >> kF >> "&" >> kA >> psDelim;  ist.second = FPredictor(d,kF,kA);  return o; }
  }
  friend bool operator>> ( pair<istream&,FPredictor&> ist, const vector<const char*>& vpsDelim ) {
    D d;  ist.first >> "d" >> d >> "&"; 
    if ( ist.first.peek()=='t' ) { Delimited<T> t;       auto o = ist.first >> "t" >> t        >> vpsDelim;  ist.second = FPredictor(d,t);      return o; }
    else                         { Delimited<K> kF, kA;  auto o = ist.first >> kF >> "&" >> kA >> vpsDelim;  ist.second = FPredictor(d,kF,kA);  return o; }
  }
  friend ostream& operator<< ( ostream& os, const FPredictor& t ) {
    if      ( mit.end()  != mit.find(t.id)  ) return os << "d" << mid[t.id] << "&" << "t" << mit[t.id];
    else if ( mikA.end() != mikA.find(t.id) ) return os << "d" << mid[t.id] << "&" << mikF[t.id] << "&" << mikA[t.id];
    else                                      return os << "NON_STRING_ID_" << t.id;
  }
  // output no-sem fpredictors---duan
  public:
  string getFPStringNoSem(){
  stringstream ss;
  if ( mikA.end() == mikA.find(id) && mit.end() != mit.find(id) ) ss << "d" << mid[id] << "&" << "t" <<  mit[id]; 
  else if (mikA.end() == mikA.find(id))                                  ss << "NON_STRING_ID_" << id;
  return ss.str();
}
};
uint                  FPredictor::nextid = 1;   // space for bias "" predictor
map<uint,D>           FPredictor::mid;
map<uint,T>           FPredictor::mit;
map<uint,K>           FPredictor::mikF;
map<uint,K>           FPredictor::mikA;
map<pair<D,T>,uint>   FPredictor::mdti;
map<trip<D,K,K>,uint> FPredictor::mdkki;

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
//if ( mfkfr.end()!=mfkfr.find(pair<F,K>(f,k)) ) *this = mfkfr[pair<F,K>(f,k)]; else *this=mfkfr[pair<F,K>(f,K::kBot)]; }

  F getFork ( ) const { return mfrf[*this]; }
  E getE    ( ) const { return mfre[*this]; }
  K getK    ( ) const { return mfrk[*this]; }
};
map<FResponse,F>           FResponse::mfrf;
map<FResponse,E>           FResponse::mfre;
map<FResponse,K>           FResponse::mfrk;
map<trip<F,E,K>,FResponse> FResponse::mfekfr;
//const FResponse FRESP_F0BOT("f0&bot");
//const FResponse FRESP_F1BOT("f1&bot");

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
    // output no-sem jpredictors---duan
  public:
  string getJPStringNoSem(){
  stringstream ss;
    if ( mikA.end() == mikA.find(id) && mitA.end() != mitA.find(id) ) ss << "d" << mid[id] << "&" << "t" <<  mitA[id]<< "&" << "t" << mitL[id]; 
    else if (mikA.end() == mikA.find(id))                                  ss << "NON_STRING_ID_" << id;
    return ss.str();
}

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
// if ( mjoojr.end()!=mjoojr.find(trip<J,O,O>(j,oL,oR)) ) *this = mjoojr[trip<J,O,O>(j,oL,oR)]; else *this=mjoojr[trip<J,O,O>(j,'I','I')]; }
  J getJoin ( ) const { return mjrj[*this]; }
  E getE    ( ) const { return mjre[*this]; }
  O getLOp  ( ) const { return mjroL[*this]; }
  O getROp  ( ) const { return mjroR[*this]; }
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
  static KSet ksDummy;
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
KSet KSet::ksDummy;
KSet ksTop = KSet( K::kTop );
KSet ksBot = KSet( K::kBot );

////////////////////////////////////////////////////////////////////////////////

class Sign : public DelimitedTrip<psX,KSet,psColon,T,psX,S,psX> {
 public:
  Sign ( )                           : DelimitedTrip<psX,KSet,psColon,T,psX,S,psX> ( )           { }
  Sign ( const KSet& ks1, T t, S s ) : DelimitedTrip<psX,KSet,psColon,T,psX,S,psX> ( ks1, t, s ) { }
  Sign ( const KSet& ks1, const KSet& ks2, T t, S s ) {
    first.reserve( ks1.size() + ks2.size() );
    first.insert( first.end(), ks1.begin(), ks1.end() );
    first.insert( first.end(), ks2.begin(), ks2.end() );
    second = t;
    thrid = s;
  }
  KSet&       setKSet ( )       { return first;  }
  T&          setType ( )       { return second; }
  S&          setSide ( )       { return third;  }
  const KSet& getKSet ( ) const { return first;  }
  T           getType ( ) const { return second; }
  S           getSide ( ) const { return third;  }
  bool        isDitto ( ) const { return getKSet().isDitto(); }
};

////////////////////////////////////////////////////////////////////////////////

class LeftChildSign : public Sign {
 public:
  LeftChildSign ( const StoreState& qPrev, F f, E eF, const Sign& aPretrm ) {
    // Additions and subtractions to store depth...
    uint iForkSub = 0; uint iForkAdd = 0;
    N nP;
    // If no fork, remove lowest (b) and all lowest nonlocal a's...
    if( f==0 ) { iForkSub=1; for( uint i=qPrev.size()-3; i>=0; i-- ) if( qPrev[i].getSide()==S_A ) iForkSub++; }
    // If yes fork, add p as a and if new adding new nonloc, add as b...
    if( f==1 ) { iForkAdd=1; nP = aPretrm.getType().getLastNolo(); bool b=false; for( Sign& a : *this ) b |= a.getType()==nP; if( !b ) iForkAdd=2; }

    uint iCarrier=qPrev.size(); for( uint i=0; i<qPrev.size()-iForkSub+iForkAdd; i++ ) if( qPrev[i].getType()==aPretrm.getType().getLastNonlocal() ) iCarrier=i;
    const KSet& ksExtrtn = (eF=='N' || iCarrier>=qPrev.size()) ? KSet::ksDummy : qPrev[iCarrier];

    // Calculate dependency direction (for calculation of contexts) for extraction operation...
    int dirE = (eF>='0' && eF<='9') ? '0'-eF :  // (numbered argument)
               (eF=='M')            ? 1      :  // (modifier)
                                      10;       // (will not map)

    *this = (f==1 && eF!='N')                                    ? Sign( KSet(ksExtrtn,dirE,aPretrm.getKSet()), aPretrm.getType(), S_A ) :
            (f==1)                                               ? aPretrm :            // if fork, lchild is preterm.
            (qPrev.size()<iForkSub)                              ? StoreState::aTop :   // if no fork and stack empty, lchild is T (NOTE: should not happen).
            (!qPrev[qPrev.size()-iForkSub].isDitto() && eF!='N') ? Sign( KSet(ksExtrtn,dirE,at(dLC).getA().getKSet()), qPrev[qPrev.size()-iForkSub].getType(), S_A ) :
            (!qPrev[qPrev.size()-iForkSub].isDitto())            ? qPrev[qPrev.size()-iForkSub] :      // if no fork and stack exists and last apex context set is not ditto, return last apex.
                                                                   Sign( KSet(ksExtrtn,dirE,qPrev[qPrev.size()-1].getKSet()), aPretrm.getKSet(), qPrev[qPrev.size()-iForkSub].getType(), S_A );  // otherwise make new context set.
  }
};

////////////////////////////////////////////////////////////////////////////////

class StoreState : public DelimitedVector<psX,Sign,psX,psX> {  // NOTE: format can't be read in bc of internal psX delimiter, but we don't need to.
 public:

  static Sign aTop;
  static Sign aBot;

  StoreState ( ) : DelimitedVector<psX,Sign,psX,psX> ( ) { }
  StoreState ( const StoreState& qPrev, F f, J j, E eF, E eJ, O opL, O opR, T tA, T tB, const Sign& aPretrm ) {

    //// A. FIND STORE LANDMARKS AND EXISTING P,A,B CARRIERS...

    // Find store landmarks...
    int iNextLowestA = qPrev.size();  int iNextLowestB = qPrev.size();  int iLowestA = qPrev.size();  int iLowestB = qPrev.size()-1;
    for( int i=iLowestB; i>=0; i-- ) if( !qPrev[i].getType().isCarrier() ) {
      if( iNextLowestA > iNextLowestB ) iNextLowestA = i;
      if( iNextLowestB > iLowestA )     iNextLowestB = i;
      if( iLowestA     > iLowestB )     iLowestA     = i;
    }

    // Find ancestors...
    int iAncestorA = ( f==0 ) ? iNextLowestA : iLowestA;
    int iAncestorB = ( f==0 ) ? iNextLowestB : iLowestB;

    // Find existing nonlocal carriers...
    N nP = aPretrm.getType().getLastNolo();  N nL = ((f==0)?qPrev[iLowestA]:aPretrm).getType().getLastNolo();  N nA = tA.getLastNolo();  N nB = tB.getLastNolo();
    int iCarrierP = -1;                      int iCarrierL = qPrev.size();                                     int iCarrierA = -1;       int iCarrierB = -1;
    // Find preterm nonlocal carrier, traversing up from ancestorB through carriers or noncarriers containing preterminal nonlocal, until carrier found...
    if( nP!=N_NONE ) for( int i=iAncestorB; i>=0 && (qPrev[i].getType().isCarrier() || qPrev[i].getType().containsCarrier(nP); i-- ) if( qPrev[i].getType()==nP ) iCarrierP=i;
    // Find lchild nonlocal carrier, traversing up from ancestorB through carriers or noncarriers containing lchild nonlocal, until carrier found...
    if( nP!=N_NONE ) for( int i=iLowestA; i>=0 && (qPrev[i].getType().isCarrier() || qPrev[i].getType().containsCarrier(nL); i-- ) if( qPrev[i].getType()==nL ) iCarrierL=i;
    // Find apex nonlocal carrier, traversing up from ancestorB through carriers or noncarriers containing apex nonlocal, until carrier found...
    if( nA!=N_NONE ) for( int i=iAncestorB; i>=0 && (qPrev[i].getType().isCarrier() || qPrev[i].getType().containsCarrier(nA); i-- ) if( qPrev[i].getType()==nA ) iCarrierA=i;
    // Find brink nonlocal carrier, traversing up from ancestorB through carriers or noncarriers containing brink nonlocal, until carrier found...
    if( nB!=N_NONE ) for( int i=iAncestorB; i>=0 && (qPrev[i].getType().isCarrier() || qPrev[i].getType().containsCarrier(nB); i-- ) if( qPrev[i].getType()==nB ) iCarrierB=i;

    // Reserve store big enough for ancestorB + new A and B if no join + any needed carriers...
    reserve( iAncestorB + ((j==0) ? 2 : 0) + ((nP!=N_NONE && iCarrierP!=-1) ? 1 : 0) + ((nA!=N_NONE && iCarrierA==-1) ? 1 : 0) + ((nB!=N_NONE && iCarrierB==-1) ? 1 : 0) ); 



   /*
    //// A. CALC NUMBER OF ADDITIONS AND SUBTRACTIONS FROM STORE...

    // Additions and subtractions to store depth...
    uint iForkSub = 0; uint iForkAdd = 0; uint iJoinSub = 0; uint iJoinAdd = 0; uint iBAdd = 0;
    N nP; N nA; N nB;

    // If no fork, remove lowest (b) and all lowest nonlocal a's...
    if( f==0 ) { iForkSub=1; for( uint i=qPrev.size()-3; i>=0; i-- ) if( qPrev[i].getSide()==S_A ) iForkSub++; }
    // If yes fork, add p as a and if new adding new nonloc, add as b...
    if( f==1 ) { iForkAdd=1; nP = aPretrm.getType().getLastNolo(); bool b=false; for( Sign& a : *this ) b |= a.getType()==nP; if( !b ) iForkAdd=2; }

    // If yes join, remove lowest a and all lowest nonlocal b's...
    if( j==1 ) { iJoinSub=iForkAdd; for( uint i=qPrev.size()-iForkSub-2; i>=0; i-- ) if( qPrev[i].getSide()==S_B ) iJoinSub++; }
    // If no join, replace old a with new a, and if adding new nonloc, add as b...
    if( j==0 ) { iJoinAdd=0; nA = tA.getLastNolo(); bool b=false; for( uint i=qPrev.size()-iForkSub-2; i>=0; i-- ) b |= qPrev[i].getType()==nA; if( !b ) iJoinAdd=1; }

    // If adding nonloc in b, add as a...
    iBAdd=1; bB = tB.getLastNolo(); bool b=false; for( uint i=qPrev.size()-iForkSub+iForkAdd-iJoinSub-2; i>=0; i-- ) b |= qPrev[i].getType()==nB; if( !b ) iBAdd=2;

    // Reserve store big enough for new signs minus removed signs...
    reserve( qPrev.size() - iForkSub + iForkAdd - iJoinSub + iJoinAdd + iBAdd );


    //// B. CALCULATE PARENT, LEFT CHILD CONTEXT SET, AND DEPENDENCY DIRECTIONS...

    T tParent  = ( j==0 ) ? tA : ( iAncestorB==0 ) ? tTop : qPrev[iAncestorB].getType();
    uint iCarrier=qPrev.size(); for( uint i=0; i<qPrev.size()-iForkSub+iForkAdd-iJoinSub; i++ ) if( qPrev[i].getType()==tParent.getLastNonlocal() ) iCarrier=i;
    const KSet& ksExtrtn = (eJ=='N') ? KSet::ksDummy : qPrev[iCarrier];
   */


    // Calculate left child based on fork...
    const LeftChildSign aLchild( qPrev, f, eF, aPretrm );


    //// C. FILL IN NEW PARTS OF NEW STORE...

    const KSet& ksLchild = aLchild.getKSet();
    const KSet  ksParent = (iCarrierA!=-1 && eJ!='N') ? KSet( KSet( (j==0) ? KSet() : qPrev[iAncestorB], aLchild.getKSet(), -getDir(opL) ), qPrev[iCarrierA], -getDir(eJ) );
                         :                              KSet(       (j==0) ? KSet() : qPrev[iAncestorB], aLchild.getKSet(), -getDir(opL)                                  );
    const KSet  ksRchild( ksParent, getDir(opR) );

    for( int i=0; i<(f==0&&j==1)?iAncestorB:(f==0&&j==0)?iLowestA:(f==1&&j==1)?iLowestB:iLowestB+1; i++ )
      *emplace( end() ) = ( i==iAncestorA && j==1 && qPrev[i].isDitto() && opR!='I' ) ? qPrev[iAncestorB]                                                                             // End of ditto.
                        : ( i==iCarrierP && eF!='N' )                                 ? Sign( KSet(qPrev[i].getKSet(),ksLchild,getDir(eF)), qPrev[i].getType(), qPrev[i].getSide() )  // Update to P carrier.
                        : ( i==iCarrierA && eJ!='N' )                                 ? Sign( KSet(qPrev[i].getKSet(),ksParent,getDir(eJ)), qPrev[i].getType(), qPrev[i].getSide() )  // Update to A carrier. 
                        :                                                               qPrev[i];                                                                                     // Copy store element.
    if( j==0 && nP!=N_NONE && iCarrierP==-1 && eF!='N' ) *emplace( end() ) = Sign( KSet(aPretrm.getKSet(),getDir(eF)), nP, S_A );  // If no join and nonloc P with no existing carrier, add P carrier.
    if( j==0 && nA!=N_NONE && iCarrierA==-1 && eJ!='N' ) *emplace( end() ) = Sign( KSet(ksParent,         getDir(eJ)), nA, S_A );  // If no join and nonloc A with no existing carrier, add A carrier.
    if( j==0 )                                           *emplace( end() ) = Sign( ksParent, tA, S_A );                            // If no join, add A sign.
    if( opL=='N' && nB!=N_NONE && iCarrierB==-1 )        *emplace( end() ) = Sign( ksLchild, nB, S_B );                            // Add left child kset as B carrier (G rule).
    if( orR=='N' && iCarrierL>iAncestorB )               *emplace( end() ) = Sign( qPrev[iCarrierL].getKSet(), tB, S_B );          // Add right child kset as L carrier (H rule).
    else                                                 *emplace( end() ) = Sign( ksRchild, tB, S_B );                            // Add B sign.

   /*
    // Identify outgoing ditto...
    iDittoLoc=qPrev.size(); if( j==1 && oR!='I' ) for( uint i=qPrev.size()-iForkSub+iForkAdd-iJoinSub-1; i>=0; i-- ) if( qPrev[i].isDitto() ) iDittoLoc=i;

    // Copy part of store above ditto...
    for( uint i=0; i<iDittoLoc && i<qPrev.size()-iForkSub+iForkAdd-iJoinSub; i++ ) *emplace( end() ) = qPrev[i];
    // Replace ditto with kset of brink (highest b below it)...
    if( iDittoLoc<qPrev.size() ) *emplace( end() ) = Sign( qPrev[qPrev.size()-iForkSub+iForkAdd-iJoinSub].getKSet(), qPrev[iDittoLoc].getType(), qPrev[iDittoLoc].getSide );
    // Copy part of store below ditto and above new depth...
    for( uint i=iDittoLoc+1; i<qPrev.size() && i<qPrev.size()-iForkSub+iForkAdd-iJoinSub; i++ ) *emplace( end() ) = qPrev[i];
   */

    // Add new P nonloc to store...
    if( iJoinAdd==1 ) *emplace( end() ) = Sign( KSet(), nA, S_B );
    // Add new A nonloc to store...
    if( iForkAdd==2 ) *emplace( end() ) = Sign( KSet(), nP, S_B );
    // Add A to store...
    if( j==0 )        *emplace( end() ) = Sign( (opR=='I') ? KSet(K_DITTO) : KSet(aLchild.getKSet(),dirL,KSet(ksExtrtn,-dirE,qPrev[size()-1].getKSet())), tA, S_A );
    // Add new B nonloc to store...
    if( iBAdd   ==2 ) *emplace( end() ) = Sign( KSet(), nB, S_A );
    // Add B to store...
    *emplace( end() )                   = Sign( KSet(KSet(aLchild.getKSet(),dirL,KSet(ksExtrtn,-dirE,qPrev[qPrev.size()-iForkSub+iForkAdd-iJoinSub].getKSet())),dirR), tB, S_B );





    // Calculate left child based on fork, join...
    Sign aLchildTmp;
    const Sign& aLchild = qPrev.getLchild( aLchildTmp, f, eF, aPretrm );

    // Obtain carrier for disappearing non-local dependency...
    int iJoinParent = (f==1) ? qPrev.size()-1 : dRetainedApex;
    T tParent = (j==0) ? tA : (iJoinParent>=0) ? qPrev[iJoinParent].getB().getType() : tTop;
    const KSet& ksExtrtn = (eJ=='N') ? KSet::ksDummy : qPrev.getCarrierKSet( iJoinParent, tParent.getLastNonlocal() );
    //if ( eJ!='N' ) cout<<"        tParent="<<tParent<<" dFirstUncopied="<<dFirstUncopied<<" dRetainedApex="<<dRetainedApex<<" ksExtrtn="<<ksExtrtn<<endl;

    // Calculate dependency direction (for calculation of contexts) for left child operation (direction is reversed b/c transition goes upward)...
    int dirL = (opL>='1' && opL<='9') ? '0'-opL :  // (numbered argument)
               (opL=='M')             ? 1       :  // (modifier)
               (opL=='I' || opL=='V') ? 0       :  // (identity)
                                        -10;       // (will not map)
    // Calculate dependency direction (for calculation of contexts) for extraction operation...
    int dirE = (eJ>='1' && eJ<='9') ? eJ-'0' :  // (numbered argument)
               (eJ=='M')            ? -1     :  // (modifier)
                                      -10;      // (will not map)
    // Calculate dependency direction (for calculation of contexts) for right child operation...
    int dirR = (opR>='1' && opR<='9') ? opR-'0' :  // (numbered argument)
               (opR=='M')             ? -1      :  // (modifier)
               (opR=='I' || opR=='V') ? 0       :  // (identity)
                                        -10;       // (will not map)
    // Get the incomplete sign that is to be joined...
//cerr<<"SS4.5 "<<qPrev.size()-1+f-j<<" "<<qPrev.size()-1+f-j+fN-jN<<endl;

    const IncompleteSign& isToJoin = (j==0 || dFirstUncopied<0) ? IncompleteSign() : qPrev[dRetainedApex];
    const Sign& aToJoinB = (iJoinParent>=0) ? qPrev[iJoinParent].getB() : aTop;    //isToJoin.getB();
//cerr<<"SS5 isToJoin="<<isToJoin<<endl;

    // Construct new incomplete sign...
    if ( j==0 && opR=='I' ) { isNew.setA() = Sign( KSet(K_DITTO), tA );
                              isNew.setB() = Sign( KSet(aLchild.getKSet(),dirL,KSet(ksExtrtn,-dirE)), tB ); }
    if ( j==0 && opR!='I' ) { isNew.setA() = Sign( KSet(aLchild.getKSet(),dirL,KSet(ksExtrtn,-dirE)), tA );
                              isNew.setB() = Sign( KSet(isNew.getA().getKSet(),dirR), tB ); }
    if ( j==1 && isToJoin.getA().isDitto() && opR=='I' ) { isNew.setA() = isToJoin.getA();
                                                           isNew.setB() = Sign( KSet(aLchild.getKSet(),dirL,KSet(ksExtrtn,-dirE,aToJoinB.getKSet())), tB ); }
    if ( j==1 && isToJoin.getA().isDitto() && opR!='I' ) { isNew.setA() = Sign( KSet(aLchild.getKSet(),dirL,KSet(ksExtrtn,-dirE,aToJoinB.getKSet())), isToJoin.getA().getType() );
                                                           isNew.setB() = Sign( KSet(isNew.getA().getKSet(),dirR), tB ); }
    if ( j==1 && !isToJoin.getA().isDitto() && opR=='I' ) { isNew.setA() = isToJoin.getA();
                                                            isNew.setB() = Sign( KSet(aLchild.getKSet(),dirL,KSet(ksExtrtn,-dirE,aToJoinB.getKSet())), tB ); }
    if ( j==1 && !isToJoin.getA().isDitto() && opR!='I' ) { isNew.setA() = isToJoin.getA();
                                                            isNew.setB() = Sign( KSet(KSet(aLchild.getKSet(),dirL,KSet(ksExtrtn,-dirE,aToJoinB.getKSet())),dirR), tB ); }
//cerr<<"SS6 size="<<size()<<" est size="<<qPrev.size()-1+f-j+fN-jN<<" isNew="<<isNew<<" isNew.getB().getKSet()="<<isNew.getB().getKSet()<<" *this"<<*this<<endl;

    // Add contexts of disappearing non-local to carrier...
    if ( eJ!='N' ) {
      KSet& ksNewCarrier = setCarrierKSet( size()-1, tParent.getLastNonlocal() );
      if ( &ksNewCarrier!=&KSet::ksDummy && j==0 ) { ksNewCarrier = KSet(KSet(aLchild.getKSet(),dirL),dirE,ksNewCarrier); }
      if ( &ksNewCarrier!=&KSet::ksDummy && j==1 ) { ksNewCarrier = KSet(KSet(aLchild.getKSet(),dirL,aToJoinB.getKSet()),dirE,ksNewCarrier); }
      if ( STORESTATE_CHATTY ) cout<<"tried to set carrier to "<<KSet(KSet(aLchild.getKSet(),dirL),dirE,ksNewCarrier)<<endl;
    }

    // If new brink is non-carrier `N' op, set kset to lchild kset...
    if ( !tB.hasCarrierMark() && opR=='N' ) isNew.setB().setKSet() = qPrev.getCarrierKSet( qPrev.size()-1, aLchild.getType().getLastNonlocal() ); //aLchild.getKSet();

    // If new nonlocal dep appears in brink with no contexts, add carrier at new incomplete sign...
    const Sign& bLast = back().getB();
    dci = getDeepestCarrierIndex(size()-1);
    N dcn = (dci<0) ? N_NONE : at(dci).getB().getType().getLastNonlocal();  if (N_NONE==dcn && 0<=dci && dci<int(size())) dcn = at(dci+1).getA().getType().getLastNonlocal(); 
/*
    if ( nB!=N_NONE && nB!=nA && noCarrierBAbove(int(size())-2,nB) && nB!=dcn && isNew.getB().getKSet()==KSet() ) {
      if ( STORESTATE_CHATTY ) cout<<"(new nonloc "<<nB<<" in brink; adding carrier at)"<<endl;
      isNew.setB() = Sign( (opL=='N')?aLchild.getKSet():KSet(),tB.giveCarrierMark() );
    }
//cerr<<"isNew="<<isNew<<endl;
*/
    // WS NOTE: CAN'T BE dcn BELOW BC MIGHT BE EARLIER!  BAD SUPPORT FOR MULTIPLE NOLOS IN GENERAL!!!
    // If new nonlocal dep appears in brink with some contexts, add carrier after new incomlete sign...
//    if ( nB!=N_NONE ) for ( int d=0; d<size(); d++ ) { cout<<at(d).getB().getType().hasCarrierMark()<<at(d).getB().getType().getLastNonlocal()<<endl; if ( at(d).getB().getType().hasCarrierMark() && at(d).getB().getType().getLastNonlocal()==nB ) dcn=nB; }
//cout<<" nB="<<nB<<" dcn="<<dcn<<endl;
    if ( nB!=N_NONE && nB!=nA && noCarrierBAbove(int(size())-2,nB) && nB!=dcn /* && isNew.getB().getKSet()!=KSet() */ ) {
      if ( STORESTATE_CHATTY ) cout<<"(new nonloc "<<nB<<" in brink; adding carrier below)"<<endl;
      //cout<<"    "<<opL<<" "<<aLchild<<endl;
      *emplace( end() ) = IncompleteSign( Sign(KSet(K_DITTO),tB), bLast );
      //cout<<" indicators: opR="<<opR<<" tB.getLastNonlocal()="<<tB.getLastNonlocal()<<endl;
      const KSet& ksCarrier = ( opR=='N' && tB.getLastNonlocal()==N("-rN") ) ? aLchild.getKSet() :
                              ( opL=='N'                                   ) ? aLchild.getKSet() :
                                                                               KSet() ;
      at(size()-2).setB() = Sign( ksCarrier ,tB.giveCarrierMark() );
    }
//cerr<<"SS7 "<<j<<" qPrev="<<qPrev<<" isToJoin="<<isToJoin<<" isNew="<<isNew<<endl;
//cerr<<"SS7 *this="<<*this<<endl;

    // Calculate dependency direction (for calculation of contexts) for extraction operation...
    int dirEF = (eF>='1' && eF<='9') ? eF-'0' :  // (numbered argument)
                (eF=='M')            ? -1     :  // (modifier)
                                       -10;      // (will not map)
    // Add contexts of disappearing non-local to carrier...
    if ( eF!='N' ) {
      // (short circuit if carrier immediately discharged)...
      KSet& ksNewCarrier = (!isNew.getB().getType().hasCarrierMark() && opR=='N' && aLchild.getType().getLastNonlocal()==aPretrm.getType().getLastNonlocal()) ? isNew.setB().setKSet() : setCarrierKSet( size()-1, aPretrm.getType().getLastNonlocal() );
      const KSet& ksOldCarrier = (!isNew.getB().getType().hasCarrierMark() && opR=='N' && aLchild.getType().getLastNonlocal()==aPretrm.getType().getLastNonlocal()) ? qPrev.getCarrierKSet(qPrev.size()-1,aPretrm.getType().getLastNonlocal()) : getCarrierKSet( size()-1, aPretrm.getType().getLastNonlocal() );
      if ( &ksNewCarrier!=&KSet::ksDummy && f==0 ) { ksNewCarrier = KSet(KSet(aPretrm.getKSet(),qPrev.back().getB().getKSet()),dirEF,ksOldCarrier); }
      if ( &ksNewCarrier!=&KSet::ksDummy && f==1 ) { ksNewCarrier = KSet(aPretrm.getKSet(),dirEF,ksOldCarrier); }
      if ( STORESTATE_CHATTY ) cout<<"after "<<aPretrm<<" tried to set carrier to "<<KSet(aPretrm.getKSet(),dirEF,ksNewCarrier)<<" aka "<<ksNewCarrier<<endl;
    }

//    cerr << "            " << qPrev << "  " << aLchild << "  ==(f" << f << ",j" << j << "," << opL << "," << opR << ")=>  " << *this << endl;
  }

  int getDir ( char cOp ) const {
    return (cOp>='1' && cOp<='9') ? cOp-'0' :  // (numbered argument)
           (cOp=='M')             ? -1      :  // (modifier)
           (cOp=='I' || cOp=='V') ? 0       :  // (identity)
                                    -10;       // (will not map)
  }

  int getParamDepth ( ) const {
    int d = 0;
    for( int i=0; i<int(size()); i++,d++ ) {
      if ( at(i).getA().getType().hasCarrierMark() ) d--;
      if ( at(i).getB().getType().hasCarrierMark() ) d--;
    }
    return d;
  }

  const Sign& getAncstr ( F f ) const {
    int dAncstr = getAncstrDepth( f );
    return ( dAncstr>=0 ) ? at(dAncstr).getB() : aTop;
//    return (int(size())-2+f>=0) ? operator[](size()-2+f).getB() :
//                                  aTop;
  }

  const Sign& getLchild ( Sign& aLchildTmp, F f, E eF, const Sign& aPretrm ) const {           // NOTE: carrier case should be more complex
    const KSet& ksExtrtn = getCarrierKSet( size()-1, aPretrm.getType().getLastNonlocal() );
    int dLC = getLchildDepth( f );
    // Calculate dependency direction (for calculation of contexts) for extraction operation...
    int dirE = (eF>='0' && eF<='9') ? '0'-eF :  // (numbered argument)
               (eF=='M')            ? 1      :  // (modifier)
                                      10;       // (will not map)
    return (f==1 && eF!='N')                      ? aLchildTmp=Sign( KSet(ksExtrtn,dirE,aPretrm.getKSet()), aPretrm.getType() ) :
           (f==1)                                 ? aPretrm :            // if fork, lchild is preterm.
           (dLC<0)                                ? StoreState::aTop :   // if no fork and stack empty, lchild is T (NOTE: should not happen).
           //(j==1 && at(size()-2).getB().getType().hasCarrierMark()) ? at(size()-2).getA() : // if no fork and join and penultimate brink is carrier, return penultimate apex.
           (!at(dLC).getA().isDitto() && eF!='N') ? aLchildTmp=Sign( KSet(ksExtrtn,dirE,at(dLC).getA().getKSet()), at(dLC).getA().getType() ) :
           (!at(dLC).getA().isDitto())            ? at(dLC).getA() :      // if no fork and stack exists and last apex context set is not ditto, return last apex.
                                                   aLchildTmp=Sign( KSet(ksExtrtn,dirE,at(dLC).getB().getKSet()), aPretrm.getKSet(), at(dLC).getA().getType() );  // otherwise make new context set.
  }

  const KSet& getFillerKSet ( ) const {
    for( int i=size()-1; i>=0; i-- ) {
      if ( at(i).getA().getType().hasCarrierMark() ) return at(i).getA().getKSet();
      if ( at(i).getB().getType().hasCarrierMark() ) return at(i).getB().getKSet();
    }
    return ksBot;
  }

  const KSet& getAncstrKSet ( F f ) const {
    return ( int(size())-2+f<0 )                                 ? ksTop :
           ( operator[](size()-2+f).getB().getKSet().size()==0 ) ? ksBot :
                                                                   operator[](size()-2+f).getB().getKSet() ;
  }

  list<FPredictor>& calcForkPredictors ( list<FPredictor>& lfp ) const {
    if( STORESTATE_TYPE ) lfp.emplace_back( getParamDepth(), (size()>0) ? operator[](size()-1).getB().getType() : tTop );
    const KSet& ksFiller = getFillerKSet();
    const KSet& ksAncstr = getAncstrKSet(1);
    for( auto& kA : ksAncstr ) lfp.emplace_back( getParamDepth(), kNil, kA );
    if( ksFiller!=ksBot ) for( auto& kF : ksFiller ) lfp.emplace_back( getParamDepth(), kF, kNil );
/*
    if       ( size()==0 )                                       lfp.emplace_back( size(), K::kTop );
    else if  ( operator[](size()-1).getB().getKSet().size()==0 ) lfp.emplace_back( size(), K::kBot );
    else for ( auto& k : operator[](size()-1).getB().getKSet() ) lfp.emplace_back( size(), k       ); 
*/
    return lfp;
  }

  PPredictor calcPretrmTypeCondition ( F f, E e, K k_p_t ) const {
    return PPredictor( getParamDepth(), f, e, (size()>0) ? operator[](size()-1).getB().getType() : tTop, k_p_t.getType() );
  }

  list<JPredictor>& calcJoinPredictors ( list<JPredictor>& ljp, F f, E eF, const Sign& aPretrm ) const {
    Sign aLchildTmp;
    int dLC = getLchildDepth(f);
    const Sign& aLchild = getLchild ( aLchildTmp, f, eF, aPretrm );   // NOTE: assume join in cases of carrier.
    const KSet& ksFiller = getFillerKSet();
    const KSet& ksAncstr = getAncstrKSet(f);
    const KSet& ksLchild = ( aLchild.getKSet().size()==0 ) ? KSet(K::kBot) : aLchild.getKSet() ;
    if( STORESTATE_TYPE ) ljp.emplace_back( dLC+1, getAncstr(f).getType(), aLchild.getType() );
    for( auto& kA : ksAncstr ) for( auto& kL : ksLchild ) ljp.emplace_back( dLC+1, kNil, kA, kL );
    if( ksFiller!=ksBot ) for( auto& kF : ksFiller ) for( auto& kA : ksAncstr ) ljp.emplace_back( dLC+1, kF, kA, kNil );
    if( ksFiller!=ksBot ) for( auto& kF : ksFiller ) for( auto& kL : ksLchild ) ljp.emplace_back( dLC+1, kF, kNil, kL );
    return ljp;
  }

  /*
  list<JPredictor>& calcJoinPredictorsnosem ( list<JPredictor>& ljp, F f, E eF, const Sign& aPretrm ) const {
    Sign aLchildTmp;
    const Sign& aLchild = getLchild ( aLchildTmp, f, eF, aPretrm );
    if ( STORESTATE_TYPE ) ljp.emplace_back( getParamDepth()+f, getAncstr(f).getType(), aLchild.getType() );
    ljp.emplace_back( getParamDepth()+f, K::kBot, K::kBot, K::kBot );
    return ljp;
  }
  */

  APredictor calcApexTypeCondition ( F f, J j, E eF, E eJ, O opL, const Sign& aPretrm ) const {
    Sign aLchildTmp;
    return APredictor( getLchildDepth(f)+1-j, f, j, eJ, opL, getAncstr(f).getType(), (j==0) ? getLchild(aLchildTmp,f,eF,aPretrm).getType() : tBot );
  }

  BPredictor calcBrinkTypeCondition ( F f, J j, E eF, E eJ, O opL, O opR, T tParent, const Sign& aPretrm ) const {
    Sign aLchildTmp;
    return BPredictor( getLchildDepth(f)+1-j, f, j, eJ, opL, opR, tParent, getLchild(aLchildTmp,f,eF,aPretrm).getType() );
  }
};
Sign StoreState::aTop;
IncompleteSign StoreState::qTop;
IncompleteSign StoreState::qBot;

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


