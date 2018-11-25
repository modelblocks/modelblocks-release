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
#include<regex>

////////////////////////////////////////////////////////////////////////////////

char psLBrack[] = "[";
char psRBrack[] = "]";

////////////////////////////////////////////////////////////////////////////////

int getDir ( char cOp ) {
  return (cOp>='0' && cOp<='9')             ? cOp-'0' :  // (numbered argument)
         (cOp=='M' || cOp=='U')             ? -1      :  // (modifier)
         (cOp=='u')                         ? -2      :  // (auxiliary w arity 2)
         (cOp=='I' || cOp=='C' || cOp=='V') ? 0       :  // (identity)
                                              -10;       // (will not map)
}

////////////////////////////////////////////////////////////////////////////////

typedef Delimited<int>  D;  // depth
typedef Delimited<int>  F;  // fork decision
typedef Delimited<int>  J;  // join decision
typedef Delimited<char> O;  // composition operation
//typedef Delimited<char> E;  // extraction operation
typedef Delimited<char> S;  // side (A,B)
const S S_A("/");
const S S_B(";");

DiscreteDomain<int> domAdHoc;
typedef Delimited<DiscreteDomainRV<int,domAdHoc>> AdHocFeature;
const AdHocFeature corefON("acorefON");
const AdHocFeature bias("abias");

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
  static map<T,N>            mtnFirstNol;
  static map<T,T>            mttNoFirstNol;
  static map<T,N>            mtnLastNol;
  static map<T,T>            mttNoLastNol;
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
    return ('N'==l[0] and not (strlen(l)>7 and '{'==l[3] and 'N'==l[4] and 'D'==l[7]) ) ? ctr+1 : ctr;
  }
  N getFirstNolo ( const char* l ) {
    int depth = 0;
    uint beg = strlen(l);
    uint end = strlen(l);
    for ( uint i=0; i<strlen(l); i++ ) {
      if ( l[i]=='{' ) depth++;
      if ( l[i]=='}' ) depth--;
      if ( beg>i && l[i]=='-' && (l[i+1]=='g' || l[i+1]=='h'   // || l[i+1]=='i'
                                                             || l[i+1]=='r' || l[i+1]=='v') && depth==0 ) beg = i;
      if ( beg<i && end>i && depth==0 && (l[i]=='-' || l[i]=='_' || l[i]=='\\' || l[i]=='^') ) end = i;
    }
    // cerr<<"i think first nolo of "<<l<<" is "<<string(l,beg,end-beg)<<endl;
    return N( string(l,beg,end-beg).c_str() );  // l+strlen(l);
  }
  T getNoFirstNoloHelper ( const char* l ) {
    int depth = 0;
    uint beg = strlen(l);
    uint end = strlen(l);
    for ( uint i=0; i<strlen(l); i++ ) {
      if ( l[i]=='{' ) depth++;
      if ( l[i]=='}' ) depth--;
      if ( beg>i && l[i]=='-' && (l[i+1]=='g' || l[i+1]=='h'   // || l[i+1]=='i'
                                                             || l[i+1]=='r' || l[i+1]=='v') && depth==0 ) beg = i;
      if ( beg<i && end>i && depth==0 && (l[i]=='-' || l[i]=='_' || l[i]=='\\' || l[i]=='^') ) end = i;
    }
    // cerr<<"i think without first nolo of "<<l<<" is "<<string(l,0,beg)+string(l,end,strlen(l)-end)<<endl;
    return T( (string(l,0,beg)+string(l,end,strlen(l)-end)).c_str() );  // l+strlen(l);
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
    // cerr<<"i think last nolo of "<<l<<" is "<<string(l,beg,end-beg)<<endl;
    return N( string(l,beg,end-beg).c_str() );  // l+strlen(l);
  }
  T getNoLastNoloHelper ( const char* l ) {
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
    // cerr<<"i think without last nolo of "<<l<<" is "<<string(l,0,beg)<<endl;
    return T( string(l,0,beg).c_str() );  // l+strlen(l);
  }
  void calcDetermModels ( const char* ps ) {
    if( mnbArg.end()==mnbArg.find(*this) ) { mnbArg[*this]=( strlen(ps)<=4 ); }
    if( mtiArity.end()==mtiArity.find(*this) ) { mtiArity[*this]=getArity(ps); }
    if( mtbIsCarry.end()==mtbIsCarry.find(*this) ) { mtbIsCarry[*this]=( ps[0]=='-' && ps[1]>='a' && ps[1]<='z' ); }  //( ps[strlen(ps)-1]=='^' ); }
    if( mtnFirstNol.end()==mtnFirstNol.find(*this) && strlen(ps)>0 && !(ps[0]=='-'&&ps[1]>='a'&&ps[1]<='z') ) { N& n=mtnFirstNol[*this]; n=getFirstNolo(ps); }
    if( mttNoFirstNol.end()==mttNoFirstNol.find(*this) ) { T& t=mttNoFirstNol[*this]; t=getNoFirstNoloHelper(ps); }
    if( mtnLastNol.end()==mtnLastNol.find(*this) && strlen(ps)>0 && !(ps[0]=='-'&&ps[1]>='a'&&ps[1]<='z') ) { N& n=mtnLastNol[*this]; n=getLastNolo(ps); }
    if( mttNoLastNol.end()==mttNoLastNol.find(*this) ) { T& t=mttNoLastNol[*this]; t=getNoLastNoloHelper(ps); }
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
      if ( depth==0 && beg>0 && beg<i && (ps[i+1]=='-' || ps[i+1]=='_' || ps[i+1]=='\\' || ps[i+1]=='^' || ps[i+1]=='\0') ) {
        // cerr<<"i think "<<string(ps,beg,i+1-beg)<<" is in "<<ps<<endl;
        mtnbIn[pair<T,N>(*this,string(ps,beg,i+1-beg).c_str())]=true;
        beg = strlen(ps);
      }
    }
  }
 public:
  T ( )                : DiscreteDomainRV<int,domT> ( )    { }
  T ( const char* ps ) : DiscreteDomainRV<int,domT> ( ps ) { calcDetermModels(ps); }
  bool isArg            ( )       const { return mnbArg[*this]; }
  int  getArity         ( )       const { return mtiArity  [*this]; }
  bool isCarrier        ( )       const { return mtbIsCarry[*this]; }
  N    getFirstNonlocal ( )       const { return mtnFirstNol[*this]; }
  T    withoutFirstNolo ( )       const { return mttNoFirstNol[*this]; }
  N    getLastNonlocal  ( )       const { return mtnLastNol[*this]; }
  T    withoutLastNolo  ( )       const { return mttNoLastNol[*this]; }
  bool containsCarrier  ( N n )   const { return mtnbIn.find(pair<T,N>(*this,n))!=mtnbIn.end(); }
  T    getLets          ( )       const { const auto& x = mttLets.find(*this); return (x==mttLets.end()) ? *this : x->second; }
  int  getNums          ( )       const { const auto& x = mtiNums.find(*this); return (x==mtiNums.end()) ? 0 : x->second; }
  T    addNum           ( int i ) const { const auto& x = mtitLetNum.find(pair<T,int>(*this,i)); return (x==mtitLetNum.end()) ? *this : x->second; }

  T    removeLink      ( )       { 
          string mtype = this->getString();
          if (string::npos != mtype.find("-n")) {
                  std::regex re ( "(.*)-n.*");
                  std::smatch sm;
                  if (std::regex_search(mtype,sm,re) && sm.size() > 1){
                          return T(sm.str(1).c_str());
                  }
          }
          return *this;
  }
};

map<N,bool>         T::mnbArg;
map<T,int>          T::mtiArity;
map<T,bool>         T::mtbIsCarry;
map<T,N>            T::mtnLastNol;
map<T,T>            T::mttNoLastNol;
map<T,N>            T::mtnFirstNol;
map<T,T>            T::mttNoFirstNol;
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
//  static map<K,K> mkkVU;
//  static map<K,K> mkkVD;
  static map<K,K> mkkO;
  void calcDetermModels ( const char* ps ) {
    if( strchr(ps,':')!=NULL ) {
      char cSelf = ('N'==ps[0]) ? '1' : '0';
      // Add associations to label and related K's...  (NOTE: related K's need two-step constructor so as to avoid infinite recursion!)
      if( mkt.end()==mkt.find(*this) ) mkt[*this]=T(string(ps,strchr(ps,':')).c_str());
      if( mkik.end()==mkik.find(pair<K,int>(*this,-4)) && strlen(ps)>2 && ps[strlen(ps)-2]!='-' ) { K& k=mkik[pair<K,int>(*this,-4)]; k=string(ps).append("-4").c_str(); }
      if( mkik.end()==mkik.find(pair<K,int>(*this,-3)) && strlen(ps)>2 && ps[strlen(ps)-2]!='-' ) { K& k=mkik[pair<K,int>(*this,-3)]; k=string(ps).append("-3").c_str(); }
      if( mkik.end()==mkik.find(pair<K,int>(*this,-2)) && strlen(ps)>2 && ps[strlen(ps)-2]!='-' ) { K& k=mkik[pair<K,int>(*this,-2)]; k=string(ps).append("-2").c_str(); }
      if( mkik.end()==mkik.find(pair<K,int>(*this,-1)) && strlen(ps)>2 && ps[strlen(ps)-2]!='-' ) { K& k=mkik[pair<K,int>(*this,-1)]; k=string(ps).append("-1").c_str(); }
      if( mkik.end()==mkik.find(pair<K,int>(*this,0))                            ) { K& k=mkik[pair<K,int>(*this,0)]; k=ps; }
      if( mkik.end()==mkik.find(pair<K,int>(*this,1)) && ps[strlen(ps)-2]!='-' && ps[strlen(ps)-1]==cSelf ) { K& k=mkik[pair<K,int>(*this,1)]; k=string(ps,strlen(ps)-1).append("1").c_str(); }
      if( mkik.end()==mkik.find(pair<K,int>(*this,2)) && ps[strlen(ps)-2]!='-' && ps[strlen(ps)-1]==cSelf ) { K& k=mkik[pair<K,int>(*this,2)]; k=string(ps,strlen(ps)-1).append("2").c_str(); }
      if( mkik.end()==mkik.find(pair<K,int>(*this,3)) && ps[strlen(ps)-2]!='-' && ps[strlen(ps)-1]==cSelf ) { K& k=mkik[pair<K,int>(*this,3)]; k=string(ps,strlen(ps)-1).append("3").c_str(); }
      if( mkik.end()==mkik.find(pair<K,int>(*this,4)) && ps[strlen(ps)-2]!='-' && ps[strlen(ps)-1]==cSelf ) { K& k=mkik[pair<K,int>(*this,4)]; k=string(ps,strlen(ps)-1).append("4").c_str(); }
      if( mkik.end()==mkik.find(pair<K,int>(*this,1)) && ps[strlen(ps)-2]=='-' && ps[strlen(ps)-1]=='1' ) { K& k=mkik[pair<K,int>(*this,1)]; k=string(ps,strlen(ps)-2).c_str(); }
//     if( mkkVU.end()==mkkVU.find(*this) && ps[strlen(ps)-2]=='-' && ps[strlen(ps)-1]=='1' ) { K& k=mkkVU[*this]; k=string(ps,strlen(ps)-2).append("-2").c_str(); }
//     else if( mkkVU.end()==mkkVU.find(*this) )                                              { K& k=mkkO[*this]; k=ps; }
//     if( mkkVD.end()==mkkVD.find(*this) && ps[strlen(ps)-2]=='-' && ps[strlen(ps)-1]=='2' ) { K& k=mkkVU[*this]; k=string(ps,strlen(ps)-2).append("-1").c_str(); }
//     else if( mkkVD.end()==mkkVD.find(*this) )                                              { K& k=mkkO[*this]; k=ps; }
      if( mkkO.end()== mkkO.find(*this) ) {
        if     ( ps[strlen(ps)-2]=='-' && ps[strlen(ps)-1]=='1' ) { K& k=mkkO[*this]; k=string(ps,strlen(ps)-2).append("-2").c_str(); }
        else if( ps[strlen(ps)-2]=='-' && ps[strlen(ps)-1]=='2' ) { K& k=mkkO[*this]; k=string(ps,strlen(ps)-2).append("-1").c_str(); }
        else                                                      { K& k=mkkO[*this]; k=ps; }
      }
    }
    else mkt[*this] = (*this==kBot) ? tBOT : (*this==kTop) ? tTop : tBot;
  }
 public:
  K ( )                : DiscreteDomainRV<int,domK> ( )    { }
  K ( const char* ps ) : DiscreteDomainRV<int,domK> ( ps ) { calcDetermModels(ps); }
  T getType   ( )       const { auto it = mkt.find(*this); return (it==mkt.end()) ? tBot : it->second; }
  K project   ( int n ) const { auto it = mkik.find(pair<K,int>(*this,n)); return (it==mkik.end()) ? kBot : it->second; }
  K transform ( bool bUp, char c ) const { return mkkO[*this]; }
//  K transform ( bool bUp, char c ) const { return (bUp and c=='V') ? mkkVU[*this] :
//                                                  (        c=='V') ? mkkVD[*this] : kBot; }
};
map<K,T> K::mkt;
map<pair<K,int>,K> K::mkik;
//map<K,K> K::mkkVU;
//map<K,K> K::mkkVD;
map<K,K> K::mkkO;
const K K::kBot("Bot");
const K kNil("");
const K K_DITTO("\"");
const K K::kTop("Top");


////////////////////////////////////////////////////////////////////////////////

class StoreState;

/////////////////////////////////////////////////////////////////////////////////

DiscreteDomain<int> domE;
class EVar : public DiscreteDomainRV<int,domE> {   // NOTE: can't be subclass of Delimited<...> or string-argument constructor of this class won't get called!
 public:
  static const EVar eNil;
 private:
  static map<EVar,char> meoTop;
  static map<EVar,char> meoBot;
  static map<EVar,EVar> meeNoTop;
  static map<EVar,EVar> meeNoBot;
  void calcDetermModels ( const char* ps ) {       // NOTE: top is front, bot is back...
    if(   meoTop.end()==  meoTop.find(*this) )                  { char& c=  meoTop[*this]; c=ps[0]; }
    if(   meoBot.end()==  meoBot.find(*this) )                  { char& c=  meoBot[*this]; c=ps[strlen(ps)-1]; }
    if( meeNoTop.end()==meeNoTop.find(*this) and strlen(ps)>1 ) { EVar& e=meeNoTop[*this]; e=ps+1; }
    if( meeNoBot.end()==meeNoBot.find(*this) and strlen(ps)>1 ) { EVar& e=meeNoBot[*this]; e=string(ps,0,strlen(ps)-1).c_str(); }
  }
 public:
  EVar ( )                : DiscreteDomainRV<int,domE> ( )    { }
  EVar ( const char* ps ) : DiscreteDomainRV<int,domE> ( ps ) { calcDetermModels(ps); }
  char top        ( ) const { return   meoTop[*this]; }
  char bot        ( ) const { return   meoBot[*this]; }
  EVar withoutTop ( ) const { return meeNoTop[*this]; }
  EVar withoutBot ( ) const { return meeNoBot[*this]; }
  char popTop     ( )       { char c = meoTop[*this]; *this = meeNoTop[*this]; return c; }
};
map<EVar,char> EVar::meoTop;
map<EVar,char> EVar::meoBot;
map<EVar,EVar> EVar::meeNoTop;
map<EVar,EVar> EVar::meeNoBot;
const EVar EVar::eNil("");

////////////////////////////////////////////////////////////////////////////////

class NPredictor {
  /*
  Boolean predictors for antecedent model.  Generally KxK pairs or TxT pairs between anaphor and candidate antecedent. 
  */
  private:
    uint id;

    static uint                nextid;
    static map<pair<K,K>,uint> mkki; 
    //static map<pair<T,T>,uint> mtti;
    static map<AdHocFeature,uint>    mstri;
    static map<uint,K>         miantk;
    static map<uint,K>         miancestork;
    static map<uint,AdHocFeature>    mistr;
    //static map<unit,T>         miantt;
    //static map<uint,T>         miancestort;
    //static mapsfromidtootherstuff;
    //
  public:
    //Constructors
    NPredictor ( ) : id(0) { }

    NPredictor (K antK, K ancestorK) {
      const auto& it = mkki.find(pair<K,K>(antK,ancestorK));
      if (it != mkki.end() ) id = it->second;
      else { id = nextid++; miantk[id] = antK; miancestork[id] = ancestorK; mkki[pair<K,K>(antK,ancestorK)] = id; }
    }

    NPredictor (AdHocFeature mstring) {
      const auto& it = mstri.find(mstring);
      if (it != mstri.end() ) id = it->second;
      else { id = nextid++; mistr[id] = mstring; mstri[mstring] = id; }
    }
  //NPredictor ("distance", int) {
        //TODO - can't use existing XPredictor template, whose values are binary for categories
        //

  //Accessor Methods
  uint toInt() const { return id; }
  operator uint() const { return id; }
  K getAncstrK()  const { return miancestork[id]; }
  K getAntcdntK() const { return miantk[id]; } 
  AdHocFeature getfeatname() const { return mistr[id]; }
  static uint getDomainSize() { return nextid; }

  // Input / output methods...
  friend pair<istream&,NPredictor&> operator>> ( istream& is, NPredictor& t ) {
    return pair<istream&,NPredictor&>(is,t);
  }
  friend istream& operator>> ( pair<istream&,NPredictor&> ist, const char* psDelim ) {
    if ( ist.first.peek()==psDelim[0] ) { auto& o =  ist.first >> psDelim;  ist.second = NPredictor();  return o; }
    if ( ist.first.peek()=='a' ) { 
      AdHocFeature mstring;
      auto& o = ist.first >> mstring >> psDelim;  
      ist.second = NPredictor(mstring);     
      return o; 
    }
    else                         { 
      Delimited<K> kAntecedent, kAncestor;  
      auto& o = ist.first >> kAntecedent >> "&" >> kAncestor >> "&" >> psDelim;  
      ist.second = NPredictor(kAntecedent,kAncestor);  
      return o; 
    }
    //Delimited<K> kAntecedent, kAncestor;
    //auto& o = ist.first >> kAntecedent >> "&" >> kAncestor >> psDelim;  
    //ist.second = NPredictor(kAntecedent,kAncestor);
    //return o; 
  }
  friend bool operator>> ( pair<istream&,NPredictor&> ist, const vector<const char*>& vpsDelim ) {
    if ( ist.first.peek()=='a' ) { 
      AdHocFeature mstring;
      auto o = ist.first >> mstring >> vpsDelim;  
      ist.second = NPredictor(mstring);     
      return o; 
    }
    else                         { 
      Delimited<K> kAntecedent, kAncestor;
      auto o = ist.first >> kAntecedent >> "&" >> kAncestor >> vpsDelim;  
      ist.second = NPredictor(kAntecedent,kAncestor);  
      return o; 
    }
  }
  friend ostream& operator<< ( ostream& os, const NPredictor& t ) {
    //return os << miantk[t.id] << "&" << miancestork[t.id]; 
    if (miantk.find(t.id)!=miantk.end()) { return os << miantk[t.id] << "&" << miancestork[t.id]; } //output either KxK
    else { return os << mistr[t.id]; } //...or string
  }
  static bool exists ( K kAntecedent, K kAncestor )      { return( mkki.end()!=mkki.find(pair<K,K>(kAntecedent,kAncestor)) ); }
  static bool exists ( AdHocFeature mstring )                  { return( mstri.end()!=mstri.find(mstring) ); }
};
uint                NPredictor::nextid = 1;
map<pair<K,K>,uint> NPredictor::mkki; 
//map<pair<T,T>,uint> NPredictor::mtti;
map<AdHocFeature,uint>    NPredictor::mstri;
map<uint,AdHocFeature>    NPredictor::mistr;
map<uint,K>         NPredictor::miantk;
map<uint,K>         NPredictor::miancestork;


////////////////////////////////////////////////////////////////////////////////

//TODO adapt JResponse definition for NResponse. why do we need this for variable that represents 0,1? why not use bool?. William says use D depth type
DiscreteDomain<int> domNResponse;
//typedef Delimited<int>  NResponse;  // 
typedef DiscreteDomainRV<int,domNResponse> NResponse;
/*
class NResponse : public DiscreteDomainRV<int,domNResponse> {
 private:
  static map<NResponse,N>                mnrn; //not sure even this makes sense since NResponses are 0,1 - won't map to specific NPredictor
  //static map<NResponse,EVar>             mjre;
  //static map<NResponse,O>                mjroL;
  //static map<NResponse,O>                mjroR;
  //static map<quad<N,EVar,O,O>,NResponse> mjeoojr;
 
  void calcDetermModels ( const char* ps ) {
    if( mnrn.end() ==mnrn.find(*this)  ) mnrn[*this]=ps[1]-'0';
    EVar e  = string(ps+3,uint(strchr(ps+3,'&')-(ps+3))).c_str();  //ps[3];
    O    oL = ps[4+e.getString().size()];
    O    oR = ps[6+e.getString().size()];
    if( mjre.end() ==mjre.find(*this)  ) mjre[*this]=e;  //ps[3];
    if( mjroL.end()==mjroL.find(*this) ) mjroL[*this]=oL;  //ps[5];
    if( mjroR.end()==mjroR.find(*this) ) mjroR[*this]=oR;  //ps[7];
    if( mjeoojr.end()==mjeoojr.find(quad<N,EVar,O,O>(ps[1]-'0',e,oL,oR)) ) mjeoojr[quad<N,EVar,O,O>(ps[1]-'0',e,oL,oR)]=*this;
  }
  
 public:
  NResponse ( )                         : DiscreteDomainRV<int,domNResponse> ( )    { }
  //NResponse ( const char* ps )          : DiscreteDomainRV<int,domNResponse> ( ps ) { calcDetermModels(ps); }
  //NResponse ( N j, EVar e, O oL, O oR ) : DiscreteDomainRV<int,domNResponse> ( )    {
  //  *this = ( mjeoojr.end()==mjeoojr.find(quad<N,EVar,O,O>(j,e,oL,oR)) ) ? ("j" + to_string(j) + "&" + e.getString() + "&" + string(1,oL) + "&" + string(1,oR)).c_str()
  //                                                                       : mjeoojr[quad<N,EVar,O,O>(j,e,oL,oR)];
  //}
  N    getAnte ( ) const { return mnrn[*this]; }
  //EVar getE    ( ) const { return mjre[*this]; }
  //O    getLOp  ( ) const { return mjroL[*this]; }
  //O    getROp  ( ) const { return mjroR[*this]; }
  //static bool exists ( N j, EVar e, O oL, O oR ) { return( mjeoojr.end()!=mjeoojr.find(quad<N,EVar,O,O>(j,e,oL,oR)) ); }
};
map<NResponse,N>                NResponse::mnrn;
//map<NResponse,EVar>             NResponse::mjre;
//map<NResponse,O>                NResponse::mjroL;
//map<NResponse,O>                NResponse::mjroR;
//map<quad<N,EVar,O,O>,NResponse> NResponse::mjeoojr;
*/

////////////////////////////////////////////////////////////////////////////////
class FPredictor {
 private:

  // Data members...
  uint id;

  // Static data members...
  static uint                    nextid;
  static map<uint,D>             mid;
  static map<uint,T>             mit;
  static map<uint,K>             mikF;
  static map<uint,K>             mikA;
  static map<uint,K>             mikAnt;
  static map<pair<D,T>,uint>     mdti;
  static map<trip<D,K,K>,uint>   mdkki;
  static map<quad<D,K,K,K>,uint> mdkkki; //ej proposed change for coreference
  static map<pair<K,K>,uint>     mkki;

 public:

  // Constructors...
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
  FPredictor ( D d, K kF, K kA, K kAntecedent ) { //ej new constructor with antecedent 
    const auto& it = mdkkki.find(quad<D,K,K,K>(d,kF,kA,kAntecedent));
    if ( it != mdkkki.end() ) id = it->second;
    else { id = nextid++;  mid[id] = d;  mikF[id] = kF;  mikA[id] = kA;  mikAnt[id] = kAntecedent; mdkkki[quad<D,K,K,K>(d,kF,kA,kAntecedent)] = id; }
    //cout<<"did id "<<id<<"/"<<nextid<<" as "<<*this<<endl;
  }

  // Accessor methods...
  uint toInt() const { return id; }
  operator uint() const { return id; }
  D getDepth()    const { return mid[id]; }
  T getT()        const { return mit[id]; }
  K getFillerK()  const { return mikF[id]; }
  K getAncstrK()  const { return mikA[id]; }
  K getAntcdntK() const { return mikAnt[id]; } //ej change for coref
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
      else                         { Delimited<K> kF, kA, kAntecedent;  auto& o = ist.first >> kF >> "&" >> kA >> "&" >> kAntecedent >> psDelim;  ist.second = FPredictor(d,kF,kA,kAntecedent);  return o; }
    } else { 
                                     Delimited<K> kF, kA;  auto& o = ist.first >> kF >> "&" >> kA >> psDelim;  ist.second = FPredictor(kF,kA);    return o;
    }
  }
  friend bool operator>> ( pair<istream&,FPredictor&> ist, const vector<const char*>& vpsDelim ) {
    D d;  ist.first >> "d" >> d >> "&"; 
    if ( ist.first.peek()=='d' ) { 
      if ( ist.first.peek()=='t' ) { Delimited<T> t;       auto o = ist.first >> "t" >> t        >> vpsDelim;  ist.second = FPredictor(d,t);      return o; }
      else                         { Delimited<K> kF, kA, kAntecedent;  auto o = ist.first >> kF >> "&" >> kA >> "&" >> kAntecedent >> vpsDelim;  ist.second = FPredictor(d,kF,kA,kAntecedent);  return o; }
    } else { 
                                     Delimited<K> kF, kA;  auto o = ist.first >> kF >> "&" >> kA >> vpsDelim;  ist.second = FPredictor(kF,kA);    return o; 
    }
  }
  friend ostream& operator<< ( ostream& os, const FPredictor& t ) {
    if      ( mit.end()  != mit.find(t.id)  ) return os << "d" << mid[t.id] << "&" << "t" << mit[t.id];
    else if ( mid.end()  != mid.find(t.id)  ) return os << "d" << mid[t.id] << "&" << mikF[t.id] << "&" << mikA[t.id] << "&" << mikAnt[t.id];
    else if ( mikA.end() != mikA.find(t.id) ) return os << mikF[t.id] << "&" << mikA[t.id];
    else                                      return os << "NON_STRING_ID_" << t.id;
  }
  static bool exists ( D d, T t )        { return( mdti.end()!=mdti.find(pair<D,T>(d,t)) ); }
  static bool exists ( D d, K kF, K kA ) { return( mdkki.end()!=mdkki.find(trip<D,K,K>(d,kF,kA)) ); }
  static bool exists ( K kF, K kA )      { return( mkki.end()!=mkki.find(pair<K,K>(kF,kA)) ); }
  static bool exists ( D d, K kF, K kA, K kAntecedent) { return( mdkkki.end()!=mdkkki.find(quad<D,K,K,K>(d,kF,kA,kAntecedent)) ); } //ej changes for coref
  FPredictor  addNum ( int i ) const     { return( FPredictor( mid[id], mit[id].addNum(i) ) ); }
};
uint                  FPredictor::nextid = 1;   // space for bias "" predictor
map<uint,D>           FPredictor::mid;
map<uint,T>           FPredictor::mit;
map<uint,K>           FPredictor::mikF;
map<uint,K>           FPredictor::mikA;
map<uint,K>           FPredictor::mikAnt;
map<pair<D,T>,uint>   FPredictor::mdti;
map<trip<D,K,K>,uint> FPredictor::mdkki;
map<pair<K,K>,uint>   FPredictor::mkki;
map<quad<D,K,K,K>,uint> FPredictor::mdkkki; 

////////////////////////////////////////////////////////////////////////////////

DiscreteDomain<int> domFResponse;
class FResponse : public DiscreteDomainRV<int,domFResponse> {
 private:
  static map<FResponse,F>              mfrf;
  static map<FResponse,EVar>           mfre;
  static map<FResponse,K>              mfrk;
  static map<trip<F,EVar,K>,FResponse> mfekfr;
  void calcDetermModels ( const char* ps ) {
    F    f = ps[1]-'0';
    EVar e = string(ps+3,uint(strchr(ps+3,'&')-(ps+3))).c_str();  //ps[3];
    K    k = ps+4+e.getString().size();   //ps+5;
    if( mfrf.end()==mfrf.find(*this) ) mfrf[*this]=f;
    if( mfre.end()==mfre.find(*this) ) mfre[*this]=e;
    if( mfrk.end()==mfrk.find(*this) ) mfrk[*this]=k;
    if( mfekfr.end()==mfekfr.find(trip<F,EVar,K>(f,e,k)) ) mfekfr[trip<F,EVar,K>(f,e,k)]=*this;
  }
 public:
  FResponse ( )                   : DiscreteDomainRV<int,domFResponse> ( )    { }
  FResponse ( const char* ps )    : DiscreteDomainRV<int,domFResponse> ( ps ) { calcDetermModels(ps); }
  FResponse ( F f, EVar e, K k )  : DiscreteDomainRV<int,domFResponse> ( )    {
    *this = ( mfekfr.end()==mfekfr.find(trip<F,EVar,K>(f,e,k)) ) ? ("f" + to_string(f) + "&" + e.getString() + "&" + k.getString()).c_str()
                                                                 : mfekfr[trip<F,EVar,K>(f,e,k)];
  }
  static bool exists ( F f, EVar e, K k ) { return( mfekfr.end()!=mfekfr.find(trip<F,EVar,K>(f,e,k)) ); }

  F    getFork ( ) const { return mfrf[*this]; }
  EVar getE    ( ) const { return mfre[*this]; }
  K    getK    ( ) const { return mfrk[*this]; }
};
map<FResponse,F>              FResponse::mfrf;
map<FResponse,EVar>           FResponse::mfre;
map<FResponse,K>              FResponse::mfrk;
map<trip<F,EVar,K>,FResponse> FResponse::mfekfr;

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

  // Constructors...
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
  //non loop-based input stream operator
  friend istream& operator>> ( pair<istream&,JPredictor&> ist, const char* psDelim ) {
    if ( ist.first.peek()==psDelim[0] ) { auto& o =  ist.first >> psDelim;  ist.second = JPredictor();  return o; }
    D d;  ist.first >> "d" >> d >> "&";
    if ( ist.first.peek()=='t' ) { Delimited<T>     tA, tL;  auto& o = ist.first >> "t"       >> tA >> "&" >> "t" >> tL >> psDelim;  ist.second = JPredictor(d,tA,tL);     return o; }
    else                         { Delimited<K> kF, kA, kL;  auto& o = ist.first >> kF >> "&" >> kA >> "&" >>        kL >> psDelim;  ist.second = JPredictor(d,kF,kA,kL);  return o; }
  }
  //loop-based input stream operator
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
  static map<JResponse,J>                mjrj;
  static map<JResponse,EVar>             mjre;
  static map<JResponse,O>                mjroL;
  static map<JResponse,O>                mjroR;
  static map<quad<J,EVar,O,O>,JResponse> mjeoojr;
  void calcDetermModels ( const char* ps ) {
    if( mjrj.end() ==mjrj.find(*this)  ) mjrj[*this]=ps[1]-'0';
    EVar e  = string(ps+3,uint(strchr(ps+3,'&')-(ps+3))).c_str();  //ps[3];
    O    oL = ps[4+e.getString().size()];
    O    oR = ps[6+e.getString().size()];
    if( mjre.end() ==mjre.find(*this)  ) mjre[*this]=e;  //ps[3];
    if( mjroL.end()==mjroL.find(*this) ) mjroL[*this]=oL;  //ps[5];
    if( mjroR.end()==mjroR.find(*this) ) mjroR[*this]=oR;  //ps[7];
    if( mjeoojr.end()==mjeoojr.find(quad<J,EVar,O,O>(ps[1]-'0',e,oL,oR)) ) mjeoojr[quad<J,EVar,O,O>(ps[1]-'0',e,oL,oR)]=*this;
  }
 public:
  JResponse ( )                         : DiscreteDomainRV<int,domJResponse> ( )    { }
  JResponse ( const char* ps )          : DiscreteDomainRV<int,domJResponse> ( ps ) { calcDetermModels(ps); }
  JResponse ( J j, EVar e, O oL, O oR ) : DiscreteDomainRV<int,domJResponse> ( )    {
    *this = ( mjeoojr.end()==mjeoojr.find(quad<J,EVar,O,O>(j,e,oL,oR)) ) ? ("j" + to_string(j) + "&" + e.getString() + "&" + string(1,oL) + "&" + string(1,oR)).c_str()
                                                                         : mjeoojr[quad<J,EVar,O,O>(j,e,oL,oR)];
  }
  J    getJoin ( ) const { return mjrj[*this]; }
  EVar getE    ( ) const { return mjre[*this]; }
  O    getLOp  ( ) const { return mjroL[*this]; }
  O    getROp  ( ) const { return mjroR[*this]; }
  static bool exists ( J j, EVar e, O oL, O oR ) { return( mjeoojr.end()!=mjeoojr.find(quad<J,EVar,O,O>(j,e,oL,oR)) ); }
};
map<JResponse,J>                JResponse::mjrj;
map<JResponse,EVar>             JResponse::mjre;
map<JResponse,O>                JResponse::mjroL;
map<JResponse,O>                JResponse::mjroR;
map<quad<J,EVar,O,O>,JResponse> JResponse::mjeoojr;

////////////////////////////////////////////////////////////////////////////////

typedef Delimited<int> D;

class PPredictor : public DelimitedQuint<psX,D,psSpace,F,psSpace,Delimited<EVar>,psSpace,Delimited<T>,psSpace,Delimited<T>,psX> {
 public:
  PPredictor ( )                              : DelimitedQuint<psX,D,psSpace,F,psSpace,Delimited<EVar>,psSpace,Delimited<T>,psSpace,Delimited<T>,psX> ( )                 { }
  PPredictor ( D d, F f, EVar e, T tB, T tK ) : DelimitedQuint<psX,D,psSpace,F,psSpace,Delimited<EVar>,psSpace,Delimited<T>,psSpace,Delimited<T>,psX> ( d, f, e, tB, tK ) { }

};

class WPredictor : public DelimitedTrip<psX,Delimited<EVar>,psSpace,Delimited<K>,psSpace,Delimited<T>,psX> { };

class APredictor : public DelimitedSept<psX,D,psSpace,F,psSpace,J,psSpace,Delimited<EVar>,psSpace,O,psSpace,Delimited<T>,psSpace,Delimited<T>,psX> {
 public:
  APredictor ( )                                         : DelimitedSept<psX,D,psSpace,F,psSpace,J,psSpace,Delimited<EVar>,psSpace,O,psSpace,Delimited<T>,psSpace,Delimited<T>,psX> ( ) { }
  APredictor ( D d, F f, J j, EVar e, O oL, T tB, T tL ) : DelimitedSept<psX,D,psSpace,F,psSpace,J,psSpace,Delimited<EVar>,psSpace,O,psSpace,Delimited<T>,psSpace,Delimited<T>,psX> ( d, f, j, e, oL, tB, tL ) { }
};

class BPredictor : public DelimitedOct<psX,D,psSpace,F,psSpace,J,psSpace,Delimited<EVar>,psSpace,O,psSpace,O,psSpace,Delimited<T>,psSpace,Delimited<T>,psX> {
 public:
  BPredictor ( )                                               : DelimitedOct<psX,D,psSpace,F,psSpace,J,psSpace,Delimited<EVar>,psSpace,O,psSpace,O,psSpace,Delimited<T>,psSpace,Delimited<T>,psX> ( )                         { }
  BPredictor ( D d, F f, J j, EVar e, O oL, O oR, T tP, T tL ) : DelimitedOct<psX,D,psSpace,F,psSpace,J,psSpace,Delimited<EVar>,psSpace,O,psSpace,O,psSpace,Delimited<T>,psSpace,Delimited<T>,psX> ( d, f, j, e, oL, oR, tP, tL ) { }
};

////////////////////////////////////////////////////////////////////////////////

class KSet : public DelimitedVector<psLBrack,Delimited<K>,psComma,psRBrack> {
 private:
  EVar eBankedUnaryTransforms;
 public:
  static const KSet ksDummy;
  KSet ( )                                                      : DelimitedVector<psLBrack,Delimited<K>,psComma,psRBrack> ( ) { }
  KSet ( const K& k )                                           : DelimitedVector<psLBrack,Delimited<K>,psComma,psRBrack> ( ) { emplace_back(k); }
  KSet ( const KSet& ks1, const KSet& ks2 )                     : DelimitedVector<psLBrack,Delimited<K>,psComma,psRBrack> ( ) {
    reserve( ks1.size() + ks2.size() );
    insert( end(), ks1.begin(), ks1.end() );
    insert( end(), ks2.begin(), ks2.end() );
  }
  KSet ( const KSet& ks, int iProj, const KSet& ksNoProject = ksDummy ) : DelimitedVector<psLBrack,Delimited<K>,psComma,psRBrack> ( ) {
    reserve( ks.size() + ksNoProject.size() );
    for( const K& k : ks ) if( k.project(iProj)!=K::kBot ) push_back( k.project(iProj) );
    insert( end(), ksNoProject.begin(), ksNoProject.end() );
  }
  // Constructor for nolos...
  KSet ( const KSet& ks, int iProj, bool bUp, EVar e, const vector<int>& viCarriers, const StoreState& ss, const KSet& ksNoProject );
  // Specification methods...
  void addBankedUnaryTransform ( EVar e ) { eBankedUnaryTransforms=e; }
  // Accessor methods...
  EVar getBankedUnaryTransform ( ) const { return eBankedUnaryTransforms; }
  bool isDitto                 ( ) const { return ( size()>0 and front()==K_DITTO ); }
};
const KSet KSet::ksDummy;
const KSet ksTop = KSet( K::kTop );
const KSet ksBot = KSet( K::kBot );

////////////////////////////////////////////////////////////////////////////////

class Sign : public DelimitedTrip<psX,KSet,psColon,T,psX,S,psX> {
 public:
  Sign ( )                           : DelimitedTrip<psX,KSet,psColon,T,psX,S,psX> ( )           {third()=S_A; }
  Sign ( const KSet& ks1, T t, S s ) : DelimitedTrip<psX,KSet,psColon,T,psX,S,psX> ( ks1, t, s ) { }
  Sign ( const KSet& ks1, const KSet& ks2, T t, S s ) {
    first().reserve( ks1.size() + ks2.size() );
    first().insert( first().end(), ks1.begin(), ks1.end() );
    first().insert( first().end(), ks2.begin(), ks2.end() );
    first().addBankedUnaryTransform( ks1.getBankedUnaryTransform() );
    second() = t;
    third()  = s;
  }
  KSet&       setKSet ( )       { return first();  }
  T&          setType ( )       { return second(); }
  S&          setSide ( )       { return third();  }
  const KSet& getKSet ( ) const { return first();  }
  T           getType ( ) const { return second().removeLink(); }
  S           getSide ( ) const { return third();  }
  bool        isDitto ( ) const { return getKSet().isDitto(); }
};

////////////////////////////////////////////////////////////////////////////////

class LeftChildSign : public Sign {
 public:
  LeftChildSign ( const Sign& a ) : Sign(a) { }
  LeftChildSign ( const StoreState& qPrev, F f, EVar eF, const Sign& aPretrm );
};

////////////////////////////////////////////////////////////////////////////////

class StoreState : public DelimitedVector<psX,Sign,psX,psX> {  // NOTE: format can't be read in bc of internal psX delimiter, but we don't need to.
 public:

  static const Sign aTop;
  static const Sign aBot;

  StoreState ( ) : DelimitedVector<psX,Sign,psX,psX> ( ) { }
  StoreState ( const StoreState& qPrev, F f, J j, EVar evF, EVar evJ, O opL, O opR, T tA, T tB, const Sign& aPretrm, const LeftChildSign& aLchild ) {

    ////// A. FIND STORE LANDMARKS AND EXISTING P,A,B CARRIERS...

    //// A.1. Find reentrance points in old structure...
    int iAncestorA = qPrev.getAncestorAIndex(f);
    int iAncestorB = qPrev.getAncestorBIndex(f);
    int iLowerA    = (f==1) ? qPrev.size() : qPrev.getAncestorAIndex(1);

    //// A.2. Create vectors of carrier indices (one for each nonlocal in category, first to last)...
    T tCurrP=aPretrm.getType();  vector<int> viCarrierP;  viCarrierP.reserve(4);
    T tCurrL=aLchild.getType();  vector<int> viCarrierL;  viCarrierL.reserve(4);
    T tCurrA=tA;                 vector<int> viCarrierA;  viCarrierA.reserve(4);
    T tCurrB=tB;                 vector<int> viCarrierB;  viCarrierB.reserve(4);
    int nNewCarriers = 0;
    for( int i=qPrev.size()-1; i>=-1; i-- ) {
      T tI = (i>-1) ? qPrev[i].getType() : tTop;
      N nP=tCurrP.getLastNonlocal(); if( i>-1 and                 nP!=N_NONE && qPrev[i].getType()==nP                     ) { viCarrierP.push_back(i);  tCurrP=tCurrP.withoutLastNolo(); }
                                     if(                          nP!=N_NONE && !tI.isCarrier() && !tI.containsCarrier(nP) ) { viCarrierP.push_back(-1); tCurrP=tCurrP.withoutLastNolo(); nNewCarriers++; }
      N nL=tCurrL.getLastNonlocal(); if( i>-1 and i<iLowerA    && nL!=N_NONE && qPrev[i].getType()==nL                     ) { viCarrierL.push_back(i);  tCurrL=tCurrL.withoutLastNolo(); }
                                     if(          i<iLowerA    && nL!=N_NONE && !tI.isCarrier() && !tI.containsCarrier(nL) ) { viCarrierL.push_back(-1); tCurrL=tCurrL.withoutLastNolo(); nNewCarriers++; }
      N nA=tCurrA.getLastNonlocal(); if( i>-1 and i<iLowerA    && nA!=N_NONE && qPrev[i].getType()==nA                     ) { viCarrierA.push_back(i);  tCurrA=tCurrA.withoutLastNolo(); }
                                     if(          i<iLowerA    && nA!=N_NONE && !tI.isCarrier() && !tI.containsCarrier(nA) ) { viCarrierA.push_back(-1); tCurrA=tCurrA.withoutLastNolo(); nNewCarriers++; }
      N nB=tCurrB.getLastNonlocal(); if( i>-1 and i<iAncestorB && nB!=N_NONE && qPrev[i].getType()==nB                     ) { viCarrierB.push_back(i);  tCurrB=tCurrB.withoutLastNolo(); }
                                     if(          i<iAncestorB && nB!=N_NONE && !tI.isCarrier() && !tI.containsCarrier(nB) ) { viCarrierB.push_back(-1); tCurrB=tCurrB.withoutLastNolo(); nNewCarriers++; }
    }

//cout<<" viCarrierP=";
//for( int i : viCarrierP ) cout<<" "<<i;
//cout<<endl;
//cout<<" viCarrierA=";
//for( int i : viCarrierA ) cout<<" "<<i;
//cout<<endl;
//cout<<" viCarrierL=";
//for( int i : viCarrierL ) cout<<" "<<i;
//cout<<endl;
//cout<<" viCarrierB=";
//for( int i : viCarrierB ) cout<<" "<<i;
//cout<<endl;

    // Reserve store big enough for ancestorB + new A and B if no join + any needed carriers...
    reserve( iAncestorB + 1 + ((j==0) ? 2 : 0) + nNewCarriers ); 

    ////// B. FILL IN NEW PARTS OF NEW STORE...

    //// B.1. Add existing nolo contexts to parent via extraction op...
    const KSet& ksLchild = aLchild.getKSet();
    const KSet  ksParent = KSet( aLchild.getKSet(), -getDir(opL), j==0, evJ, viCarrierA, qPrev, (j==0) ? KSet() : qPrev.at(iAncestorB).getKSet() );
    const KSet  ksRchild( ksParent, getDir(opR) );

    //// B.2. Copy store state and add parent/preterm contexts to existing non-locals via extraction operation...
    for( int i=0; i<((f==0&&j==1)?iAncestorB:(f==0&&j==0)?iLowerA:(f==1&&j==1)?iAncestorB:iAncestorB+1); i++ ) {
      Sign& s = *emplace( end() );
      if( i==iAncestorA and j==1 and qPrev[i].isDitto() and opR!='I' )            { s = Sign( ksParent, qPrev[i].getType(), qPrev[i].getSide() ); }
      else if( viCarrierP.size()>0 and i==viCarrierP.back() and evF.top()!='\0' ) { viCarrierP.pop_back(); s = Sign( KSet(aPretrm.getKSet(),getDir(evF.popTop()),qPrev[i].getKSet()), qPrev[i].getType(), qPrev[i].getSide() ); }
      else if( viCarrierA.size()>0 and i==viCarrierA.back() and evJ.top()!='\0' ) { viCarrierA.pop_back(); s = Sign( KSet(ksParent,getDir(evJ.popTop()),qPrev[i].getKSet()), qPrev[i].getType(), qPrev[i].getSide() ); }
      else                                                                        { s = qPrev[i]; }
    }

    //// B.3. Add new non-locals with contexts from parent/rchild via new extraction or G/H/V operations...
    // If no join, add A carriers followed by new lowest A...
    if( j==0 ) {
      // Add A carriers...
      tCurrP = aPretrm.getType();  tCurrA = tA;
      for( int i : viCarrierP ) if( i==-1 ) { if( STORESTATE_CHATTY ) cout<<"(adding carrierP for "<<tCurrP.getFirstNonlocal()<<" bc none above "<<iAncestorB<<")"<<endl;
                                               *emplace( end() ) = Sign( KSet( aPretrm.getKSet(), getDir(evF.popTop()) ), tCurrP.getFirstNonlocal(), S_B ); tCurrP=tCurrP.withoutFirstNolo(); }
      for( int i : viCarrierA ) if( i==-1 ) { if( STORESTATE_CHATTY ) cout<<"(adding carrierA for "<<tCurrA.getFirstNonlocal()<<" bc none above "<<iAncestorB<<")"<<endl;
                                               *emplace( end() ) = Sign( KSet( ksParent,          getDir(evJ.popTop()) ), tCurrA.getFirstNonlocal(), S_B ); tCurrA=tCurrA.withoutFirstNolo(); }
      // Add lowest A...
      *emplace( end() ) = Sign( (opR=='I') ? KSet(K_DITTO) : ksParent, tA, S_A );
      if( tA.getLastNonlocal()==N("-vN") and viCarrierA[0]==-1 ) back().setKSet().addBankedUnaryTransform( "O" );
      iLowerA = size()-1;
    }
    // Add B carriers...
    N nA = tA.getLastNonlocal();  N nB = tB.getLastNonlocal();  N nL = aLchild.getType().getLastNonlocal();
    if( nB!=N_NONE && nB!=nA && viCarrierB[0]==-1 ) {  if( STORESTATE_CHATTY ) cout<<"(adding carrierB for "<<nB<<" bc none above "<<iAncestorB<<") (G/R rule)"<<endl;
                                                       *emplace( end() ) = Sign( ksLchild, nB, S_A ); }                            // Add left child kset as A carrier (G rule).
    // WS: SUPPOSED TO BE FOR C-rN EXTRAPOSITION, BUT DOESN'T QUITE WORK...
    // if( nL!=N_NONE && iCarrierL>iAncestorB )    if( STORESTATE_CHATTY ) cout<<"(adding carrierL for "<<nL<<" bc none above "<<iLowerA<<" and below "<<iAncestorB<<")"<<endl;
    // if( nL!=N_NONE && iCarrierL>iAncestorB )    *emplace( end() ) = Sign( qPrev[iCarrierL].getKSet(), nL, S_A );            // Add right child kset as L carrier (H rule).
    // For B, if left child nolo cat not listed in apex carrier signs (H rule)...
    if( nL!=N_NONE and nL!=nA and nL!=N("-vN") ) {
      if( STORESTATE_CHATTY ) cout<<"(attaching carrierL for "<<nL<<" above "<<iLowerA<<" and below "<<iAncestorB<<") (H rule)"<<endl;
      // If no join, use lowest left-child carrier as right...
      // CODE REVIEW: should not keep lowest left carrier if using H rule.
      if( j==0 ) {
        // Redefine viCarrier on current time step...
        tCurrL=aLchild.getType();
        viCarrierL.clear();
        for( int i=iLowerA-1; i>=-1; i-- ) {
          T tI = at(i).getType();
          N nL=tCurrL.getLastNonlocal(); if( i>-1 && nL!=N_NONE && tI==nL                                     ) { viCarrierL.push_back(i);  tCurrL=tCurrL.withoutLastNolo(); }
                                         if(         nL!=N_NONE && !tI.isCarrier() && !tI.containsCarrier(nL) ) { viCarrierL.push_back(-1); tCurrL=tCurrL.withoutLastNolo(); }
        }
//cout<<" viCarrierL=";
//for( int& i : viCarrierL ) cout<<" "<<i;
//cout<<endl;
        if( viCarrierL[0]>iAncestorB )              *emplace( end() ) = Sign( at(viCarrierL[0]).getKSet(),                                                        ksRchild, tB, S_B );  // Add right child kset as B (H rule).
        else cerr<<"ERROR StoreState 784: should not happen, on '"<<qPrev<<" "<<f<<" "<<j<<" "<<evF<<" "<<evJ<<" "<<opL<<" "<<opR<<" "<<tA<<" "<<tB<<" "<<aPretrm<<" "<<aLchild<<"'"<<endl;
    //    cerr << "            " << qPrev << "  " << aLchild << "  ==(f" << f << ",j" << j << "," << opL << "," << opR << ")=>  " << *this << endl;
      } else {  // If j==1...
        // If existing left carrier, integrate with sign...
        if( evF.top()!='\0' and viCarrierL[0]!=-1 ) *emplace( end() ) = Sign( KSet( aPretrm.getKSet(), getDir(evF.popTop()), qPrev.at(viCarrierL[0]).getKSet() ), ksRchild, tB, S_B );
        // If new left carrier...
        else if(evF.top()!='\0' )                   *emplace( end() ) = Sign( KSet( aPretrm.getKSet(), getDir(evF.popTop()) ),                                    ksRchild, tB, S_B );
        // If no extraction...
        else                                        *emplace( end() ) = Sign( qPrev.at(viCarrierL[0]).getKSet(),                                                  ksRchild, tB, S_B );
      }
    }
    // Add lowest B...
    else if( size()>0 )            { *emplace( end() ) = Sign( ksRchild, tB, S_B ); }
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

  list<FPredictor>& calcForkPredictors ( list<FPredictor>& lfp, const KSet& ksAnt, bool bAdd=true ) const {
    int d = (FEATCONFIG & 1) ? 0 : getDepth(); // max used depth - (dbar)
    const KSet& ksB = at(size()-1).getKSet(); //contexts of lowest b (bdbar)
    int iCarrier = getAncestorBCarrierIndex( 1 ); // get lowest nonlocal above bdbar
    if( STORESTATE_TYPE ) lfp.emplace_back( d, at(size()-1).getType().removeLink() ); // flag to add depth and category label as predictor, default is true
    if( !(FEATCONFIG & 2) ) {
      for( auto& kA : (ksB.size()==0) ? ksBot  : ksB                    ) if( bAdd || FPredictor::exists(d,kNil,kA,kNil) ) lfp.emplace_back( d, kNil, kA, kNil ); 
      //for( auto& kA : (ksB.size()==0) ? ksBot  : ksB                    ) if( bAdd || FPredictor::exists(d,kNil,kA,kNil) ) lfp.emplace_back( d, kNil, kA, kNil ); //ej proposed change to add coreference
      for( auto& kF : (iCarrier<0)    ? KSet() : at(iCarrier).getKSet() ) if( bAdd || FPredictor::exists(d,kF,kNil,kNil) ) lfp.emplace_back( d, kF, kNil, kNil ); 
      //for( auto& kF : (iCarrier<0)    ? KSet() : at(iCarrier).getKSet() ) if( bAdd || FPredictor::exists(d,kF,kNil,kNil) ) lfp.emplace_back( d, kF, kNil, kNil ); //ej proposed change to add coreference 
      // for (auto& kAntecedent : ksetfrombackpointers) if ( bAdd || Fpredictor::exists(d,kNil,kNil,kAntecedent) ) lfp.emplace_back( d, kNil, kNil, kAntecedent); // ej proposed change to add coreference
      for (auto& kAntecedent : ksAnt) {
          //cerr << "cfp processing ksAnt: " << kAntecedent << endl;
          if ( bAdd || FPredictor::exists(d,kNil,kNil,kAntecedent) ) {
              //cerr << "cfp adding kAntecedent: " << kAntecedent << endl;
              lfp.emplace_back( d, kNil, kNil, kAntecedent); // ej change to add coreference
          }
      }
//    } else if( FEATCONFIG & 1 ) {
//      for( auto& kA : (ksB.size()==0) ? ksBot  : ksB                    ) if( bAdd || FPredictor::exists(kNil,kA) ) lfp.emplace_back( kNil, kA );
//      for( auto& kF : (iCarrier<0)    ? KSet() : at(iCarrier).getKSet() ) if( bAdd || FPredictor::exists(kF,kNil) ) lfp.emplace_back( kF, kNil );
    }
    return lfp;
  }

  PPredictor calcPretrmTypeCondition ( F f, EVar e, K k_p_t ) const {
    if( FEATCONFIG & 1 ) return PPredictor( 0, f, (FEATCONFIG & 4) ? EVar("-") : e, at(size()-1).getType(), (FEATCONFIG & 16384) ? tBot : k_p_t.getType() );
    return             PPredictor( getDepth(), f, (FEATCONFIG & 4) ? EVar("-") : e, at(size()-1).getType(), (FEATCONFIG & 16384) ? tBot : k_p_t.getType() );
  }

  list<JPredictor>& calcJoinPredictors ( list<JPredictor>& ljp, F f, EVar eF, const LeftChildSign& aLchild, bool bAdd=true ) const {
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

  APredictor calcApexTypeCondition ( F f, J j, EVar eF, EVar eJ, O opL, const LeftChildSign& aLchild ) const {
    if( FEATCONFIG & 1 ) return APredictor( 0, 0, j, (FEATCONFIG & 64) ? EVar("-") : eJ, (FEATCONFIG & 128) ? O('-') : opL, at(getAncestorBIndex(f)).getType(), (j==0) ? aLchild.getType() : tBot );
    return         APredictor( getDepth()+f-j, f, j, (FEATCONFIG & 64) ? EVar("-") : eJ, (FEATCONFIG & 128) ? O('-') : opL, at(getAncestorBIndex(f)).getType(), (j==0) ? aLchild.getType() : tBot );
  }

  BPredictor calcBrinkTypeCondition ( F f, J j, EVar eF, EVar eJ, O opL, O opR, T tParent, const LeftChildSign& aLchild ) const {
    if( FEATCONFIG & 1 ) return  BPredictor( 0, 0, 0, (FEATCONFIG & 64) ? EVar("-") : eJ, (FEATCONFIG & 128) ? O('-') : opL, (FEATCONFIG & 128) ? O('-') : opR, tParent, aLchild.getType() );
    return          BPredictor( getDepth()+f-j, f, j, (FEATCONFIG & 64) ? EVar("-") : eJ, (FEATCONFIG & 128) ? O('-') : opL, (FEATCONFIG & 128) ? O('-') : opR, tParent, aLchild.getType() );
  }

  list<NPredictor>& calcNPredictors (list<NPredictor>& npreds, const Sign& candidate, bool bcorefON ) {
    //cerr << "entered calcNPredictors..." << endl;
    //probably will look like Join model feature generation.ancestor is a sign, sign has T and Kset.
    //TODO add unary category feats for anaphor and antecedentt - just basic label, no semantics. 
    //TODO add unary semantic feats for anaphor and antecedent - just predarg structure, no syntactic category.
    //TODO add dependence to P model.  P category should be informed by which antecedent category was chosen here.
    //cout << "calcNPredictors received candidate: " << candidate << endl;
    //cout << "candidate kset: " << candidate.getKSet() << endl;
    const KSet& ksB = at(size()-1).getKSet(); //contexts of lowest b (bdbar)
    //cout << "ksb: " << ksB << endl;
    for (auto& antk:candidate.getKSet()){ //add k x k feats
      for (auto& currk:ksB) {
        npreds.emplace_back(antk, currk);
      }
    }
    npreds.emplace_front(bias); //add bias term

    //corefON feature
    if (bcorefON == true) { 
      npreds.emplace_back(corefON);
      //cerr << "corefON added to npreds..." << endl;
      //for (auto& npred : npreds) {cerr << npred << " ";}
      //cerr << endl;
    }

    return npreds;
  }
};
const Sign StoreState::aTop( KSet(K::kTop), tTop, S_B );

////////////////////////////////////////////////////////////////////////////////

KSet::KSet ( const KSet& ksToProject, int iProj, bool bUp, EVar e, const vector<int>& viCarrierIndices, const StoreState& ss, const KSet& ksNoProject ) {
//cout<<"tomake "<<ks<<" iProj="<<iProj<<" bup="<<bUp<<" e="<<e<<" "<<viCarrierIndices.size()<<" "<<ss<<" "<<ksNoProject<<endl;
  // Determine number of carrier contexts...
  int nCarrierContexts=0;  for( int iCarrierIndex : viCarrierIndices ) if( iCarrierIndex>=0 ) nCarrierContexts += ss.at(iCarrierIndex).getKSet().size();
  // Reserve size to avoid costly reallocation.
  reserve( ksToProject.size() + nCarrierContexts + ksNoProject.size() );

//cout<<"made kTo="<<ksToProject<<" with bank="<<ksToProject.eBankedUnaryTransforms<<" iProj="<<iProj<<" bUp="<<bUp<<" e="<<e<<" ksNo="<<ksNoProject;

  // If going down (join)...
  if( not bUp ) {
    // Add untransformed (parent if up, nothing if down) contexts...
    insert( end(), ksNoProject.begin(), ksNoProject.end() );
    // Build parent/pretrm top to bottom...
    for( uint i=0; e!=EVar::eNil; e=e.withoutTop() ) {
      // For extractions...
      if( e.top()>='0' and e.top()<='9' and i<viCarrierIndices.size() and viCarrierIndices[i]!=-1 ) {
        for( const K& k : ss.at(viCarrierIndices[i++]).getKSet() ) if( k.project(-getDir(e.top()))!=K::kBot ) push_back( k.project(-getDir(e.top())) );
      }
      // For reorderings...
      else if( e.top()=='O' or e.top()=='V' ) {
        for(       K& k : *this                                  ) k = k.transform(bUp,e.top());
        //if( eBankedUnaryTransforms!=EVar() ) cerr << "ERROR StoreState:869: " << eBankedUnaryTransforms << " piled with " << e.top() << endl;
        eBankedUnaryTransforms = "O"; // e.top();
      }
    }
  }

  // If swap op is banked, swap participants...
  if     ( eBankedUnaryTransforms==EVar("O") and iProj==-1 ) iProj=-2;
  else if( eBankedUnaryTransforms==EVar("O") and iProj==-2 ) iProj=-1;
  // If projection source swap op is banked, swap participant label...
  if     ( ksToProject.eBankedUnaryTransforms==EVar("O") and iProj==1 ) iProj=2;
  else if( ksToProject.eBankedUnaryTransforms==EVar("O") and iProj==2 ) iProj=1;
  // Add projected (child if up, parent if down) contexts...
  for( const K& k : ksToProject ) if( k.project(iProj)!=K::kBot ) push_back( k.project(iProj) );

  // If going up (no join)...
  if( bUp ) {
    // Build parent/pretrm bottom to top...
    for( int i=viCarrierIndices.size()-1; e!=EVar::eNil; e=e.withoutBot() ) {
      // For extractions...
      if( e.bot()>='0' and e.bot()<='9' and i>=0 and viCarrierIndices[i]!=-1 ) {
        for( const K& k : ss.at(viCarrierIndices[i--]).getKSet() ) if( k.project(-getDir(e.bot()))!=K::kBot ) push_back( k.project(-getDir(e.bot())) );
      }
      // For reorderings...
      else if( e.bot()=='O' or e.bot()=='V' ) {
        for(       K& k : *this                                  ) k = k.transform(bUp,e.bot());
        //if( eBankedUnaryTransforms!=EVar() ) cerr << "ERROR StoreState:859: " << eBankedUnaryTransforms << " piled with " << e.bot() << endl;
        eBankedUnaryTransforms = "O"; // e.bot();
      }
    }
    // Add untransformed (parent if down/join, nothing if up/no-join) contexts...
    insert( end(), ksNoProject.begin(), ksNoProject.end() );
  }

//cout<<" to get "<<*this<<" with iProj="<<iProj<<" banked="<<getBankedUnaryTransform()<<endl;
}

////////////////////////////////////////////////////////////////////////////////

LeftChildSign::LeftChildSign ( const StoreState& qPrev, F f, EVar eF, const Sign& aPretrm ) {
//    int         iCarrierB  = qPrev.getAncestorBCarrierIndex( 1 );
    int         iAncestorB = qPrev.getAncestorBIndex(f);
    T           tCurrB = qPrev.at(iAncestorB).getType();
    vector<int> viCarrierB;  viCarrierB.reserve(4);
    for( int i=qPrev.size(); i>=0; i-- ) {
      N nB=tCurrB.getLastNonlocal(); if( i<iAncestorB && nB!=N_NONE && qPrev[i].getType()==nB                  ) { viCarrierB.push_back(i);  tCurrB=tCurrB.withoutLastNolo(); }
                                     if( i<iAncestorB && nB!=N_NONE && !qPrev[i].getType().isCarrier() && !qPrev[i].getType().containsCarrier(nB) ) { viCarrierB.push_back(-1); tCurrB=tCurrB.withoutLastNolo(); }
    }
//cout<<" viCarrierB=";
//for( int i : viCarrierB ) cout<<" "<<i;
//cout<<endl;
    const Sign& aAncestorA = qPrev.at( qPrev.getAncestorAIndex(1) );
    const Sign& aAncestorB = qPrev.at( qPrev.getAncestorBIndex(1) );
//    const KSet& ksExtrtn   = (iCarrierB<0) ? KSet() : qPrev.at(iCarrierB).getKSet();
    *this = (f==1 && eF!=EVar::eNil)                  ? Sign( KSet(KSet(),0,true,eF,viCarrierB,qPrev,aPretrm.getKSet()), aPretrm.getType(), S_A )
                                                  //Sign( KSet(ksExtrtn,-getDir(eF),aPretrm.getKSet()), aPretrm.getType(), S_A )
          : (f==1)                                    ? aPretrm                             // if fork, lchild is preterm.
          : (qPrev.size()<=0)                         ? StoreState::aTop                    // if no fork and stack empty, lchild is T (NOTE: should not happen).
          : (!aAncestorA.isDitto() && eF!=EVar::eNil) ? Sign( KSet(KSet(),0,true,eF,viCarrierB,qPrev,aAncestorA.getKSet()), aAncestorA.getType(), S_A )
                                                  //Sign( KSet(ksExtrtn,-getDir(eF),aAncestorA.getKSet()), aAncestorA.getType(), S_A )
          : (!aAncestorA.isDitto())                   ? aAncestorA                          // if no fork and stack exists and last apex context set is not ditto, return last apex.
          :                                             Sign( KSet(KSet(),0,true,eF,viCarrierB,qPrev,aAncestorB.getKSet()), aPretrm.getKSet(), aAncestorA.getType(), S_A );
                                                  //Sign( KSet(ksExtrtn,-getDir(eF),aAncestorB.getKSet()), aPretrm.getKSet(), aAncestorA.getType(), S_A );  // otherwise make new context set.
  if( aPretrm.getType().getLastNonlocal()==N("-vN") and viCarrierB[0]==-1 ) setKSet().addBankedUnaryTransform( "O" );
//cout<<"lchild created with banked "<<getKSet().getBankedUnaryTransform()<<endl;
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

////////////////////////////////////////////////////////////////////////////////

char psSpaceF[]       = " f";
char psAmpersand[]    = "&";

class HiddState : public DelimitedSept<psX,Sign,psSpaceF,F,psAmpersand,EVar,psAmpersand,K,psSpace,JResponse,psSpace,StoreState,psSpace,Delimited<int>,psX> {
public:
  HiddState ( )                                                                    : DelimitedSept<psX,Sign,psSpaceF,F,psAmpersand,EVar,psAmpersand,K,psSpace,JResponse,psSpace,StoreState,psSpace,Delimited<int>,psX>()             { }
  HiddState ( const Sign& a, F f, EVar e, K k, JResponse jr, const StoreState& q , int i=0 ) : DelimitedSept<psX,Sign,psSpaceF,F,psAmpersand,EVar,psAmpersand,K,psSpace,JResponse,psSpace,StoreState,psSpace,Delimited<int>,psX>(a,f,e,k,jr,q,i) { }
  const Sign& getPrtrm ()     { return first(); }
  F getF ()                   { return second(); }
  EVar getForkE ()            { return third(); }
  K getForkK ()               { return fourth(); }
  const JResponse& getJResp() { return fifth(); }
  const StoreState& getStoreState() { return sixth(); }
};

////////////////////////////////////////////////////////////////////////////////

