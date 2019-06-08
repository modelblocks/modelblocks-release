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

typedef Delimited<int>  D;  // depth
typedef Delimited<int>  F;  // fork decision
typedef Delimited<int>  J;  // join decision
typedef Delimited<char> O;  // composition operation
const O O_N("N");
const O O_I("."); //(".");
typedef Delimited<char> S;  // side (A,B)
const S S_A("/");
const S S_B(";");

DiscreteDomain<int> domAdHoc;
typedef Delimited<DiscreteDomainRV<int,domAdHoc>> AdHocFeature;
const AdHocFeature corefON("acorefON");
const AdHocFeature corefOFF("acorefOFF");
const AdHocFeature bias("abias");

////////////////////////////////////////////////////////////////////////////////

int getDir ( O cOp ) {
  return (cOp>='0' && cOp<='9')             ? cOp-'0' :  // (numbered argument)
         (cOp=='M' || cOp=='U')             ? -1      :  // (modifier)
         (cOp=='u')                         ? -2      :  // (auxiliary w arity 2)
         (cOp==O_I || cOp=='C' || cOp=='V') ? 0       :  // (identity)
                                              -10;       // (will not map)
}

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

class CVar;
typedef CVar N;

////////////////////////////////////////////////////////////////////////////////

DiscreteDomain<int> domCVar;
class CVar : public DiscreteDomainRV<int,domCVar> {
 private:
  static map<N,bool>               mnbArg;
  static map<CVar,uint>            mciSynArgs;
  static map<CVar,int>             mciArity;
  static map<CVar,int>             mciNoloArity;
  static map<CVar,bool>            mcbIsCarry;
  static map<CVar,N>               mcnFirstNol;
  static map<CVar,CVar>            mccNoFirstNol;
  static map<CVar,N>               mcnLastNol;
  static map<CVar,CVar>            mccNoLastNol;
  static map<pair<CVar,N>,bool>    mcnbIn;
  static map<CVar,CVar>            mccLets;
  static map<CVar,int>             mciNums;
  static map<pair<CVar,int>,CVar>  mcicLetNum;
  uint getSynArgs ( const char* l ) {
    int depth = 0;
    int ctr   = 0;
    for ( uint i=0; i<strlen(l); i++ ) {
      if ( l[i]=='{' ) depth++;
      if ( l[i]=='}' ) depth--;
      if ( l[i]=='-' && l[i+1]>='a' && l[i+1]<='b' && depth==0 ) ctr++;
    }
    return ctr;
  }
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
  int getNoloArity ( const char* l ) {
    int depth = 0;
    int ctr   = 0;
    for ( uint i=0; i<strlen(l); i++ ) {
      if ( l[i]=='{' ) depth++;
      if ( l[i]=='}' ) depth--;
      if ( l[i]=='-' && (l[i+1]=='g' or l[i+1]=='h' or l[i+1]=='i' or l[i+1]=='r' or l[i+1]=='v') && depth==0 ) ctr++;
    }
    return ctr;
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
  CVar getNoFirstNoloHelper ( const char* l ) {
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
    return CVar( (string(l,0,beg)+string(l,end,strlen(l)-end)).c_str() );  // l+strlen(l);
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
  CVar getNoLastNoloHelper ( const char* l ) {
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
    return CVar( string(l,0,beg).c_str() );  // l+strlen(l);
  }
  void calcDetermModels ( const char* ps ) {
    if( mnbArg.end()==mnbArg.find(*this) ) { mnbArg[*this]=( strlen(ps)<=4 ); }
    if( mciSynArgs.end()==mciSynArgs.find(*this) ) { mciSynArgs[*this]=getSynArgs(ps); }
    if( mciArity.end()==mciArity.find(*this) ) { mciArity[*this]=getArity(ps); }
    if( mciNoloArity.end()==mciNoloArity.find(*this) ) { mciNoloArity[*this]=getNoloArity(ps); }
    if( mcbIsCarry.end()==mcbIsCarry.find(*this) ) { mcbIsCarry[*this]=( ps[0]=='-' && ps[1]>='a' && ps[1]<='z' ); }  //( ps[strlen(ps)-1]=='^' ); }
    if( mcnFirstNol.end()==mcnFirstNol.find(*this) && strlen(ps)>0 && !(ps[0]=='-'&&ps[1]>='a'&&ps[1]<='z') ) { N& n=mcnFirstNol[*this]; n=getFirstNolo(ps); }
    if( mccNoFirstNol.end()==mccNoFirstNol.find(*this) ) { CVar& c=mccNoFirstNol[*this]; c=getNoFirstNoloHelper(ps); }
    if( mcnLastNol.end()==mcnLastNol.find(*this) && strlen(ps)>0 && !(ps[0]=='-'&&ps[1]>='a'&&ps[1]<='z') ) { N& n=mcnLastNol[*this]; n=getLastNolo(ps); }
    if( mccNoLastNol.end()==mccNoLastNol.find(*this) ) { CVar& c=mccNoLastNol[*this]; c=getNoLastNoloHelper(ps); }
    if( mccLets.end()==mccLets.find(*this) ) { const char* ps_=strchr(ps,'_');
                                               if( ps_!=NULL ) { mccLets[*this] = string(ps,0,ps_-ps).c_str(); mciNums[*this] = atoi(ps_+1);
                                                                 mcicLetNum[pair<CVar,int>(mccLets[*this],mciNums[*this])]=*this; } }
                                               //else { mccLets[*this]=*this; mciNums[*this]=0; mcicLetNum[pair<CVar,int>(*this,0)]=*this; } }
    uint depth = 0;  uint beg = strlen(ps);
    for( uint i=0; i<strlen(ps); i++ ) {
      if ( ps[i]=='{' ) depth++;
      if ( ps[i]=='}' ) depth--;
      if ( depth==0 && ps[i]=='-' && (ps[i+1]=='g' || ps[i+1]=='h'    // || ps[i+1]=='i'
                                                                         || ps[i+1]=='r' || ps[i+1]=='v') ) beg = i;
      if ( depth==0 && beg>0 && beg<i && (ps[i+1]=='-' || ps[i+1]=='_' || ps[i+1]=='\\' || ps[i+1]=='^' || ps[i+1]=='\0') ) {
        // cerr<<"i think "<<string(ps,beg,i+1-beg)<<" is in "<<ps<<endl;
        mcnbIn[pair<CVar,N>(*this,string(ps,beg,i+1-beg).c_str())]=true;
        beg = strlen(ps);
      }
    }
  }
 public:
  CVar ( )                : DiscreteDomainRV<int,domCVar> ( )    { }
  CVar ( const char* ps ) : DiscreteDomainRV<int,domCVar> ( ps ) { calcDetermModels(ps); }
  bool isArg            ( )       const { return mnbArg[*this]; }
  uint getSynArgs       ( )       const { return mciSynArgs[*this]; }
  int  getArity         ( )       const { return mciArity[*this]; }
  int  getNoloArity     ( )       const { return mciNoloArity[*this]; }
  bool isCarrier        ( )       const { return mcbIsCarry[*this]; }
  N    getFirstNonlocal ( )       const { return mcnFirstNol[*this]; }
  CVar withoutFirstNolo ( )       const { return mccNoFirstNol[*this]; }
  N    getLastNonlocal  ( )       const { return mcnLastNol[*this]; }
  CVar withoutLastNolo  ( )       const { return mccNoLastNol[*this]; }
  bool containsCarrier  ( N n )   const { return mcnbIn.find(pair<CVar,N>(*this,n))!=mcnbIn.end(); }
  CVar getLets          ( )       const { const auto& x = mccLets.find(*this); return (x==mccLets.end()) ? *this : x->second; }
  int  getNums          ( )       const { const auto& x = mciNums.find(*this); return (x==mciNums.end()) ? 0 : x->second; }
  CVar addNum           ( int i ) const { const auto& x = mcicLetNum.find(pair<CVar,int>(*this,i)); return (x==mcicLetNum.end()) ? *this : x->second; }
};
map<N,bool>               CVar::mnbArg;
map<CVar,uint>            CVar::mciSynArgs;
map<CVar,int>             CVar::mciArity;
map<CVar,int>             CVar::mciNoloArity;
map<CVar,bool>            CVar::mcbIsCarry;
map<CVar,N>               CVar::mcnLastNol;
map<CVar,CVar>            CVar::mccNoLastNol;
map<CVar,N>               CVar::mcnFirstNol;
map<CVar,CVar>            CVar::mccNoFirstNol;
map<pair<CVar,N>,bool>    CVar::mcnbIn;
map<CVar,CVar>            CVar::mccLets;
map<CVar,int>             CVar::mciNums;
map<pair<CVar,int>,CVar>  CVar::mcicLetNum;
const CVar cTop("T");
const CVar cBot("-");
const CVar cBOT("bot");  // not sure if this really needs to be distinct from cBot
const CVar cFail("FAIL");
const CVar cNone("NONE");
const N N_NONE("");

////////////////////////////////////////////////////////////////////////////////

DiscreteDomain<int> domX;
typedef DiscreteDomainRV<int,domX> XVar;
const XVar xBot("Bot");

////////////////////////////////////////////////////////////////////////////////

DiscreteDomain<int> domK;
class K : public DiscreteDomainRV<int,domK> {   // NOTE: can't be subclass of Delimited<...> or string-argument constructor of this class won't get called!
 public:
  static const K kTop;
  static const K kBot;
  private:
  static map<K,CVar> mkc;
  static map<pair<K,int>,K> mkik;
  static map<K,XVar> mkx;
  static map<XVar,K> mxk;
  static map<K,int> mkdir;
//  static map<K,K> mkkVU;
//  static map<K,K> mkkVD;
  static map<K,K> mkkO;
  void calcDetermModels ( const char* ps ) {
    if( strchr(ps,':')!=NULL ) {
      char cSelf = ('N'==ps[0]) ? '1' : '0';
      // Add associations to label and related K's...  (NOTE: related K's need two-step constructor so as to avoid infinite recursion!)
      if( mkc.end()==mkc.find(*this) ) mkc[*this]=CVar(string(ps,strchr(ps,':')).c_str());
      if( mkik.end()==mkik.find(pair<K,int>(*this,-4)) && strlen(ps)>2 && ps[strlen(ps)-2]!='-' ) { K& k=mkik[pair<K,int>(*this,-4)]; k=string(ps).append("-4").c_str(); }
      if( mkik.end()==mkik.find(pair<K,int>(*this,-3)) && strlen(ps)>2 && ps[strlen(ps)-2]!='-' ) { K& k=mkik[pair<K,int>(*this,-3)]; k=string(ps).append("-3").c_str(); }
      if( mkik.end()==mkik.find(pair<K,int>(*this,-2)) && strlen(ps)>2 && ps[strlen(ps)-2]!='-' ) { K& k=mkik[pair<K,int>(*this,-2)]; k=string(ps).append("-2").c_str(); }
      if( mkik.end()==mkik.find(pair<K,int>(*this,-1)) && strlen(ps)>2 && ps[strlen(ps)-2]!='-' ) { K& k=mkik[pair<K,int>(*this,-1)]; k=string(ps).append("-1").c_str(); }
      if( mkik.end()==mkik.find(pair<K,int>(*this,0))                                                     ) { K& k=mkik[pair<K,int>(*this,0)]; k=ps; }
      if( mkik.end()==mkik.find(pair<K,int>(*this,1)) && ps[strlen(ps)-2]!='-' && ps[strlen(ps)-1]==cSelf ) { K& k=mkik[pair<K,int>(*this,1)]; k=string(ps,strlen(ps)-1).append("1").c_str(); }
      if( mkik.end()==mkik.find(pair<K,int>(*this,2)) && ps[strlen(ps)-2]!='-' && ps[strlen(ps)-1]==cSelf ) { K& k=mkik[pair<K,int>(*this,2)]; k=string(ps,strlen(ps)-1).append("2").c_str(); }
      if( mkik.end()==mkik.find(pair<K,int>(*this,3)) && ps[strlen(ps)-2]!='-' && ps[strlen(ps)-1]==cSelf ) { K& k=mkik[pair<K,int>(*this,3)]; k=string(ps,strlen(ps)-1).append("3").c_str(); }
      if( mkik.end()==mkik.find(pair<K,int>(*this,4)) && ps[strlen(ps)-2]!='-' && ps[strlen(ps)-1]==cSelf ) { K& k=mkik[pair<K,int>(*this,4)]; k=string(ps,strlen(ps)-1).append("4").c_str(); }
      if( mkik.end()==mkik.find(pair<K,int>(*this,1)) && ps[strlen(ps)-2]=='-' && ps[strlen(ps)-1]=='1' ) { K& k=mkik[pair<K,int>(*this,1)]; k=string(ps,strlen(ps)-2).c_str(); }
      if( mkx.end()==mkx.find(*this) ) { const char* psU=strchr(ps,'_'); XVar x=(psU)?string(ps,psU-ps).c_str():ps; mkx[*this]=x; mkdir[*this]=(psU-ps==uint(strlen(ps)-2))?stoi(psU+1):0; if(mxk.end()==mxk.find(x)) { mxk[x]=(ps[strlen(ps)-1]=='0'||!psU)?*this:(string(ps,psU-ps)+"_0").c_str(); } }
      if( ps[strlen(ps)-1]=='0' ) { K& k =mkik[pair<K,int>(*this,1)]; k =string(ps,strlen(ps)-1).append("1").c_str();
                                    K& k2=mkik[pair<K,int>(*this,2)]; k2=string(ps,strlen(ps)-1).append("2").c_str();
                                    K& k3=mkik[pair<K,int>(*this,3)]; k3=string(ps,strlen(ps)-1).append("3").c_str();
                                    K& k4=mkik[pair<K,int>(*this,4)]; k4=string(ps,strlen(ps)-1).append("4").c_str(); }

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
    else mkc[*this] = (*this==kBot) ? cBOT : (*this==kTop) ? cTop : cBot;
  }
 public:
  K ( )                : DiscreteDomainRV<int,domK> ( )    { }
  K ( const char* ps ) : DiscreteDomainRV<int,domK> ( ps ) { calcDetermModels(ps); }
  K ( XVar x )         : DiscreteDomainRV<int,domK> ( )    { auto it = mxk.find(x); *this = (it==mxk.end()) ? kBot : it->second; }
  CVar getCat    ( )                  const { auto it = mkc.find(*this); return (it==mkc.end()) ? cBot : it->second; }
  XVar getXVar   ( )                  const { auto it = mkx.find(*this); return (it==mkx.end()) ? XVar() : it->second; }
  int  getDir    ( )                  const { auto it = mkdir.find(*this); return (it==mkdir.end()) ? 0 : it->second; }
  K    project   ( int n )            const { auto it = mkik.find(pair<K,int>(*this,n)); return (it==mkik.end()) ? kBot : it->second; }
  K    transform ( bool bUp, char c ) const { return mkkO[*this]; }
//  K transform ( bool bUp, char c ) const { return (bUp and c=='V') ? mkkVU[*this] :
//                                                  (        c=='V') ? mkkVD[*this] : kBot; }
};
map<K,CVar>        K::mkc;
map<pair<K,int>,K> K::mkik;
map<K,XVar>        K::mkx;
map<XVar,K>        K::mxk;
map<K,int>         K::mkdir;
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
  static map<EVar,int>  meiNoloDelta;
  static map<EVar,char> meoTop;
  static map<EVar,char> meoBot;
  static map<EVar,EVar> meeNoTop;
  static map<EVar,EVar> meeNoBot;
  void calcDetermModels ( const char* ps ) {       // NOTE: top is front, bot is back...
    if( meiNoloDelta.end()==meiNoloDelta.find(*this) ) { int a=0; for(uint i=0; i<strlen(ps); i++) a += (ps[i]=='O') ? 0 : (ps[i]=='V') ? -1 : 1;
                                                         meiNoloDelta[*this]=a; }
    if(       meoTop.end()==      meoTop.find(*this) and strlen(ps)>0 ) { char& c=  meoTop[*this]; c=ps[0]; }
    if(       meoBot.end()==      meoBot.find(*this) and strlen(ps)>0 ) { char& c=  meoBot[*this]; c=ps[strlen(ps)-1]; }
    if(     meeNoTop.end()==    meeNoTop.find(*this) and strlen(ps)>0 ) { EVar& e=meeNoTop[*this]; e=ps+1; }
    if(     meeNoBot.end()==    meeNoBot.find(*this) and strlen(ps)>0 ) { EVar& e=meeNoBot[*this]; e=string(ps,0,strlen(ps)-1).c_str(); }
  }
 public:
  EVar ( )                : DiscreteDomainRV<int,domE> ( )    { }
  EVar ( const char* ps ) : DiscreteDomainRV<int,domE> ( ps ) { calcDetermModels(ps); }
  int  getNoloDelta ( ) const { auto it = meiNoloDelta.find( *this ); assert( it != meiNoloDelta.end() ); return ( it!=meiNoloDelta.end() ) ? it->second : 0; }
  char top          ( ) const { auto it =       meoTop.find( *this ); assert( it !=       meoTop.end() ); return ( it!=      meoTop.end() ) ? it->second : '?'; }
  char bot          ( ) const { auto it =       meoBot.find( *this ); assert( it !=       meoBot.end() ); return ( it!=      meoBot.end() ) ? it->second : '?'; }
  EVar withoutTop   ( ) const { auto it =     meeNoTop.find( *this ); assert( it !=     meeNoTop.end() ); return ( it!=    meeNoTop.end() ) ? it->second : eNil; }
  EVar withoutBot   ( ) const { auto it =     meeNoBot.find( *this ); assert( it !=     meeNoBot.end() ); return ( it!=    meeNoBot.end() ) ? it->second : eNil; }
  char popTop       ( )       { auto it =       meoTop.find( *this ); assert( it !=       meoTop.end() ); *this = meeNoTop[*this]; return ( it!=meoTop.end() ) ? it->second : '?'; }
};
map<EVar,int>  EVar::meiNoloDelta;
map<EVar,char> EVar::meoTop;
map<EVar,char> EVar::meoBot;
map<EVar,EVar> EVar::meeNoTop;
map<EVar,EVar> EVar::meeNoBot;
const EVar EVar::eNil("");

////////////////////////////////////////////////////////////////////////////////

typedef unsigned int NResponse;
typedef unsigned int FResponse;
typedef unsigned int JResponse;

////////////////////////////////////////////////////////////////////////////////

typedef Delimited<int> D;

////////////////////////////////////////////////////////////////////////////////

#ifdef DENSE_VECTORS
#include<KVec_dense.hpp>
#else
#include<KVec_sparse.hpp>
#endif

////////////////////////////////////////////////////////////////////////////////

class HVec : public DelimitedVector<psX,KVec,psX,psX> {
 public:

  static const HVec hvDitto;

  // Constructors...
  HVec ( )                                             : DelimitedVector<psX,KVec,psX,psX>()                            {             }
  HVec ( int i )                                       : DelimitedVector<psX,KVec,psX,psX>( i )                         {             }
  HVec ( const KVec& kv )                              : DelimitedVector<psX,KVec,psX,psX>( 1 )                         { at(0) = kv; }
  HVec ( K k, const EMat& matE, const OFunc& funcO )   : DelimitedVector<psX,KVec,psX,psX>( k.getCat().getSynArgs()+1 ) {
    int dir = k.getDir();
    at(0) = (dir) ? funcO( dir, matE( k.getXVar() ) ) : matE( k.getXVar() );
    for( unsigned int arg=1; arg<k.getCat().getSynArgs()+1; arg++ )
      at(arg) = funcO( dir + arg, matE( k.getXVar() ) );  //funcO(arg, at(0));
  }
  HVec& add( const HVec& hv ) {
    for( unsigned int arg=0; arg<size() and arg<hv.size(); arg++ ) at(arg).add( hv.at(arg) );
    return *this;
  }
  HVec& addSynArg( int iDir, const HVec& hv ) {
//cout<<"trying addSynArg(" << iDir << "," << hv << ")" <<endl;
    if     ( iDir == 0                  )          add( hv );
    else if( iDir < 0 and 0 < hv.size() )          { if( -iDir >= int(size()) ) resize( -iDir + 1 );
                                                     at( -iDir ).add( hv.at( 0  ) ); }
    else if( iDir >= 0 and iDir < int(hv.size()) ) { if( 0 >= size() ) resize( 1 );
                                                     at( 0   ).add( hv.at(iDir) ); }
//    else cout<<"i failed."<< endl;
    return *this;
  }
  HVec& swap( int i, int j ) {
    if     ( size() >= 3 ) { auto kv = at(i);  at(i) = at(j);  at(j) = kv; }
    else if( size() >= 2 ) at(i) = KVec();
    return *this;
  }
#ifndef SIMPLE_STORE
  HVec& applyUnariesTopDn( EVar e, const vector<int>& viCarrierIndices, const StoreState& ss );
  HVec& applyUnariesBotUp( EVar e, const vector<int>& viCarrierIndices, const StoreState& ss );
#endif
  bool isDitto ( ) const { return ( *this == hvDitto ); }
};

const HVec hvTop( kvTop );
const HVec hvBot( kvBot );
const HVec HVec::hvDitto( kvDitto );

////////////////////////////////////////////////////////////////////////////////

class Sign : public DelimitedTrip<psX,HVec,psColon,CVar,psX,S,psX> {
 public:
  Sign ( )                              : DelimitedTrip<psX,HVec,psColon,CVar,psX,S,psX> ( )           { third()=S_A; }
  Sign ( const HVec& hv1, CVar c, S s ) : DelimitedTrip<psX,HVec,psColon,CVar,psX,S,psX> ( hv1, c, s ) { }
  HVec&       setHVec ( )       { return first();  }
  CVar&       setCat  ( )       { return second(); }
  S&          setSide ( )       { return third();  }
  const HVec& getHVec ( ) const { return first();  }
  CVar        getCat  ( ) const { return second(); } //.removeLink(); }
  S           getSide ( ) const { return third();  }
  bool        isDitto ( ) const { return getHVec().isDitto(); }
};
const Sign aTop( hvTop, cTop, S_A );
const Sign bTop( hvTop, cTop, S_B );

////////////////////////////////////////////////////////////////////////////////

#ifdef SIMPLE_STORE

class LeftChildSign : public Sign {
 public:
  LeftChildSign ( ) : Sign() { }
  LeftChildSign ( const Sign& a ) : Sign(a) { }
 };

////////////////////////////////////////////////////////////////////////////////

class SignWithCarriers : public DelimitedVector<psX,Sign,psX,psX> {
 public:
  Sign&       back ( unsigned int i = 0 )       { assert( size() > i );  return at( size() - 1 - i ); }
  const Sign& back ( unsigned int i = 0 ) const { return ( size() > i ) ? at( size() - 1 - i ) : aTop; }
};

class ApexWithCarriers : public SignWithCarriers {
 public:
  void set ( CVar cB, CVar cA, O opL, O opR, const Sign& aLchild = Sign() );
  Sign&       back ( unsigned int i = 0 )       { assert( size() > i );  return at( size() - 1 - i ); }
  const Sign& back ( unsigned int i = 0 ) const { return ( size() > i ) ? at( size() - 1 - i ) : aTop; }
};
ApexWithCarriers awcDummy;

class BaseWithCarriers : public SignWithCarriers {
 public:
  void set ( CVar cP, CVar cB, O opL, O opR, StoreState& ss, const SignWithCarriers& swcParent, const ApexWithCarriers& awcLchild );
  Sign&       back ( unsigned int i = 0 )       { assert( size() > i );  return at( size() - 1 - i ); }
  const Sign& back ( unsigned int i = 0 ) const { return ( size() > i ) ? at( size() - 1 - i ) : bTop; }
};

////////////////////////////////////////////////////////////////////////////////

class DerivationFragment : public DelimitedPair<psX,ApexWithCarriers,psX,BaseWithCarriers,psX> {
 public:
  ApexWithCarriers&       apex( )       { return pair<ApexWithCarriers,BaseWithCarriers>::first;  }
  BaseWithCarriers&       base( )       { return pair<ApexWithCarriers,BaseWithCarriers>::second; }
  const ApexWithCarriers& apex( ) const { return pair<ApexWithCarriers,BaseWithCarriers>::first;  }
  const BaseWithCarriers& base( ) const { return pair<ApexWithCarriers,BaseWithCarriers>::second; }
};
const DerivationFragment dfTop;

////////////////////////////////////////////////////////////////////////////////

class StoreState : public DelimitedVector<psX,DerivationFragment,psX,psX> {
 public:

  static       Sign aDummy;  // for set, to compile

  StoreState ( ) : DelimitedVector<psX,DerivationFragment,psX,psX> ( ) { } 

  StoreState ( CVar cA, CVar cB ) { emplace( end() ); back().apex().emplace_back( hvBot, cA, S_A ); back().base().emplace_back( hvBot, cB, S_B ); }

  // Preterminal constructor...
  StoreState ( const StoreState& qPrev, const HVec& hvAnt, EVar evF, K k, CVar cP, const EMat& matE, const OFunc& funcO ) {
    // Add preterm and apply unaries...
    reserve( qPrev.size() + 1 );
    insert( end(), qPrev.begin(), qPrev.end() );
    emplace( end() )->apex().set( getBase().getCat(), cP, 'I', 'I', Sign() );
    back().apex().back().setHVec() = HVec( k, matE, funcO );
    back().apex().back().setHVec().add( hvAnt );
    applyUnariesBotUp( back().apex(), evF );
  }

  // Constructor for terminal phase in left-corner parser...
  StoreState ( const StoreState& qPrev, F f ) {
    if( f==0 ) {
      // Close most recent derivation fragment and make new complete sign (lowest apex)...
      reserve( qPrev.size()-1 );
      insert( end(), qPrev.begin(), qPrev.end()-2 );
      emplace( end() )->apex().set( getBase().getCat(), qPrev.back(1).apex().back().getCat(), 'N', 'N', Sign() );
      if( qPrev.getApex(1).isDitto() ) back().apex().back().setHVec() = qPrev.back(1).base().back().getHVec();
      else                             back().apex().back().setHVec() = qPrev.back(1).apex().back().getHVec();
    }
    else {
      // Leave preterm as complete sign...
      *this = qPrev;
    }
  }

  // Constructor for nonterminal phase in left-corner parser...
  StoreState ( const StoreState& qPrev, J j, EVar evJ, O opL, O opR, CVar cA, CVar cB ) {
    if( j==1 and qPrev.getDepth()>1 ) {
      // Grow prev derivation fragment downward to subsume new apex...
      reserve( qPrev.size()-1 );
      if( qPrev.size() >= 2 ) insert( end(), qPrev.begin(), qPrev.end()-2 );
      emplace( end() )->apex() = qPrev.back(1).apex();

      // Create intermediate base parent...
      BaseWithCarriers bwcParent = qPrev.back(1).base();  if( evJ != EVar::eNil and evJ.bot()=='V' ) bwcParent.emplace( bwcParent.end()-1 );
      applyUnariesTopDn( bwcParent, evJ );                                                                   // Calc parent contexts (below unaries).
      if( (opL>='1' and opL<='9') or (opR>='1' and opR<='9') ) bwcParent.back().setHVec().emplace_back();    // Add space for satisfied argument. 
      if( getDir(opL)!=-10 ) bwcParent.back().setHVec().addSynArg( -getDir(opL), qPrev.getApex().getHVec() );

      // Create right child base...
      back().base() = bwcParent;  back().base().pop_back();
      back().base().set( bwcParent.back().getCat(), cB, opL, opR, *this, bwcParent, qPrev.back().apex() );
      if( getApex().isDitto() and opR!=O_I ) setApex().setHVec() = bwcParent.back().getHVec();               // If base != apex, end ditto.
    }
    else if( j==0 ) {
      // Grow apex upward into new derivation fragment...
      reserve( qPrev.size() );
      insert( end(), qPrev.begin(), qPrev.end()-1 );
      emplace( end() );                                                                                      // Add depth level d.

      // Create new apex...
      unsigned int iSubtracted = ( opL=='R' or opR=='H' or opR=='N' ) ? 1 : 0;   // Subtract newest nolos that are discharged on way up in current branch.
      if( qPrev.back().apex().size() > iSubtracted ) back().apex().insert( back().apex().end(), qPrev.back().apex().begin(), qPrev.back().apex().end() - 1 - iSubtracted );                // Add nolos from left child as older.
      back().apex().set( getBase().getCat(), cA, opL, opR, qPrev.getApex() );                                                 // Fill in apex at d.
      applyUnariesBotUp( back().apex(), evJ );                   // Calc apex contexts.

      // Create right child base...
      back().base().set( cA, cB, opL, opR, *this, back().apex(), qPrev.back().apex() );                      // Fill in base at d.
      if( opR==O_I ) setApex().setHVec() = HVec::hvDitto;                                                    // Init ditto.
    }
  }

  // Extraction method for depth...
  unsigned int getDepth( ) const { return size(); }

  // Bounds-checking extraction methods for derivation fragments...
  DerivationFragment&       back ( unsigned int i = 0 )       { assert( size() > i );  return at( size() - 1 - i ); }
  const DerivationFragment& back ( unsigned int i = 0 ) const { return ( size() > i ) ? at( size() - 1 - i ) : dfTop; }

  // Bounds-checking extraction methods for apex and base signs...
  Sign&       setApex ( unsigned int iDepthBack = 0 )       {  return back( iDepthBack ).apex().back();                             }
  const Sign& getApex ( unsigned int iDepthBack = 0 ) const {  return back( iDepthBack ).apex().back();                             }
  const Sign& getBase ( unsigned int iDepthBack = 0 ) const {  return back( iDepthBack + (back().base().size()==0) ).base().back(); }

  // Specification and extraction methods for nonlocal dependencies...
  Sign& setNoloBack ( unsigned int iCarrBack = 0, SignWithCarriers& awc = awcDummy ) {                 // NOTE: getNoloBack(0) is most recent nonlocal dep; i.e. furthest left.
    for( int i = int(awc.size())-1; i-->0; )  if( iCarrBack-- == 0 ) return awc.at(i);
    // Count down from bot...   // Count back from end...                   // Decrement counter and if finished, report...
    for( int d=size(); d--; ) { for( int i=at(d).base().size()-1; i-->0; )  if( iCarrBack-- == 0 ) return( at(d).base().at(i) );
                                for( int i=at(d).apex().size()-1; i-->0; )  if( iCarrBack-- == 0 ) return( at(d).apex().at(i) ); }
    cout << "FAILING: " << *this << " " << iCarrBack << " " << awc << endl;
    assert( false );
    return( aDummy );
  }
  const Sign& getNoloBack ( int iCarrBack = 0, const SignWithCarriers& awc = SignWithCarriers() ) const {     // NOTE: getNoloBack(0) is most recent nonlocal dep; i.e. furthest left.
    for( int i = int(awc.size())-1; i-->0; )  if( iCarrBack-- == 0 ) return awc.at(i);

    // Count down from bot...   // Count back from end...                        // Decrement counter and if finished, report...
    int D = ( back().base().size()==0 ) ? size() - 1 : size();
    for( int d=D; d--; )      { for( int i=int(at(d).base().size())-1; i-->0; )  if( iCarrBack-- == 0 ) return( at(d).base().at(i) );
                                for( int i=int(at(d).apex().size())-1; i-->0; )  if( iCarrBack-- == 0 ) return( at(d).apex().at(i) ); }
    return( aTop );
  }

  // Specification methods for unary operations...
  void applyUnariesBotUp( ApexWithCarriers& awc, EVar e ) {                    // From bottom up, extract least recent nolos first...
    HVec hvTemp;
    for( unsigned int iBack = e.getNoloDelta()-1; e != EVar::eNil; e = e.withoutBot() ) {
      if(      e.bot() == 'O' )   awc.back().setHVec().swap( 1, 2 );
      else if( e.bot() >= '0' and e.bot() <= '9' and  e.withoutBot() != EVar::eNil and e.withoutBot().top() == 'V' )
                                { hvTemp = HVec(1);  hvTemp.addSynArg( getDir(e.bot()), awc.back().getHVec() );
                                  e = e.withoutTop();  awc.back().setHVec().addSynArg( -1, hvTemp ); }
      else if( e.bot() == 'V' ) { awc.back().setHVec().addSynArg( -1, getNoloBack(0,awc).getHVec() ); } //awc.erase(awc.end()-1); }
      else                      { awc.back().setHVec().addSynArg( -getDir(e.bot()), getNoloBack(iBack,awc).getHVec() );
                                  setNoloBack(iBack--,awc).setHVec().addSynArg( getDir(e.bot()), awc.back().getHVec() ); }
    }
  }
  void applyUnariesTopDn( BaseWithCarriers& bwc, EVar e ) {
    for( unsigned int iBack = (e != EVar::eNil and e.bot()=='V') ? 1 : 0; e != EVar::eNil; e = e.withoutTop() ) {       // From top down, extract most recent nolos first...
      HVec hvTemp;
      if(      e.top() == 'O' )   bwc.back().setHVec().swap( 1, 2 );
      else if( e.top() == 'V' and e.withoutTop() != EVar::eNil and e.withoutTop().top() >= '0' and e.withoutTop().top() <= '9' )
                                { hvTemp = HVec( 1 );  hvTemp.addSynArg( 1, bwc.back().getHVec() );  bwc.back().setHVec().at( 1 ) = KVec();
                                  e = e.withoutTop();  bwc.back().setHVec().addSynArg( -getDir(e.top()), hvTemp ); }
      else if( e.top() == 'V' ) { setNoloBack(0,bwc).setHVec().addSynArg( 1, bwc.back().getHVec() );  bwc.back().setHVec().at(1) = KVec(); }
      else                      { bwc.back().setHVec().addSynArg( -getDir(e.top()), getNoloBack(iBack,bwc).getHVec() );
                                  setNoloBack(iBack++,bwc).setHVec().addSynArg( getDir(e.top()), bwc.back().getHVec() ); }
    }
  }
};
Sign StoreState::aDummy( hvTop, cTop, S_B );

////////////////////////////////////////////////////////////////////////////////

/*
void SignWithCarriers::setSign ( CVar cA, O opL, O opR, const Sign& aLchild ) {
  back().setHVec().resize( cA.getSynArgs() + ( ((opL>='1' and opL<='9') or (opR>='1' and opR<='9')) ? 2 : 1 ) );
  if( aLchild!=Sign() and getDir(opL)!=-10 ) back().setHVec().addSynArg( -getDir(opL), aLchild.getHVec() );  // Apply operator from lchild to parent.
}
*/

// Implementation of specifier method to allocate apex and carriers...
void ApexWithCarriers::set ( CVar cB, CVar cA, O opL, O opR, const Sign& aLchild ) {
  int iAdding = cA.getNoloArity() - cB.getNoloArity() - size();
  if( iAdding > 0 ) insert( end(), iAdding, Sign() );                                                        // Add nolos not in lchild as more recent.
//  *emplace( end() ) = Sign( HVec(), cA, S_A );  setSign( cA, opL, opR, aLchild );
  *emplace( end() ) = Sign( HVec(), cA, S_A );  back().setHVec() = HVec( cA.getSynArgs() + ( ((opL>='1' and opL<='9') or (opR>='1' and opR<='9')) ? 2 : 1 ) );

  if( aLchild!=Sign() and getDir(opL)!=-10 ) back().setHVec().addSynArg( -getDir(opL), aLchild.getHVec() );  // Apply operator from lchild to parent.
}

// Implementation of specifier method to allocate base and carriers...
void BaseWithCarriers::set ( CVar cP, CVar cB, O opL, O opR, StoreState& ss, const SignWithCarriers& swcParent, const ApexWithCarriers& awcLchild ) {
  int iAdding = cB.getNoloArity() - cP.getNoloArity() - size();
  if( iAdding > 0 ) insert( end(), iAdding, Sign() );                                                        // Add nolos not in parent as more recent.
  *emplace( end() ) = Sign( HVec(), cB, S_B );  back().setHVec() = HVec( cB.getSynArgs() + 1 );

  if( getDir(opR)!=-10 ) back().setHVec().addSynArg( getDir(opR), swcParent.back().getHVec() );              // Apply operator from parent to rchild.
  if( opL=='G' or opR=='R' ) { ss.setNoloBack( 0, *this ).setHVec() = HVec( awcLchild.back().getCat().getSynArgs() + 1 );
                               ss.setNoloBack( 0, *this ).setHVec().add( awcLchild.back().getHVec() ); }
  if( opL=='R' or opR=='H' ) back().setHVec().add( ss.getNoloBack( 0, awcLchild ).getHVec() );
  if(             opR=='I' ) { ss.setNoloBack( 0, *this ).setHVec() = HVec(1);
                               ss.setNoloBack( 0, *this ).setHVec().addSynArg( awcLchild.back().getCat().getSynArgs(), awcLchild.back().getHVec() ); }
}

#else

////////////////////////////////////////////////////////////////////////////////

class LeftChildSign : public Sign {
 public:
  LeftChildSign ( ) : Sign() { }
  LeftChildSign ( const Sign& a ) : Sign(a) { }
  LeftChildSign ( const StoreState& qPrev, F f, EVar eF, const Sign& aPretrm );
};

////////////////////////////////////////////////////////////////////////////////

class StoreState : public DelimitedVector<psX,Sign,psX,psX> {  // NOTE: format can't be read in bc of internal psX delimiter, but we don't need to.
 public:

  static const Sign aTop;

  StoreState ( ) : DelimitedVector<psX,Sign,psX,psX> ( ) { }
  StoreState ( const StoreState& qPrev, F f, J j, EVar evF, EVar evJ, O opL, O opR, CVar cA, CVar cB, const Sign& aPretrm, const LeftChildSign& aLchild ) {

    ////// A. FIND STORE LANDMARKS AND EXISTING P,A,B CARRIERS...

    //// A.1. Find reentrance points in old structure...
    int iAncestorA = qPrev.getAncestorAIndex(f);
    int iAncestorB = qPrev.getAncestorBIndex(f);
    int iLowerA    = (f==1) ? qPrev.size() : qPrev.getAncestorAIndex(1);

    //// A.2. Create vectors of carrier indices (one for each nonlocal in category, first to last)...
    CVar cCurrP=aPretrm.getCat();  vector<int> viCarrierP;  viCarrierP.reserve(4);
    CVar cCurrL=aLchild.getCat();  vector<int> viCarrierL;  viCarrierL.reserve(4);
    CVar cCurrA=cA;                vector<int> viCarrierA;  viCarrierA.reserve(4);
    CVar cCurrB=cB;                vector<int> viCarrierB;  viCarrierB.reserve(4);
    int nNewCarriers = 0;
    for( int i=qPrev.size()-1; i>=-1; i-- ) {
      CVar cI = (i>-1) ? qPrev[i].getCat() : cTop;
      N nP=cCurrP.getLastNonlocal();
      if( i>-1 and                  nP!=N_NONE && qPrev[i].getCat()==nP                      ) { viCarrierP.push_back(i);  cCurrP=cCurrP.withoutLastNolo(); }
      if(                           nP!=N_NONE && !cI.isCarrier() && !cI.containsCarrier(nP) ) { viCarrierP.push_back(-1); cCurrP=cCurrP.withoutLastNolo(); nNewCarriers++; }
      N nL=cCurrL.getLastNonlocal();
      if( i>-1 and i<iLowerA     && nL!=N_NONE && qPrev[i].getCat()==nL                      ) { viCarrierL.push_back(i);  cCurrL=cCurrL.withoutLastNolo(); }
      if(          i<iLowerA     && nL!=N_NONE && !cI.isCarrier() && !cI.containsCarrier(nL) ) { viCarrierL.push_back(-1); cCurrL=cCurrL.withoutLastNolo(); nNewCarriers++; }
      N nA=cCurrA.getLastNonlocal();
      if( i>-1 and i<iLowerA     && nA!=N_NONE && qPrev[i].getCat()==nA                      ) { viCarrierA.push_back(i);  cCurrA=cCurrA.withoutLastNolo(); }
      if(          i<iLowerA     && nA!=N_NONE && !cI.isCarrier() && !cI.containsCarrier(nA) ) { viCarrierA.push_back(-1); cCurrA=cCurrA.withoutLastNolo(); nNewCarriers++; }
      N nB=cCurrB.getLastNonlocal();
      if( i>-1 and i<iAncestorB  && nB!=N_NONE && qPrev[i].getCat()==nB                      ) { viCarrierB.push_back(i);  cCurrB=cCurrB.withoutLastNolo(); }
      if(          i<=iAncestorB && nB!=N_NONE && !cI.isCarrier() && !cI.containsCarrier(nB) ) { viCarrierB.push_back(-1); cCurrB=cCurrB.withoutLastNolo(); nNewCarriers++; }
    }

    //cout<<" viCarrierP="; for( int i : viCarrierP ) cout<<" "<<i; cout<<endl;
    //cout<<" viCarrierA="; for( int i : viCarrierA ) cout<<" "<<i; cout<<endl;
    //cout<<" viCarrierL="; for( int i : viCarrierL ) cout<<" "<<i; cout<<endl;
    //cout<<" viCarrierB="; for( int i : viCarrierB ) cout<<" "<<i; cout<<endl;

    // Reserve store big enough for ancestorB + new A and B if no join + any needed carriers...
    reserve( iAncestorB + 1 + ((j==0) ? 2 : 0) + nNewCarriers ); 

    ////// B. FILL IN NEW PARTS OF NEW STORE...

    //// B.1. Add existing nolo contexts to parent via extraction op...
    HVec hvParent( cA.getSynArgs()+( ( (opL>='1' and opL<='9') or (opR>='1' and opR<='9') ) ? 2 : 1 ) );
    HVec hvRchild( cB.getSynArgs()+1 );
    // If join, apply unaries going down from ancestor, then merge redirect of left child...
    if( j ) {
      hvParent.add( qPrev.at(iAncestorB).getHVec() ).applyUnariesTopDn( evJ, viCarrierA, qPrev ).addSynArg( -getDir(opL), aLchild.getHVec() );
      hvRchild.addSynArg( getDir(opR), hvParent );
    }
    // If not join, merge redirect of left child...
    else {
      hvParent.addSynArg( -getDir(opL), aLchild.getHVec() );
      hvRchild.addSynArg( getDir(opR), hvParent ); 
      hvParent.applyUnariesBotUp( evJ, viCarrierA, qPrev );
    }

    //// B.2. Copy store state and add parent/preterm contexts to existing non-locals via extraction operation...
    for( int i=0; i<((f==0&&j==1)?iAncestorB:(f==0&&j==0)?iLowerA:(f==1&&j==1)?iAncestorB:iAncestorB+1); i++ ) {
      Sign& s = *emplace( end() ) = qPrev[i];
      if( i==iAncestorA and j==1 and qPrev[i].isDitto() and opR!=O_I )            { s.setHVec() = hvParent; } 
      else if( viCarrierP.size()>0 and i==viCarrierP.back() and evF!=EVar::eNil ) { viCarrierP.pop_back();
                                                                                    s.setHVec() = HVec( s.getCat().getSynArgs()+1 );
                                                                                    s.setHVec().addSynArg( getDir(evF.popTop()), aPretrm.getHVec() ); }
      else if( viCarrierA.size()>0 and i==viCarrierA.back() and evJ!=EVar::eNil ) { viCarrierA.pop_back();
                                                                                    s.setHVec() = HVec( s.getCat().getSynArgs()+1 );
                                                                                    s.setHVec().addSynArg( getDir(evJ.popTop()), hvParent ); }
      else                                                                        { s = qPrev[i]; }
    }

    //// B.3. Add new non-locals with contexts from parent/rchild via new extraction or G/H/V operations...
    // If no join, add A carriers followed by new lowest A...
    if( j==0 ) {
      // Add A carriers...
      cCurrP = aPretrm.getCat();  cCurrA = cA;
      for( int i : viCarrierP ) if( i==-1 and evF!=EVar::eNil ) { if( STORESTATE_CHATTY ) cout<<"(adding carrierP for "<<cCurrP.getFirstNonlocal()<<" bc none above "<<iAncestorB<<")"<<endl;
                                                                  Sign& s = *emplace( end() ) = Sign( HVec(1), cCurrP.getFirstNonlocal(), S_B );
                                                                  s.setHVec().addSynArg( getDir(evF.popTop()), aPretrm.getHVec() );
                                                                  cCurrP=cCurrP.withoutFirstNolo(); }
      for( int i : viCarrierA ) if( i==-1 and evJ!=EVar::eNil ) { if( STORESTATE_CHATTY ) cout<<"(adding carrierA for "<<cCurrA.getFirstNonlocal()<<" bc none above "<<iAncestorB<<")"<<endl;
                                                                  Sign& s = *emplace( end() ) = Sign( HVec(1), cCurrA.getFirstNonlocal(), S_B );
                                                                  s.setHVec().addSynArg( getDir(evJ.popTop()), hvParent );
                                                                  cCurrA=cCurrA.withoutFirstNolo(); }
      // Add lowest A...
      *emplace( end() ) = Sign( (opR==O_I) ? HVec::hvDitto : hvParent, cA, S_A );
      iLowerA = size()-1;
    }
    // Add B carriers...
    N nA = cA.getLastNonlocal();  N nB = cB.getLastNonlocal();  N nL = aLchild.getCat().getLastNonlocal();
    if( nB!=N_NONE and nB!=nA and viCarrierB[0]==-1 ) {  if( STORESTATE_CHATTY ) cout<<"(adding carrierB for "<<nB<<" bc none above "<<iAncestorB<<") (G/R rule)"<<endl;
                                                         *emplace( end() ) = Sign( aLchild.getHVec(), nB, S_A ); }                            // Add left child kset as A carrier (G rule).
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
        cCurrL=aLchild.getCat();
        viCarrierL.clear();
        for( int i=iLowerA-1; i>=-1; i-- ) {
          CVar cI = at(i).getCat();
          N nL=cCurrL.getLastNonlocal(); if( i>-1 && nL!=N_NONE && cI==nL                                     ) { viCarrierL.push_back(i);  cCurrL=cCurrL.withoutLastNolo(); }
                                         if(         nL!=N_NONE && !cI.isCarrier() && !cI.containsCarrier(nL) ) { viCarrierL.push_back(-1); cCurrL=cCurrL.withoutLastNolo(); }
        }
        //cout<<" viCarrierL="; for( int& i : viCarrierL ) cout<<" "<<i; cout<<endl;
        if( viCarrierL[0]>iAncestorB ) { Sign& s = *emplace( end() ) = Sign( hvRchild, cB, S_B );  s.setHVec().add( at(viCarrierL[0]).getHVec() ); }  // Add right child kset as B (H rule).
        else cerr<<"ERROR StoreState 1019: should not happen, on '"<<qPrev<<" "<<f<<" "<<j<<" "<<evF<<" "<<evJ<<" "<<opL<<" "<<opR<<" "<<cA<<" "<<cB<<" "<<aPretrm<<" "<<aLchild<<"'"<<endl;
      } else {  // If j==1...
        Sign& s = *emplace( end() ) = Sign( hvRchild, cB, S_B );
        // If existing left carrier, integrate with sign...
        if( viCarrierL[0]!=-1 ) s.setHVec().add( qPrev.at(viCarrierL[0]).getHVec() );
        // If extraction...
        if( evF!=EVar::eNil )   s.setHVec().addSynArg( getDir(evF.popTop()), aPretrm.getHVec() );
      }
    }
    // Add lowest B...
    else if( size()>0 )            { *emplace( end() ) = Sign( hvRchild, cB, S_B ); }
  }

  const Sign& at ( int i ) const { assert(i<int(size())); return (i<0) ? aTop : operator[](i); }

  int getDepth ( ) const {
    int d = 0; for( int i=size()-1; i>=0; i-- ) if( !operator[](i).getCat().isCarrier() && operator[](i).getSide()==S_B ) d++;
    return d;
  }

  int getAncestorBIndex ( F f ) const {
    if( f==1 ) return size()-1;
    for( int i=size()-2; i>=0; i-- ) if( !operator[](i).getCat().isCarrier() && operator[](i).getSide()==S_B ) return i;
    return -1;
  }

  int getAncestorAIndex ( F f ) const {
    for( int i=getAncestorBIndex(f)-1; i>=0; i-- ) if( !operator[](i).getCat().isCarrier() && operator[](i).getSide()==S_A ) return i;
    return -1;
  }

  int getAncestorBCarrierIndex ( F f ) const {
    int iAncestor = getAncestorBIndex( f );
    N nB = at(iAncestor).getCat().getLastNonlocal();
    if( nB!=N_NONE ) for( int i=iAncestor-1; i>=0 && (operator[](i).getCat().isCarrier() || operator[](i).getCat().containsCarrier(nB)); i-- ) if( operator[](i).getCat()==nB ) return i;
    return -1;
  } 

  /*
  const Sign& getBase( unsigned int iDepthOffset ) const {
    return at( getDepth() - iDepthOffset );
  }

  const Sign& getBaseCarrier( unsigned int iDepthOffset, unsigned int iCarrierProximity ) {
    return at( 
  */
};
const Sign StoreState::aTop( hvTop, cTop, S_B );

////////////////////////////////////////////////////////////////////////////////

HVec& HVec::applyUnariesTopDn( EVar e, const vector<int>& viCarrierIndices, const StoreState& ss ) {
  for( int i=0; e!=EVar::eNil; e=e.withoutTop() ) {
    if( e.top()>='0' and e.top()<='9' and i<int(viCarrierIndices.size()) and viCarrierIndices[i++]!=-1 )  addSynArg( -getDir(e.top()), ss.at(viCarrierIndices[i-1]).getHVec() );
    else if( e.top()=='O' or e.top()=='V' )  swap(1,2);
  }
  return *this;
}

HVec& HVec::applyUnariesBotUp( EVar e, const vector<int>& viCarrierIndices, const StoreState& ss ) {
  for( int i=viCarrierIndices.size()-1; e!=EVar::eNil; e=e.withoutBot() ) {
    if( e.bot()>='0' and e.bot()<='9' and i>=0 and viCarrierIndices[i--]!=-1 )  addSynArg( -getDir(e.bot()), ss.at(viCarrierIndices[i+1]).getHVec() );
    else if( e.bot()=='O' or e.bot()=='V' )  swap(1,2);
  }
  return *this;
}

////////////////////////////////////////////////////////////////////////////////

LeftChildSign::LeftChildSign ( const StoreState& qPrev, F f, EVar eF, const Sign& aPretrm ) {
//    int         iCarrierB  = qPrev.getAncestorBCarrierIndex( 1 );
    int         iAncestorB = qPrev.getAncestorBIndex(f);
    CVar        cCurrB     = qPrev.at(iAncestorB).getCat();
    vector<int> viCarrierB;  viCarrierB.reserve(4);
    for( int i=qPrev.size()-1; i>=-1; i-- ) {
      N nB=cCurrB.getLastNonlocal();
      if( i>-1 && i<iAncestorB  && nB!=N_NONE && qPrev[i].getCat()==nB                                                               ) { viCarrierB.push_back(i);  cCurrB=cCurrB.withoutLastNolo(); }
      if(         i<=iAncestorB && nB!=N_NONE && (i<0 || (!qPrev[i].getCat().isCarrier() && !qPrev[i].getCat().containsCarrier(nB))) ) { viCarrierB.push_back(-1); cCurrB=cCurrB.withoutLastNolo(); }
    }
    //cout<<" viCarrierB="; for( int i : viCarrierB ) cout<<" "<<i; cout<<endl;
    const Sign& aAncestorA = qPrev.at( qPrev.getAncestorAIndex(1) );
    const Sign& aAncestorB = qPrev.at( qPrev.getAncestorBIndex(1) );
//    const KSet& ksExtrtn   = (iCarrierB<0) ? KSet() : qPrev.at(iCarrierB).getKSet();
    setSide() = S_A;
    if( f==1 )                       { setCat()  = aPretrm.getCat();
                                       setHVec() = HVec(getCat().getSynArgs()+1);  setHVec().add( aPretrm.getHVec() ).applyUnariesBotUp( eF, viCarrierB, qPrev ); }
    else if( qPrev.size()<=0 )       { *this = StoreState::aTop; }
    else if( !aAncestorA.isDitto() ) { setCat()  = aAncestorA.getCat();
                                       setHVec() = HVec(getCat().getSynArgs()+1);  setHVec().add( aAncestorA.getHVec() ).applyUnariesBotUp( eF, viCarrierB, qPrev ); }
    else                             { setCat()  = aAncestorA.getCat();
                                       setHVec() = HVec(getCat().getSynArgs()+1);  setHVec().add( aPretrm.getHVec() ).applyUnariesBotUp( eF, viCarrierB, qPrev ).add( aAncestorB.getHVec() ); }
}

#endif

////////////////////////////////////////////////////////////////////////////////

W wUnkIng ( "!unk!ing" );
W wUnkEd  ( "!unk!ed"  );
W wUnkS   ( "!unk!s"   );
W wUnkIon ( "!unk!ion" );
W wUnkEr  ( "!unk!er"  );
W wUnkEst ( "!unk!est" );
W wUnkLy  ( "!unk!ly"  );
W wUnkIty ( "!unk!ity" );
W wUnkY   ( "!unk!y"   );
W wUnkAl  ( "!unk!al"  );
W wUnkCap ( "!unk!cap" );
W wUnkNum ( "!unk!num" );
W wUnk    ( "!unk!"    );

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
    const Sign& getPrtrm ()           const { return first(); }
    F getF ()                         const { return second(); }
    EVar getForkE ()                  const { return third(); }
    K getForkK ()                     const { return fourth(); }
    const JResponse& getJResp()       const { return fifth(); }
    const StoreState& getStoreState() const { return sixth(); }
    const Delimited<int>& getI()      const { return seventh(); }
};

////////////////////////////////////////////////////////////////////////////////

typedef Delimited<CVar> P;
typedef Delimited<CVar> A;
typedef Delimited<CVar> B;

////////////////////////////////////////////////////////////////////////////////

class PPredictorVec : public DelimitedQuint<psX,D,psSpace,F,psSpace,Delimited<EVar>,psSpace,Delimited<CVar>,psSpace,Delimited<CVar>,psX> {
 public:
  PPredictorVec ( ) { }
  PPredictorVec ( F f, EVar e, K k_p_t, const StoreState& ss ) :
#ifdef SIMPLE_STORE
    DelimitedQuint<psX,D,psSpace,F,psSpace,Delimited<EVar>,psSpace,Delimited<CVar>,psSpace,Delimited<CVar>,psX>( ss.getDepth(), f, e, ss.getBase().getCat(), k_p_t.getCat() ) { }
#else
    DelimitedQuint<psX,D,psSpace,F,psSpace,Delimited<EVar>,psSpace,Delimited<CVar>,psSpace,Delimited<CVar>,psX>( ss.getDepth(), f, e, ss.at(ss.size()-1).getCat(), k_p_t.getCat() ) { }
#endif
};

class PModel : public map<PPredictorVec,map<P,double>> {
 public:
  PModel ( ) { }
  PModel ( istream& is ) {
    // Process P lines in stream...
    while( is.peek()=='P' ) {
      PPredictorVec ppv;  P p;
      is >> "P " >> ppv >> " : " >> p >> " = ";
      is >> (*this)[ppv][p] >> "\n"; 
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

class WPredictor : public DelimitedTrip<psX,Delimited<EVar>,psSpace,Delimited<K>,psSpace,Delimited<CVar>,psX> { };

class WModel : public map<W,list<DelimitedPair<psX,WPredictor,psSpace,Delimited<double>,psX>>> {
 public:
  WModel ( ) { }
  WModel ( istream& is ) {
    while( is.peek()=='W' ) {
      WPredictor wp;  W w;  Delimited<double> pr;
      is >> "W " >> wp >> " : " >> w >> " = " >> pr >> "\n";
      (*this)[w].emplace_back(wp,pr);
    }

    // Add unk...
    for( auto& entry : *this ) {
      // for each word:{<category:prob>}
      for( auto& unklistelem : (*this)[unkWord(entry.first.getString().c_str())] ) {
        // for each possible unked(word) category:prob pair
        bool BAIL = false;
        for( auto& listelem : entry.second ) {
          if (listelem.first == unklistelem.first) {
            BAIL = true;
            listelem.second = listelem.second + ( 0.000001 * unklistelem.second ); // merge actual likelihood and unk likelihood
          }
        }
        if (not BAIL) entry.second.push_back( DelimitedPair<psX,WPredictor,psSpace,Delimited<double>,psX>(unklistelem.first,0.000001*unklistelem.second) );
      }
    } //closes for auto& entry : lexW
  }

  const list<DelimitedPair<psX,WPredictor,psSpace,Delimited<double>,psX>>& calcPredictorLikelihoods( W w_t ) const {
    if( end() == find( unkWord( w_t.getString().c_str() ) ) )  cerr << "ERROR: unable to find unk form: " << unkWord( w_t.getString().c_str() ) << endl;
    return( ( end() != find(w_t) ) ? find(w_t)->second : find( unkWord( w_t.getString().c_str() ) )->second );
  }
};

////////////////////////////////////////////////////////////////////////////////

class APredictorVec : public DelimitedSept<psX,D,psSpace,F,psSpace,J,psSpace,Delimited<EVar>,psSpace,O,psSpace,Delimited<CVar>,psSpace,Delimited<CVar>,psX> {
 public:
  APredictorVec ( ) { }
  APredictorVec ( D d, F f, J j, EVar e, O o, CVar cP, CVar cL ) :
    DelimitedSept<psX,D,psSpace,F,psSpace,J,psSpace,Delimited<EVar>,psSpace,O,psSpace,Delimited<CVar>,psSpace,Delimited<CVar>,psX>( d, f, j, e, o, cP, cL ) { }
  APredictorVec ( F f, J j, EVar eF, EVar eJ, O opL, const LeftChildSign& aLchild, const StoreState& ss ) :
#ifdef SIMPLE_STORE
    DelimitedSept<psX,D,psSpace,F,psSpace,J,psSpace,Delimited<EVar>,psSpace,O,psSpace,Delimited<CVar>,psSpace,Delimited<CVar>,psX>( ss.getDepth()-j, f, j, eJ, opL, ss.getBase().getCat(), (j==0) ? aLchild.getCat() : cBot ) { } 
#else
    DelimitedSept<psX,D,psSpace,F,psSpace,J,psSpace,Delimited<EVar>,psSpace,O,psSpace,Delimited<CVar>,psSpace,Delimited<CVar>,psX>( ss.getDepth()+f-j, f, j, eJ, opL, ss.at(ss.getAncestorBIndex(f)).getCat(), (j==0) ? aLchild.getCat() : cBot ) { } 
#endif
};

class AModel : public map<APredictorVec,map<A,double>> {
 public:
  AModel ( ) { }
  AModel ( istream& is ) {
    // Add top-level rule...
    (*this)[ APredictorVec(1,0,1,EVar::eNil,'S',CVar("T"),CVar("-")) ][ A("-") ] = 1.0;      // should be CVar("S")
    // Process A lines in stream...
    while( is.peek()=='A' ) {
      APredictorVec apv;  A a;
      is >> "A " >> apv >> " : " >> a >> " = ";
      is >> (*this)[apv][a] >> "\n"; 
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

class BPredictorVec : public DelimitedOct<psX,D,psSpace,F,psSpace,J,psSpace,Delimited<EVar>,psSpace,O,psSpace,O,psSpace,Delimited<CVar>,psSpace,Delimited<CVar>,psX> {
 public:
  BPredictorVec ( ) { }
  BPredictorVec ( D d, F f, J j, EVar e, O oL, O oR, CVar cP, CVar cL ) :
    DelimitedOct<psX,D,psSpace,F,psSpace,J,psSpace,Delimited<EVar>,psSpace,O,psSpace,O,psSpace,Delimited<CVar>,psSpace,Delimited<CVar>,psX>( d, f, j, e, oL, oR, cP, cL ) { }
  BPredictorVec ( F f, J j, EVar eF, EVar eJ, O opL, O opR, CVar cParent, const LeftChildSign& aLchild, const StoreState& ss ) :
#ifdef SIMPLE_STORE
    DelimitedOct<psX,D,psSpace,F,psSpace,J,psSpace,Delimited<EVar>,psSpace,O,psSpace,O,psSpace,Delimited<CVar>,psSpace,Delimited<CVar>,psX>( ss.getDepth()-j, f, j, eJ, opL, opR, cParent, aLchild.getCat() ) { }
#else
    DelimitedOct<psX,D,psSpace,F,psSpace,J,psSpace,Delimited<EVar>,psSpace,O,psSpace,O,psSpace,Delimited<CVar>,psSpace,Delimited<CVar>,psX>( ss.getDepth()+f-j, f, j, eJ, opL, opR, cParent, aLchild.getCat() ) { }
#endif
};

class BModel : public map<BPredictorVec,map<B,double>> {
 public:
  BModel ( ) { }
  BModel ( istream& is ) {
    // Add top-level rule...
    (*this)[ BPredictorVec(1,0,1,EVar::eNil,'S','1',CVar("-"),CVar("S")) ][ B("T") ] = 1.0;
    // Process B lines in stream...
    while( is.peek()=='B' ) {
      BPredictorVec bpv;  B b;
      is >> "B " >> bpv >> " : " >> b >> " = ";
      is >> (*this)[bpv][b] >> "\n"; 
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

