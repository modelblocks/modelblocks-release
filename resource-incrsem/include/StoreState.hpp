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
const AdHocFeature corefOFF("acorefOFF");
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

class CVar;
typedef CVar N;

////////////////////////////////////////////////////////////////////////////////

DiscreteDomain<int> domCVar;
class CVar : public DiscreteDomainRV<int,domCVar> {
 private:
  static map<N,bool>               mnbArg;
  static map<CVar,int>             mciArity;
  static map<CVar,bool>            mcbIsCarry;
  static map<CVar,N>               mcnFirstNol;
  static map<CVar,CVar>            mccNoFirstNol;
  static map<CVar,N>               mcnLastNol;
  static map<CVar,CVar>            mccNoLastNol;
  static map<pair<CVar,N>,bool>    mcnbIn;
  static map<CVar,CVar>            mccLets;
  static map<CVar,int>             mciNums;
  static map<pair<CVar,int>,CVar>  mcicLetNum;
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
    if( mciArity.end()==mciArity.find(*this) ) { mciArity[*this]=getArity(ps); }
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
  int  getArity         ( )       const { return mciArity  [*this]; }
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
map<CVar,int>             CVar::mciArity;
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
const N N_NONE("");

////////////////////////////////////////////////////////////////////////////////

DiscreteDomain<int> domK;
class K : public DiscreteDomainRV<int,domK> {   // NOTE: can't be subclass of Delimited<...> or string-argument constructor of this class won't get called!
 public:
  static const K kTop;
  static const K kBot;
  private:
  static map<K,CVar> mkc;
  static map<pair<K,int>,K> mkik;
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
    else mkc[*this] = (*this==kBot) ? cBOT : (*this==kTop) ? cTop : cBot;
  }
 public:
  K ( )                : DiscreteDomainRV<int,domK> ( )    { }
  K ( const char* ps ) : DiscreteDomainRV<int,domK> ( ps ) { calcDetermModels(ps); }
  CVar getCat    ( )                  const { auto it = mkc.find(*this); return (it==mkc.end()) ? cBot : it->second; }
  K    project   ( int n )            const { auto it = mkik.find(pair<K,int>(*this,n)); return (it==mkik.end()) ? kBot : it->second; }
  K    transform ( bool bUp, char c ) const { return mkkO[*this]; }
//  K transform ( bool bUp, char c ) const { return (bUp and c=='V') ? mkkVU[*this] :
//                                                  (        c=='V') ? mkkVD[*this] : kBot; }
};
map<K,CVar>        K::mkc;
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
  Boolean predictors for antecedent model.  Generally KxK pairs or CVarxCVar pairs between anaphor and candidate antecedent. 
  */
  private:
    uint id;

    static uint                      nextid;
    static map<pair<K,K>,uint>       mkki; 
    static map<pair<CVar,CVar>,uint> mcci; //pairwise CVarCVar? probably too sparse...
    static map<uint,AdHocFeature>    mistr;
    static map<AdHocFeature,uint>    mstri;
    static map<uint,K>               miantk;         //pairwise
    static map<uint,K>               miancestork;    //pairwise
    static map<uint,CVar>            miantecedentc;
    static map<uint,CVar>            miancestorc;
    static map<CVar,uint>            mantecedentci;
    static map<CVar,uint>            mancestorci;

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

    NPredictor (CVar antecedentCVar, CVar ancestorCVar) {
      const auto& it = mcci.find(pair<CVar,CVar>(antecedentCVar, ancestorCVar));
      if (it != mcci.end() ) id = it->second;
      else { id = nextid++; miantecedentc[id] = antecedentCVar; miancestorc[id] = ancestorCVar; mcci[pair<CVar,CVar>(antecedentCVar,ancestorCVar)] = id; }
    }

  //Accessor Methods
  uint toInt() const { return id; }
  operator uint() const { return id; }
  K getAncstrK()  const { return miancestork[id]; }
  K getAntcdntK() const { return miantk[id]; } 
  CVar getAntecedentCVar() const { return miantecedentc[id]; }
  CVar getAncestorCVar() const { return miancestorc[id]; }
  AdHocFeature getfeatname() const { return mistr[id]; }
  static uint getDomainSize() { return nextid; }

  // Input / output methods...
  // need to add handling of unary t and k features for IO...
  friend pair<istream&,NPredictor&> operator>> ( istream& is, NPredictor& t ) {
    return pair<istream&,NPredictor&>(is,t);
  }
  friend istream& operator>> ( pair<istream&,NPredictor&> ist, const char* psDelim ) {
    if ( ist.first.peek()==psDelim[0] ) { 
      auto& o =  ist.first >> psDelim;  
      ist.second = NPredictor();  
      return o; 
    }
    else if ( ist.first.peek()=='a' ) { 
      AdHocFeature mstring;
      auto& o = ist.first >> mstring >> psDelim;  
      ist.second = NPredictor(mstring);     
      return o; 
    }
    else if (ist.first.peek()=='t') {
      Delimited<CVar> antecedentc, ancestorc;
      auto& o = ist.first >> "t" >> antecedentc >> "&" >> ancestorc >> psDelim;
      ist.second = NPredictor(antecedentc, ancestorc);
      return o;
    }
    else  { 
      Delimited<K> kAntecedent, kAncestor;  
      auto& o = ist.first >> kAntecedent >> "&" >> kAncestor >> psDelim;  
      ist.second = NPredictor(kAntecedent, kAncestor);  
      return o; 
    }
  }
  friend bool operator>> ( pair<istream&,NPredictor&> ist, const vector<const char*>& vpsDelim ) {
    if ( ist.first.peek()=='a' ) { 
      AdHocFeature mstring;
      auto o = ist.first >> mstring >> vpsDelim;  
      ist.second = NPredictor(mstring);     
      return o; 
    }
    else if (ist.first.peek()=='t') {
      Delimited<CVar> antecedentc, ancestorc;
      auto o = ist.first >> "t" >> antecedentc >> "&" >> ancestorc >> vpsDelim;
      ist.second = NPredictor(antecedentc, ancestorc);
      return o;
    }
    else { 
      Delimited<K> kAntecedent, kAncestor;
      auto o = ist.first >> kAntecedent >> "&" >> kAncestor >> vpsDelim;  
      ist.second = NPredictor(kAntecedent,kAncestor);  
      return o; 
    }
  }
  friend ostream& operator<< ( ostream& os, const NPredictor& t ) {
    //return os << miantk[t.id] << "&" << miancestork[t.id]; 
    if (miantk.find(t.id) != miantk.end()) { return os << miantk[t.id] << "&" << miancestork[t.id]; } //output either KxK
    else if (miantecedentc.find(t.id) != miantecedentc.end()) { return os << "t" << miantecedentc[t.id] << "&" << miancestorc[t.id]; } // or t x t
    else if (mistr.find(t.id) != mistr.end()) { return os << mistr[t.id]; } //check for string
    else { return os << "NON_STRING_ID_" << t.id; } 
  }
  static bool exists ( K kAntecedent, K kAncestor )       { return( mkki.end()!=mkki.find(pair<K,K>(kAntecedent,kAncestor)) ); }
  static bool exists ( AdHocFeature mstring )             { return( mstri.end()!=mstri.find(mstring) ); }
  static bool exists ( CVar cAntecedent, CVar cAncestor ) { return( mcci.end()!=mcci.find(pair<CVar,CVar>(cAntecedent,cAncestor)) ); }
};
uint                      NPredictor::nextid = 1;
map<pair<K,K>,uint>       NPredictor::mkki; 
map<pair<CVar,CVar>,uint> NPredictor::mcci;
map<AdHocFeature,uint>    NPredictor::mstri;
map<uint,AdHocFeature>    NPredictor::mistr;
map<uint,K>               NPredictor::miantk;
map<uint,K>               NPredictor::miancestork;
map<uint,CVar>            NPredictor::miantecedentc;
map<uint,CVar>            NPredictor::miancestorc;
map<CVar,uint>            NPredictor::mantecedentci;
map<CVar,uint>            NPredictor::mancestorci;

////////////////////////////////////////////////////////////////////////////////

class NPredictorSet {
//need to be able to output real-valued distance integers, NPreds
//TODO maybe try quadratic distance
  private:
     int mdist;
     DelimitedList<psX,NPredictor,psComma,psX> mnpreds;

  public:
    //constructor
    NPredictorSet ( ) : mdist(0), mnpreds() { }
    DelimitedList<psX,NPredictor,psComma,psX>& setList ( ) {
      return mnpreds;
    }
    int& setAntDist ( ) {
      return mdist;
    }

    void printOut ( ostream& os ) {
        os << "N "; 
        os << "antdist=" << mdist;
        for (auto& npred : mnpreds) {
          os << ","; 
          os << npred << "=1";
        } 
    }
   
    arma::vec NLogResponses ( const arma::mat& matN) {
      arma::vec nlogresponses = arma::zeros( matN.n_rows );
      nlogresponses += mdist * matN.col(NPredictor("antdist").toInt());
      for ( auto& npredr : mnpreds) {
          //if (VERBOSE>1) { cout << npredr << " " << npredr.toInt() << endl; }
        if ( npredr.toInt() < matN.n_cols ) { 
          nlogresponses += matN.col( npredr.toInt() ); 
          //if (VERBOSE>1) { cout << npredr << " " << npredr.toInt() << " matN.n_cols:" << matN.n_cols << " logprob: " << matN.col( npredr.toInt())(NResponse("1").toInt()) << endl; }
        }
      }
      return nlogresponses;
    }
};

////////////////////////////////////////////////////////////////////////////////

DiscreteDomain<int> domNResponse;
typedef DiscreteDomainRV<int,domNResponse> NResponse;

////////////////////////////////////////////////////////////////////////////////

typedef unsigned int FResponse;
typedef unsigned int JResponse;

////////////////////////////////////////////////////////////////////////////////

typedef Delimited<int> D;

class PPredictor : public DelimitedQuint<psX,D,psSpace,F,psSpace,Delimited<EVar>,psSpace,Delimited<CVar>,psSpace,Delimited<CVar>,psX> {
 public:
  PPredictor ( )                                    : DelimitedQuint<psX,D,psSpace,F,psSpace,Delimited<EVar>,psSpace,Delimited<CVar>,psSpace,Delimited<CVar>,psX> ( )                 { }
  PPredictor ( D d, F f, EVar e, CVar tB, CVar tK ) : DelimitedQuint<psX,D,psSpace,F,psSpace,Delimited<EVar>,psSpace,Delimited<CVar>,psSpace,Delimited<CVar>,psX> ( d, f, e, tB, tK ) { }

};

class WPredictor : public DelimitedTrip<psX,Delimited<EVar>,psSpace,Delimited<K>,psSpace,Delimited<CVar>,psX> { };

class APredictor : public DelimitedSept<psX,D,psSpace,F,psSpace,J,psSpace,Delimited<EVar>,psSpace,O,psSpace,Delimited<CVar>,psSpace,Delimited<CVar>,psX> {
 public:
  APredictor ( )                                               : DelimitedSept<psX,D,psSpace,F,psSpace,J,psSpace,Delimited<EVar>,psSpace,O,psSpace,Delimited<CVar>,psSpace,Delimited<CVar>,psX> ( ) { }
  APredictor ( D d, F f, J j, EVar e, O oL, CVar tB, CVar tL ) : DelimitedSept<psX,D,psSpace,F,psSpace,J,psSpace,Delimited<EVar>,psSpace,O,psSpace,Delimited<CVar>,psSpace,Delimited<CVar>,psX> ( d, f, j, e, oL, tB, tL ) { }
};

class BPredictor : public DelimitedOct<psX,D,psSpace,F,psSpace,J,psSpace,Delimited<EVar>,psSpace,O,psSpace,O,psSpace,Delimited<CVar>,psSpace,Delimited<CVar>,psX> {
 public:
  BPredictor ( )                                                     : DelimitedOct<psX,D,psSpace,F,psSpace,J,psSpace,Delimited<EVar>,psSpace,O,psSpace,O,psSpace,Delimited<CVar>,psSpace,Delimited<CVar>,psX> ( )                         { }
  BPredictor ( D d, F f, J j, EVar e, O oL, O oR, CVar tP, CVar tL ) : DelimitedOct<psX,D,psSpace,F,psSpace,J,psSpace,Delimited<EVar>,psSpace,O,psSpace,O,psSpace,Delimited<CVar>,psSpace,Delimited<CVar>,psX> ( d, f, j, e, oL, oR, tP, tL ) { }
};

////////////////////////////////////////////////////////////////////////////////

/*
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
  KSet ( const KSet& ksToProject, int iProj, const KSet& ksNoProject = ksDummy ) : DelimitedVector<psLBrack,Delimited<K>,psComma,psRBrack> ( ) {
    // If swap op is banked, swap participants...
    if     ( eBankedUnaryTransforms==EVar("O") and iProj==-1 ) iProj=-2;
    else if( eBankedUnaryTransforms==EVar("O") and iProj==-2 ) iProj=-1;
    // If projection source swap op is banked, swap participant label...
    if     ( ksToProject.eBankedUnaryTransforms==EVar("O") and iProj==1 ) iProj=2;
    else if( ksToProject.eBankedUnaryTransforms==EVar("O") and iProj==2 ) iProj=1;
    reserve( ksToProject.size() + ksNoProject.size() );
    for( const K& k : ksToProject ) if( k.project(iProj)!=K::kBot ) push_back( k.project(iProj) );
    insert( end(), ksNoProject.begin(), ksNoProject.end() );
    if( 0==iProj ) eBankedUnaryTransforms = ksToProject.eBankedUnaryTransforms;
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
*/

////////////////////////////////////////////////////////////////////////////////

/*
class HVec;

class Redirect : public pair<int,const HVec&> {
 public:
  Redirect( int dir, const HVec& hv ) : pair<int,const HVec&>(dir,hv) { }
};
*/

class HVec : public DelimitedVector<psX,DelimitedVector<psLBrack,Delimited<K>,psComma,psRBrack>,psX,psX> {
 public:
  // Constructors...
  HVec ( )     : DelimitedVector<psX,DelimitedVector<psLBrack,Delimited<K>,psComma,psRBrack>,psX,psX>() { }
  HVec ( K k ) : DelimitedVector<psX,DelimitedVector<psLBrack,Delimited<K>,psComma,psRBrack>,psX,psX>() { emplace( end() )->emplace_back(k); }
//  HVec& operator+= ( const HVec& hv ) {
  HVec& add( const HVec& hv ) {
    for( unsigned int arg=0; arg<hv.size(); arg++ ) at(arg).insert( at(arg).end(), hv.at(arg).begin(), hv.at(arg).end() );
    return *this;
  }
//  HVec& operator+= ( const Redirect& r ) {
  HVec& addSynArg( int iDir, const HVec& hv ) {
    if(      iDir == 0 ) add( hv );
    else if( iDir >  0 ) at(iDir).insert( at(iDir).end(), hv.at( 0   ).begin(), hv.at( 0   ).end() );
    else                 at( 0  ).insert( at( 0  ).end(), hv.at(-iDir).begin(), hv.at(-iDir).end() );
    return *this;
  }
  HVec& swap( int i, int j ) {
    auto kv = at(i);  at(i) = at(j);  at(j) = kv;
    return *this;
  }
  HVec& applyUnariesTopDown( EVar e, const vector<int>& viCarrierIndices, const StoreState& ss );
  HVec& applyUnariesBottomUp( EVar e, const vector<int>& viCarrierIndices, const StoreState& ss );
  bool isDitto ( ) const { return ( size()>0 and front().size()>0 and front().front()==K_DITTO ); }
};
const HVec hvTop = HVec( K::kTop );
const HVec hvBot = HVec( K::kBot );

////////////////////////////////////////////////////////////////////////////////

class Sign : public DelimitedTrip<psX,HVec,psColon,CVar,psX,S,psX> {
 public:
  Sign ( )                              : DelimitedTrip<psX,HVec,psColon,CVar,psX,S,psX> ( )           { third()=S_A; }
  Sign ( const HVec& hv1, CVar c, S s ) : DelimitedTrip<psX,HVec,psColon,CVar,psX,S,psX> ( hv1, c, s ) { }
  Sign ( const HVec& hv1, const HVec& hv2, CVar c, S s ) {
    first().reserve( hv1.size() + hv2.size() );
    first().insert( first().end(), hv1.begin(), hv1.end() );
    first().insert( first().end(), hv2.begin(), hv2.end() );
//    first().addBankedUnaryTransform( hv1.getBankedUnaryTransform() );
    second() = c;
    third()  = s;
  }
  HVec&       setHVec ( )       { return first();  }
  CVar&       setCat  ( )       { return second(); }
  S&          setSide ( )       { return third();  }
  const HVec& getHVec ( ) const { return first();  }
  CVar        getCat  ( ) const { return second(); } //.removeLink(); }
  S           getSide ( ) const { return third();  }
  bool        isDitto ( ) const { return getHVec().isDitto(); }
};

////////////////////////////////////////////////////////////////////////////////

class LeftChildSign : public Sign {
 public:
  LeftChildSign ( const Sign& a ) : Sign(a) { }
  LeftChildSign ( const StoreState& qPrev, F f, EVar eF, const Sign& aPretrm );
};

////////////////////////////////////////////////////////////////////////////////

class StoreState : public DelimitedVector<psX,Sign,psX,psX> {  // NOTE: format can't be read in bc of internal psX delimicer, but we don't need to.
 public:

  static const Sign aTop;
  static const Sign aBot;
  // storestate records antecedent choices, excluding previous indices to force most recent mentions to be considered. closer to entity tracking
  std::vector<int> excludedIndices;

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
      N nP=cCurrP.getLastNonlocal(); if( i>-1 and                  nP!=N_NONE && qPrev[i].getCat()==nP                      ) { viCarrierP.push_back(i);  cCurrP=cCurrP.withoutLastNolo(); }
                                     if(                           nP!=N_NONE && !cI.isCarrier() && !cI.containsCarrier(nP) ) { viCarrierP.push_back(-1); cCurrP=cCurrP.withoutLastNolo(); nNewCarriers++; }
      N nL=cCurrL.getLastNonlocal(); if( i>-1 and i<iLowerA     && nL!=N_NONE && qPrev[i].getCat()==nL                      ) { viCarrierL.push_back(i);  cCurrL=cCurrL.withoutLastNolo(); }
                                     if(          i<iLowerA     && nL!=N_NONE && !cI.isCarrier() && !cI.containsCarrier(nL) ) { viCarrierL.push_back(-1); cCurrL=cCurrL.withoutLastNolo(); nNewCarriers++; }
      N nA=cCurrA.getLastNonlocal(); if( i>-1 and i<iLowerA     && nA!=N_NONE && qPrev[i].getCat()==nA                      ) { viCarrierA.push_back(i);  cCurrA=cCurrA.withoutLastNolo(); }
                                     if(          i<iLowerA     && nA!=N_NONE && !cI.isCarrier() && !cI.containsCarrier(nA) ) { viCarrierA.push_back(-1); cCurrA=cCurrA.withoutLastNolo(); nNewCarriers++; }
      N nB=cCurrB.getLastNonlocal(); if( i>-1 and i<iAncestorB  && nB!=N_NONE && qPrev[i].getCat()==nB                      ) { viCarrierB.push_back(i);  cCurrB=cCurrB.withoutLastNolo(); }
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
    /*
    const KSet& ksLchild = aLchild.getKSet();
    const KSet  ksParent = KSet( aLchild.getKSet(), -getDir(opL), j==0, evJ, viCarrierA, qPrev, (j==0) ? KSet() : qPrev.at(iAncestorB).getKSet() );
    const KSet  ksRchild( ksParent, getDir(opR) );
    */

    HVec hvParent, hvRchild;
    // If join, apply unaries going down from ancestor, then merge redirect of left child...
    if( j ) {
      hvParent.add( qPrev.at(iAncestorB).getHVec() ).applyUnariesTopDown( evJ, viCarrierA, qPrev ).addSynArg( -getDir(opL), aLchild.getHVec() );
      hvRchild.addSynArg(  getDir(opR), hvParent );
    }
    // If not join, merge redirect of left child...
    else {
      hvParent.addSynArg( -getDir(opL), aLchild.getHVec() );
      hvRchild.addSynArg(  getDir(opR), hvParent ); 
      hvParent.applyUnariesBottomUp( evJ, viCarrierA, qPrev );
    }

    //// B.2. Copy store state and add parent/preterm contexts to existing non-locals via extraction operation...
    for( int i=0; i<((f==0&&j==1)?iAncestorB:(f==0&&j==0)?iLowerA:(f==1&&j==1)?iAncestorB:iAncestorB+1); i++ ) {
      Sign& s = *emplace( end() ) = qPrev[i];
      if( i==iAncestorA and j==1 and qPrev[i].isDitto() and opR!='I' )            { s.setHVec() = hvParent; }  //s = Sign( ksParent, qPrev[i].getCat(), qPrev[i].getSide() ); }
      else if( viCarrierP.size()>0 and i==viCarrierP.back() and evF.top()!='\0' ) { viCarrierP.pop_back();
                                                                                    s.setHVec().addSynArg( getDir(evF.popTop()), aPretrm.getHVec() ); }
                                                                                    //s = Sign( KSet(aPretrm.getKSet(),getDir(evF.popTop()),qPrev[i].getKSet()), qPrev[i].getCat(), qPrev[i].getSide() ); }
      else if( viCarrierA.size()>0 and i==viCarrierA.back() and evJ.top()!='\0' ) { viCarrierA.pop_back();
                                                                                    s.setHVec().addSynArg( getDir(evJ.popTop()), hvParent ); }
                                                                                    //s = Sign( KSet(ksParent,getDir(evJ.popTop()),qPrev[i].getKSet()), qPrev[i].getCat(), qPrev[i].getSide() ); }
      else                                                                        { s = qPrev[i]; }
    }

    //// B.3. Add new non-locals with contexts from parent/rchild via new extraction or G/H/V operations...
    // If no join, add A carriers followed by new lowest A...
    if( j==0 ) {
      // Add A carriers...
      cCurrP = aPretrm.getCat();  cCurrA = cA;
      for( int i : viCarrierP ) if( i==-1 ) { if( STORESTATE_CHATTY ) cout<<"(adding carrierP for "<<cCurrP.getFirstNonlocal()<<" bc none above "<<iAncestorB<<")"<<endl;
                                              //*emplace( end() ) = Sign( KSet( aPretrm.getKSet(), getDir(evF.popTop()) ), cCurrP.getFirstNonlocal(), S_B );
                                              Sign& s = *emplace( end() ) = Sign( HVec(), cCurrP.getFirstNonlocal(), S_B );
                                              s.setHVec().addSynArg( getDir(evF.popTop()), aPretrm.getHVec() );
                                              cCurrP=cCurrP.withoutFirstNolo(); }
      for( int i : viCarrierA ) if( i==-1 ) { if( STORESTATE_CHATTY ) cout<<"(adding carrierA for "<<cCurrA.getFirstNonlocal()<<" bc none above "<<iAncestorB<<")"<<endl;
                                              //*emplace( end() ) = Sign( KSet( ksParent,          getDir(evJ.popTop()) ), cCurrA.getFirstNonlocal(), S_B );
                                              Sign& s = *emplace( end() ) = Sign( HVec(), cCurrA.getFirstNonlocal(), S_B );
                                              s.setHVec().addSynArg( getDir(evJ.popTop()), hvParent );
                                              cCurrA=cCurrA.withoutFirstNolo(); }
      // Add lowest A...
      *emplace( end() ) = Sign( (opR=='I') ? HVec(K_DITTO) : hvParent, cA, S_A );
//      if( cA.getLastNonlocal()==N("-vN") and viCarrierA[0]==-1 ) back().setHVec().addBankedUnaryTransform( "O" );
      iLowerA = size()-1;
    }
    // Add B carriers...
    N nA = cA.getLastNonlocal();  N nB = cB.getLastNonlocal();  N nL = aLchild.getCat().getLastNonlocal();
//    if( nB!=N_NONE and nB!=nA and viCarrierB[0]==-1 and viCarrierB.size()==0 ) cerr<<"WEIRD: "<<cB<<" got "<<nB<<" with iAncestorB="<<iAncestorB<<" in "<<qPrev<<" !"<<endl;
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
//        if( viCarrierL[0]>iAncestorB )              *emplace( end() ) = Sign( at(viCarrierL[0]).getHVec(),                                                        hvRchild, cB, S_B );  // Add right child kset as B (H rule).
        else cerr<<"ERROR StoreState 1019: should not happen, on '"<<qPrev<<" "<<f<<" "<<j<<" "<<evF<<" "<<evJ<<" "<<opL<<" "<<opR<<" "<<cA<<" "<<cB<<" "<<aPretrm<<" "<<aLchild<<"'"<<endl;
    //    cerr << "            " << qPrev << "  " << aLchild << "  ==(f" << f << ",j" << j << "," << opL << "," << opR << ")=>  " << *this << endl;
      } else {  // If j==1...
        Sign& s = *emplace( end() ) = Sign( hvRchild, cB, S_B );
        // If existing left carrier, integrate with sign...
        if( viCarrierL[0]!=-1 ) s.setHVec().add( qPrev.at(viCarrierL[0]).getHVec() );
        // If extraction...
        if( evF.top()!='\0' )   s.setHVec().addSynArg( getDir(evF.popTop()), aPretrm.getHVec() );
        /*
        // If existing left carrier, integrate with sign...
        if( evF.top()!='\0' and viCarrierL[0]!=-1 ) *emplace( end() ) = Sign( KSet( aPretrm.getKSet(), getDir(evF.popTop()), qPrev.at(viCarrierL[0]).getKSet() ), ksRchild, cB, S_B );
        // If new left carrier...
        else if(evF.top()!='\0' )                   *emplace( end() ) = Sign( KSet( aPretrm.getKSet(), getDir(evF.popTop()) ),                                    ksRchild, cB, S_B );
        // If no extraction...
        else                                        *emplace( end() ) = Sign( qPrev.at(viCarrierL[0]).getKSet(),                                                  ksRchild, cB, S_B );
        */
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

  PPredictor calcPretrmCatCondition ( F f, EVar e, K k_p_t ) const {
    if( FEATCONFIG & 1 ) return PPredictor( 0, f, (FEATCONFIG & 4) ? EVar("-") : e, at(size()-1).getCat(), (FEATCONFIG & 16384) ? cBot : k_p_t.getCat() );
    return             PPredictor( getDepth(), f, (FEATCONFIG & 4) ? EVar("-") : e, at(size()-1).getCat(), (FEATCONFIG & 16384) ? cBot : k_p_t.getCat() );
  }

  APredictor calcApexCatCondition ( F f, J j, EVar eF, EVar eJ, O opL, const LeftChildSign& aLchild ) const {
    if( FEATCONFIG & 1 ) return APredictor( 0, 0, j, (FEATCONFIG & 64) ? EVar("-") : eJ, (FEATCONFIG & 128) ? O('-') : opL, at(getAncestorBIndex(f)).getCat(), (j==0) ? aLchild.getCat() : cBot );
    return         APredictor( getDepth()+f-j, f, j, (FEATCONFIG & 64) ? EVar("-") : eJ, (FEATCONFIG & 128) ? O('-') : opL, at(getAncestorBIndex(f)).getCat(), (j==0) ? aLchild.getCat() : cBot );
  }

  BPredictor calcBaseCatCondition ( F f, J j, EVar eF, EVar eJ, O opL, O opR, CVar cParent, const LeftChildSign& aLchild ) const {
    if( FEATCONFIG & 1 ) return  BPredictor( 0, 0, 0, (FEATCONFIG & 64) ? EVar("-") : eJ, (FEATCONFIG & 128) ? O('-') : opL, (FEATCONFIG & 128) ? O('-') : opR, cParent, aLchild.getCat() );
    return          BPredictor( getDepth()+f-j, f, j, (FEATCONFIG & 64) ? EVar("-") : eJ, (FEATCONFIG & 128) ? O('-') : opL, (FEATCONFIG & 128) ? O('-') : opR, cParent, aLchild.getCat() );
  }

  void calcNPredictors( NPredictorSet& nps, const Sign& candidate, bool bcorefON, int antdist, bool bAdd=true ) const {
    //probably will look like Join model feature generation.ancestor is a sign, sign has T and Kset.
    //TODO add dependence to P model.  P category should be informed by which antecedent category was chosen here

    const HVec& hvB = at(size()-1).getHVec(); //contexts of lowest b (bdbar)
    for( int iA=0; iA<candidate.getHVec().size(); iA++ )  for( auto& antk : candidate.getHVec()[iA] ) { 
      if( bAdd || NPredictor::exists(antk.project(-iA),kNil) ) nps.setList().emplace_back( antk.project(-iA), kNil ); //add unary antecedent k feat, using kxk template
      for( int iB=0; iB<hvB.size(); iB++)  for( auto& currk : hvB[iB] ) {
        if( bAdd || NPredictor::exists(antk.project(-iA),currk.project(-iB)) ) nps.setList().emplace_back( antk.project(-iA), currk.project(-iB) ); //pairwise kxk feat
      }
    }
    for( int iB=0; iB<hvB.size(); iB++ )  for( auto& currk : hvB[iB] ) {
      if( bAdd || NPredictor::exists(kNil,currk.project(-iB)) ) nps.setList().emplace_back( kNil, currk.project(-iB) ); //unary ancestor k feat
    }

    if( bAdd || NPredictor::exists(candidate.getCat(),N_NONE) )                nps.setList().emplace_back( candidate.getCat(), N_NONE                ); // antecedent CVar
    if( bAdd || NPredictor::exists(N_NONE,at(size()-1).getCat()) )             nps.setList().emplace_back( N_NONE, at(size()-1).getCat()             ); // ancestor CVar
    if( bAdd || NPredictor::exists(candidate.getCat(),at(size()-1).getCat()) ) nps.setList().emplace_back( candidate.getCat(), at(size()-1).getCat() ); // pairwise T

    nps.setAntDist() = antdist;
    /*
    //loop over curr k:ksB
      npreds.emplace_back(currk, 1); //add unary anaphor (ancestor) k feat. 1 meants currk type feature
    CVar antt = candidate.getCat();
    npreds.emplace_back(antt,2); //add unary antecedent T 
    CVar currt = at(size()-1).getCat(); //type of lowest b (bdbar)
    npreds.emplace_back(currt,3); //add unary anaphor (ancestor) CVar
    */
    nps.setList().emplace_front(bias); //add bias term

    //corefON feature
    if (bcorefON == true) { 
      nps.setList().emplace_back(corefON);
    }
  }
};
const Sign StoreState::aTop( HVec(K::kTop), cTop, S_B );

////////////////////////////////////////////////////////////////////////////////

HVec& HVec::applyUnariesTopDown( EVar e, const vector<int>& viCarrierIndices, const StoreState& ss ) {
  for( uint i=0; e!=EVar::eNil; e=e.withoutTop() ) {
    if( e.top()>='0' and e.top()<='9' and i<viCarrierIndices.size() and viCarrierIndices[i++]!=-1 )  addSynArg( -getDir(e.top()), ss.at(viCarrierIndices[i]).getHVec() );
    else if( e.top()=='O' or e.top()=='V' )  swap(1,2);
  }
  return *this;
}

HVec& HVec::applyUnariesBottomUp( EVar e, const vector<int>& viCarrierIndices, const StoreState& ss ) {
  for( uint i=viCarrierIndices.size()-1; e!=EVar::eNil; e=e.withoutBot() ) {
    if( e.bot()>='0' and e.bot()<='9' and i>=0 and viCarrierIndices[i--]!=-1 )  addSynArg( -getDir(e.bot()), ss.at(viCarrierIndices[i]).getHVec() );
    else if( e.bot()=='O' or e.bot()=='V' )  swap(1,2);
  }
  return *this;
}

/*
KSet::KSet ( const KSet& ksToProject, int iProj, bool bUp, EVar e, const vector<int>& viCarrierIndices, const StoreState& ss, const KSet& ksNoProject ) {
//cout<<"tomake "<<ksToProject<<" with eBanked="<<ksToProject.eBankedUnaryTransforms<<" iProj="<<iProj<<" bup="<<bUp<<" e="<<e<<" "<<viCarrierIndices.size()<<" "<<ss<<" "<<ksNoProject<<endl;
  // Determine number of carrier contexts...
  int nCarrierContexts=0;  for( int iCarrierIndex : viCarrierIndices ) if( iCarrierIndex>=0 ) nCarrierContexts += ss.at(iCarrierIndex).getKSet().size();
  // Reserve size to avoid costly reallocation...
  reserve( ksToProject.size() + nCarrierContexts + ksNoProject.size() );
  // Propagate banked unary transforms...
  if( 0==iProj ) eBankedUnaryTransforms = ksToProject.eBankedUnaryTransforms;

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
*/

////////////////////////////////////////////////////////////////////////////////

LeftChildSign::LeftChildSign ( const StoreState& qPrev, F f, EVar eF, const Sign& aPretrm ) {
//    int         iCarrierB  = qPrev.getAncestorBCarrierIndex( 1 );
    int         iAncestorB = qPrev.getAncestorBIndex(f);
    CVar        cCurrB     = qPrev.at(iAncestorB).getCat();
    vector<int> viCarrierB;  viCarrierB.reserve(4);
    for( int i=qPrev.size(); i>=-1; i-- ) {
      N nB=cCurrB.getLastNonlocal(); if( i>-1 && i<iAncestorB  && nB!=N_NONE && qPrev[i].getCat()==nB                  ) { viCarrierB.push_back(i);  cCurrB=cCurrB.withoutLastNolo(); }
                                     if(         i<=iAncestorB && nB!=N_NONE && !qPrev[i].getCat().isCarrier() && !qPrev[i].getCat().containsCarrier(nB) ) { viCarrierB.push_back(-1); cCurrB=cCurrB.withoutLastNolo(); }
    }
    //cout<<" viCarrierB="; for( int i : viCarrierB ) cout<<" "<<i; cout<<endl;
    const Sign& aAncestorA = qPrev.at( qPrev.getAncestorAIndex(1) );
    const Sign& aAncestorB = qPrev.at( qPrev.getAncestorBIndex(1) );
//    const KSet& ksExtrtn   = (iCarrierB<0) ? KSet() : qPrev.at(iCarrierB).getKSet();
    setSide() = S_A;
    if( f==1 )                          { setCat() = aPretrm.getCat();  setHVec().add( aPretrm.getHVec() ).applyUnariesBottomUp( eF, viCarrierB, qPrev ); }
    else if( qPrev.size()<=0 )          { *this = StoreState::aTop; }
    else if( not aAncestorA.isDitto() ) { setCat() = aAncestorA.getCat();  setHVec().add( aAncestorA.getHVec() ).applyUnariesBottomUp( eF, viCarrierB, qPrev ); }
    else                                { setCat() = aAncestorA.getCat();  setHVec().add( aPretrm.getHVec() ).applyUnariesBottomUp( eF, viCarrierB, qPrev ).add( aAncestorB.getHVec() ); }

    /*
    *this = (f==1 && eF!=EVar::eNil)                  ? Sign( KSet(KSet(),0,true,eF,viCarrierB,qPrev,aPretrm.getKSet()), aPretrm.getCat(), S_A )
                                                  //Sign( KSet(ksExtrtn,-getDir(eF),aPretrm.getKSet()), aPretrm.getCat(), S_A )
          : (f==1)                                    ? aPretrm                             // if fork, lchild is preterm.
          : (qPrev.size()<=0)                         ? StoreState::aTop                    // if no fork and stack empty, lchild is T (NOTE: should not happen).
          : (!aAncestorA.isDitto() && eF!=EVar::eNil) ? Sign( KSet(KSet(),0,true,eF,viCarrierB,qPrev,aAncestorA.getKSet()), aAncestorA.getCat(), S_A )
                                                  //Sign( KSet(ksExtrtn,-getDir(eF),aAncestorA.getKSet()), aAncestorA.getCat(), S_A )
          : (!aAncestorA.isDitto())                   ? aAncestorA                          // if no fork and stack exists and last apex context set is not ditto, return last apex.
          :                                             Sign( KSet(KSet(),0,true,eF,viCarrierB,qPrev,aAncestorB.getKSet()), aPretrm.getKSet(), aAncestorA.getCat(), S_A );
                                                  //Sign( KSet(ksExtrtn,-getDir(eF),aAncestorB.getKSet()), aPretrm.getKSet(), aAncestorA.getCat(), S_A );  // otherwise make new context set.
//  if( aPretrm.getCat().getLastNonlocal()==N("-vN") and viCarrierB.size()==0 ) cerr<<"WEIRD2: "<<aPretrm.getCat().getLastNonlocal()<<" with iAncestorB="<<iAncestorB<<" in "<<qPrev<<" !"<<endl;
  if( aPretrm.getCat().getLastNonlocal()==N("-vN") and (viCarrierB.size()==0 or viCarrierB[0]==-1) ) setKSet().addBankedUnaryTransform( "O" );
//cout<<"lchild created with banked "<<getKSet().getBankedUnaryTransform()<<endl;
  */
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
    const Sign& getPrtrm ()           const { return first(); }
    F getF ()                         const { return second(); }
    EVar getForkE ()                  const { return third(); }
    K getForkK ()                     const { return fourth(); }
    const JResponse& getJResp()       const { return fifth(); }
    const StoreState& getStoreState() const { return sixth(); }
    const Delimited<int>& getI()      const { return seventh(); }
};

////////////////////////////////////////////////////////////////////////////////

class FPredictorVec : public list<unsigned int> {

  public:

    template<class FM>  // J model is template variable to allow same behavior for const and non-const up until getting predictor indices
    FPredictorVec( FM& fm, const HVec& hvAnt, bool nullAnt, const StoreState& ss ) {
      int d = (FEATCONFIG & 1) ? 0 : ss.getDepth(); // max used depth - (dbar)
      const HVec& hvB = ( ss.at(ss.size()-1).getHVec().size() > 0 ) ? ss.at(ss.size()-1).getHVec() : hvBot; //contexts of lowest b (bdbar)
      int iCarrier = ss.getAncestorBCarrierIndex( 1 ); // get lowest nonlocal above bdbar
      const HVec& hvF = ( iCarrier >= 0 ) ? ss.at(iCarrier).getHVec() : HVec();
      emplace_back( fm.getPredictorIndex( "Bias" ) );  // add bias
      if( STORESTATE_TYPE ) emplace_back( fm.getPredictorIndex( d, ss.at(ss.size()-1).getCat() ) ); 
      if( !(FEATCONFIG & 2) ) {
        for( uint iB=0; iB<hvB.size();   iB++ )  for( auto& kB : hvB[iB] )   emplace_back( fm.getPredictorIndex( d, kNil,            kB.project(-iB), kNil ) );
        for( uint iF=0; iF<hvF.size();   iF++ )  for( auto& kF : hvF[iF] )   emplace_back( fm.getPredictorIndex( d, kF.project(-iF), kNil,            kNil ) );
        for( uint iA=0; iA<hvAnt.size(); iA++ )  for( auto& kA : hvAnt[iA] ) emplace_back( fm.getPredictorIndex( d, kNil,            kNil,            kA.project(-iA) ) );
      }
      if( nullAnt ) emplace_back( fm.getPredictorIndex( "corefOFF" ) );
      else          emplace_back( fm.getPredictorIndex( "corefON"  ) ); 
    }
};

////////////////////////////////////////////////////////////////////////////////

class FModel {

  typedef DelimitedTrip<psX,F,psAmpersand,Delimited<EVar>,psAmpersand,Delimited<K>,psX> FEK;

  private:

    arma::mat matF;                              // matrix itself

    unsigned int iNextPredictor = 0;             // predictor and response next-pointers
    unsigned int iNextResponse  = 0;

    map<string,unsigned int> msi;                // predictor indices for ad-hoc feature
    map<unsigned int,string> mis;
    map<quad<D,K,K,K>,unsigned int> mdkkki;      // predictor indices for k-context tuples
    map<unsigned int,quad<D,K,K,K>> midkkk;
    map<pair<D,CVar>,unsigned int> mdci;         // predictor indices for category tuples
    map<unsigned int,pair<D,CVar>> midc;

    map<FEK,unsigned int> mfeki;                 // response indices
    map<unsigned int,FEK> mifek;

  public:

    FModel( ) { }
    FModel( istream& is ) {
      list< trip< unsigned int, unsigned int, double > > l;    // store elements on list until we know dimensions of matrix
      while( is.peek()=='F' ) {
        auto& prw = *l.emplace( l.end() );
        is >> "F ";
        if( is.peek()=='a' )   { Delimited<string> s;   is >> "a" >> s >> " : ";                          prw.first()  = getPredictorIndex( s );             }
        else{
          D d;                                          is >> "d" >> d >> "&";
          if( is.peek()=='t' ) { Delimited<CVar> c;     is >> "t" >> c >> " : ";                          prw.first()  = getPredictorIndex( d, c );          }
          else                 { Delimited<K> kN,kF,kA; is >> kN >> "&" >> kF >> "&" >> kA >> " : ";      prw.first()  = getPredictorIndex( d, kN, kF, kA ); }
        }
        F f; Delimited<EVar> e; Delimited<K> k;         is >> "f" >> f >> "&" >> e >> "&" >> k >> " = ";  prw.second() = getResponseIndex( f, e, k );
        Delimited<double> w;                            is >> w >> "\n";                                  prw.third()  = w;
      }

      if( l.size()==0 ) cerr << "ERROR: No F items found." << endl;
      matF.zeros ( mifek.size(), iNextPredictor );
      for( auto& prw : l ) { matF( prw.second(), prw.first() ) = prw.third(); }
    }

    unsigned int getPredictorIndex( const string& s ) {
      const auto& it = msi.find( s );  if( it != msi.end() ) return( it->second );
      msi[ s ] = iNextPredictor;  mis[ iNextPredictor ] = s;  return( iNextPredictor++ );
    }
    unsigned int getPredictorIndex( const string& s ) const {                  // const version with closed predictor domain
      const auto& it = msi.find( s );  return( ( it != msi.end() ) ? it->second : 0 );
    }

    unsigned int getPredictorIndex( D d, K kF, K kA, K kL ) {
      const auto& it = mdkkki.find( quad<D,K,K,K>(d,kF,kA,kL) );  if( it != mdkkki.end() ) return( it->second );
      mdkkki[ quad<D,K,K,K>(d,kF,kA,kL) ] = iNextPredictor;  midkkk[ iNextPredictor ] = quad<D,K,K,K>(d,kF,kA,kL);  return( iNextPredictor++ );
    }
    unsigned int getPredictorIndex( D d, K kF, K kA, K kL ) const {            // const version with closed predictor domain
      const auto& it = mdkkki.find( quad<D,K,K,K>(d,kF,kA,kL) );  return( ( it != mdkkki.end() ) ? it->second : 0 );
    }

    unsigned int getPredictorIndex( D d, CVar c ) {
      const auto& it = mdci.find( pair<D,CVar>(d,c) );  if( it != mdci.end() ) return( it->second );
      mdci[ pair<D,CVar>(d,c) ] = iNextPredictor;  midc[ iNextPredictor ] = pair<D,CVar>(d,c);  return( iNextPredictor++ );
    }
    unsigned int getPredictorIndex( D d, CVar c ) const {                      // const version with closed predictor domain
      const auto& it = mdci.find( pair<D,CVar>(d,c) );  return( ( it != mdci.end() ) ? it->second : 0 );
    }

    unsigned int getResponseIndex( F f, EVar e, K k ) {
      const auto& it = mfeki.find( FEK(f,e,k) );  if( it != mfeki.end() ) return( it->second );
      mfeki[ FEK(f,e,k) ] = iNextResponse;  mifek[ iNextResponse ] = FEK(f,e,k);  return( iNextResponse++ );
    }
    unsigned int getResponseIndex( F f, EVar e, K k ) const {                  // const version with closed predictor domain
      const auto& it = mfeki.find( FEK(f,e,k) );  return( ( it != mfeki.end() ) ? it->second : uint(-1) );
    }

    const FEK& getFEK( unsigned int i ) const {
      return mifek.find( i )->second;
    }

    arma::vec calcResponses( const FPredictorVec& lfpredictors ) const {
      arma::vec flogresponses = arma::zeros( matF.n_rows );
      for ( auto& fpredr : lfpredictors ) if ( fpredr < matF.n_cols ) flogresponses += matF.col( fpredr );
      arma::vec fresponses = arma::exp( flogresponses );
      double fnorm = arma::accu( fresponses );                                 // fork normalization term (denominator)

      // Replace overflowing distribs by max...
      if( fnorm == 1.0/0.0 ) {
        uint ind_max=0; for( uint i=0; i<flogresponses.size(); i++ ) if( flogresponses(i)>flogresponses(ind_max) ) ind_max=i;
        flogresponses -= flogresponses( ind_max );
        fresponses = arma::exp( flogresponses );
        fnorm = arma::accu( fresponses ); //accumulate is sum over elements
      } //closes if fnorm
      return fresponses / fnorm;
    }

    friend ostream& operator<<( ostream& os, const pair< const FModel&, const FPredictorVec& >& mv ) {
      for( const auto& i : mv.second ) {
        if( &i != &mv.second.front() ) os << ",";
        const auto& itK = mv.first.midkkk.find(i);
       	if( itK != mv.first.midkkk.end() ) { os << "d" << itK->second.first() << "&" << itK->second.second() << "&" << itK->second.third() << "&" << itK->second.fourth() << "=1"; continue; }
        const auto& itC = mv.first.midc.find(i);
        if( itC != mv.first.midc.end()   ) { os << "d" << itC->second.first << "&t" << itC->second.second << "=1"; continue; }
        const auto& itS = mv.first.mis.find(i);
        if( itS != mv.first.mis.end()    ) { os << "a" << itS->second << "=1"; }
      }
      return os;
    }

    unsigned int getNumPredictors( ) { return iNextPredictor; }
    unsigned int getNumResponses(  ) { return iNextResponse;  }
};

////////////////////////////////////////////////////////////////////////////////

class JPredictorVec : public list<unsigned int> {

  public:

    template<class JM>  // J model is template variable to allow same behavior for const and non-const up until getting predictor indices
    JPredictorVec( JM& jm, F f, EVar eF, const LeftChildSign& aLchild, const StoreState& ss ) {
      int d = (FEATCONFIG & 1) ? 0 : ss.getDepth()+f;
      int iCarrierB = ss.getAncestorBCarrierIndex( f );
      const Sign& aAncstr  = ss.at( ss.getAncestorBIndex(f) );
      const HVec& hvAncstr = ( aAncstr.getHVec().size()==0 ) ? hvBot : aAncstr.getHVec();
      const HVec& hvFiller = ( iCarrierB<0                 ) ? hvBot : ss.at( iCarrierB ).getHVec();
      const HVec& hvLchild = ( aLchild.getHVec().size()==0 ) ? hvBot : aLchild.getHVec() ;
      emplace_back( jm.getPredictorIndex( "Bias" ) );  // add bias
      if( STORESTATE_TYPE ) emplace_back( jm.getPredictorIndex( d, aAncstr.getCat(), aLchild. getCat() ) );
      if( !(FEATCONFIG & 32) ) {
        for( uint iA=0; iA<hvAncstr.size(); iA++ ) for( auto& kA : hvAncstr[iA] )
          for( uint iL=0; iL<hvLchild.size(); iL++ ) for( auto& kL : hvLchild[iL] ) emplace_back( jm.getPredictorIndex( d, kNil, kA.project(-iA), kL.project(-iL) ) );
        for( uint iF=0; iF<hvFiller.size(); iF++ ) for( auto& kF : hvFiller[iF] )
          for( uint iA=0; iA<hvAncstr.size(); iA++ ) for( auto& kA : hvAncstr[iA] ) emplace_back( jm.getPredictorIndex( d, kF.project(-iF), kA.project(-iA), kNil ) );
        for( uint iF=0; iF<hvFiller.size(); iF++ ) for( auto& kF : hvFiller[iF] )
          for( uint iL=0; iL<hvLchild.size(); iL++ ) for( auto& kL : hvLchild[iL] ) emplace_back( jm.getPredictorIndex( d, kF.project(-iF), kNil, kL.project(-iL) ) );
      }
    }
};

////////////////////////////////////////////////////////////////////////////////

class JModel {

  typedef DelimitedQuad<psX,J,psAmpersand,Delimited<EVar>,psAmpersand,O,psAmpersand,O,psX> JEOO;

  private:

    arma::mat matJ;                              // matrix itself

    unsigned int iNextPredictor = 0;             // predictor and response next-pointers
    unsigned int iNextResponse  = 0;

    map<string,unsigned int> msi;                // predictor indices for ad-hoc feature
    map<unsigned int,string> mis;
    map<quad<D,K,K,K>,unsigned int> mdkkki;      // predictor indices for k-context tuples
    map<unsigned int,quad<D,K,K,K>> midkkk;
    map<trip<D,CVar,CVar>,unsigned int> mdcci;   // predictor indices for category tuples
    map<unsigned int,trip<D,CVar,CVar>> midcc;

    map<JEOO,unsigned int> mjeooi;               // response indices
    map<unsigned int,JEOO> mijeoo;

  public:

    JModel( ) { }
    JModel( istream& is ) {
      list< trip< unsigned int, unsigned int, double > > l;    // store elements on list until we know dimensions of matrix
      while( is.peek()=='J' ) {
        auto& prw = *l.emplace( l.end() );
	is >> "J ";
	if( is.peek()=='a' )   { Delimited<string> s;   is >> "a" >> s >> " : ";                                        prw.first()  = getPredictorIndex( s );             }
        else{
          D d;                                          is >> "d" >> d >> "&";
          if( is.peek()=='t' ) { Delimited<CVar> cA,cL; is >> "t" >> cA >> "&t" >> cL >> " : ";                         prw.first()  = getPredictorIndex( d, cA, cL );     }
          else                 { Delimited<K> kF,kA,kL; is >> kF >> "&" >> kA >> "&" >> kL >> " : ";                    prw.first()  = getPredictorIndex( d, kF, kA, kL ); }
        }
        J j; Delimited<EVar> e; O oL,oR;                is >> "j" >> j >> "&" >> e >> "&" >> oL >> "&" >> oR >> " = ";  prw.second() = getResponseIndex( j, e, oL, oR );
        Delimited<double> w;                            is >> w >> "\n";                                                prw.third()  = w;
      }

      if( l.size()==0 ) cerr << "ERROR: No J items found." << endl;
      matJ.zeros ( mijeoo.size(), iNextPredictor );
      for( auto& prw : l ) { matJ( prw.second(), prw.first() ) = prw.third(); }
    }

    unsigned int getPredictorIndex( const string& s ) {
      const auto& it = msi.find( s );  if( it != msi.end() ) return( it->second );
      msi[ s ] = iNextPredictor;  mis[ iNextPredictor ] = s;  return( iNextPredictor++ );
    }
    unsigned int getPredictorIndex( const string& s ) const {                  // const version with closed predictor domain
      const auto& it = msi.find( s );  return( ( it != msi.end() ) ? it->second : 0 );
    }

    unsigned int getPredictorIndex( D d, K kF, K kA, K kL ) {
      const auto& it = mdkkki.find( quad<D,K,K,K>(d,kF,kA,kL) );  if( it != mdkkki.end() ) return( it->second );
      mdkkki[ quad<D,K,K,K>(d,kF,kA,kL) ] = iNextPredictor;  midkkk[ iNextPredictor ] = quad<D,K,K,K>(d,kF,kA,kL);  return( iNextPredictor++ );
    }
    unsigned int getPredictorIndex( D d, K kF, K kA, K kL ) const {            // const version with closed predictor domain
      const auto& it = mdkkki.find( quad<D,K,K,K>(d,kF,kA,kL) );  return( ( it != mdkkki.end() ) ? it->second : 0 );
    }

    unsigned int getPredictorIndex( D d, CVar cA, CVar cL ) {
      const auto& it = mdcci.find( trip<D,CVar,CVar>(d,cA,cL) );  if( it != mdcci.end() ) return( it->second );
      mdcci[ trip<D,CVar,CVar>(d,cA,cL) ] = iNextPredictor;  midcc[ iNextPredictor ] = trip<D,CVar,CVar>(d,cA,cL);  return( iNextPredictor++ );
    }
    unsigned int getPredictorIndex( D d, CVar cA, CVar cL ) const {            // const version with closed predictor domain
      const auto& it = mdcci.find( trip<D,CVar,CVar>(d,cA,cL) );  return( ( it != mdcci.end() ) ? it->second : 0 );
    }

    unsigned int getResponseIndex( J j, EVar e, O oL, O oR ) {
      const auto& it = mjeooi.find( JEOO(j,e,oL,oR) );  if( it != mjeooi.end() ) return( it->second );
      mjeooi[ JEOO(j,e,oL,oR) ] = iNextResponse;  mijeoo[ iNextResponse ] = JEOO(j,e,oL,oR);  return( iNextResponse++ );
    }
    unsigned int getResponseIndex( J j, EVar e, O oL, O oR ) const {           // const version with closed predictor domain
      const auto& it = mjeooi.find( JEOO(j,e,oL,oR) );  return( ( it != mjeooi.end() ) ? it->second : uint(-1) );
    }

    const JEOO& getJEOO( unsigned int i ) const {
      return mijeoo.find( i )->second;
    }

    arma::vec calcResponses( const JPredictorVec& ljpredictors ) const {
      arma::vec jlogresponses = arma::zeros( matJ.n_rows );
      for ( auto& jpredr : ljpredictors ) if ( jpredr < matJ.n_cols ) jlogresponses += matJ.col( jpredr );
      arma::vec jresponses = arma::exp( jlogresponses );
      double jnorm = arma::accu( jresponses );                                 // join normalization term (denominator)

      // Replace overflowing distribs by max...
      if( jnorm == 1.0/0.0 ) {
        uint ind_max=0; for( uint i=0; i<jlogresponses.size(); i++ ) if( jlogresponses(i)>jlogresponses(ind_max) ) ind_max=i;
        jlogresponses -= jlogresponses( ind_max );
        jresponses = arma::exp( jlogresponses );
        jnorm = arma::accu( jresponses ); //accumulate is sum over elements
      } //closes if jnorm
      return jresponses / jnorm;
    }

    friend ostream& operator<<( ostream& os, const pair< const JModel&, const JPredictorVec& >& mv ) {
      for( const auto& i : mv.second ) {
        if( &i != &mv.second.front() ) os << ",";
        const auto& itK = mv.first.midkkk.find(i);
       	if( itK != mv.first.midkkk.end() ) { os << "d" << itK->second.first() << "&" << itK->second.second() << "&" << itK->second.third() << "&" << itK->second.fourth() << "=1"; continue; }
        const auto& itC = mv.first.midcc.find(i);
        if( itC != mv.first.midcc.end()  ) { os << "d" << itC->second.first() << "&t" << itC->second.second() << "&t" << itC->second.third() << "=1"; continue; }
        const auto& itS = mv.first.mis.find(i);
        if( itS != mv.first.mis.end()    ) { os << "a" << itS->second << "=1"; }
      }
      return os;
    }

    unsigned int getNumPredictors( ) { return iNextPredictor; }
    unsigned int getNumResponses(  ) { return iNextResponse;  }
};

////////////////////////////////////////////////////////////////////////////////

