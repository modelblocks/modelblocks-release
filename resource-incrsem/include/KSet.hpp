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

////////////////////////////////////////////////////////////////////////////////

/*

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

*/

////////////////////////////////////////////////////////////////////////////////

typedef DelimitedVector<psLBrack,Delimited<K>,psComma,psRBrack> KVec;

class EMat {
 public:
  EMat( )             { }
  EMat( istream& is ) { }
//  KVec operator() ( K k ) { KVec kv; kv.push_back(k); return kv; }
};

class OFunc {
 public:
  OFunc( )             { }
  OFunc( istream& is ) { }
//  KVec operator() ( int iDir, const KVec& kv ) { KVec kvOut; for( auto& k : kv ) kvOut.push_back( k.project(iDir) ); }
};

class HVec : public DelimitedVector<psX,KVec,psX,psX> {

 public:

  static const HVec hvDitto;

  // Constructors...
  HVec ( )       : DelimitedVector<psX,KVec,psX,psX>() { }
  HVec ( int i ) : DelimitedVector<psX,KVec,psX,psX>( i ) { }
  HVec ( K k, const EMat& em = EMat(), const OFunc& of = OFunc() )   : DelimitedVector<psX,KVec,psX,psX>( k.getCat().getSynArgs()+1 ) {
    at(0).emplace_back( k );  for( unsigned int arg=1; arg<k.getCat().getSynArgs()+1; arg++ ) at(arg).emplace_back( k.project(arg) );
  }
//  HVec& operator+= ( const HVec& hv ) {
  HVec& add( const HVec& hv ) {
    for( unsigned int arg=0; arg<size() and arg<hv.size(); arg++ ) at(arg).insert( at(arg).end(), hv.at(arg).begin(), hv.at(arg).end() );
    return *this;
  }
//  HVec& operator+= ( const Redirect& r ) {
  HVec& addSynArg( int iDir, const HVec& hv ) {
    if     ( iDir == 0                 ) add( hv );
    else if( iDir < 0 and -iDir<size() ) at(-iDir).insert( at(-iDir).end(), hv.at( 0  ).begin(), hv.at( 0  ).end() );
    else if( iDir<hv.size()            ) at( 0   ).insert( at( 0   ).end(), hv.at(iDir).begin(), hv.at(iDir).end() );
    return *this;
  }
  HVec& swap( int i, int j ) {
    if     ( size() >= 3 ) { auto kv = at(i);  at(i) = at(j);  at(j) = kv; }
    else if( size() >= 2 ) at(i).clear();
    return *this;
  }
  HVec& applyUnariesTopDn( EVar e, const vector<int>& viCarrierIndices, const StoreState& ss );
  HVec& applyUnariesBotUp( EVar e, const vector<int>& viCarrierIndices, const StoreState& ss );
  bool isDitto ( ) const { return ( size()>0 and front().size()>0 and front().front()==K_DITTO ); }
};
const HVec hvTop = HVec( K::kTop );
const HVec hvBot = HVec( K::kBot );
const HVec HVec::hvDitto( K_DITTO );

////////////////////////////////////////////////////////////////////////////////

