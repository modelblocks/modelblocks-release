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

#include <iostream>
#include <fstream>
#include "nl-cpt.h"

const int MAX_ITER = 20;

template<class C,class V,class P>
//class UnorderedModel : public SimpleHash<C,SimpleHash<V,P> > {
class UnorderedModel : public SimpleMap<C,SimpleMap<V,P> > {
 private:
  String sLbl;
 public:
  // Constructor methods...
  UnorderedModel(const char* ps) : sLbl(ps) { }
  // Extractor methods...
  const SimpleMap<V,P>& getDistrib ( const C& c ) const { return get(c); }
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
  void dump ( ) {
    for ( typename SimpleMap<C,SimpleMap<V,P> >::const_iterator i=SimpleMap<C,SimpleMap<V,P> >::begin(); i!=SimpleMap<C,SimpleMap<V,P> >::end(); i++ )
      for ( typename SimpleMap<V,P>::const_iterator j=i->second.begin(); j!=i->second.end(); j++ )
        cout << sLbl << " " << i->first << " : " <<j->first << " = " << j->second << "\n";
  }
  void normalize ( ) {
    for ( typename SimpleMap<C,SimpleMap<V,P> >::iterator i=SimpleMap<C,SimpleMap<V,P> >::begin(); i!=SimpleMap<C,SimpleMap<V,P> >::end(); i++ ) {
      P prTot = 0.0;
      for ( typename SimpleMap<V,P>::const_iterator j=i->second.begin(); j!=i->second.end(); j++ )
        prTot += j->second;
      for ( typename SimpleMap<V,P>::iterator j=i->second.begin(); j!=i->second.end(); j++ )
        j->second /= prTot;
    }
  }
};

template<class C,class V,class P>
class OrderedModel : public SimpleMap<C,Array<pair<V,P> > > {
 private:
  String sLbl;
 public:
  // Constructor methods...
  OrderedModel(const char* ps) : sLbl(ps) { }
  // Extractor methods...
  const Array<pair<V,P> >& getDistrib ( const C& c ) const { return get(c); }
  // I/O methods...
  friend pair<StringInput,OrderedModel<C,V,P>*> operator>> ( const StringInput ps, OrderedModel<C,V,P>& m ) { return pair<StringInput,OrderedModel<C,V,P>*>(ps,&m); }
  friend StringInput operator>>( pair<StringInput,OrderedModel<C,V,P>*> si_m, const char* psD) {
    StringInput          si =  si_m.first;
    OrderedModel<C,V,P>& m  = *si_m.second;
    if (si == StringInput()) return si;
    C c;  V v;  LogProb pr;
    si = si >> m.sLbl.c_array() >> " " >> c >> " : " >> v >> " = " >> pr >> "\0";
    if ( si != StringInput() ) m.set(c).add() = pair<V,P>(v,pr);
    return si;
  }
  void dump ( ) {
    for ( typename SimpleMap<C,Array<pair<V,P> > >::const_iterator i=SimpleMap<C,Array<pair<V,P> > >::begin(); i!=SimpleMap<C,Array<pair<V,P> > >::end(); i++ )
      for ( unsigned int j=0; j<i->second.size(); j++ )
        cout << sLbl << " " << i->first << " : " << i->second.get(j).first << " = " << i->second.get(j).second << "\n";
  }
  void normalize ( ) {
    for ( typename SimpleMap<C,Array<pair<V,P> > >::iterator i=SimpleMap<C,Array<pair<V,P> > >::begin(); i!=SimpleMap<C,Array<pair<V,P> > >::end(); i++ ) {
      P prTot = 0.0;
      for ( unsigned int j=0; j<i->second.size(); j++ )
        prTot += i->second.get(j).second;
      for ( unsigned int j=0; j<i->second.size(); j++ )
        i->second.set(j).second /= prTot;
    }
  }
};


///////////////////////////////////////////////////////////////////////////////


typedef int PhaseNum;
typedef int ChoiceNum;

char psX[]     = "";
char psSpace[] = " ";
char psSlash[] = "/";
char psDot[]   = ".";
char psSemi[]  = ";";
char psComma[] = ",";


DiscreteDomain<int> domB;
class B : public DiscreteDomainRV<int,domB> {
 public:
  B ( )                : DiscreteDomainRV<int,domB> ( )    { }
  B ( const char* ps ) : DiscreteDomainRV<int,domB> ( ps ) { }
};
const B B_L("L");
const B B_R("R");


DiscreteDomain<int> domD;
class D : public DiscreteDomainRV<int,domD> {
 public:
  D ( )                : DiscreteDomainRV<int,domD> ( )    { }
  D ( int i )          : DiscreteDomainRV<int,domD> ( i )  { }
  D ( const char* ps ) : DiscreteDomainRV<int,domD> ( ps ) { }
};
const D D_0("0");
const D D_1("1");
const D D_2("2");
const D D_3("3");
const D D_4("4");
const D D_5("5");


DiscreteDomain<int> domC;
class C : public DiscreteDomainRV<int,domC> {
 private:
  static SimpleMap<C,int> mcbB;
  static SimpleMap<C,int> mcbE;
 public:
  C ( )                : DiscreteDomainRV<int,domC> ( )    { }
  C ( const char* ps ) : DiscreteDomainRV<int,domC> ( ps ) { if(!mcbB.contains(*this)){mcbB[*this]=(NULL!=strstr(ps,"-b")); mcbE[*this]=(NULL!=strstr(ps,"-e"));} } // don't need??
  bool isB() const { return mcbB[*this]; }
  bool isE() const { return mcbE[*this]; }
};
SimpleMap<C,int> C::mcbB;
SimpleMap<C,int> C::mcbE;
const C C_NIL("-");


DiscreteDomain<int> domE;
class E : public DiscreteDomainRV<int,domE> {
 public:
  E ( )                : DiscreteDomainRV<int,domE> ( )    { }
  E ( const char* ps ) : DiscreteDomainRV<int,domE> ( ps ) { }
};


DiscreteDomain<int> domL;
class L : public DiscreteDomainRV<int,domL> {
 public:
  L ( )                : DiscreteDomainRV<int,domL> ( )    { }
  L ( const char* ps ) : DiscreteDomainRV<int,domL> ( ps ) { }
};


typedef C F;
const F F_NIL("-");


class Q : public DelimitedJoint2DRV<psX,F,psSlash,F,psX> {
 public:
  Q ( )                : DelimitedJoint2DRV<psX,F,psSlash,F,psX> ( )    { }
  Q ( const char* ps ) : DelimitedJoint2DRV<psX,F,psSlash,F,psX> ( ps ) { }
  F&       setAct ( )       { return first;  }
  F&       setAwa ( )       { return second; }
  const F& getAct ( ) const { return first;  }
  const F& getAwa ( ) const { return second; }
};
const Q Q_BOT("-/-");
const Q Q_TOP("REST/REST");
Q Q_SCRATCH;


class Y : public DelimitedJoint3DRV<psX,B,psSemi,DelimitedJointArrayRV<4,psSemi,Q>,psSemi,F,psX> {
 public:
  Y ( )                : DelimitedJoint3DRV<psX,B,psSemi,DelimitedJointArrayRV<4,psSemi,Q>,psSemi,F,psX> ( )    { }
  Y ( const char* ps ) : DelimitedJoint3DRV<psX,B,psSemi,DelimitedJointArrayRV<4,psSemi,Q>,psSemi,F,psX> ( ps ) { }
  int      getD ( )       const { return ( (F()!=second.get(3).getAwa()) ? 4 :
                                           (F()!=second.get(2).getAwa()) ? 3 :
                                           (F()!=second.get(1).getAwa()) ? 2 :
                                           (F()!=second.get(0).getAwa()) ? 1 : 0 ); }
  B&       setB ( )             { return first; }
  Q&       setQ ( int d )       { return (d<0) ? Q_SCRATCH : (d<4) ? second.set(d) : Q_SCRATCH; }
  F&       setP ( )             { return third; }
  const Q& getQ ( int d ) const { return (d<0) ? Q_TOP     : (d<4) ? second.get(d) : Q_BOT;     }
  const F& getP ( )       const { return third; }
};
const Y Y_INIT("0;-/-;-/-;-/-;-/-;-");


DiscreteDomain<int> domX;
class X : public DiscreteDomainRV<int,domX> {
 public:
  X ( )                : DiscreteDomainRV<int,domX> ( )    { }
  X ( const char* ps ) : DiscreteDomainRV<int,domX> ( ps ) { }
};
const X X_NIL("");
const X X_FINAL("the");



////////////////////////////////////////////////////////////////////////////////


int main ( int nArgs, char* argv[] ) {

  typedef UnorderedModel<MapKey3D<B,D,C>,MapKey2D<C,C>,Prob> BDCtoCC;
  typedef UnorderedModel<MapKey2D<D,C>,C,Prob> DCtoC;
  typedef UnorderedModel<MapKey2D<D,C>,MapKey2D<C,C>,Prob> DCtoCC;
  typedef UnorderedModel<MapKey3D<D,C,C>,C,Prob> DCCtoC;
  typedef UnorderedModel<MapKey3D<D,C,C>,MapKey2D<C,C>,Prob> DCCtoCC;

  BDCtoCC mCC("CC");

  char psBuff[1000];
  // For each line of model file...
  for ( int line=1; cin.getline(psBuff,1000); line++ ) {
    StringInput si(psBuff);
    // Process comments or model lines...
    if ( !( psBuff[0]=='#' ||
            (si=si>>mCC>>"\0")!=NULL) )
      cout<<psBuff<<"\n";
    // Print progress (for big models)...
    if ( line%1000000==0 ) cerr<<"  "<<line<<" lines read...\n";
  }

  DCtoC Ch0_giv_BL_D_Ch("");
  DCtoC Ch0_giv_BR_D_Ch("");
  DCtoCC Ch0_Ch1_giv_BL_D_Ch("");
  DCtoCC Ch0_Ch1_giv_BR_D_Ch("");
  // read in CC model and obtain relevant models
  for ( BDCtoCC::const_iterator i=mCC.begin(); i!=mCC.end(); i++ ) {
    B b  = i->first.getX1();
    D d  = i->first.getX2();
    C ch = i->first.getX3();
    for ( SimpleMap<MapKey2D<C,C>,Prob>::const_iterator j=i->second.begin(); j!=i->second.end(); j++ ) {
      C ch0 = j->first.getX1();
      C ch1 = j->first.getX2();
      if ( B_L == b ) {
        Ch0_giv_BL_D_Ch.set(MapKey2D<D,C>(d,ch)).set(ch0) += j->second;
        Ch0_Ch1_giv_BL_D_Ch.set(MapKey2D<D,C>(d,ch)).set(MapKey2D<C,C>(ch0,ch1)) += j->second;
      }
      else {
        Ch0_giv_BR_D_Ch.set(MapKey2D<D,C>(d.toInt()+1,ch)).set(ch0) += j->second;
        Ch0_Ch1_giv_BR_D_Ch.set(MapKey2D<D,C>(d,ch)).set(MapKey2D<C,C>(ch0,ch1)) += j->second;
      }
    }
  }
  mCC.clear();  
  Ch0_giv_BL_D_Ch.normalize();
  Ch0_giv_BR_D_Ch.normalize();
  Ch0_Ch1_giv_BL_D_Ch.normalize();
  Ch0_Ch1_giv_BR_D_Ch.normalize();

  cerr<<"1/8\n";

  ////////// obtain intermediate models

  // obtain expected counts for unbounded left descendants
  DCtoC Chi_giv_D_Ch_prev("");
  DCtoC Chi_giv_D_Ch_curr("");
  DCtoC Chi_giv_D_Ch_star("Crl*");
  // add zero iteration to star model
  for ( DCtoC::const_iterator i=Ch0_giv_BR_D_Ch.begin(); i!=Ch0_giv_BR_D_Ch.end(); i++ ) {
    D d  = i->first.getX1();
    C ch = i->first.getX2();
    for ( SimpleMap<C,Prob>::const_iterator j=i->second.begin(); j!=i->second.end(); j++ ) {
      C chi = j->first;
      Chi_giv_D_Ch_curr.set(i->first).set(j->first) += j->second;
      Chi_giv_D_Ch_star.set(i->first).set(j->first) += j->second;
    }
  }
  DCtoC* pChi_giv_D_Ch_prev = &Chi_giv_D_Ch_prev;
  DCtoC* pChi_giv_D_Ch_curr = &Chi_giv_D_Ch_curr;
  // add subsequent iterations to star model
  for ( int h=0; h<MAX_ITER; h++ ) {
    cerr<<"h="<<h<<"/"<<MAX_ITER<<"\n";
    DCtoC* temp        = pChi_giv_D_Ch_prev;
    pChi_giv_D_Ch_prev = pChi_giv_D_Ch_curr;
    pChi_giv_D_Ch_curr = temp;
    pChi_giv_D_Ch_curr->clear();
    for ( DCtoC::const_iterator i=pChi_giv_D_Ch_prev->begin(); i!=pChi_giv_D_Ch_prev->end(); i++ ) {
      D d  = i->first.getX1();
      C ch = i->first.getX2();
      for ( SimpleMap<C,Prob>::const_iterator j=i->second.begin(); j!=i->second.end(); j++ ) {
        C chi = j->first;
        const SimpleMap<C,Prob>& dist = Ch0_giv_BL_D_Ch.get(MapKey2D<D,C>(d,chi));
        for ( SimpleMap<C,Prob>::const_iterator k=dist.begin(); k!=dist.end(); k++ ) {
          pChi_giv_D_Ch_curr->set(i->first).set(k->first) += j->second * k->second;
          Chi_giv_D_Ch_star.set(i->first).set(k->first) += j->second * k->second;
        }
      }
    }
  }
  Chi_giv_D_Ch_prev.clear();
  Chi_giv_D_Ch_curr.clear();

  ////////// obtain hhmm models

  cerr<<"2/8\n";

  DCtoC mP("P");
  // obtain expansion model
  for ( DCtoC::const_iterator i=Chi_giv_D_Ch_star.begin(); i!=Chi_giv_D_Ch_star.end(); i++ ) {
    D d  = i->first.getX1();
    C ch = i->first.getX2();
    Prob pr = Ch0_giv_BR_D_Ch.get(MapKey2D<D,C>(d,ch)).get(C_NIL);
    if ( pr>0.0 )
      mP.set(MapKey2D<D,C>(d,ch)).set(ch) += pr;
    for ( SimpleMap<C,Prob>::const_iterator j=i->second.begin(); j!=i->second.end(); j++ ) {
      C chi = j->first;
      pr = j->second * Ch0_giv_BL_D_Ch.get(MapKey2D<D,C>(d,chi)).get(C_NIL);
      //if ( pr <= 0.0 ) cerr<<d<<" "<<ch<<" "<<chi<<" "<<j->second<<" "<<Ch0_giv_BL_D_Ch.get(MapKey2D<D,C>(d,chi)).get(C_NIL)<<"\n";
      if ( pr > 0.0 )
        mP.set(MapKey2D<D,C>(d,ch)).set(chi) += pr;
    }
  }
  mP.normalize();
  mP.dump();
  mP.clear();

  cerr<<"3/8\n";

  DCCtoC mF("F");
  // obtain reduction model
  for ( DCtoC::const_iterator i=Chi_giv_D_Ch_star.begin(); i!=Chi_giv_D_Ch_star.end(); i++ ) {
    D d  = i->first.getX1();
    C ch = i->first.getX2();
    Prob pr = Ch0_giv_BR_D_Ch.get(MapKey2D<D,C>(d,ch)).get(C_NIL);
    if ( pr>0.0 )
      mF.set(MapKey3D<D,C,C>(d,ch,ch)).set(C_NIL) += pr;
    for ( SimpleMap<C,Prob>::const_iterator j=i->second.begin(); j!=i->second.end(); j++ ) {
      C chi = j->first;
      pr = j->second * Ch0_giv_BL_D_Ch.get(MapKey2D<D,C>(d,chi)).get(C_NIL);
      if ( pr > 0.0 )
        mF.set(MapKey3D<D,C,C>(d,ch,chi)).set(ch) += pr;
    }
  }
  Ch0_giv_BR_D_Ch.clear();
  mF.normalize();
  mF.dump();
  mF.clear();

  cerr<<"4/8\n";

  DCCtoC mWa("Wa");
  // obtain lower awaited component of active transition model
  for ( DCtoCC::const_iterator i=Ch0_Ch1_giv_BL_D_Ch.begin(); i!=Ch0_Ch1_giv_BL_D_Ch.end(); i++ ) {
    D d  = i->first.getX1();
    C ch = i->first.getX2();
    for ( SimpleMap<MapKey2D<C,C>,Prob>::const_iterator j=i->second.begin(); j!=i->second.end(); j++ ) {
      C ch0 = j->first.getX1();
      C ch1 = j->first.getX2();
      if ( ch0 != C_NIL && ch1 != C_NIL )
        if ( j->second > 0.0 )
          mWa.set(MapKey3D<D,C,C>(d,ch,ch0)).set(ch1) += j->second;
    }
  }
  Ch0_Ch1_giv_BL_D_Ch.clear();
  mWa.normalize();
  mWa.dump();
  mWa.clear();

  cerr<<"5/8\n";

  DCCtoC mWb("Wb");

  // obtain upper awaited component of awaited transition model, and null lower active component
  for ( DCtoCC::const_iterator i=Ch0_Ch1_giv_BR_D_Ch.begin(); i!=Ch0_Ch1_giv_BR_D_Ch.end(); i++ ) {
    D d  = i->first.getX1();
    C ch = i->first.getX2();
    for ( SimpleMap<MapKey2D<C,C>,Prob>::const_iterator j=i->second.begin(); j!=i->second.end(); j++ ) {
      C ch0 = j->first.getX1();
      C ch1 = j->first.getX2();
      if ( ch0 != C_NIL && ch1 != C_NIL )
        if ( j->second > 0.0 ) {
          mWb.set(MapKey3D<D,C,C>(d,ch,ch0)).set(ch1) += j->second;
        }
    }
  }
  mWb.normalize();
  mWb.dump();
  mWb.clear();

  cerr<<"6/8\n";

  DCCtoC mA("A");
  //OrderedModel<MapKey3D<D,C,C>,MapKey2D<C,C>,Prob> mA("A");
  // obtain upper awaited component of awaited transition model, and null lower active component
  for ( DCtoCC::const_iterator i=Ch0_Ch1_giv_BR_D_Ch.begin(); i!=Ch0_Ch1_giv_BR_D_Ch.end(); i++ ) {
    D d  = i->first.getX1();
    C ch = i->first.getX2();
    for ( SimpleMap<MapKey2D<C,C>,Prob>::const_iterator j=i->second.begin(); j!=i->second.end(); j++ ) {
      C ch0 = j->first.getX1();
      C ch1 = j->first.getX2();
      if ( ch0 != C_NIL && ch1 != C_NIL )
        if ( j->second > 0.0 ) {
          mA.set(MapKey3D<D,C,C>(d.toInt()+1,ch,ch0)).set(C_NIL) += j->second;
        }
    }
  }
  Ch0_Ch1_giv_BR_D_Ch.clear();

  cerr<<"7/8\n";

  // obtain copy upper awaited component, and lower active component of active transition model
  for ( DCtoC::const_iterator i=Chi_giv_D_Ch_star.begin(); i!=Chi_giv_D_Ch_star.end(); i++ ) {
    D d  = i->first.getX1();
    C ch = i->first.getX2();
    for ( SimpleMap<C,Prob>::const_iterator j=i->second.begin(); j!=i->second.end(); j++ ) {
      C chi = j->first;
      const SimpleMap<C,Prob>& dist = Ch0_giv_BL_D_Ch.get(MapKey2D<D,C>(d,chi));
      for ( SimpleMap<C,Prob>::const_iterator k=dist.begin(); k!=dist.end(); k++ ) {
        C chi0 = k->first;
        if ( chi0 != C_NIL ) {
          Prob pr = j->second * k->second;
          if ( pr > 0.0000001 )
            mA.set(MapKey3D<D,C,C>(d,ch,chi0)).set(chi) += pr;
        }
      }
    }
  }
  Ch0_giv_BL_D_Ch.clear();
  Chi_giv_D_Ch_star.clear();
  mA.normalize();
  mA.dump();
  mA.clear();

  cerr<<"8/8\n";
}
    
