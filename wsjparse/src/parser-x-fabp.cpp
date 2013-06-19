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

#include <nl-cpt.h>
#include <Model.h>
#include <XModelBerkeley.h>
#include <PhaseParser.h>


///////////////////////////////////////////////////////////////////////////////


char psX[]     = "";
char psSpace[] = " ";
char psSlash[] = "/";
char psDot[]   = ".";
char psSemi[]  = ";";
char psComma[] = ",";


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


DiscreteDomain<int> domB;
class B : public DiscreteDomainRV<int,domB> {
 public:
  B ( )                : DiscreteDomainRV<int,domB> ( )    { }
  B ( const char* ps ) : DiscreteDomainRV<int,domB> ( ps ) { }
};
const B B_0("0");
const B B_1("1");


DiscreteDomain<int> domE;
class E : public DiscreteDomainRV<int,domE> {
 public:
  E ( )                : DiscreteDomainRV<int,domE> ( )    { }
  E ( const char* ps ) : DiscreteDomainRV<int,domE> ( ps ) { }
};
const E E_0("0");


DiscreteDomain<int> domC;
class C : public DiscreteDomainRV<int,domC> {
 private:
  static SimpleMap<C,int> mcbEB;                                      /* EPDA marker */
  static SimpleMap<C,int> mcbEE;                                      /* EPDA marker */
  void calcDetModels ( const char* ps ) {
    //cerr<<"encoding "<<*this<<" as "<<(NULL!=strstr(ps,"-eb"))<<"\n";
    if(!mcbEB.contains(*this)) mcbEB[*this]=(NULL!=strstr(ps,"-eb"));  /* EPDA marker */
    if(!mcbEE.contains(*this)) mcbEE[*this]=(NULL!=strstr(ps,"-ee"));  /* EPDA marker */
  }

 public:
  C ( )                : DiscreteDomainRV<int,domC> ( )    { }
  C ( const char* ps ) : DiscreteDomainRV<int,domC> ( ps ) { calcDetModels(ps); }
  bool isEB() const { return mcbEB[*this]; }                          /* EPDA marker */
  bool isEE() const { return mcbEE[*this]; }                          /* EPDA marker */
  friend pair<StringInput,C*> operator>> ( StringInput si, C& c ) { return pair<StringInput,C*>(si,&c); }
  friend StringInput operator>> ( pair<StringInput,C*> si_c, const char* psD ) {
    if ( si_c.first == NULL ) return NULL;
    StringInput si=si_c.first>>(DiscreteDomainRV<int,domC>&)*si_c.second>>psD;
    si_c.second->calcDetModels(si_c.second->getString().c_str()); return si; }
  const C& getC ( ) const { return *this; }
  const E& getE ( ) const { return E_0; }
};
SimpleMap<C,int> C::mcbEB;                                            /* EPDA marker */
SimpleMap<C,int> C::mcbEE;                                            /* EPDA marker */
const C C_NIL("-");


DiscreteDomain<int> domL;
class L : public DiscreteDomainRV<int,domL> {
 public:
  L ( )                : DiscreteDomainRV<int,domL> ( )    { }
  L ( const char* ps ) : DiscreteDomainRV<int,domL> ( ps ) { }
};


typedef C F;
const F F_NIL("-");
const F F_BOG("BOG");
const F F_REST("REST");


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
  int      getD ( )       const { return ( (F()!=second.get(3).getAwa()) ? 3 :
                                           (F()!=second.get(2).getAwa()) ? 2 :
                                           (F()!=second.get(1).getAwa()) ? 1 :
                                           (F()!=second.get(0).getAwa()) ? 0 : -1 ); }
  B&       setB ( )             { return first; }
  Q&       setQ ( int d )       { return (d<0) ? Q_SCRATCH : (d<4) ? second.set(d) : Q_SCRATCH; }
  F&       setP ( )             { return third; }
  const Q& getQ ( int d ) const { return (d<0) ? Q_TOP     : (d<4) ? second.get(d) : Q_BOT;     }
  const F& getP ( )       const { return third; }
};
const Y Y_INIT("0;BOG/BOG;-/-;-/-;-/-;-");


////////////////////////////////////////////////////////////////////////////////

class YModel {

 private:

  // Data members...
  OrderedModel<MapKey3D<D,F,F>,F,LogProb> mF; //mF returns whether to reduce or not
  OrderedModel<MapKey3D<D,F,F>,F,LogProb> mA; //mA generates best guess of next act trans
  OrderedModel<MapKey3D<D,F,F>,F,LogProb> mWa; //mW generates best guess of next awa trans
  OrderedModel<MapKey3D<D,F,F>,F,LogProb> mWb; //mW generates best guess of next awa trans
  OrderedModel<MapKey2D<D,F>,F,LogProb>   mP; //mP generates best guess for next preterm
  XModel<F>                               mX; //mX contains vocabulary
// create class Xmodel as subclass of unorderedmodel<f,x,lp> getProb will do jazz to lookup unk
// and combine that with regular prob
// create X_UNK_LC... as constants, use those

 public:
  typedef C PhaseType;
  typedef Y HidType;
  typedef X ObsType;

  // Static definitions...
  static const PhaseNum NUM_PHASES = 5;
  class SearchPath : public DelimitedStaticSafeArray<NUM_PHASES,psComma,ChoiceNum> {
   public:
    SearchPath ( ) : DelimitedStaticSafeArray<NUM_PHASES,psComma,ChoiceNum> ( 0 ) { }
  };

  class WeightedY;

  class PhaseValues : public penta<F,F,F,F,LogProb> {
   public:
    PhaseValues ( ) : penta<F,F,F,F,LogProb> ( ) { }
    PhaseValues ( const WeightedY& wy ) : penta<F,F,F,F,LogProb> ( F_NIL, F_NIL, F_NIL, F_NIL, wy.getProb() ){ }
    PhaseValues ( const F& f, const F& a, const F& w, const F& p, const LogProb pr ) : penta<F,F,F,F,LogProb> ( f,a,w,p,pr ) { }
    const F& getF   ( ) const { return first; }
    const F& getA     ( ) const { return second; }
    const F& getW     ( ) const { return third; }
    const F& getP      ( ) const { return fourth; }
    LogProb  getProb   ( ) const { return fifth; }

    friend ostream& operator<< ( ostream& os, const PhaseValues& pv ) { os<<"("<<pv.getF()<<","<<pv.getA()<<","<<pv.getW()<<","<<pv.getP()<<":"<<pv.getProb()<<")"; return os; }
  };

class Measures {
   public:
    LogProb lgprMax;
    double prNorm;
    double prfl;
    double prFl;
    double prfL;
    double prFL;
    double prd1fl;
    double prd1Fl;
    double prd1fL;
    double prd1FL;
    double prd2fl;
    double prd2Fl;
    double prd2fL;
    double prd2FL;
    double prd3fl;
    double prd3Fl;
    double prd3fL;
    double prd3FL;
    double prd4fl;
    double prd4Fl;
    double prd4fL;
    double prd4FL;

    Measures ( ) : lgprMax(0), prNorm(0.0),
                   prfl(0.0), prFl(0.0), prfL(0.0), prFL(0.0),
                   prd1fl(0.0), prd1Fl(0.0), prd1fL(0.0), prd1FL(0.0),
                   prd2fl(0.0), prd2Fl(0.0), prd2fL(0.0), prd2FL(0.0),
                   prd3fl(0.0), prd3Fl(0.0), prd3fL(0.0), prd3FL(0.0),
                   prd4fl(0.0), prd4Fl(0.0), prd4fL(0.0), prd4FL(0.0) {  }
    void write ( ) { //The following metrics output cost for previous timestep
      cerr << "F-L- Cost: " << (prfl  / prNorm) << "\n";
      cerr << "F+L- Cost: " << (prFl  / prNorm) << "\n";
      cerr << "F-L+ Cost: " << (prfL  / prNorm) << "\n";
      cerr << "F+L+ Cost: " << (prFL  / prNorm) << "\n";
      cerr << "Depth 1 F-L- Cost: " << (prd1fl  / prNorm) << "\n";
      cerr << "Depth 1 F+L- Cost: " << (prd1Fl  / prNorm) << "\n";
      cerr << "Depth 1 F-L+ Cost: " << (prd1fL  / prNorm) << "\n";
      cerr << "Depth 1 F+L+ Cost: " << (prd1FL  / prNorm) << "\n";
      cerr << "Depth 2 F-L- Cost: " << (prd2fl  / prNorm) << "\n";
      cerr << "Depth 2 F+L- Cost: " << (prd2Fl  / prNorm) << "\n";
      cerr << "Depth 2 F-L+ Cost: " << (prd2fL  / prNorm) << "\n";
      cerr << "Depth 2 F+L+ Cost: " << (prd2FL  / prNorm) << "\n";
      cerr << "Depth 3 F-L- Cost: " << (prd3fl  / prNorm) << "\n";
      cerr << "Depth 3 F+L- Cost: " << (prd3Fl  / prNorm) << "\n";
      cerr << "Depth 3 F-L+ Cost: " << (prd3fL  / prNorm) << "\n";
      cerr << "Depth 3 F+L+ Cost: " << (prd3FL  / prNorm) << "\n";
      cerr << "Depth 4 F-L- Cost: " << (prd4fl  / prNorm) << "\n";
      cerr << "Depth 4 F+L- Cost: " << (prd4Fl  / prNorm) << "\n";
      cerr << "Depth 4 F-L+ Cost: " << (prd4fL  / prNorm) << "\n";
      cerr << "Depth 4 F+L+ Cost: " << (prd4FL  / prNorm) << "\n";
    }
  };

  class WeightedY : public trip<Y,PhaseValues,LogProb> {
   public:
    typedef Y VAR_TYPE;
    WeightedY ( ) : trip<Y,PhaseValues,LogProb> ( ) { }
    WeightedY ( Measures& meas, const WeightedY& wy, const PhaseValues& pv ) : trip<Y,PhaseValues,LogProb> (wy) {

                                              // for d < new d': q^d_t copied from q^d_t-1 by default
      int d = wy.getY().getD();
      d += ((F_NIL == pv.getF()) ? 1 : 0);                                     // expansion increases depth
      d -= ((F_NIL == pv.getA()) ? 1 : 0);                                     // reduction decreases depth
      setY().setB() = (F_NIL != pv.getF() && F_BOG != pv.getF()) ? B_1 : B_0;  // boolean for tree reconstruction
      setPV() = PhaseValues(pv);

//        if (d>0)
//          cerr<<"result of "<<getY().getQ(d-1).getAwa()<<" is "<<getY().getQ(d-1).getAwa().isEB()<<" so "<<(d>0 && getY().getQ(d-1).getAwa().isEB() && pv.getW().isEE())<<"\n";
      d -= ((d>0 && getY().getQ(d-1).getAwa().isEB() && pv.getW().isEE()) ? 1 : 0);  // epda reduction decreases depth

      if (F_NIL != pv.getA()) {               // if ya reduction, a^d_t stays copied from a^d_t-1
        setY().setQ(d).setAct() = pv.getA();  // if no reduction, a^d_t = A
      }
      setY().setQ(d).setAwa() = pv.getW();    // b^d_t = B
      if (d < 3){
        setY().setQ(d+1) = Q_BOT;             // for d > new d': q^d_t = '-'
      }
      setY().setP() = pv.getP();              // p_t = P
      setProb() = pv.getProb();               // copy prob

      if ( 0.0==meas.prNorm ) meas.lgprMax = pv.getProb();   // scale all probs by first (highest) outcome of queue
      double prCurr = (pv.getProb()/meas.lgprMax).toDouble();

      if (pv.getF() == C_NIL) { // F+
        if (pv.getA() == C_NIL) { // L+
          meas.prFL += prCurr;
          if (wy.getY().getD() == 0) meas.prd1FL += prCurr;
          else if (wy.getY().getD() == 1) meas.prd2FL += prCurr;
          else if (wy.getY().getD() == 2) meas.prd3FL += prCurr;
          else meas.prd4FL += prCurr;
        }
        else { // L-
          meas.prFl += prCurr;
          if (getY().getD() == 0) meas.prd1Fl += prCurr;
          else if (getY().getD() == 1) meas.prd2Fl += prCurr;
          else if (getY().getD() == 2) meas.prd3Fl += prCurr;
          else meas.prd4Fl += prCurr;
        }
      }
      else { // F-
        if (pv.getA() == C_NIL) { // L+
          meas.prfL += prCurr;
          if (wy.getY().getD() == 0) meas.prd1fL += prCurr;
          else if (wy.getY().getD() == 1) meas.prd2fL += prCurr;
          else if (wy.getY().getD() == 2) meas.prd3fL += prCurr;
          else meas.prd4fL += prCurr;
        }
        else { // L-
          if (wy.getY().getD() == 0) meas.prd1fl += prCurr;
          else if (wy.getY().getD() == 1) meas.prd2fl += prCurr;
          else if (wy.getY().getD() == 2) meas.prd3fl += prCurr;
          else meas.prd4fl += prCurr;
          meas.prfl += prCurr;
        }
      }

      meas.prNorm += prCurr;
    }
    WeightedY ( const Y& y, const LogProb pr ) : trip<Y,PhaseValues,LogProb> ( y, PhaseValues(), pr ) { }
    Y&       setY     ( )       { return first;  }
    PhaseValues& setPV ( )    { return second;  }
    LogProb& setProb  ( )       { return third; }
    const Y& getY     ( ) const { return first;  }
    const PhaseValues& getPV ( ) const { return second;  }
    LogProb  getProb  ( ) const { return third; }
    friend ostream& operator<< ( ostream& os, const WeightedY& wy ) { os<<wy.first<<":"<<wy.third; return os; }
  };

  static bool EOS;

  // Constructor methods...
  YModel ( ) : mF("F"), mA("A"), mWa("Wa"), mWb("Wb"), mP("P"), mX("X") { }

  // Extractor methods...
  const XModel<C>& getXModel() const { return mX; }

  PhaseValues getNext ( const PhaseValues& pv, PhaseNum pn, SearchPath& sp, Array<WeightedY>& beam, const X& x ) const {
    if (VERBOSE)
      cerr<<"pv: "<<pv<<" pn: "<<pn<<" sp: "<<sp<<" tmin: "<<beam[sp[0]]<<" x: "<<x<< "\n";
    if ( 0==pn ) {
      //cerr<<"    pull from beam...\n";
      return PhaseValues(beam[sp[0]]);
    }
    else if ( 1==pn ) { //D-phase
      int d = beam[sp[0]].getY().getD();
      //Pretend you're at the end of the previous sentence.
      if ( F_NIL == beam[sp[0]].getY().getP() ) return (0==sp[pn]) ? PhaseValues(F_BOG,F_NIL,F_NIL,F_NIL,1.0) : PhaseValues();
      //Consult model (remember model file indices start at 1, whereas store Y indices start at 0, so d+1 syncs)
      // also, d shows lowest level occupied, so add another +1 depth for destination: d+2
      pair<F,LogProb> fpr = mF.getDistrib(MapKey3D<D,F,F>(d+2,beam[sp[0]].getY().getQ(d).getAwa(),beam[sp[0]].getY().getP()))[sp[pn]];
      if (VERBOSE)
        cerr<<"    trying F "<< d+2 <<" "<< beam[sp[0]].getY().getQ(d).getAwa()<<" "<<beam[sp[0]].getY().getP()<< "... "<<fpr.first<<" = "<<fpr.second<<"\n";
      //'If' No expansion; 'else' is an expansion
      if (F_NIL == fpr.first) {
        return PhaseValues ( beam[sp[0]].getY().getQ(d).getAct(),F_NIL,F_NIL,F_NIL, pv.getProb() * fpr.second );
      }
      else return PhaseValues ( F_NIL,F_NIL,F_NIL,F_NIL, pv.getProb() * fpr.second );
    }
    else if ( 2==pn ) {
      int d = beam[sp[0]].getY().getD();
      //Fill pv.F with t-1.Act if F_NIL is seen as code for expansion
      F f = (F_NIL == pv.getF()) ? beam[sp[0]].getY().getP() : pv.getF();
      //Add a 1 or 0 to d based on F-phase expansion or not
      d += ((F_NIL == pv.getF()) ? 1 : 0);
      //if ( F_NIL == beam[sp[0]].getY().getQ(d).getAct() ) return (0==sp[pn]) ? PhaseValues(pv.getF(),F_BOG,F_NIL,F_NIL,1) : PhaseValues();
      if ( F_BOG == pv.getF() ) return (0==sp[pn]) ? PhaseValues(F_BOG,F_NIL,F_NIL,F_NIL,1.0) : PhaseValues();
      //Consult model (remember model file indices start at 1, whereas store Y indices start at 0, so d+1 syncs)
      pair<F,LogProb> fpr = mA.getDistrib(MapKey3D<D,F,F>(d+1,beam[sp[0]].getY().getQ(d-1).getAwa(),f))[sp[pn]];
      // upper active/awaited transition must agree with EOS flag
      //cerr<<"not happening "<<d<<" "<<EOS<<" "<<ffpr.first<<" "<<(F_NIL==ffpr.first.getX2())<<" "<<ffpr.second<<"\n";
      while ( (!EOS && d==0 && F_NIL==fpr.first && fpr.second>LogProb()) ||
              (EOS  && (d!=0 || F_NIL!=fpr.first) && fpr.second>LogProb()) ) {
        //cerr<<"aah! it happened! "<<d<<" "<<EOS<<" "<<ffpr.first<<" "<<ffpr.second<<"\n";
        sp[pn]++;
        fpr = mA.getDistrib(MapKey3D<D,F,F>(d+1,beam[sp[0]].getY().getQ(d-1).getAwa(),f))[sp[pn]];
      }
      if (VERBOSE)
        cerr<<"    trying A "<< d+1 <<" "<<beam[sp[0]].getY().getQ(d-1).getAwa()<<" "<<f <<"... "<<fpr.first<<" = "<<fpr.second<<"\n";
      //If reduction, X2==NIL: use d-1.Act; otherwise use active transition X2
      //If reduction (Awaited transition), then fill in Awa from mA; else wait for mW
      return PhaseValues ( pv.getF(),fpr.first,F_NIL,F_NIL,pv.getProb() * fpr.second );
    }
    else if ( 3==pn ) {
      int d = beam[sp[0]].getY().getD();
      //Fill pv.F with t-1.Act if F_NIL is seen as code for expansion
      F f = (F_NIL == pv.getF()) ? beam[sp[0]].getY().getP() : pv.getF();
      //Add a 1 or 0 to d based on F-phase expansion or not
      d += ((F_NIL == pv.getF()) ? 1 : 0);
      //Subtract a 1 or 0 to d based on A-phase reduction or not
      d -= ((F_NIL == pv.getA()) ? 1 : 0);
      if ( F_BOG == pv.getF() ) return (0==sp[pn]) ? PhaseValues(F_BOG,F_NIL,F_REST,F_NIL,1.0) : PhaseValues();
      //Consult model (remember model file indices start at 1, whereas store Y indices start at 0, so d+1 syncs)
      const pair<F,LogProb>& fpr = ((F_NIL!=pv.getA())?mWa:mWb).getDistrib(MapKey3D<D,F,F>(d+1,(F_NIL==pv.getA())?beam[sp[0]].getY().getQ(d).getAwa():pv.getA(),f))[sp[pn]] ;
      if (VERBOSE)
        cerr<<"    trying W "<< d+1 <<" "<< ((F_NIL==pv.getA())?beam[sp[0]].getY().getQ(d).getAwa():pv.getA()) <<" "<< f <<"... "<<fpr.first<<" = "<<fpr.second<<"\n";

      return PhaseValues ( pv.getF(),pv.getA(),fpr.first,F_NIL, pv.getProb() * fpr.second );
    }
    else if ( 4==pn ) {
      int d = beam[sp[0]].getY().getD();
      //Fill pv.F with t-1.Act if F_NIL is seen as code for expansion
      F f = (F_NIL == pv.getF()) ? beam[sp[0]].getY().getP() : pv.getF();
      //Add a 1 or 0 to d based on F-phase expansion or not
      d += ((F_NIL == pv.getF()) ? 1 : 0);
      //Subtract a 1 or 0 to d based on A-phase reduction or not
      d -= ((F_NIL == pv.getA()) ? 1 : 0);
      //Consult model (remember model file indices start at 1, whereas store Y indices start at 0, so d+1 syncs)
      // also, d shows lowest level occupied, so add another +1 depth for destination: d+2
      const pair<F,LogProb>& fpr = mP.getDistrib(MapKey2D<D,F>(d+2,pv.getW()))[sp[pn]];
      if (VERBOSE)
        cerr<<"    trying P " << d+2 <<" "<< pv.getW() << "... "<<fpr.first<<" = "<<fpr.second<<"\n";
      return PhaseValues ( pv.getF(),pv.getA(),pv.getW(),fpr.first, pv.getProb() * fpr.second );
    }
    else {
      if (VERBOSE)
        cerr<<"    trying X... "<< mX.getProb(pv.getP(),x) << "\n";
      return PhaseValues ( pv.getF(),pv.getA(),pv.getW(),pv.getP(), pv.getProb() * mX.getProb(pv.getP(), x) );
    }
  }

  // I/O methods...
  friend pair<StringInput,YModel*> operator>> ( const StringInput ps, YModel& m ) { return pair<StringInput,YModel*>(ps,&m); }
  friend StringInput operator>>( pair<StringInput, YModel*> si_m, const char* psD) {
    StringInput si;
    return ( (si=si_m.first>>si_m.second->mF>>psD)!=NULL ||
             (si=si_m.first>>si_m.second->mA>>psD)!=NULL ||
             (si=si_m.first>>si_m.second->mWa>>psD)!=NULL ||
             (si=si_m.first>>si_m.second->mWb>>psD)!=NULL ||
             (si=si_m.first>>si_m.second->mP>>psD)!=NULL ||
             (si=si_m.first>>si_m.second->mX>>psD)!=NULL ) ? si : StringInput(NULL);
  }
};
bool YModel::EOS = false;


////////////////////////////////////////////////////////////////////////////////

int main ( int nArgs, char* argv[] ) {
  PhaseParser<YModel> ( nArgs, argv, Y_INIT );
}

