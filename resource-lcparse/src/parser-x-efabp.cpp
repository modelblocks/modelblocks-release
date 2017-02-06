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
#include <Model.hpp>
#include <XModelBerkeley.hpp>
#include <PhaseParser.hpp>

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


DiscreteDomain<int> domC;
class C : public DiscreteDomainRV<int,domC> {
 private:
  static SimpleMap<C,int> mcbEB;                                      /* EPDA marker */
  static SimpleMap<C,int> mcbEE;                                      /* EPDA marker */
  static SimpleMap<C,int> mcbO;
  static SimpleMap<C,int> mcbG;
  static SimpleMap<C,int> mcbGm;
  static SimpleMap<C,int> mcbA;
  static SimpleMap<C,int> mcbC;
  static SimpleMap<C,int> mcbM;
  static SimpleMap<C,int> mcbI;
  static SimpleMap<C,int> mcb1;
  static SimpleMap<C,int> mcb2;
  static SimpleMap<C,int> mcb3;
  void calcDetModels ( const char* ps ) {
    //cerr<<"encoding "<<*this<<" as "<<(NULL!=strstr(ps,"-lA") && NULL!=strstr(ps,"-u") && NULL==strstr(strstr(ps,"-u")+1,"-u"))<<"\n";
    if(!mcbEB.contains(*this)) mcbEB[*this]=(NULL!=strstr(ps,"-eb"));  /* EPDA marker */
    if(!mcbEE.contains(*this)) mcbEE[*this]=(NULL!=strstr(ps,"-ee"));  /* EPDA marker */
    if(!mcbO .contains(*this)) mcbO [*this]=(NULL!=strstr(ps,"-oR"));
    if(!mcbG .contains(*this)) mcbG [*this]=(NULL!=strstr(ps,"-g"));
    if(!mcbGm.contains(*this)) mcbGm[*this]=(NULL!=strstr(ps,"-gRP") || NULL!=strstr(ps,"-gAC"));
    if(!mcbA .contains(*this)) mcbA [*this]=(NULL!=strstr(ps,"-lA"));
    if(!mcbC .contains(*this)) mcbC [*this]=(NULL!=strstr(ps,"-lC"));
    if(!mcbM .contains(*this)) mcbM [*this]=(NULL!=strstr(ps,"-lM"));
    if(!mcbI .contains(*this)) mcbI [*this]=(NULL!=strstr(ps,"-lI"));
    if(!mcb1.contains(*this)) mcb1[*this]=(NULL==strstr(ps,"-b") && NULL==strstr(ps,"-s"));
    if(!mcb2.contains(*this)) mcb2[*this]=((NULL!=strstr(ps,"-b") && NULL==strstr(strstr(ps,"-b")+1,"-b")) ||
					     (NULL!=strstr(ps,"-s") && NULL==strstr(strstr(ps,"-s")+1,"-b")) );
    if(!mcb3.contains(*this)) mcb3[*this]=((NULL!=strstr(ps,"-b") && NULL!=strstr(strstr(ps,"-b")+1,"-b")) ||
					     (NULL!=strstr(ps,"-s") && NULL!=strstr(strstr(ps,"-s")+1,"-b")) );
  }

 public:
  C ( )                : DiscreteDomainRV<int,domC> ( )    { }
  C ( const char* ps ) : DiscreteDomainRV<int,domC> ( ps ) { calcDetModels(ps); }
  bool isEB() const { return mcbEB[*this]; }                          /* EPDA marker */
  bool isEE() const { return mcbEE[*this]; }                          /* EPDA marker */
  bool isO () const { return mcbO [*this]; }
  bool isG () const { return mcbG [*this]; }
  bool isGm() const { return mcbGm[*this]; }
  bool isA () const { return mcbA [*this]; }
  bool isC () const { return mcbC [*this]; }
  bool isM () const { return mcbM [*this]; }
  bool isI () const { return mcbI [*this]; }
  bool is1() const { return mcb1[*this]; }
  bool is2() const { return mcb2[*this]; }
  bool is3() const { return mcb3[*this]; }
  friend pair<StringInput,C*> operator>> ( StringInput si, C& c ) { return pair<StringInput,C*>(si,&c); }
  friend StringInput operator>> ( pair<StringInput,C*> si_c, const char* psD ) {
    if ( si_c.first == NULL ) return NULL;
    StringInput si=si_c.first>>(DiscreteDomainRV<int,domC>&)*si_c.second>>psD;
    si_c.second->calcDetModels(si_c.second->getString().c_str()); return si; }
};
SimpleMap<C,int> C::mcbEB;                                            /* EPDA marker */
SimpleMap<C,int> C::mcbEE;                                            /* EPDA marker */
SimpleMap<C,int> C::mcbO;
SimpleMap<C,int> C::mcbG;
SimpleMap<C,int> C::mcbGm;
SimpleMap<C,int> C::mcbA;
SimpleMap<C,int> C::mcbC;
SimpleMap<C,int> C::mcbM;
SimpleMap<C,int> C::mcbI;
SimpleMap<C,int> C::mcb1;
SimpleMap<C,int> C::mcb2;
SimpleMap<C,int> C::mcb3;
const C C_NIL("-");
const C C_BOG("BOG");
const C C_REST("REST");


DiscreteDomain<int> domE;
class E : public DiscreteDomainRV<int,domE> {
 public:
  E ( )                : DiscreteDomainRV<int,domE> ( )    { }
  E ( const char* ps ) : DiscreteDomainRV<int,domE> ( ps ) { }
  friend ostream& operator<< ( ostream& os, const E& e ) { return os<<e.toInt(); }
};
const E E_0("0");
const E E_1("1");


DiscreteDomain<int> domL;
class L : public DiscreteDomainRV<int,domL> {
 public:
  L ( )                : DiscreteDomainRV<int,domL> ( )    { }
  L ( const char* ps ) : DiscreteDomainRV<int,domL> ( ps ) { }
};
const L L_0("0");
const L L_1("1");
const L L_2("2");
const L L_3("3");
const L L_C("C");
const L L_P("P");
const L L_EQ("=");


class R : public DelimitedJoint3DRV<psX,E,psSlash,L,psSlash,E,psX> {
 public:
  R ( )                : DelimitedJoint3DRV<psX,E,psSlash,L,psSlash,E,psX> ( )    { }
  R ( const char* ps ) : DelimitedJoint3DRV<psX,E,psSlash,L,psSlash,E,psX> ( ps ) { }
  R ( const E& eS, const L& l, const E& eT ) : DelimitedJoint3DRV<psX,E,psSlash,L,psSlash,E,psX> ( eS,l,eT ) { }
};
const R R_NIL("0/0/0");


class F : public DelimitedJoint2DRV<psX,C,psDot,E,psX> {
 public:
  F ( )                : DelimitedJoint2DRV<psX,C,psDot,E,psX> ( )    { }
  F ( const char* ps ) : DelimitedJoint2DRV<psX,C,psDot,E,psX> ( ps ) { }
  const C& getC ( ) const { return first; }
  const E& getE ( ) const { return second; }
};
const F F_NIL("-.0");
const F F_BOG("BOG.0");
const F F_REST("REST.0");


class Q : public DelimitedJoint2DRV<psX,F,psSlash,F,psX> {
 public:
  Q ( )                : DelimitedJoint2DRV<psX,F,psSlash,F,psX> ( )    { }
  Q ( const char* ps ) : DelimitedJoint2DRV<psX,F,psSlash,F,psX> ( ps ) { }
  F&       setAct ( )       { return first;  }
  F&       setAwa ( )       { return second; }
  const F& getAct ( ) const { return first;  }
  const F& getAwa ( ) const { return second; }
};
const Q Q_BOT("-.0/-.0");
const Q Q_TOP("REST.0/REST.0");
Q Q_SCRATCH;


class Y : public DelimitedJoint4DRV<psX,B,psSemi,DelimitedJointArrayRV<4,psSemi,Q>,psSemi,F,psSemi,DelimitedJointArrayRV<4,psComma,R>,psX> {
 public:
  Y ( )                : DelimitedJoint4DRV<psX,B,psSemi,DelimitedJointArrayRV<4,psSemi,Q>,psSemi,F,psSemi,DelimitedJointArrayRV<4,psComma,R>,psX> ( )    { }
  Y ( const char* ps ) : DelimitedJoint4DRV<psX,B,psSemi,DelimitedJointArrayRV<4,psSemi,Q>,psSemi,F,psSemi,DelimitedJointArrayRV<4,psComma,R>,psX> ( ps ) { }
  int      getD ( )       const { return ( (F()!=second.get(3).getAwa()) ? 3 :
                                           (F()!=second.get(2).getAwa()) ? 2 :
                                           (F()!=second.get(1).getAwa()) ? 1 :
                                           (F()!=second.get(0).getAwa()) ? 0 : -1 ); }
  B&       setB ( )             { return first; }
  Q&       setQ ( int d )       { return (d<0) ? Q_SCRATCH : (d<4) ? second.set(d) : Q_SCRATCH; }
  F&       setP ( )             { return third; }
  R&       setR ( int i )       { return fourth.set(i); }
  const Q& getQ ( int d ) const { return (d<0) ? Q_TOP     : (d<4) ? second.get(d) : Q_BOT;     }
  const F& getP ( )       const { return third; }
  const R& getR ( int i ) const { return fourth.get(i); }
  const bool hasG ( ) const {
    for ( int d = getD(); d >= 0; d -- ) {
      if ( second.get(d).getAwa().getC().isG() ) { return 1; }
    }
    if ( second.get(0).getAct().getC().isG() ) { return 1; }
    return 0;
  }
};
const Y Y_INIT("0;BOG.0/BOG.0;-.0/-.0;-.0/-.0;-.0/-.0;-.0");


////////////////////////////////////////////////////////////////////////////////

class YModel {

 private:

  // Data members...
  OrderedModel<MapKey3D<D,C,C>,C,LogProb> mF; //mF returns whether to reduce or not
  OrderedModel<MapKey3D<D,C,C>,C,LogProb> mA; //mA generates best guess of next act trans
  OrderedModel<MapKey3D<D,C,C>,C,LogProb> mWa; //mW generates best guess of next awa trans
  OrderedModel<MapKey3D<D,C,C>,C,LogProb> mWb; //mW generates best guess of next awa trans
  OrderedModel<MapKey2D<D,C>,C,LogProb>   mP; //mP generates best guess for next preterm
  XModel<C>                               mX; //mX contains vocabulary

 public:

  typedef Y HidType;
  typedef X ObsType;
  typedef C PhaseType;

  // Static definitions...
  static const PhaseNum NUM_PHASES = 5;
  class SearchPath : public DelimitedStaticSafeArray<NUM_PHASES,psComma,ChoiceNum> {
   public:
    SearchPath ( ) : DelimitedStaticSafeArray<NUM_PHASES,psComma,ChoiceNum> ( 0 ) { }
  };

  class WeightedY;

  class PhaseValues : public penta<C,C,C,C,LogProb> {
   public:
    PhaseValues ( ) : penta<C,C,C,C,LogProb> ( ) { }
    PhaseValues ( const WeightedY& wy ) : penta<C,C,C,C,LogProb> ( C_NIL, C_NIL, C_NIL, C_NIL, wy.getProb() ){ }
    PhaseValues ( const C& f, const C& a, const C& w, const C& p, const LogProb pr ) : penta<C,C,C,C,LogProb> ( f,a,w,p,pr ) { }
    const C& getF    ( ) const { return first;  }
    const C& getA    ( ) const { return second; }
    const C& getW    ( ) const { return third;  }
    const C& getP    ( ) const { return fourth; }
    LogProb  getProb ( ) const { return fifth;  }

    friend ostream& operator<< ( ostream& os, const PhaseValues& pv ) { os<<"("<<pv.getF()<<","<<pv.getA()<<","<<pv.getW()<<","<<pv.getP()<<":"<<pv.getProb()<<")"; return os; }
  };

  class Measures {
   public:
    LogProb lgprMax;
    double  prNorm;
    double  prfl;
    double  prFl;
    double  prfL;
    double  prFL;
    double  prd1fl;
    double  prd1Fl;
    double  prd1fL;
    double  prd1FL;
    double  prd2fl;
    double  prd2Fl;
    double  prd2fL;
    double  prd2FL;
    double  prd3fl;
    double  prd3Fl;
    double  prd3fL;
    double  prd3FL;
    double  prd4fl;
    double  prd4Fl;
    double  prd4fL;
    double  prd4FL;
    double  prDfL;
    double  prBadd;
    double  prBplus;
    double  prBsto;
    double  prBcdr;
    double  prBmin;
    double  prBmindep;
    double  prBplusdep;
    double  prBplusdist;

    double  prflB;
    double  prfLB;
    double  prFlB;
    double  prFLB;
    double  prfla;
    double  prflc;
    double  prflo;
    double  prfLa;
    double  prfLc;
    double  prfLo;
    double  prFla;
    double  prFlc;
    double  prFlo;
    double  prFLa;
    double  prFLc;
    double  prFLo;
    Measures ( ) : lgprMax(0), prNorm(0.0),
                   prfl(0.0), prFl(0.0), prfL(0.0), prFL(0.0),
                   prd1fl(0.0), prd1Fl(0.0), prd1fL(0.0), prd1FL(0.0),
                   prd2fl(0.0), prd2Fl(0.0), prd2fL(0.0), prd2FL(0.0),
                   prd3fl(0.0), prd3Fl(0.0), prd3fL(0.0), prd3FL(0.0),
                   prd4fl(0.0), prd4Fl(0.0), prd4fL(0.0), prd4FL(0.0),
                   prDfL(0.0),
                   prBadd(0.0), prBplus(0.0), prBsto(0.0), prBcdr(0.0), prBmin(0.0), prBmindep(0.0),prBplusdep(0.0),prBplusdist(0.0),
                   prflB(0.0), prfLB(0.0), prFlB(0.0), prFLB(0.0), prfla(0.0), prflc(0.0), prflo(0.0), prfLa(0.0), prfLc(0.0), prfLo(0.0),
                   prFla(0.0), prFlc(0.0), prFlo(0.0), prFLa(0.0), prFLc(0.0), prFLo(0.0) { }
    void headwrite( ) { //Output the complexity metric header
      cout <<"word totsurp lexsurp synsurp entred embdep embdif ";

      if ( ENTROPY_VERBOSE ) {
        cout << "vocabentropy prevvocabentropy ";
      }

      cout << "F-L- F+L- F-L+ F+L+ ";
      cout << "d1F-L- d1F+L- d1F-L+ d1F+L+ ";
      cout << "d2F-L- d2F+L- d2F-L+ d2F+L+ ";
      cout << "d3F-L- d3F+L- d3F-L+ d3F+L+ ";
      cout << "d4F-L- d4F+L- d4F-L+ d4F+L+ ";
      cout << "distF-L+ ";

      cout << "Badd B+ Bsto Bcdr B- ";
      cout << "dB+ dB- DB+ ";

      cout << "F-L-Badd F-L-Bcdr F-L-BNil F-L-B+ ";
      cout << "F+L-Badd F+L-Bcdr F+L-BNil F+L-B+ ";
      cout << "F-L+Badd F-L+Bcdr F-L+BNil F-L+B+ ";
      cout << "F+L+Badd F+L+Bcdr F+L+BNil F+L+B+ ";

      cout << "\n";
    }
    void write ( ) { //The following metrics output cost for previous timestep
      cout << (prfl / prNorm) << " " << (prFl / prNorm) << " " << (prfL / prNorm) << " " << (prFL / prNorm) << " ";
      cout << (prd1fl / prNorm) << " " << (prd1Fl / prNorm) << " " << (prd1fL / prNorm) << " " << (prd1FL / prNorm) << " ";
      cout << (prd2fl / prNorm) << " " << (prd2Fl / prNorm) << " " << (prd2fL / prNorm) << " " << (prd2FL / prNorm) << " ";
      cout << (prd3fl / prNorm) << " " << (prd3Fl / prNorm) << " " << (prd3fL / prNorm) << " " << (prd3FL / prNorm) << " ";
      cout << (prd4fl / prNorm) << " " << (prd4Fl / prNorm) << " " << (prd4fL / prNorm) << " " << (prd4FL / prNorm) << " ";
      cout << (prDfL / prNorm) << " ";

      cout << (prBadd / prNorm) << " " << (prBplus / prNorm) << " " << (prBsto / prNorm) << " " << (prBcdr / prNorm) << " " << (prBmin / prNorm) << " ";
      cout << (prBplusdep / prNorm) << " " << (prBmindep / prNorm) << " " << (prBplusdist / prNorm) << " ";

      cout << (prfla / prNorm) << " " << (prflc / prNorm) << " " << (prflo / prNorm) << " " << (prflB / prNorm) << " ";
      cout << (prfLa / prNorm) << " " << (prfLc / prNorm) << " " << (prfLo / prNorm) << " " << (prfLB / prNorm) << " ";
      cout << (prFla / prNorm) << " " << (prFlc / prNorm) << " " << (prFlo / prNorm) << " " << (prFlB / prNorm) << " ";
      cout << (prFLa / prNorm) << " " << (prFLc / prNorm) << " " << (prFLo / prNorm) << " " << (prFLB / prNorm) << "\n";
    }
  };

  class WeightedY : public pair<Y,LogProb> {
   public:
    typedef Y VAR_TYPE;
    WeightedY ( ) : pair<Y,LogProb> ( ) { }
    WeightedY ( Measures& meas, const WeightedY& wy, const PhaseValues& pv ) : pair<Y,LogProb> (wy) {

      // Calculate measures...
      bool flagfl(0);
      bool flagFl(0);
      bool flagfL(0);

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

          flagFl = 1;
        }
      }
      else { // F-
        if (pv.getA() == C_NIL) { // L+
          meas.prfL += prCurr;

          if (wy.getY().getD() == 0) meas.prd1fL += prCurr;
          else if (wy.getY().getD() == 1) meas.prd2fL += prCurr;
          else if (wy.getY().getD() == 2) meas.prd3fL += prCurr;
          else meas.prd4fL += prCurr;

          meas.prDfL += prCurr * (wy.getY().getQ(wy.getY().getD()).getAwa().getE().toInt() -
                                  wy.getY().getQ(wy.getY().getD() - 1).getAwa().getE().toInt());

          flagfL = 1;
        }
        else { // L-
          meas.prfl += prCurr;

          if (wy.getY().getD() == 0) meas.prd1fl += prCurr;
          else if (wy.getY().getD() == 1) meas.prd2fl += prCurr;
          else if (wy.getY().getD() == 2) meas.prd3fl += prCurr;
          else meas.prd4fl += prCurr;

          flagfl = 1;
        }
      }

      meas.prNorm += prCurr;

      E eCtr = wy.getY().getP().second;
      E eCtrPlusOne = eCtr; eCtrPlusOne += 1;
      E eCtrNew     = eCtr; eCtrNew     += 2;

      // define l as either p_t-1 (if expansion) or b^d+1_t-1 (if none)
      F fL = ( C_NIL == pv.getF() ) ? wy.getY().getP() : wy.getY().getQ(wy.getY().getD()).getAct();  // set eL depending on whether expansion
      C& cL = fL.first;
      E& eL = fL.second;

      //cerr<<"aligning "<<cL<<" as "<<cL.isA1()<<" or "<<cL.isA2()<<"\n";

                                                // for d < new d': q^d_t copied from q^d_t-1 by default
      int d = wy.getY().getD();
      d += ((C_NIL == pv.getF()) ? 1 : 0);                                     // expansion increases depth
      d -= ((C_NIL == pv.getA()) ? 1 : 0);                                     // reduction decreases depth
      setY().setB() = (C_NIL != pv.getF() && C_BOG != pv.getF()) ? B_1 : B_0;  // boolean for tree reconstruction
      //setPV() = PhaseValues(pv);

      C cP = (C_NIL == pv.getA()) ? wy.getY().getQ(d).getAwa().first  : getY().getQ(d).getAct().first;
      E eP = (C_NIL == pv.getA()) ? wy.getY().getQ(d).getAwa().second : (cL.isI()) ? eL : eCtrPlusOne;
      C cR = pv.getW();
      E eR = ( (cP.isEB() && cR.isEE()) ? eCtrPlusOne :
               (cR.isEB() || (!cP.isO() && cR.isO()) ) ? eL :       // note, transfer left referent on -o b/c of split referent at gap introduction
               (cR.isI()) ? eP : eCtrPlusOne );
      E eG = ( (wy.getY().getQ(3).getAwa().first.isEB()) ? wy.getY().getQ(3).getAwa().second :
               (wy.getY().getQ(2).getAwa().first.isEB()) ? wy.getY().getQ(2).getAwa().second :
               (wy.getY().getQ(1).getAwa().first.isEB()) ? wy.getY().getQ(1).getAwa().second : wy.getY().getQ(0).getAwa().second );

      //E eG = ( (!wy.getY().getQ(d  ).getAwa().first.isG()) ? E_0 :
      //         (!wy.getY().getQ(d-1).getAwa().first.isG()) ? wy.getY().getQ(d  ).getAwa().second :
      //         (!wy.getY().getQ(d-2).getAwa().first.isG()) ? wy.getY().getQ(d-1).getAwa().second : wy.getY().getQ(d-2).getAwa().second );
      // set R values...
      setY().setR(0) = ( ( C_NIL != pv.getF() ) ? // if no expansion, identify b^d'_t-1 and p_t-1 referents
                         R(wy.getY().getQ(wy.getY().getD()).getAwa().second,L_EQ,wy.getY().getP().second) : R_NIL );
      setY().setR(1) = ( ( C_NIL == pv.getA() && cL.isI() ) ? // if reduction and left is head, identify parent and left referents
                         R(wy.getY().getQ(d).getAwa().second,L_EQ,eL) : R_NIL );

//        if (d>0)
//          cerr<<"result of "<<getY().getQ(d-1).getAwa()<<" is "<<getY().getQ(d-1).getAwa().isEB()<<" so "<<(d>0 && getY().getQ(d-1).getAwa().isEB() && pv.getW().isEE())<<"\n";
      d -= ((d>0 && getY().getQ(d-1).getAwa().first.isEB() && pv.getW().isEE()) ? 1 : 0);  // epda reduction decreases depth

      // compute new store state...
      if (C_NIL != pv.getA()) {                     // if is reduction, a^d_t stays copied from a^d_t-1
        setY().setQ(d).setAct().first  = pv.getA(); // if no reduction, a^d_t = A
        setY().setQ(d).setAct().second = eP;
      }
      setY().setQ(d).setAwa().first  = pv.getW();   // b^d_t = B
      setY().setQ(d).setAwa().second = eR;
      if (d < 3){
        setY().setQ(d+1) = Q_BOT;                   // for d > new d': q^d_t = '-'
      }
      setY().setP().first  = pv.getP();             // p_t = P
      setY().setP().second = eCtrNew;
      setProb() = pv.getProb();                     // copy prob

      // set function-argument R values...
      setY().setR(2) = ( (pv.getP().isO()      ) ? R(eR,L_P,eCtrNew) :
                         (cL.isC()             ) ? R(eR,L_C,eL) :
                         (cL.isM()             ) ? R(eL,L_1,eR) :
                         (cL.isA() && cR.is1()) ? R(eR,L_1,eL) :
                         (cL.isA() && cR.is2()) ? R(eR,L_2,eL) :
                         (cL.isA() && cR.is3()) ? R(eR,L_3,eL) :
                         (             cR.isC()) ? R(eL,L_C,eR) :
                         (             cR.isM()) ? R(eR,L_1,eL) :
                         (cL.is1() && cR.isA()) ? R(eL,L_1,eR) :
                         (cL.is2() && cR.isA()) ? R(eL,L_2,eR) :
                         (cL.is3() && cR.isA()) ? R(eL,L_3,eR) : R_NIL );
      // if gap disappears from parent to children, set R value for filler...
      setY().setR(3) = ( ( cP.isG() && !cL.isG() && !cR.isG() ) ? ( (cP.isGm() && cL.isI()) ? R(eG,L_1,eL) :
                                                                    (cP.isGm() && cR.isI()) ? R(eG,L_1,eR) :
                                                                    (cP.isGm()) ? R(eG,L_1,eP) :
								    ((cL.is2() && cR.isA()) || (cL.is1() && cL.isI())) ? R(eL,L_1,eG) :
								    ((cL.isA() && cR.is2()) || (cR.is1() && cR.isI())) ? R(eR,L_1,eG) :
								    ((cL.is3() && cR.isA()) || (cL.is2() && cL.isI())) ? R(eL,L_2,eG) :
								    ((cL.isA() && cR.is3()) || (cR.is2() && cR.isI())) ? R(eR,L_2,eG) :
								    (cL.is1()) ? R(eL,L_1,eG) :
								    (cR.is1()) ? R(eL,L_1,eG) :
                                                                    (cL.is2()) ? R(eL,L_2,eG) :
                                                                    (cR.is2()) ? R(eL,L_2,eG) :
                                                                    (cL.is3()) ? R(eL,L_3,eG) :
                                                                    (cR.is3()) ? R(eL,L_3,eG) :
                                                                    R_NIL ) :
                         ( getY().getP().first.isGm() ) ? R(eG,L_1,getY().getP().second) :
			 ( getY().getP().first.isG()  ) ? ( (getY().getP().first.is1()) ? R(getY().getP().second,L_2,eG) :
                                                            (getY().getP().first.is2()) ? R(getY().getP().second,L_3,eG) : R_NIL ) :
			 R_NIL );

      //If a gap has been filled at this timestep, measure the cost (B+)
      if ( wy.getY().hasG() && !getY().hasG() ) { meas.prBplus += prCurr;
                                       meas.prBplusdep += prCurr * (wy.getY().getD() + 1);
                                       if ( flagfl ) { meas.prflB += prCurr; }
                                       else if ( flagfL ) { meas.prfLB += prCurr; }
                                       else if ( flagFl ) { meas.prFlB += prCurr; }
                                       else { meas.prFLB += prCurr; }
      }
      else if ( getY().getR(3) != R_NIL ) {//This may not work since the Rs aren't guaranteed to be synchronized...
        meas.prBplusdist += prCurr * ( abs(getY().getR(3).third.toInt() - getY().getR(3).first.toInt()) );
      }
      else {
             //If a gap exists at this timestep,
             if ( getY().hasG() ){
                 meas.prBsto += prCurr;
                 //but a gap didn't exist in the previous timestep,
                 if ( !wy.getY().hasG() ){
                     meas.prBadd += prCurr;
                     if ( flagfl ) { meas.prfla += prCurr; }
                     else if ( flagfL ) { meas.prfLa += prCurr; }
                     else if ( flagFl ) { meas.prFla += prCurr; }
                     else { meas.prFLa += prCurr; }
                 }
                 //but a gap did exist in the previous timestep,
                 else {
                     meas.prBcdr += prCurr;
                     if ( flagfl ) { meas.prflc += prCurr; }
                     else if ( flagfL ) { meas.prfLc += prCurr; }
                     else if ( flagFl ) { meas.prFlc += prCurr; }
                     else { meas.prFLc += prCurr; }
                 }
             }
             else {
                 if ( flagfl ) { meas.prflo += prCurr; }
                 else if ( flagfL ) { meas.prfLo += prCurr; }
                 else if ( flagFl ) { meas.prFlo += prCurr; }
                 else { meas.prFLo += prCurr; }
             }
             meas.prBmin += prCurr;
             meas.prBmindep += prCurr * (wy.getY().getD() + 1);
      }
    }

    WeightedY ( const Y& y, const LogProb pr ) : pair<Y,LogProb> ( y, pr ) { }
    Y&       setY     ( )       { return first;  }
    //PhaseValues& setPV ( )    { return second;  }
    LogProb& setProb  ( )       { return second; }
    const Y& getY     ( ) const { return first;  }
    //const PhaseValues& getPV ( ) const { return second;  }
    LogProb  getProb  ( ) const { return second; }

    friend ostream& operator<< ( ostream& os, const WeightedY& wy ) { os<<wy.first<<":"<<wy.second; return os; }
  };

  static bool EOS;

  // Constructor methods...
  YModel ( ) : mF("F"), mA("A"), mWa("Wa"), mWb("Wb"), mP("P"), mX("X") { }

  // Extractor methods...
  const XModel<C>& getXModel() const { return mX; }

  PhaseValues getNext ( const PhaseValues& pv, PhaseNum pn, SearchPath& sp, Array<WeightedY>& beam, const X& x ) const {
    if ( 0==pn ) {
      //cerr<<"    pull from beam...\n";
      return PhaseValues(beam[sp[0]]);
    }
    else if ( 1==pn ) { //D-phase
      int d = beam[sp[0]].getY().getD();
      //cerr<<"    trying F "<< d+2 <<" "<< beam[sp[0]].getY().getQ(d).getAwa()<<" "<<beam[sp[0]].getY().getP()<< "... "<<"\n";
      //Pretend you're at the end of the previous sentence.
      if ( F_NIL == beam[sp[0]].getY().getP() ) return (0==sp[pn]) ? PhaseValues(C_BOG,C_NIL,C_NIL,C_NIL,1.0) : PhaseValues();
      //Consult model (remember model file indices start at 1, whereas store Y indices start at 0, so d+1 syncs)
      // also, d shows lowest level occupied, so add another +1 depth for destination: d+2
      pair<C,LogProb> fpr = mF.getDistrib(MapKey3D<D,C,C>(d+2,beam[sp[0]].getY().getQ(d).getAwa().first,beam[sp[0]].getY().getP().first))[sp[pn]];
      //'If' No expansion; 'else' is an expansion
      if (C_NIL == fpr.first) {
        return PhaseValues ( beam[sp[0]].getY().getQ(d).getAct().first,C_NIL,C_NIL,C_NIL, pv.getProb() * fpr.second );
      }
      else return PhaseValues ( C_NIL,C_NIL,C_NIL,C_NIL, pv.getProb() * fpr.second );
    }
    else if ( 2==pn ) {
      int d = beam[sp[0]].getY().getD();
      //Fill pv.F with t-1.Act if C_NIL is seen as code for expansion
      C c = (C_NIL == pv.getF()) ? beam[sp[0]].getY().getP().first : pv.getF();
      //Add a 1 or 0 to d based on F-phase expansion or not
      d += ((C_NIL == pv.getF()) ? 1 : 0);
      //cerr<<"    trying A "<< d+1 <<" "<<beam[sp[0]].getY().getQ(d-1).getAwa()<<" "<<c <<"...\n";
      if ( C_BOG == pv.getF() ) return (0==sp[pn]) ? PhaseValues(C_BOG,C_NIL,C_NIL,C_NIL,1.0) : PhaseValues();
      //Consult model (remember model file indices start at 1, whereas store Y indices start at 0, so d+1 syncs)
      pair<C,LogProb> fpr = mA.getDistrib(MapKey3D<D,C,C>(d+1,beam[sp[0]].getY().getQ(d-1).getAwa().first,c))[sp[pn]];
      // upper active/awaited transition must agree with EOS flag
      //cerr<<"not happening "<<d<<" "<<EOS<<" "<<ffpr.first<<" "<<(C_NIL==ffpr.first.getX2())<<" "<<ffpr.second<<"\n";
      while ( (!EOS && d==0 && C_NIL==fpr.first && fpr.second>LogProb()) ||
              (EOS  && (d!=0 || C_NIL!=fpr.first) && fpr.second>LogProb()) ) {
        //cerr<<"aah! it happened! "<<d<<" "<<EOS<<" "<<ffpr.first<<" "<<ffpr.second<<"\n";
        sp[pn]++;
        fpr = mA.getDistrib(MapKey3D<D,C,C>(d+1,beam[sp[0]].getY().getQ(d-1).getAwa().first,c))[sp[pn]];
      }
      //If reduction, X2==NIL: use d-1.Act; otherwise use active transition X2
      //If reduction (Awaited transition), then fill in Awa from mA; else wait for mW
      return PhaseValues ( pv.getF(),fpr.first,C_NIL,C_NIL,pv.getProb() * fpr.second );
    }
    else if ( 3==pn ) {
      int d = beam[sp[0]].getY().getD();
      //Fill pv.F with t-1.Act if C_NIL is seen as code for expansion
      C c = (C_NIL == pv.getF()) ? beam[sp[0]].getY().getP().first : pv.getF();
      //Add a 1 or 0 to d based on F-phase expansion or not
      d += ((C_NIL == pv.getF()) ? 1 : 0);
      //Subtract a 1 or 0 to d based on A-phase reduction or not
      d -= ((C_NIL == pv.getA()) ? 1 : 0);
      //cerr<<"    trying W "<< d+1 <<" "<< ((C_NIL==pv.getA())?beam[sp[0]].getY().getQ(d).getAwa():pv.getA()) <<" "<< c <<"...\n";
      if ( C_BOG == pv.getF() ) return (0==sp[pn]) ? PhaseValues(C_BOG,C_NIL,C_REST,C_NIL,1.0) : PhaseValues();
      //Consult model (remember model file indices start at 1, whereas store Y indices start at 0, so d+1 syncs)
      const pair<C,LogProb>& fpr = ((C_NIL!=pv.getA())?mWa:mWb).getDistrib(MapKey3D<D,C,C>(d+1,(C_NIL==pv.getA())?beam[sp[0]].getY().getQ(d).getAwa().first:pv.getA(),c))[sp[pn]];
      return PhaseValues ( pv.getF(),pv.getA(),fpr.first,C_NIL, pv.getProb() * fpr.second );
    }
    else if ( 4==pn ) {
      int d = beam[sp[0]].getY().getD();
      //Add a 1 or 0 to d based on F-phase expansion or not
      d += ((C_NIL == pv.getF()) ? 1 : 0);
      //Subtract a 1 or 0 to d based on A-phase reduction or not
      d -= ((C_NIL == pv.getA()) ? 1 : 0);
      //cerr<<"    trying P " << d+2 <<" "<< pv.getW() << "... "<<"\n";
      //Consult model (remember model file indices start at 1, whereas store Y indices start at 0, so d+1 syncs)
      // also, d shows lowest level occupied, so add another +1 depth for destination: d+2
      const pair<C,LogProb>& fpr = mP.getDistrib(MapKey2D<D,C>(d+2,pv.getW()))[sp[pn]];
      return PhaseValues ( pv.getF(),pv.getA(),pv.getW(),fpr.first, pv.getProb() * fpr.second );
    }
    else {
      //cerr<<"    trying X\n";
      return PhaseValues ( pv.getF(),pv.getA(),pv.getW(),pv.getP(), pv.getProb() * mX.getProb(pv.getP(),x) );
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

