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

#include <getopt.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <sys/time.h>
#include <nl-heap.h>
#include <Beam.hpp>
#include <ModelReader.hpp>

////////////////////////////////////////////////////////////////////////////////

static struct option long_options[] = {
  {"beam-width", required_argument, 0, 'b'},
  {"complexity", no_argument, 0, 'c'},
  {"verbose", no_argument, 0, 'v'},
  {"very-verbose", required_argument, 0, 'V'},
  {0, 0, 0, 0}};

unsigned int       BEAM_WIDTH     = 2000;
const unsigned int TRELLIS_LENGTH = 200;
bool               VERBOSE        = false;
int                VERY_VERBOSE   = 0;
bool               REPORT_COMPLEX   = false;

typedef int PhaseNum;
typedef int ChoiceNum;

////////////////////////////////////////////////////////////////////////////////

template<class YM>
class PhaseParser : public ModelReader<YM> {

 private:

  // Static definitions...
  typedef typename YM::PhaseValues PhaseValues;
  typedef typename YM::WeightedY   WeightedY;
  typedef typename YM::SearchPath  SearchPath;
  typedef typename YM::PhaseType   C;
  typedef typename YM::HidType     Y;
  typedef typename YM::ObsType     X;

  class QueueElement : public quad<PhaseValues,PhaseNum,SearchPath,PhaseValues> {
   public:
    QueueElement ( ) : quad<PhaseValues,PhaseNum,SearchPath,PhaseValues> ( ) { }
    QueueElement ( const PhaseValues& pvPrev, PhaseNum pn, const SearchPath& sp, const PhaseValues& pv ) : quad<PhaseValues,PhaseNum,SearchPath,PhaseValues>(pvPrev,pn,sp,pv) { }
    PhaseNum          getPhase      ( ) const { return this->second; }
    const SearchPath& getSearchPath ( ) const { return this->third;  }
    const PhaseValues&  getPhaseValues  ( ) const { return this->fourth; }
    LogProb           getProb       ( ) const { return this->fourth.getProb(); }
    friend ostream&   operator<<    ( ostream& os, const QueueElement& qe ) { os<<qe.first<<"|"<<qe.second<<"|"<<qe.third<<"|"<<qe.fourth; return os; }
  };
  static bool outRank ( const QueueElement& qe1, const QueueElement& qe2 ) {
    return ( qe1.getProb() > qe2.getProb() );
  }

  // Data members...
  BackPointerBestFirstBeam<WeightedY> beams[TRELLIS_LENGTH];  // set of complete hypotheses at each time step
  X                                   xs   [TRELLIS_LENGTH];  // set of observations at each time step
  int                                 t;

  double prevembeddep;
  double preventropy;

  double prevtotsurp;
  double prevlexsurp;
  double prevsynsurp;
  double preventred;
  double prevembedif;


  // Private methods...
  QueueElement getNext ( const QueueElement& qe, PhaseNum pnAdd, ChoiceNum cnAdd, BestFirstBeam<WeightedY>& beam, const X& x ) const {
    PhaseNum   pn = qe.second + pnAdd;
    SearchPath sp = qe.third; if ( pn<YM::NUM_PHASES ) { sp[pn] += cnAdd; if(cnAdd) for(PhaseNum pn1=pn+1; pn1<YM::NUM_PHASES; pn1++) sp[pn1]=0; }
    //cerr<<"from "<<qe.fourth<<"|"<<pn<<"|"<<sp<<"|...\n";
    return QueueElement ( (pnAdd)?qe.fourth:qe.first, pn, sp, ModelReader<YM>::getModel().getNext((pnAdd)?qe.fourth:qe.first,pn,sp,beam,x) );
  }
  void update ( const X& x ) {
    // Init new beam...
    beams[t] = BackPointerBestFirstBeam<WeightedY> ( BEAM_WIDTH );               // %2
    xs   [t] = x;                                                                // %2

    typename YM::Measures meas;

    // Create new queue and add initial partial hypoth...
    Heap<QueueElement,PhaseParser::outRank> queue;                  // set of partial hypotheses in best-first (A*) search

    //cerr<<"    pull from beam...\n";
    //cerr<<"      init step: "<<QueueElement(PhaseValues(),0,SearchPath(),PhaseValues(beams[(t-1)][0]))<<"\n";
    queue.enqueue ( QueueElement(PhaseValues(),0,SearchPath(),PhaseValues(beams[(t-1)][0])) );  // %2

    // Iterate over set of active hypotheses...
    while ( queue.getSize()>0 and beams[t].size()<BEAM_WIDTH ) {                 // %2

      if ( VERY_VERBOSE == t ) {
        cerr<<"------size:"<<queue.getSize()<<"\n";
        cerr<<queue;
        cerr<<"------\n";
      }

      // Obtain the leading hypothesis...
      QueueElement qe = queue.dequeueTop();
      QueueElement qe2;

      // If last phase...
      if ( YM::NUM_PHASES == qe.getPhase() )
        beams[t].add ( WeightedY ( meas, beams[t-1].get(qe.getSearchPath().get(0)), qe.getPhaseValues() ), qe.getSearchPath().get(0) );           // %2
      // If not last phase..
      else {
        // Take side step...
        qe2 = getNext(qe,0,1,beams[(t-1)],x);                                    // %2
        //cerr<<"      side step: "<<qe2<<"\n";
        if ( qe2.getPhaseValues().getProb() > LogProb() ) queue.enqueue ( qe2 );
        // Take forward step...
        qe2 = getNext(qe,1,0,beams[(t-1)],x);                                    // %2
        //cerr<<"      fwrd step: "<<qe2<<"\n";
        if ( qe2.getPhaseValues().getProb() > LogProb() ) queue.enqueue ( qe2 );
      }
    }

    if ( VERBOSE ) cerr<<"\n"<<beams[t];
    if ( REPORT_COMPLEX ) {
      /* Output complexity from previous timestep (since F was first phase of this timestep)*/
      if (t == 1 ) {
        meas.headwrite();
        cout << x;
      }
      else { /* ignore first timestep since no previous metrics have been computed yet */
        cout << " " << prevtotsurp << " " << prevlexsurp << " " << prevsynsurp << " " << preventred << " " << prevembeddep << " " << prevembedif << " ";

        meas.write();
        cout << x;
      }

      /* Update previous complexity metrics for next timestep */
      double aveembeddep = 0.0;
      double beamprob = 0.0;
      double prevbeamprob = 0.0;
      double entropy = 0.0;
      double thisprob = 0.0;
      double synonly = 0.0;

      if ( beams[t].size() != 0 ) { //Since beams is an array of hashes, accessing [0] creates it, so make sure it exists
        LogProb mymax = beams[t][0].getProb(); //Scale everything by the largest thing in the beam to obtain same relative metrics without underflow
        double mymaxNorm = 0.0;
        for ( unsigned int ix=0; ix<beams[t].size(); ix++ ) {
          mymaxNorm += (beams[t][ix].getProb()/mymax).toDouble(); //Create a scaled normalization constant to prevent entred underflow
        }
        for ( unsigned int ix=0; ix<beams[t-1].size(); ix++ ) {
          prevbeamprob += (beams[t-1][ix].getProb()/mymax).toDouble(); //Recalculate prevbeamprob to use same normalization constant
        }

        for ( unsigned int ix=0; ix<beams[t].size(); ix++ ) {
          thisprob = (beams[t][ix].getProb()/mymax).toDouble();
          aveembeddep += ( beams[t][ix].getY().getD() + 1 ) * thisprob;
          beamprob += thisprob;
          entropy -= (thisprob/mymaxNorm)*log2(thisprob/mymaxNorm); //Use mymaxNorm to make the beam sum to 1 since entred isn't scale-invariant
          synonly += ((beams[t][ix].getProb()/mymax) / ModelReader<YM>::getModel().getXModel().getProb(beams[t][ix].getY().getP().getC(),x)).toDouble();
        }

        prevtotsurp = -log2(beamprob/prevbeamprob);
        prevlexsurp = (log2(beamprob/synonly) == 0)?log2(1.0):-log2(beamprob/synonly);
        prevsynsurp = -log2(synonly/prevbeamprob);
        preventred = fmax(0.0,preventropy-entropy);
        preventropy = entropy;
        aveembeddep /= beamprob;
        prevembedif = aveembeddep-prevembeddep;
        prevembeddep = aveembeddep;
      }
      else if (REPORT_COMPLEX) {
        /* Output complexity from previous timestep (since F was first phase of this timestep)*/
        if (t == 1 ) {
          cout << "word\n";
        }
        cout << x << "\n";
      }
    }
/**/cerr<<x<<" ("<<beams[t].size()<<") ";                                    // %2
  }

 public:

  // Constructor methods...
  PhaseParser ( int nArgs, char* argv[], const Y& Y_INIT ) : ModelReader<YM>(nArgs,argv) {

    cerr<<"start!\n";

    // Check flags...
    // char opt;
    for( int opt, option_index = 0; (opt = getopt_long(nArgs, argv, "vV:cb:", long_options, &option_index)) != -1; ) {
      switch(opt) {
      //case 't': TRELLIS_LENGTH = atoi(optarg); if(TRELLIS_LENGTH<=0)cerr<<"\nERROR: trellis length set to zero!\n\n"; break;
      case 'b': BEAM_WIDTH = atoi(optarg); if(BEAM_WIDTH<=0)cerr<<"\nERROR: beam width set to zero!\n\n"; break;
      case 'v': VERBOSE = true; break;
      case 'V': VERBOSE = true; VERY_VERBOSE = atoi(optarg); break;
      case 'c': REPORT_COMPLEX = true; break;
      default: break;
      }
    }

    //        // For each letter in input...
    //        int c;
    //        for ( int t=1; EOS!=c=getc(); t++ ) {

    // For each sentence...
    //time_t start, end, istart, iend;
    struct timeval start, end, istart, iend;
    gettimeofday (&start, NULL);

    const X X_NIL("");  // This is an empty X, indicating failure to read a word from input stream.
    X X_FINAL;          // This is a dummy first word of following sentence, equal to the first word in the current sentence.

    C C_NIL;

    char psBuff[1000];

    for ( int n=1; cin.getline(psBuff,1000); n++ ) {
      gettimeofday (&istart, NULL);
      prevembeddep = 0.0;
      preventropy = 0.0;
      /**/ cerr<<n<<":"; //Input line number

      // Initialize beam with start state...
      beams[0] = BackPointerBestFirstBeam<WeightedY> ( BEAM_WIDTH );
      beams[0].add ( WeightedY(Y_INIT,LogProb(1.0)), 0 );
      /**/ cerr<<" ("<<beams[0].size()<<") ";                                         // %2
      if ( VERBOSE ) cerr<<"\n"<<beams[0];
      YM::EOS = true;

      // For each word in sentence...
      int numwords = 0;
      StringInput si(psBuff), siNext;
      X x;
      for ( t=1; ((siNext=si>>x>>" ")!=NULL or (siNext=si>>x>>"\0")!=NULL) and x!=X_NIL; si=siNext, t++ ) {
	if ( t==1 ) X_FINAL = x;
        // Update hmm...
        update ( x );
        YM::EOS = false;
        numwords++;
      }
      // Update hmm one last time (for final reduction)...
      YM::EOS = true;
      update ( X_FINAL );
      /**/ cerr<<"<FINAL> ("<<beams[t].size()<<")\n";                                 // %2

      if (REPORT_COMPLEX) {
        //Separate complexity measures from output
        cout<<"\n----------\n";
      }

      if (beams[t].size() == 0){
       numwords = 0;
      }

      // Print most likely sequence...
      unsigned int best = 0;
      double pr = 0.0;
      while ( best<beams[t].size() and beams[t][best].getY().getD()>=0 ) best++; // %2
      if ( best < beams[t].size() ) {                                            // %2
        pr = beams[t][best].getProb().toDouble();
        for ( ; t>=0; t-- ) {
          cout << t << " " << beams[t][best].getY() << " " << xs[t] << "\n";     // %2
          best = beams[t].getBack(best);
        }
      }

      gettimeofday (&iend, NULL);

      // Print metadata...
      cout<<"---------- line="<<n<<" pr="<<pr<<" msec="<<(iend.tv_sec-istart.tv_sec)*1000+(iend.tv_usec-istart.tv_usec)/1000.0+.05<<" wds="<<numwords<<"\n";

      cerr<<"Line parse time: "<<(iend.tv_sec-istart.tv_sec)*1000+(iend.tv_usec-istart.tv_usec)/1000.0+.05<<" milliseconds for "<<numwords<<" words.\n";
    }
    gettimeofday (&end, NULL);
    cerr<<"Total parse time: "<<(end.tv_sec-start.tv_sec)*1000+(end.tv_usec-start.tv_usec)/1000.0+.05<<" milliseconds\n";
  }
};
