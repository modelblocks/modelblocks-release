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

#define ARMA_64BIT_WORD
#include <iostream>
#include <fstream>
#include <list>
using namespace std;
#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression_function.hpp>
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>
using namespace arma;
using namespace mlpack;
using namespace mlpack::regression;
using namespace mlpack::optimization;
#include <thread>
#include <mutex>
#define DONT_USE_UNMAPPABLE_TUPLES
#include <nl-randvar.h>
#include <nl-string.h>
#include <Delimited.hpp>
#include<time.h>

////////////////////////////////////////////////////////////////////////////////

DiscreteDomain<int> domXFeat;
class XFeat : public DiscreteDomainRV<int,domXFeat> {
 public:
  XFeat ( )                : DiscreteDomainRV<int,domXFeat> ( )    { }
  XFeat ( int i )          : DiscreteDomainRV<int,domXFeat> ( i )  { }
  XFeat ( const char* ps ) : DiscreteDomainRV<int,domXFeat> ( ps ) { }
};

DiscreteDomain<int> domYVal;
class YVal : public DiscreteDomainRV<int,domYVal> {
 public:
  YVal ( )                : DiscreteDomainRV<int,domYVal> ( )    { }
  YVal ( int i )          : DiscreteDomainRV<int,domYVal> ( i )  { }
  YVal ( const char* ps ) : DiscreteDomainRV<int,domYVal> ( ps ) { }
};

////////////////////////////////////////////////////////////////////////////////

class SpMatLogisticRegressionFunction {

 private:

  double        lambda;
  uint          numThreads;
  arma::sp_mat  predictors;
  arma::sp_mat  responses;
  arma::mat     initialpoint;
  //arma::mat     norms;         // cache norms during Evaluate for use in Gradient
  arma::mat     cooccurrences; // cache cooccurrences from beginning
  arma::mat     expectations;  // cache expectationa during Evaluate for use in Gradient
  arma::vec     counts;        // for (superposed) fractional or repeated items in data
  vector<mutex> vmExpectationRows;

 public:

  SpMatLogisticRegressionFunction ( uint nX, uint nY, uint nT = 1, double l = 0.0 ) : lambda(l), numThreads(nT), vmExpectationRows(nX) {
    initialpoint.randn( nX, nY );
    initialpoint *= 0.01;
    expectations.zeros( nX, nY );
    cerr << "L2 regularization parameter: " << lambda << "\n";
  }

  arma::sp_mat&       Predictors ( )        { return predictors; }
  arma::sp_mat&       Responses  ( )        { return responses;  }
  const arma::sp_mat& Responses  ( ) const  { return responses;  }
  arma::vec&          Counts     ( )        { return counts;     }

  double Evaluate(const arma::mat& parameters)  {

////    cerr<<"inEval"<<endl;

    const double regularization = 0.5 * lambda * arma::accu( parameters % parameters );

////    const arma::mat logscoredistrs = parameters * predictors;
////    const arma::mat logscores      = arma::ones<rowvec>(parameters.n_rows) * (responses % logscoredistrs);
////    const arma::mat logprobs       = logscores - arma::log( arma::ones<rowvec>(parameters.n_rows) * arma::exp(logscoredistrs) );
////    cerr<<"regularized logP = "<<-arma::accu(logprobs)<<"\n";
////    cerr<<parameters(0,0)<<"\n";
////    return -arma::accu(logprobs) + regularization;

    if ( cooccurrences.n_cols == 0 ) {
//?!      predictors *= dUnderflowScaler;
      arma::sp_mat weighted_responses( responses );
      for( uint i=0; i<weighted_responses.n_cols; i++ )
//      weighted_responses.col(i) *= counts(i) * dUnderflowScaler;
       for( uint j=weighted_responses.col_ptrs[i]; j<weighted_responses.col_ptrs[i+1]; j++ )
          weighted_responses(weighted_responses.row_indices[j],i) *= counts(i);
      cooccurrences = predictors * weighted_responses.t();
    }
//    if ( cooccurrences.n_cols == 0 ) cooccurrences = predictors * arma::diagmat(counts) * dUnderflowScaler * responses.t();
    double totlogprob = 0;
    mutex mTotlogprob;
    expectations.zeros ( );
    vector<thread> vtWorkers; // ( numThreads, 
    for ( uint jglobal=0; jglobal<numThreads; jglobal++ ) vtWorkers.push_back( thread( [&] (int j) {
      // multi-threaded -- each worker processes a division of the data items (columns)...
      ////cerr<<"thread "<<j<<" "<<((predictors.n_cols*j)/numThreads)<<" "<<((predictors.n_cols*(j+1))/numThreads)<<" started...\n";
      for ( uint c=(predictors.n_cols*j)/numThreads; c<(predictors.n_cols*(j+1))/numThreads; c++ ) {
        arma::vec logscoredistr = arma::zeros( parameters.n_cols );
        // Add effect of each (sparse) predictor as vector of output weights...
        for ( uint i=predictors.col_ptrs[c]; i<predictors.col_ptrs[c+1]; i++ )
          logscoredistr += parameters.row(predictors.row_indices[i]).t() * predictors.values[i];
        arma::vec scoredistr = arma::exp( logscoredistr );
        double norm = arma::accu( scoredistr );
        // Prevent overflow...
        if ( norm == 1.0/0.0 ) {
          ////cerr<<"WARNING: infinite norm in data item "<<c<<"; substituting Dirac delta at max."<<endl;
          uint ind_max=0; for( uint i=0; i<logscoredistr.size(); i++ ) if( logscoredistr(i)>logscoredistr(ind_max) ) ind_max=i;
          logscoredistr -= logscoredistr( ind_max );
          scoredistr = arma::exp( logscoredistr );
          norm = arma::accu( scoredistr );
//          logscoredistr.fill(0.0); logscoredistr(ind_max) = exp(1.0);
//          scoredistr.fill(0.0);    scoredistr(ind_max)    = 1.0;
//          norm = exp(1.0);
        }
        if ( norm == -1.0/0.0 ) cerr<<"WARNING: neg inf norm!"<<endl;
        if ( norm == 0.0 ) cerr<<"WARNING: zero norm!"<<endl;
        { lock_guard<mutex> guard ( mTotlogprob );
          totlogprob += counts(c) * ( arma::accu( logscoredistr % responses.col(c) ) - log(norm) );
//cerr<<"?!?! "<< counts(c) << " * " << dUnderflowScaler << " * ( " << arma::accu( logscoredistr % responses.col(c) ) << " - " << log(norm) << " )" << endl;
        }
        for ( uint i=predictors.col_ptrs[c]; i<predictors.col_ptrs[c+1]; i++ ) {
//          if( arma::accu( expectations.row(predictors.row_indices[i]) + (counts(c) * dUnderflowScaler * predictors.values[i] * scoredistr.t()) ) != 1.0/0.0 ) {
            lock_guard<mutex> guard ( vmExpectationRows[predictors.row_indices[i]] );
            expectations.row(predictors.row_indices[i]) += counts(c) * predictors.values[i] * (scoredistr.t() / norm);
//          } else {
//            cerr<<"WARNING: infinite expectation row; skipping."<<endl;
//          }
	}
        ////if ( c%100000==0 ) cerr<<c<<"/"<<predictors.n_cols<<" done\n";	
      }
    }, jglobal ));
    for ( auto& t : vtWorkers ) t.join();
    cerr<<"regularized logP (inverse cost) = "<<totlogprob - regularization<<"\n";

////    cerr<<"outEval"<<endl;

    return -totlogprob + regularization;
  }

  void Gradient(const arma::mat& parameters, arma::mat& gradient)  {

    // Regularization term.
    arma::mat regularization = lambda * parameters;

////    cerr<<"inGrad"<<endl;

////    const arma::mat scoredistrs = arma::exp( parameters * predictors );
////    const arma::mat norms       = arma::ones<rowvec>(parameters.n_rows) * scoredistrs;
////    const arma::mat predictions = scoredistrs / (arma::ones<vec>(parameters.n_rows) * norms);
////    gradient                    = - (responses - predictions) * predictors.t();

    gradient = - ( cooccurrences - expectations ) + regularization;

////    cerr<<"outGrad "<<arma::accu(expectations)<<endl;
  }

  const arma::mat& GetInitialPoint ( ) const { return initialpoint; }
};

////////////////////////////////////////////////////////////////////////////////

int main ( int nArgs, char* argv[] ) {

  uint maxiters = nArgs>4 ? atoi(argv[3]) : 0;
  cerr << "Max iters (0 = no bound): " << maxiters << "\n";

  list<trip<DelimitedList<psX,DelimitedPair<psX,Delimited<XFeat>,psEquals,Delimited<double>,psX>,psComma,psX>,Delimited<YVal>,Delimited<double>>> lplpfdy;

  // Read data...
  cerr << "Reading data...\n";
  int numfeattokens = 0;
  while ( cin && EOF!=cin.peek() ) {
    auto& plpfdy = *lplpfdy.emplace(lplpfdy.end());
    cin >> plpfdy.first() >> " : " >> plpfdy.second() >> vector<const char*>({"\n"," "});
    if( '=' == cin.peek() ) cin >> "= " >> plpfdy.third() >> "\n";
    else plpfdy.third() = 1;
    numfeattokens += plpfdy.first().size();
  }
  cerr << "Data read: x=" << domXFeat.getSize() << " y=" << domYVal.getSize() << ".\n";

  if( 0 == domXFeat.getSize() or 0 == domYVal.getSize() ) exit( 0 );   // Empty model is easy.

  // Populate predictor matrix and result vector...
  SpMatLogisticRegressionFunction f ( domXFeat.getSize(), domYVal.getSize(), nArgs>1 ? atoi(argv[1]) : 1, nArgs>2 ? atof(argv[2]) : 0.0 );
  sp_mat& DbyFX = f.Predictors();
  sp_mat& DbyY  = f.Responses();
  vec& Dcounts  = f.Counts();
  umat xlocs ( 2, numfeattokens );
  vec  xvals = arma::zeros( numfeattokens );
  umat ylocs ( 2, lplpfdy.size() );
  vec  yvals (    lplpfdy.size() );
  Dcounts.zeros ( lplpfdy.size() );
  int t = 0; int i = 0;
  for ( auto& plpfdy : lplpfdy ) {
    for ( auto& pfd : plpfdy.first() ) {
      xlocs(0,i) = pfd.first.toInt();
      xlocs(1,i) = t;
      xvals(i)   += pfd.second;
      i++;
    }
    ylocs(0,t) = plpfdy.second().toInt();
    ylocs(1,t) = t;
    yvals(t) = 1.0;
    Dcounts(t) = plpfdy.third();
    t++;
  }
  DbyFX = sp_mat ( true, xlocs, xvals, domXFeat.getSize(), lplpfdy.size() );
  DbyY  = sp_mat ( true, ylocs, yvals, domYVal.getSize(),  lplpfdy.size() );
  cerr<<"Populated "<<DbyFX.n_cols<<" data items."<<endl;

  // Regress and print params...
  auto o = L_BFGS<SpMatLogisticRegressionFunction> ( f, 5, maxiters );
  auto W = f.GetInitialPoint();
  o.Optimize(W);
  cerr<<"Done.\n";

////  // Regress and print params...
////  auto& w = LogisticRegression<>(X,y).Parameters();
//  cout << "(const) = " << w(0) << "\n";
//  for ( size_t i=1; i<w.size(); i++ ) {
//    cout << Feat::getDomain().getString(i-1) << " = " << w(i) << "\n";
//  }

  for ( size_t xf=0; xf<W.n_rows; xf++ )
    for ( size_t y=0; y<W.n_cols; y++ )
      cout << XFeat::getDomain().getString(xf) << " : " << YVal::getDomain().getString(y) << " = " << W(xf,y) << "\n";
}

