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
#include <mlpack/core/optimizers/sgd/sgd.hpp>
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
  vector<mutex> vmExpectationRows;

 public:

  SpMatLogisticRegressionFunction ( uint nX, uint nY, double l = 0.0, uint nT = 1 ) : lambda(l), numThreads(nT), vmExpectationRows(nX) {
    initialpoint.randn ( nX, nY );
    initialpoint *= 0.01 / 10.0;    /* /= double(predictors.n_cols);  /* *= 0.01 */;
    expectations.zeros ( nX, nY );
    cerr << "L2 regularization parameter: " << lambda << "\n";
  }

  uint                NumFunctions ( ) const { return 1; }
  arma::sp_mat&       Predictors   ( )       { return predictors; }
  arma::sp_mat&       Responses    ( )       { return responses;  }
  const arma::sp_mat& Responses    ( ) const { return responses;  }

  double Evaluate( const arma::mat& parameters, uint i )  {

//    cerr<<"inEval\n";

    const double regularization = 0.5 * lambda * arma::accu( parameters % parameters );
/*
    const arma::mat logscoredistrs = parameters.t() * predictors;
    const arma::mat logscores      = arma::ones<rowvec>(parameters.n_cols) * (responses % logscoredistrs);
    const arma::mat logprobs       = logscores - arma::log( arma::ones<rowvec>(parameters.n_cols) * arma::exp(logscoredistrs) );
    cerr<<"regularized logP = "<<-arma::accu(logprobs)<<"\n";
    cerr<<parameters(0,0)<<"\n";
    return -arma::accu(logprobs) + regularization;
*/

//    if ( cooccurrences.n_cols == 0 ) initialpoint /= double(predictors.n_cols);
    if ( cooccurrences.n_cols == 0 ) { predictors /= 10.0; /* responses /= 10.0; */ }
    if ( cooccurrences.n_cols == 0 ) cooccurrences = predictors * responses.t()  /*double(predictors.n_cols)*/;
    double totlogprob = 0;
    mutex mTotlogprob;
    expectations.zeros ( );
    vector<thread> vtWorkers; // ( numThreads, 
    for ( uint jglobal=0; jglobal<numThreads; jglobal++ ) vtWorkers.push_back( thread( [&] (int j) {
      // multi-threaded...
      ////cerr<<"thread "<<j<<" "<<((predictors.n_cols*j)/numThreads)<<" "<<((predictors.n_cols*(j+1))/numThreads)<<" started...\n";
      for ( uint c=(predictors.n_cols*j)/numThreads; c<(predictors.n_cols*(j+1))/numThreads; c++ ) {
        arma::vec logscoredistr = arma::zeros( parameters.n_cols );
        for ( uint i=predictors.col_ptrs[c]; i<predictors.col_ptrs[c+1]; i++ )
          logscoredistr += parameters.row(predictors.row_indices[i]).t() * predictors.values[i];
        arma::vec scoredistr = arma::exp( logscoredistr );
        double norm = arma::accu( scoredistr );
//for( size_t j=0; j<parameters.n_cols; j++ )
//  if( scoredistr[j]>1000.0 ) cerr << "logscoredistr[" << j << "] = " << logscoredistr[j] << endl;
//if( norm > 1000.0 ) cerr << "norm[" << c << "] = " << norm << endl;
        { lock_guard<mutex> guard ( mTotlogprob );
          totlogprob += arma::accu( logscoredistr % responses.col(c) ) - log(norm);
          }
        for ( uint i=predictors.col_ptrs[c]; i<predictors.col_ptrs[c+1]; i++ ) {
          lock_guard<mutex> guard ( vmExpectationRows[predictors.row_indices[i]] );
          expectations.row(predictors.row_indices[i]) += predictors.values[i] * (scoredistr.t() / norm);
          }
        ////if ( c%100000==0 ) cerr<<c<<"/"<<predictors.n_cols<<" done\n";	
        }
      }, jglobal ));
    for ( auto& t : vtWorkers ) t.join();
    cerr<<"regularized logP = "<<totlogprob - regularization<<"\n";
//    cerr<<"trace param: "<<parameters(0,0)<<"\n";

    for ( size_t xf=0; xf<expectations.n_rows; xf++ )
      for ( size_t y=0; y<expectations.n_cols; y++ )
        if ( expectations(xf,y) < -10000.0 || expectations(xf,y) > 10000.0 ) cerr << "expectations(" << xf << "," << y << ") = " << expectations(xf,y) <<endl;;

    return -totlogprob + regularization;
  }

  void Gradient( const arma::mat& parameters, uint i, arma::mat& gradient )  {

    // Regularization term.
    arma::mat regularization = lambda * parameters;

//    cerr<<"inGrad\n";

/*
    const arma::mat scoredistrs = arma::exp( parameters.t() * predictors );                       // Y*D
    const arma::mat norms       = arma::ones<rowvec>(parameters.n_cols) * scoredistrs;            // 1*D
    const arma::mat predictions = scoredistrs / (arma::ones<vec>(parameters.n_cols) * norms);     // Y*D
    gradient                    = - predictors * (responses - predictions).t() + regularization;  // FX*Y

    for( size_t i=0; i<predictions.n_rows; i++ )
      for( size_t j=0; j<predictions.n_cols; j++ )
        if( predictions(i,j)<-1.0 || predictions(i,j)>1.0 ) cerr << "predictions(" << i << "," << j << ") = " << predictions(i,j) << endl;
*/

    gradient = - ( cooccurrences - expectations ) + regularization;
//    cerr<<"trace param gradient: "<<gradient(0,0)<<"\n";
  }

  const arma::mat& GetInitialPoint ( ) const { return initialpoint; }
};

////////////////////////////////////////////////////////////////////////////////

int main ( int nArgs, char* argv[] ) {

  uint maxiters = nArgs>3 ? atoi(argv[3]) : 0;
  cerr << "Max iters (0 = no bound): " << maxiters << "\n";

  list<pair<DelimitedList<psX,DelimitedPair<psX,Delimited<XFeat>,psEquals,Delimited<double>,psX>,psComma,psX>,Delimited<YVal>>> lplpfdy;

  // Read data...
  cerr << "Reading data...\n";
  int numfeattokens = 0;
  while ( cin && EOF!=cin.peek() ) {
    auto& plpfdy = *lplpfdy.emplace(lplpfdy.end());
    cin >> plpfdy.first >> " : " >> plpfdy.second >> "\n";
    numfeattokens += plpfdy.first.size();
  }
  cerr << "Data read: x=" << domXFeat.getSize() << " y=" << domYVal.getSize() << ".\n";

  // Populate predictor matrix and result vector...
  SpMatLogisticRegressionFunction f ( domXFeat.getSize(), domYVal.getSize(), nArgs>2 ? atof(argv[2]) : 0.0, nArgs>1 ? atoi(argv[1]) : 1 );
  sp_mat& FXbyD = f.Predictors();
  sp_mat& YbyD  = f.Responses();
  umat xlocs ( 2, numfeattokens );
  vec  xvals (    numfeattokens );
  umat ylocs ( 2, lplpfdy.size() );
  vec  yvals (    lplpfdy.size() );
  int t = 0; int i = 0;
  for ( auto& plpfdy : lplpfdy ) {
    for ( auto& pfd : plpfdy.first ) {
      xlocs(0,i) = pfd.first.toInt();
      xlocs(1,i) = t;
      xvals(i)   = pfd.second;
      i++;
    }
    ylocs(0,t) = plpfdy.second.toInt();
    ylocs(1,t) = t;
    yvals(t) = 1.0;
    t++;
  }
  FXbyD = sp_mat ( true, xlocs, xvals, domXFeat.getSize(), lplpfdy.size() );
  YbyD  = sp_mat ( true, ylocs, yvals, domYVal.getSize(),  lplpfdy.size() );
  cerr<<"populated.\n";

  // Regress and print params...
  auto o = SGD<SpMatLogisticRegressionFunction> ( f, 0.01, maxiters );
  auto W = f.GetInitialPoint();
//  o.Optimize(W);
  arma::mat gradient;
  for( uint i=0; i<maxiters; i++ ) {
    cerr << "Iteration " << i << "..." << endl;
    f.Evaluate(W,i);
    f.Gradient(W,i,gradient);
//    for ( size_t xf=0; xf<gradient.n_rows; xf++ )
//      for ( size_t y=0; y<gradient.n_cols; y++ )
//        if ( gradient(xf,y) < -10000.0 || gradient(xf,y) > 10000.0 ) cerr << "gradient(" << xf << "," << y << ") = " << gradient(xf,y) <<endl;;
    W -= /*0.01 * */   gradient * 0.00005 ;    /*0.000005*/
//    for ( size_t xf=0; xf<W.n_rows; xf++ )
//      for ( size_t y=0; y<W.n_cols; y++ )
//        if ( W(xf,y) < -10000.0 || W(xf,y) > 10000.0 ) cerr << "W(" << xf << "," << y << ") = " << W(xf,y) <<endl;;
  }
  cerr<<"done.\n";

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

