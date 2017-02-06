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
#define DONT_USE_UNMAPPABLE_TUPLES
#include <nl-randvar.h>
#include <nl-string.h>
#include <Delimited.hpp>

////////////////////////////////////////////////////////////////////////////////

DiscreteDomain<int> domFeat;
class Feat : public DiscreteDomainRV<int,domFeat> {
 public:
  Feat ( )                : DiscreteDomainRV<int,domFeat> ( )    { }
  Feat ( int i )          : DiscreteDomainRV<int,domFeat> ( i )  { }
  Feat ( const char* ps ) : DiscreteDomainRV<int,domFeat> ( ps ) { }
};

////////////////////////////////////////////////////////////////////////////////

class SpMatLogisticRegressionFunction {  //: public LogisticRegressionFunction {

 private:

  double       lambda = 0.0;
  arma::sp_mat predictors;
  arma::vec    responses;
  arma::vec    initialpoint;

 public:

  SpMatLogisticRegressionFunction ( double l = 0.0 ) : lambda(l) {
    initialpoint.randn ( domFeat.getSize()+1 );
    cerr << "L2 regularization parameter: " << lambda << "\n";
  }

  arma::sp_mat&    Predictors ( )        { return predictors; }
  arma::vec&       Responses  ( )        { return responses;  }
  const arma::vec& Responses  ( ) const  { return responses;  }

  double Evaluate(const arma::mat& parameters) const {
    // The objective function is the log-likelihood function (w is the parameters
    // vector for the model; y is the responses; x is the predictors; sig() is the
    // sigmoid function):
    //   f(w) = sum(y log(sig(w'x)) + (1 - y) log(sig(1 - w'x))).
    // We want to minimize this function.  L2-regularization is just lambda
    // multiplied by the squared l2-norm of the parameters then divided by two.

    // For the regularization, we ignore the first term, which is the intercept
    // term.
    const double regularization = 0.5 * lambda *
        arma::dot(parameters.col(0).subvec(1, parameters.n_elem - 1),
                  parameters.col(0).subvec(1, parameters.n_elem - 1));

    // Calculate vectors of sigmoids.  The intercept term is parameters(0, 0) and
    // does not need to be multiplied by any of the predictors.
    const arma::vec exponents = parameters(0, 0) + predictors.t() *
        parameters.col(0).subvec(1, parameters.n_elem - 1);
    const arma::vec sigmoid = 1.0 / (1.0 + arma::exp(-exponents));

    // Assemble full objective function.  Often the objective function and the
    // regularization as given are divided by the number of features, but this
    // doesn't actually affect the optimization result, so we'll just ignore those
    // terms for computational efficiency.
    double result = 0.0;
    for (size_t i = 0; i < responses.n_elem; ++i)
    {
      if (responses[i] == 1)
        result += log(sigmoid[i]);
      else
        result += log(1.0 - sigmoid[i]);
    }

    cerr<<"logP(data)="<<result<<"\n";

    // Invert the result, because it's a minimization.
    return -result + regularization;

    /*
    // Calculate vectors of sigmoids...
    arma::vec exponents = predictors.t() * parameters;
    arma::vec sigmoids = 1.0 / (1.0 + arma::exp(-exponents));
    // Assemble full objective function...
    double result = 0.0;
    for (size_t i = 0; i < responses.n_elem; ++i) {
      if (responses[i] > 0.5) result += log(sigmoids[i]);
      else                    result += log(1.0 - sigmoids[i]);
    }

    cerr<<"logP(data)="<<result<<"\n";

    // Invert the result, because it's a minimization...
    return -result;
    */
  }

//  double Evaluate(const arma::mat& parameters, const size_t i) const {
//  }

  void Gradient(const arma::mat& parameters, arma::mat& gradient) const {
    // Regularization term.
    arma::mat regularization;
    regularization = lambda * parameters.col(0).subvec(1, parameters.n_elem - 1);

    const arma::vec sigmoids = 1 / (1 + arma::exp(-parameters(0, 0)
        - predictors.t() * parameters.col(0).subvec(1, parameters.n_elem - 1)));

    gradient.set_size(parameters.n_elem);
    gradient[0] = -arma::accu(responses - sigmoids);
    gradient.col(0).subvec(1, parameters.n_elem - 1) = -predictors * (responses -
        sigmoids) + regularization;

    /*
    // Calculate vectors of sigmoids...
    arma::vec exponents = predictors.t() * parameters;
    arma::vec sigmoids = 1.0 / (1.0 + arma::exp(-exponents));
    // Generate diagonal sparse matrix...
    static umat locs ( 2, responses.n_elem );
    static vec  vals ( responses.n_elem );
    for ( size_t i=0; i<responses.n_elem; i++ ) {
      locs(0,i) = i;
      locs(1,i) = i;
      vals(i) = responses(i) - sigmoids(i);
    }
    arma::sp_mat RminS ( locs, vals, responses.n_elem, responses.n_elem );
    gradient = - (predictors1 * RminS * predictors2.t() + predictors2 * RminS * predictors1.t()) * parameters;
    //cerr<<"tracked feature -- p:"<<parameters(Feat("B-aN-bA:keep,1").toInt(),2)<<" g:"<<gradient(Feat("B-aN-bA:keep,1").toInt(),2)<<"\n";
    */
  }

//  void Gradient(const arma::mat& parameters,
//                const size_t i,
//                arma::mat& gradient) const {
//  }

  const arma::mat& GetInitialPoint ( ) const { return initialpoint; }
};

////////////////////////////////////////////////////////////////////////////////

int main ( int nArgs, char* argv[] ) {

  int maxiters = nArgs>2 ? atoi(argv[2]) : 0;
  cerr << "Max iters (0 = no bound): " << maxiters << "\n";

  list<pair<DelimitedList<psX,DelimitedPair<psX,Delimited<Feat>,psEquals,Delimited<double>,psX>,psComma,psX>,Delimited<double>>> lplpfdy;

  // Read data...
  cerr << "Reading data...\n";
  int numpreds = 0;
  while ( cin && EOF!=cin.peek() ) {
    auto& plpfdy = *lplpfdy.emplace(lplpfdy.end());
    cin >> plpfdy.first >> " : " >> plpfdy.second >> "\n";
    numpreds += plpfdy.first.size();
  }
  cerr << "Data read.\n";

  // Build data matrices...
  /*
  mat X ( domFeat.getSize(), lplpfdy.size(), fill::zeros );
  vec y ( lplpfdy.size() );
  int t = -1;
  for ( auto& plpfdy : lplpfdy ) {
    t++;
    for ( auto& pfd : plpfdy.first )
      X(pfd.first.toInt(),t) = pfd.second;
    y[t] = plpfdy.second;
  }
  */

  // Populate predictor matrix and result vector...
  SpMatLogisticRegressionFunction f(nArgs>1 ? atof(argv[1]) : 0.0);
  sp_mat& X = f.Predictors();
  vec&    y = f.Responses();
  y = vec ( lplpfdy.size() );
  umat locs ( 2, numpreds );
  vec  vals ( numpreds );
  int t = 0; int i = 0;
  for ( auto& plpfdy : lplpfdy ) {
    for ( auto& pfd : plpfdy.first ) {
      locs(0,i) = pfd.first.toInt();
      locs(1,i) = t;
      vals(i)   = pfd.second;
      i++;
    }
    y(t) = plpfdy.second;
    t++;
  }
  X = sp_mat ( locs, vals, domFeat.getSize(), lplpfdy.size() );
  cerr<<"populated.\n";

  // Regress and print params...
  auto o = L_BFGS<SpMatLogisticRegressionFunction>(f,5,maxiters);
  auto w = f.GetInitialPoint();
  if ( maxiters>-1 ) o.Optimize(w);
  cerr<<"done.\n";

//  // Regress and print params...
//  auto& w = LogisticRegression<>(X,y).Parameters();
  cout << "(const) = " << w(0) << "\n";
  for ( size_t i=1; i<w.size(); i++ ) {
    cout << Feat::getDomain().getString(i-1) << " = " << w(i) << "\n";
  }
}

