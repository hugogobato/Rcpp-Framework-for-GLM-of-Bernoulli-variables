#include <iostream>
#include <cmath>
#include <Rcpp.h>
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <vector>
#include <algorithm>


using namespace std;
using namespace Eigen;



// Vectorized initialization of eta
void initiating_eta(VectorXd& eta, const VectorXd& y_initial) {
    eta = (y_initial.array().log() - (1-y_initial.array()).log()).matrix(); 
}

// Vectorized computation of mu
void estimating_mu(VectorXd& mu, VectorXd& eta) {
    mu = (eta.array().exp() / (1+eta.array().exp())).matrix();  
}

// Vectorized computation of w’s diagonal elements
void estimating_w(DiagonalMatrix<double, Dynamic>& w, VectorXd& mu) {
    w.diagonal() = (mu.array()*(1-mu.array())).matrix();  
}

// Vectorized computation of z
void estimating_z(VectorXd& z,VectorXd& mu, const VectorXd y) {
    z = ((mu.array().log() - (1-mu.array()).log())+(y.array()-mu.array())/(mu.array()*(1-mu.array()))).matrix();  
}

void updating_beta(DiagonalMatrix<double, Dynamic>& w, const MatrixXd& x, VectorXd& z, VectorXd& beta_vector){
  beta_vector = (x.transpose() * w * x).ldlt().solve(x.transpose() * w * z);

}

void updating_eta(VectorXd& eta, VectorXd& beta_vector, const MatrixXd& x){
    eta =(x*beta_vector);

}

// [[Rcpp::export]]
Rcpp::NumericVector beta_estimation(const Eigen::Map<Eigen::VectorXd>& y, const Eigen::Map<Eigen::MatrixXd>& x,int number_iterations = 10) {
    const int size_list = y.size();
    int beta_size=x.cols();
     // Initialize y_initial with modified values directly
    VectorXd y_initial = y.unaryExpr([](double val) { return val == 1 ? 0.95 : 0.05; });
    VectorXd mu(size_list);
VectorXd eta(size_list);  // Use VectorXd for eta to leverage Eigen’s vectorization
DiagonalMatrix<double, Dynamic> w(size_list);   // Initialize a zero matrix 
VectorXd z(size_list); 
VectorXd beta_vector(beta_size); 
 // Run the iterations
    initiating_eta(eta, y_initial);
  for (int i = 0; i < number_iterations; ++i) {
        estimating_mu(mu, eta);
        estimating_w(w, mu);
        estimating_z(z, mu, y);
        updating_beta(w, x, z, beta_vector);
        updating_eta(eta, beta_vector, x);
    }

    return Rcpp::wrap(beta_vector);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix create_x_sequence(const Eigen::Map<Eigen::MatrixXd>& x, int length_out = 100) {
    int num_cols = x.cols();
    MatrixXd x_sequence(length_out, num_cols);

    // Generate a sequence for each column
    for (int col = 0; col < num_cols; ++col) {
        double min_x = x.col(col).minCoeff();
        double max_x = x.col(col).maxCoeff();
        double step = (max_x - min_x) / (length_out - 1);

        // Fill the column in x_sequence with the generated sequence
        for (int i = 0; i < length_out; ++i) {
            x_sequence(i, col) = min_x + i * step;
        }
    }

    return Rcpp::wrap(x_sequence);
}

// [[Rcpp::export]]
Rcpp::NumericVector predicted_y(const Eigen::Map<Eigen::MatrixXd>& x, const Eigen::Map<Eigen::VectorXd>& beta_vector) {
    // Ensure x has 2 columns as expected by create_x_sequence
    if (x.cols() != 2) {
        Rcpp::stop("Input matrix x must have exactly 2 columns.");
    }
    int pred_size=x.rows();
    VectorXd pred_eta(pred_size);
    pred_eta = x*beta_vector;
    VectorXd pred_mu(pred_size);
    pred_mu = (pred_eta.array().exp() / (1+pred_eta.array().exp())).matrix();
    return Rcpp::wrap(pred_mu);
}

double log_likelihood(const VectorXd& y, const MatrixXd& x, VectorXd& beta_vector){
    double log_likelihood = (y.array()*(x*beta_vector).array()-(1+((x*beta_vector).array()).exp()).log()).sum();
    return log_likelihood;
}

VectorXd mu_estimation(const Eigen::Map<Eigen::VectorXd>& y, const Eigen::Map<Eigen::MatrixXd>& x,int number_iterations = 10) {
    const int size_list = y.size();
    int beta_size=x.cols();
     // Initialize y_initial with modified values directly
    VectorXd y_initial = y.unaryExpr([](double val) { return val == 1 ? 0.95 : 0.05; });
    VectorXd mu(size_list);
VectorXd eta(size_list);  // Use VectorXd for eta to leverage Eigen’s vectorization
DiagonalMatrix<double, Dynamic> w(size_list);   // Initialize a zero matrix 
VectorXd z(size_list); 
VectorXd beta_vector(beta_size); 
 // Run the iterations
    initiating_eta(eta, y_initial);
  for (int i = 0; i < number_iterations; ++i) {
        estimating_mu(mu, eta);
        estimating_w(w, mu);
        estimating_z(z, mu, y);
        updating_beta(w, x, z, beta_vector);
        updating_eta(eta, beta_vector, x);
    }

    // Print the result
    // cout << "Phi value: " << phi << endl;

    return mu;

}

VectorXd U_b(const Eigen::Map<Eigen::VectorXd>& y, const Eigen::Map<Eigen::MatrixXd>& x, VectorXd& beta_vector,int number_iterations = 10){
    VectorXd mu = mu_estimation(y,x,number_iterations);
    // cout << "Mu:\n" << mu<< std::endl;
   VectorXd U_b =x.transpose()*(y-mu);
   return U_b;
}

DiagonalMatrix<double, Dynamic> W_estimation(const Eigen::Map<Eigen::VectorXd>& y, const Eigen::Map<Eigen::MatrixXd>& x,int number_iterations = 10) {
    const int size_list = y.size();
    int beta_size=x.cols();
     // Initialize y_initial with modified values directly
    VectorXd y_initial = y.unaryExpr([](double val) { return val == 1 ? 0.95 : 0.05; });
    VectorXd mu(size_list);
VectorXd eta(size_list);  // Use VectorXd for eta to leverage Eigen’s vectorization
DiagonalMatrix<double, Dynamic> w(size_list);   // Initialize a zero matrix 
VectorXd z(size_list); 
VectorXd beta_vector(beta_size); 
 // Run the iterations
    initiating_eta(eta, y_initial);
  for (int i = 0; i < number_iterations; ++i) {
        estimating_mu(mu, eta);
        estimating_w(w, mu);
        estimating_z(z, mu, y);
        updating_beta(w, x, z, beta_vector);
        updating_eta(eta, beta_vector, x);
    }



    // Print the result
    // cout << "Phi value: " << phi << endl;

    return w;

}

MatrixXd K_bb_matrix(const Eigen::Map<Eigen::VectorXd>& y, const Eigen::Map<Eigen::MatrixXd>& x, VectorXd& beta_vector,int number_iterations = 10){ 
    MatrixXd W = W_estimation(y,x,number_iterations);
    MatrixXd K_bb = x.transpose()*W*x;
    return K_bb;
} 

MatrixXd K_bb_matrix_inverse(const Eigen::Map<Eigen::VectorXd>& y, const Eigen::Map<Eigen::MatrixXd>& x, VectorXd& beta_vector,int number_iterations = 10){
    MatrixXd K_bb_inverse = K_bb_matrix(y,x,beta_vector,number_iterations).inverse();
    return K_bb_inverse;
} 


VectorXd beta_estimation_original(const Eigen::Map<Eigen::VectorXd>& y,const Eigen::Map<Eigen::MatrixXd>& x,int number_iterations = 10){
    Rcpp::NumericVector beta_rcpp = beta_estimation(y,x,number_iterations);
    Eigen::VectorXd beta_vector = Eigen::Map<Eigen::VectorXd>(beta_rcpp.begin(), beta_rcpp.size());
   return beta_vector;
}

// [[Rcpp::export]]
double Q_LR(const Eigen::Map<Eigen::VectorXd>& y, const Eigen::Map<Eigen::MatrixXd>& x_original, const Eigen::Map<Eigen::MatrixXd>& x_test,const int number_iterations = 10){
    VectorXd beta_vector= beta_estimation_original(y,x_original,number_iterations);
    VectorXd beta_vector_test= beta_estimation_original(y,x_test,number_iterations);
    double Q_LR = 2*(log_likelihood(y,x_original,beta_vector)-log_likelihood(y,x_test,beta_vector_test));
    return Q_LR;
}

// [[Rcpp::export]]
double Q_SR(const Eigen::Map<Eigen::VectorXd>& y, const Eigen::Map<Eigen::MatrixXd>& x_test,const int number_iterations = 10){
    VectorXd beta_vector_test= beta_estimation_original(y,x_test,number_iterations);
    VectorXd U_b_test = U_b(y,x_test,beta_vector_test,number_iterations);
    // cout << "U_b:\n" << U_b_test << std::endl;
    double Q_SR= U_b_test(1) * K_bb_matrix_inverse(y, x_test,beta_vector_test,number_iterations)(1,1)* U_b_test(1);
    return Q_SR;
}

// [[Rcpp::export]]
double Q_W(const Eigen::Map<Eigen::VectorXd>& y, const Eigen::Map<Eigen::MatrixXd>& x_original, const int tested_x,const int number_iterations = 10){
    VectorXd beta_vector= beta_estimation_original(y,x_original,number_iterations);
    double Q_W = (beta_vector(tested_x-1)-0) *(beta_vector(tested_x-1)-0)/K_bb_matrix_inverse(y, x_original,beta_vector,number_iterations)(tested_x-1,tested_x-1);
    //cout << "Beta vector:\n" << beta_vector(1) << std::endl;
    //cout << "Var beta vector:\n" << K_bb_matrix_inverse(y, x_original,beta_vector)(1,1) << std::endl;
    return Q_W;
}

// [[Rcpp::export]]
double Q_G(const Eigen::Map<Eigen::VectorXd>& y, const Eigen::Map<Eigen::MatrixXd>& x_test, const Eigen::Map<Eigen::MatrixXd>& x_original,const int number_iterations = 10){
    VectorXd beta_vector= beta_estimation_original(y,x_original,number_iterations);
    VectorXd beta_vector_test= beta_estimation_original(y,x_test,number_iterations);
    VectorXd U_b_test = U_b(y,x_test,beta_vector_test,number_iterations);
    double Q_G = U_b_test(1)*(beta_vector(1)-0);
    return Q_G;
}
