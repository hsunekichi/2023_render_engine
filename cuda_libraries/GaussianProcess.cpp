#pragma once

#include <iostream>
#include "types.h"
#include <Eigen/Dense>

using MatrixXF = Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXF = Eigen::Matrix<Float, Eigen::Dynamic, 1>;

class GaussianProcess2D {
public:
    GaussianProcess2D(Float noise_variance = 0.1) : noise_variance_(noise_variance) {}

    // Function to train the GP with training data
    void train(const MatrixXF& X_train, const VectorXF& y_train);

    // Function to predict mean and variance at a given test point
    void predict(const VectorXF& X_test, Float& mean, Float& variance) const;

private:
    MatrixXF X_train_;          // Training input data (matrix with samples as rows and features as columns)
    VectorXF y_train_;          // Training output data (vector)
    Float noise_variance_;     // Noise variance parameter
    MatrixXF K_inv_;            // Inverse of the kernel matrix
};

void GaussianProcess2D::train(const MatrixXF& X_train, const VectorXF& y_train) {
    X_train_ = X_train;            // Store the training input data
    y_train_ = y_train;            // Store the training output data

    int n = X_train.rows();
    MatrixXF K = MatrixXF(n, n);  // Initialize the kernel matrix
    for (int i = 0; i < n; ++i) 
    {
        for (int j = 0; j < n; ++j) 
        {
            Float dist = (X_train.row(i) - X_train.row(j)).norm(); // Calculate the Euclidean distance in 2D
            K(i, j) = exp(-0.5 * dist * dist); // Compute the kernel value
        }
    }

    K.array() += noise_variance_ * noise_variance_ * MatrixXF::Identity(n, n).array();  // Add noise

    // Calculate the inverse of the kernel matrix
    K_inv_ = K.inverse();
}

void GaussianProcess2D::predict(const VectorXF& X_test, Float& mean, Float& variance) const 
{
    int n = X_train_.rows();

    // Calculate the kernel between test and training data
    VectorXF k(n);
    for (int j = 0; j < n; ++j) 
    {
        Float dist = (X_test.transpose() - X_train_.row(j)).norm(); // Calculate the Euclidean distance in 2D
        k(j) = exp(-0.5 * dist * dist); // Compute the kernel value
    }

    // Calculate the predictive mean
    mean = k.transpose() * K_inv_ * y_train_; // Mean of the predictive distribution

    // Calculate the kernel for the test point
    Float k_ss = 1.0; // Kernel value for the test point

    // Calculate the predictive variance
    variance = k_ss + noise_variance_ - k.transpose() * K_inv_ * k; // Variance of the predictive distribution
}