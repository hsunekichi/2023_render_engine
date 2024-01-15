#pragma once

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <vector>
#include <random>

#include "../cuda_libraries/types.h"
#include "../radiometry/Spectrum.cu"
#include "../light/Photon.cu"

struct DenoiserSample
{
    Point2f surfacePoint;
    Spectrum indirectLight;
    long long sampleId;
    Spectrum lightFactor;
};

class PolynomialRegression 
{
    public:

    std::vector<Float> coefficientsX;
    std::vector<Float> coefficientsY;
    Float baseCoefficient;

    unsigned int degree;

    PolynomialRegression(int degree=5) 
    {
        this->degree = degree;
        coefficientsX.resize(degree, 0);
        coefficientsY.resize(degree, 0);


        // Create c++ rand generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<Float> dis(-1.0, 1.0);

        baseCoefficient = dis(gen);

        // Initialize coefficients with random values
        for (int i = 0; i < coefficientsX.size(); i++)
        {
            coefficientsX[i] = dis(gen);
            coefficientsY[i] = dis(gen);
        }

        //baseCoefficient = 0.5;
        //coefficientsX[0] = 4;
        //coefficientsX[1] = -4;

        //coefficientsY[0] = coefficientsX[0];
        //coefficientsY[1] = coefficientsX[1];
        //coefficientsY[0] = 0;
        //coefficientsY[1] = 0;

    }


    // Calculate the Root Mean Squared Error (RMSE) for a given set of x and y values
    Float calculateRMSE(
            const std::vector<std::vector<Float>>& x, 
            const std::vector<Float>& y) 
    {
        std::vector<Float> predictions = predict(x);
        size_t num_samples = x.size();
        Float sum_squared_errors = 0.0;

        for (size_t i = 0; i < num_samples; i++) 
        {
            Float error = predictions[i] - y[i];
            sum_squared_errors += error * error;
        }

        Float rmse = sqrt(sum_squared_errors / num_samples);
        return rmse;
    }


    // Train the model using gradient descent and RMSE loss
    void train(const std::vector<std::vector<Float>>& x, 
            const std::vector<Float>& y, 
            Float learning_rate, 
            int num_iterations) 
    {
        size_t num_samples = x.size();

        if (num_samples == 0)
        {
            for (int i = 0; i < coefficientsX.size(); i++)
            {
                coefficientsX[i] = 0.0;
                coefficientsY[i] = 0.0;
            }

            baseCoefficient = 0.0;

            return;
        }

        for (int iter = 0; iter < num_iterations; iter++) 
        {
            std::vector<Float> predictions = predict(x);
            std::vector<Float> distances(num_samples);

            // Calculate errors
            for (size_t i = 0; i < num_samples; i++) 
            {
                //std::cout << "Prediction: " << predictions[i] << " - Real: " << y[i] << std::endl;
                distances[i] = predictions[i] - y[i];
            }

            std::vector<Float> gradientsX(coefficientsX.size(), 0.0);

            // Update coefficients using gradient descent
            for (int i = 0; i < coefficientsX.size(); i++) 
            {
                for (int sample = 0; sample < num_samples; sample++) 
                {
                    Float distance = distances[sample];
                    Float power = pow(x[sample][0], i+1);
                    Float gradient = distance * power;

                    gradientsX[i] += gradient;
                }
            }

            for (int i = 0; i < coefficientsX.size(); i++) 
            {
                coefficientsX[i] -= learning_rate * (gradientsX[i]/num_samples);
            }

            std::vector<Float> gradientsY(coefficientsY.size(), 0.0);

            // Update coefficients using gradient descent
            for (int i = 0; i < coefficientsY.size(); i++) 
            {
                for (int sample = 0; sample < num_samples; sample++) 
                {
                    Float gradient = distances[sample] * pow(x[sample][1], i+1);
                    gradientsY[i] += gradient;
                }
            }

            for (int i = 0; i < coefficientsY.size(); i++) 
            {
                coefficientsY[i] -= learning_rate * (gradientsY[i]/num_samples);
            }

            // Update base coefficient
            Float gradient = 0.0;

            for (int sample = 0; sample < num_samples; sample++) 
            {
                gradient += distances[sample];
            }

            baseCoefficient -= learning_rate * (gradient/num_samples);

            // Print RMSE every 100 iterations
            if (iter % 200 == 0) 
            {
                Float rmse = calculateRMSE(x, y);
                //std::cout << "Iteration: " << iter << " RMSE: " << rmse << std::endl;
            }
        }
    }

    // Calculate the predicted y values for a given set of x values
    std::vector<Float> predict(const std::vector<std::vector<Float>>& x) const
    {
        std::vector<Float> predictions(x.size(), 0.0);

        for (int i = 0; i < coefficientsX.size(); i++) 
        {
            for (size_t sample = 0; sample < x.size(); sample++) 
            {
                Float x_value = x[sample][0];
                Float power = pow(x_value, i+1);
                Float coefficient = coefficientsX[i];

                predictions[sample] += coefficient * power;
            }
        }

        for (int i = 0; i < coefficientsY.size(); i++) 
        {
            for (size_t sample = 0; sample < x.size(); sample++) 
            {
                predictions[sample] += coefficientsY[i] * pow(x[sample][1], i+1); // Assuming x is a vector of vectors
            }
        }

        for (size_t sample = 0; sample < x.size(); sample++) 
        {
            predictions[sample] += baseCoefficient;
        }

        return predictions;
    }

    // Operator <<
    friend std::ostream& operator<<(std::ostream& os, const PolynomialRegression& model) 
    {
        os << "f(x) = ";

        os << model.baseCoefficient;

        for (int coef = 0; coef < model.coefficientsX.size(); coef++) 
        {
            if (model.coefficientsX[coef] > 0)
                os << " + " << model.coefficientsX[coef] << "x^" << coef+1;
            else
                os << " - " << -model.coefficientsX[coef] << "x^" << coef+1;
            
            if (model.coefficientsY[coef] > 0)
                os << " + " << model.coefficientsY[coef] << "y^" << coef+1;
            else
                os << " - " << -model.coefficientsY[coef] << "y^" << coef+1;
        }


        return os;
    }
};



class ShapeLightDistribution
{
    std::vector<Float> coefficients;

    public:

    PolynomialRegression modelR, modelG, modelB;

    ShapeLightDistribution() 
    : modelR(), modelG(), modelB()
    {}

    void train(std::vector<std::vector<Float>>& x, 
                std::vector<Spectrum>& y)
    {
        if (x.size() > 0)
        {
            // Get single channel
            std::vector<Float> y_R(y.size());
            std::vector<Float> y_G(y.size());
            std::vector<Float> y_B(y.size());

            for (int i = 0; i < y.size(); i++)
            {
                y_R[i] = y[i].getR();
                y_G[i] = y[i].getG();
                y_B[i] = y[i].getB();
            }

            modelR.train(x, y_R, 0.3, 500);
            modelG.train(x, y_G, 0.3, 500);
            modelB.train(x, y_B, 0.3, 500);
        }
        else
        {
            // Initialize zero linear regresion
            modelR = PolynomialRegression();
            modelG = PolynomialRegression();
            modelB = PolynomialRegression();
        }
    }

    void train(thrust::host_vector<Photon> photons)
    {
        std::vector<std::vector<Float>> X_data(photons.size());
        std::vector<Spectrum> y_data(photons.size());

        for (int i = 0; i < photons.size(); i++)
        {
            X_data[i] = {photons[i].surfacePoint.x, photons[i].surfacePoint.y};
            y_data[i] = photons[i].radiance;
        }

        train(X_data, y_data);
    
        std::cout << modelR << std::endl;
    }

    thrust::host_vector<Spectrum> predict(std::vector<std::vector<Float>>& X_data) const
    {
        std::vector<Float> y_predR = modelR.predict(X_data);
        std::vector<Float> y_predG = modelG.predict(X_data);
        std::vector<Float> y_predB = modelB.predict(X_data);

        thrust::host_vector<Spectrum> y_pred_spectrum(y_predR.size());

        for (int i = 0; i < y_predR.size(); i++)
        {
            y_pred_spectrum[i] = Spectrum(y_predR[i], y_predG[i], y_predB[i]);
            //std::cout << "Point: " << X_data[i][0] << ", " << X_data[i][1] << " - " << y_predR[i] << std::endl;
        }

        return y_pred_spectrum;
    }

    thrust::host_vector<Spectrum> predict(thrust::host_vector<Point2f> points) const
    {
        std::vector<std::vector<Float>> X_data(points.size());

        for (int i = 0; i < points.size(); i++)
        {
            X_data[i] = {points[i].x, points[i].y};
        }

        return predict(X_data);
    }

    Spectrum predict(Point2f point)
    {
        std::vector<std::vector<Float>> X_data_vector = {{point.x, point.y}};
        thrust::host_vector<Spectrum> y_pred = predict(X_data_vector);
        return y_pred[0];
    }

    void predict(std::vector<DenoiserSample> &Xdata) const
    {
        std::vector<std::vector<Float>> surfacePoints(Xdata.size());

        for (int i = 0; i < Xdata.size(); i++)
        {
            surfacePoints[i] = {Xdata[i].surfacePoint.x, Xdata[i].surfacePoint.y};
        }

        std::cout << "Predicting indirect light" << std::endl;
        std::vector<Float> y_predR = modelR.predict(surfacePoints);
        std::vector<Float> y_predG = modelG.predict(surfacePoints);
        std::vector<Float> y_predB = modelB.predict(surfacePoints);

        for (int i = 0; i < Xdata.size(); i++)
        {
            Xdata[i].indirectLight = Spectrum(y_predR[i], y_predG[i], y_predB[i]);
        }
    }

};