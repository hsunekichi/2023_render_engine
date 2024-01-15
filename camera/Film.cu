#pragma once

#include <fstream>
//#include "../cuda_libraries/GaussianProcess.cpp"

#include "../cuda_libraries/types.h"
#include "../radiometry/Spectrum.cu"
#include "../geometry/Point.cu"
#include "../cuda_libraries/geometricMath.cu"
#include "BMP_Format.cu"
#include "../cuda_libraries/Matrix3x3.cu"

enum ToneMappingType
{
    equalize,
    reinhard,
    reinhardJodie,
    ACESfilmic,
    uncharted2,
    gammaClamp
};

class Film
{
    protected:
    Spectrum *pixelLight = nullptr;

    uint2 current_resolution;
    long long samplesLoadedFromFile = 0;
    long long samplesComputedPixel0 = 0;

    public:

    Film () 
    {
        current_resolution = {resolution.x, resolution.y};
        pixelLight = new Spectrum[current_resolution.x * current_resolution.y];
    }

    //Pre: True
    //Post: Copia el objeto al completo
    Film(Film& original){

        //Calculamos el tamanyo de la nueva imagen
        uint32_t image_size = original.current_resolution.x * original.current_resolution.y;

        //Dedicamos memoria para la misma
        this->pixelLight = new Spectrum[image_size];

        //Para cada componente de la imagen original
        for(uint32_t i = 0; i < image_size; i++){

            //La copiamos a la nueva imagen
            this->pixelLight[i] = original.pixelLight[i];
        }

        //Copiamos el resto de componentes
        this->current_resolution = original.current_resolution;
        this->samplesLoadedFromFile = original.samplesLoadedFromFile;
        this->samplesComputedPixel0 = original.samplesComputedPixel0;
    }

    //Pre: True
    //Post: Mueve el objeto de un lado a otro
    Film (Film&& original){
        //Movemos el puntero
        this->pixelLight = original.pixelLight;

        //Anulamos el original para que C++ no libere la memoria
        original.pixelLight = nullptr;

        //Copiamos el resto de componentes
        this->current_resolution = original.current_resolution;
        this->samplesLoadedFromFile = original.samplesLoadedFromFile;
        this->samplesComputedPixel0 = original.samplesComputedPixel0;
    }

    ~Film()
    { 
        delete[] pixelLight;
    }

    Film* toGPU()
    {
        Spectrum *gpu_directLight;

        cudaMalloc(&gpu_directLight, current_resolution.x * current_resolution.y * sizeof(Spectrum));
        cudaMemcpy(gpu_directLight, pixelLight, current_resolution.x * current_resolution.y * sizeof(Spectrum), cudaMemcpyHostToDevice);


        Film newFilm;
        Film *gpuFilm;
        
        Spectrum *tempDirect = newFilm.pixelLight;
        newFilm.pixelLight = gpu_directLight;

        cudaMalloc(&gpuFilm, sizeof(Film));
        cudaMemcpy(gpuFilm, &newFilm, sizeof(Film), cudaMemcpyHostToDevice);

        newFilm.pixelLight = tempDirect;

        return gpuFilm;
    }

    void copyPixelsFromGPU(Spectrum *gpu_pixels)
    {
        //cudaMemcpy(pixels, gpu_pixels, current_resolution.x * current_resolution.y * sizeof(Spectrum), cudaMemcpyDeviceToHost);
    }


    __host__ 
    void addSample(Point2i pixel, Spectrum radiance,  int nSamples)
    {
        if (pixel.x == 0 && pixel.y == 0)
            samplesComputedPixel0 += nSamples;

        pixelLight[pixel.x + pixel.y * current_resolution.x] += radiance;
    }

    double getMaxChannel()
    {
        // Get max value between all r g and b values
        double max = 0;
        for (int i = 0; i < current_resolution.x * current_resolution.y; i++)
        {
            Spectrum pixel = pixelLight[i];

            if (pixel.getR() > max)
                max = pixel.getR();
            if (pixel.getG() > max)
                max = pixel.getG();
            if (pixel.getB() > max)
                max = pixel.getB();
        }

        return max/(samplesLoadedFromFile + samplesComputedPixel0);
    }

    double getMaxLuminance()
    {
        // Get max luminance value
        double maxL = 0;
        for (int i = 0; i < current_resolution.x * current_resolution.y; i++)
        {
            Spectrum pixel = pixelLight[i];

            double luminance = pixel.brightness();
            if (luminance > maxL)
                maxL = luminance;
        }

        return maxL/(samplesLoadedFromFile + samplesComputedPixel0);
    
    }

    Spectrum equalizeTonemap(Spectrum pixel, double max)
    {
        pixel /= (samplesLoadedFromFile + samplesComputedPixel0);
        pixel = pixel/max;

        return pixel;
    }

    Spectrum reinhardTonemap(Spectrum pixel, double maxL)
    {
        pixel /= (samplesLoadedFromFile + samplesComputedPixel0);

        //return clamp(pixel / (1 + pixel.brightness()), 0, 1);

        if (pixel.brightness() == 0)
            return pixel;
            
        double l_old = pixel.brightness();
        double numerator = l_old * (1.0 + (l_old / (maxL * maxL)));
        double l_new = numerator / (1.0 + l_old);
        Spectrum result = pixel * (l_new / l_old);

        return clamp(result, 0, 1);
    }

    Spectrum reinhardJodieTonemap(Spectrum pixel)
    {
        pixel /= (samplesLoadedFromFile + samplesComputedPixel0);

        double luminance = pixel.brightness();
        Spectrum reinhard_p = pixel/(1 + luminance);

        pixel = lerp(pixel / (1+pixel), reinhard_p, reinhard_p);
        pixel = clamp(pixel, 0, 1);

        return pixel;
    }

    Spectrum uncharted2_tonemap_partial(Spectrum x)
    {
        float A = 0.15;
        float B = 0.50;
        float C = 0.10;
        float D = 0.20;
        float E = 0.02;
        float F = 0.30;
        return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
    }

    Spectrum uncharted2_filmic(Spectrum pixel)
    {
        pixel = pixel/(samplesLoadedFromFile + samplesComputedPixel0);

        float exposure_bias = 2.0;
        Spectrum curr = uncharted2_tonemap_partial(pixel * exposure_bias);

        Spectrum W = Spectrum(11.2);
        Spectrum white_scale = Spectrum(1.0) / uncharted2_tonemap_partial(W);
        return curr * white_scale;
    }

    Spectrum rtt_and_odt_fit(Spectrum pixel)
    {
        Spectrum a = pixel * (pixel + 0.0245786) - 0.000090537;
        Spectrum b = pixel * (0.983729 * pixel + 0.4329510) + 0.238081;
        return a / b;
    }

    Spectrum ACESfilmicTonemap(Spectrum pixel)
    {
        pixel = pixel/(samplesLoadedFromFile + samplesComputedPixel0);

        const Matrix3x3<double> ACESInputMat(0.59719, 0.35458, 0.04823,
                                        0.07600, 0.90834, 0.01566,
                                        0.02840, 0.13383, 0.83777);

        const Matrix3x3<double> ACESOutputMat(1.60475, -0.53108, -0.07367,
                                        -0.10208, 1.10813, -0.00605,
                                        -0.00327, -0.07276, 1.07602);

        Spectrum curr = ACESInputMat * pixel;
        curr = rtt_and_odt_fit(curr);
        curr = ACESOutputMat * curr;

        return clamp(curr, 0, 1);
    }

    Spectrum aces_approx(Spectrum pixel)
    {
        pixel = pixel/(samplesLoadedFromFile + samplesComputedPixel0);

        pixel *= 0.6f;
        float a = 2.51f;
        float b = 0.03f;
        float c = 2.43f;
        float d = 0.59f;
        float e = 0.14f;
        return clamp((pixel*(a*pixel+b))/(pixel*(c*pixel+d)+e), 0.0f, 1.0f);
    }

    Spectrum gammaClampToneMap (Spectrum pixel, double max)
    {
        double gammaVal = 2.2;
        pixel = pixel/(samplesLoadedFromFile + samplesComputedPixel0);

        pixel /= max;
        pixel = pow(pixel, 1.0/gammaVal);
        return pixel;
    }

    Spectrum toneMaxPixel(Spectrum pixel, double max, double maxL, ToneMappingType toneMode)
    {
        if (toneMode == equalize)
            pixel = equalizeTonemap(pixel, max);
        else if (toneMode == reinhard)
            pixel = reinhardTonemap(pixel, maxL);
        else if (toneMode == reinhardJodie)
            pixel = reinhardJodieTonemap(pixel);
        else if (toneMode == ACESfilmic)
            pixel = ACESfilmicTonemap(pixel);
        else if (toneMode == uncharted2)
            pixel = uncharted2_filmic(pixel);
        else if (toneMode == gammaClamp)
            pixel = gammaClampToneMap(pixel, max);

        return pixel;
    }
    

    void toneMapPPM(std::ofstream &file, ToneMappingType toneMode)
    {
        double max = 0;
        max = getMaxChannel();

        double maxL = 0;
        maxL = getMaxLuminance();
    
        
        // Write each pixel to file
        for (int i = 0; i < current_resolution.x * current_resolution.y; i++)
        {
            Spectrum pixel = pixelLight[i];

            pixel = toneMaxPixel(pixel, max, maxL, toneMode);

            file << (unsigned char)(pixel.getR() * 255) 
            << (unsigned char)(pixel.getG() * 255) 
            << (unsigned char)(pixel.getB() * 255);
        }
    }

    void toneMapBMP(std::ofstream &file, ToneMappingType toneMode)
    {
        BitmapImage imagen(current_resolution.x, current_resolution.y);

        double max = 0;
        max = getMaxChannel();

        double maxL = 0;
        maxL = getMaxLuminance();
        

        // Write each pixel to file
        for(int j = 0; j < current_resolution.y; j++)
        {
            int offset = (current_resolution.y - 1 - j)*current_resolution.x;
            
            for(int i = 0; i < current_resolution.x; i++)
            {
                Spectrum pixel = pixelLight[i + offset];
                
                pixel = toneMaxPixel(pixel, max, maxL, toneMode);

                imagen.image_bytes[3*(i + j*current_resolution.x)] = (unsigned char)(pixel.getB() * 255);
                imagen.image_bytes[3*(i + j*current_resolution.x) + 1] = (unsigned char)(pixel.getG() * 255);
                imagen.image_bytes[3*(i + j*current_resolution.x) + 2] = (unsigned char)(pixel.getR() * 255);
            }
        }

        imagen.write_image(file);
    }


    void writeToPPM(std::string filename, ToneMappingType toneMode)
    {
        filename += ".ppm";
        std::ofstream file;
        file.open(filename, std::ios::binary);

        file << "P6\n" << current_resolution.x << " " << current_resolution.y << "\n255\n";
        
        toneMapPPM(file, toneMode);
        
        file.close();
    }


    void writeToBMP(std::string filename, ToneMappingType toneMode)
    {
        filename += ".bmp";
        std::ofstream file;
        file.open(filename, std::ios::binary);

        toneMapBMP(file, toneMode);
        
        file.close();
    }



    void storeContributionsToFile(std::string filename)
    {
        filename += ".hdrbmp";

        std::ofstream file;
        file.open(filename, std::ios::binary);

        file << "P6\n" << current_resolution.x << " " << current_resolution.y;
        
        // Data will be stored as d -> double, f -> float or F -> generic Float
        // Stores the number of samples each pixel has recieved
        file << "F\n" << samplesLoadedFromFile + samplesComputedPixel0 << "\n";  

        // Store the direct and indirect light contributions
        // Pixel values are not divided by the number of samples
        for (int i = 0; i < current_resolution.x * current_resolution.y; i++)
        {
            //values[i * 6] = pixelLight[i].getR();
            //values[i * 6 + 1] = pixelLight[i].getG();
            //values[i * 6 + 2] = pixelLight[i].getB();

            file << pixelLight[i].getR() << " " 
                << pixelLight[i].getG() << " " 
                << pixelLight[i].getB() << " ";
        }

        // Close the file
        file.close();
    }

    bool loadContributionsFromFile(std::string filename)
    {
        filename += ".hdrbmp";

        std::ifstream file;
        file.open(filename, std::ios::binary);

        // If file does not exist, return false
        if (!file.is_open())
            return false;

        // Read the header
        std::string header;
        file >> header;

        // Read the resolution
        int width, height;
        file >> width >> height;

        if (width != current_resolution.x || height != current_resolution.y) 
        {
            std::cerr << "Error: The resolution of the file does not match the resolution of the film" << std::endl;
            return false;
        }

        // Read the data type
        char dataType;
        file >> dataType;

        long long tmp_samplesLoadedFromFile;

        // Read the number of samples
        file >> tmp_samplesLoadedFromFile;

        samplesLoadedFromFile += tmp_samplesLoadedFromFile;

        std::string data;
        // Discard the newline character
        std::getline(file, data);

        // Get the direct and indirect light contributions
        // Pixel values are not divided by the number of samples
        for (int i = 0; i < current_resolution.x * current_resolution.y; i++)
        {
            double values[3];
            file >> values[0] >> values[1] >> values[2];
            pixelLight[i] += Spectrum(values[0], values[1], values[2]);
        }

        // Close the file
        file.close();

        return true;
    }

    // Function to perform bicubic interpolation
    Spectrum bicubicInterpolation(const Spectrum *img, double x, double y) 
    {
        int x1 = static_cast<int>(x);
        int y1 = static_cast<int>(y);

        double dx = x - x1;
        double dy = y - y1;

        Spectrum interpolatedPixel = {0, 0, 0};

        for (int j = -1; j <= 2; j++) 
        {
            for (int i = -1; i <= 2; i++) 
            {
                int x_idx = min(max(x1 + i, 0), current_resolution.x - 1);
                int y_idx = min(max(y1 + j, 0), current_resolution.y - 1);

                double weight_x = cubicWeight(dx - i);
                double weight_y = cubicWeight(dy - j);

                int pixelIdx = y_idx * current_resolution.x + x_idx;

                Spectrum contribution(weight_x * weight_y * img[pixelIdx].getR(),
                                        weight_x * weight_y * img[pixelIdx].getG(),
                                        weight_x * weight_y * img[pixelIdx].getB());

                interpolatedPixel += contribution;
            }
        }

        // Handle negative values generated by the interpolation
        if (interpolatedPixel.getR() < 0)
            interpolatedPixel.setR(0);
        
        if (interpolatedPixel.getG() < 0)
            interpolatedPixel.setG(0);
        
        if (interpolatedPixel.getB() < 0)
            interpolatedPixel.setB(0);

        return interpolatedPixel;
    }

    double cubicWeight(double t) 
    {
        const double a = -0.5;
        double tAbs = abs(t);

        if (tAbs <= 1.0) {
            return ((a + 2.0) * tAbs - (a + 3.0)) * tAbs * tAbs + 1.0;
        } else if (tAbs <= 2.0) {
            return (a * tAbs - 5.0 * a) * tAbs * tAbs + 8.0 * a * tAbs - 4.0 * a;
        } else {
            return 0.0;
        }
    }

    void bicubicRescale(int newWidth, int newHeight) 
    {
        // Create a new vector for the resized image
        Spectrum *resizedLight = new Spectrum[newWidth * newHeight];

        // Perform bicubic interpolation for each pixel in the resized image
        for (int y = 0; y < newHeight; y++) 
        {
            for (int x = 0; x < newWidth; x++) 
            {
                double originalX = static_cast<double>(x) * (static_cast<double>(current_resolution.x) / newWidth);
                double originalY = static_cast<double>(y) * (static_cast<double>(current_resolution.y) / newHeight);

                Spectrum interpolatedLight = bicubicInterpolation(pixelLight, originalX, originalY);

                int pixelIdx = y * newWidth + x;
                resizedLight[pixelIdx] = interpolatedLight;
            }
        }

        // Delete the old image data
        delete[] pixelLight;

        // Set the new image data
        pixelLight = resizedLight;
        current_resolution.x = newWidth;
        current_resolution.y = newHeight;
    }

    // Function to perform nearest-neighbor interpolation
    Spectrum nearestNeighborInterpolation(const Spectrum *img, double x, double y) 
    {
        int x_nearest = static_cast<int>(round(x));
        int y_nearest = static_cast<int>(round(y));

        // Ensure the nearest pixel coordinates are within the image bounds
        x_nearest = max(0, min(x_nearest, current_resolution.x - 1));
        y_nearest = max(0, min(y_nearest, current_resolution.y - 1));

        int pixelIdx = y_nearest * current_resolution.x + x_nearest;

        return img[pixelIdx];
    }

    void nearestNeighbourRescale(int newWidth, int newHeight) 
    {
        // Load your image data into a 2D vector (img)
        // Define the new size (newWidth, newHeight)

        // Create a new vector for the resized image
        Spectrum *resizedLight = new Spectrum[newWidth * newHeight];

        // Perform nearest-neighbor interpolation for each pixel in the resized image
        for (int y = 0; y < newHeight; y++) 
        {
            for (int x = 0; x < newWidth; x++) 
            {
                double srcX = static_cast<double>(x) * (static_cast<double>(current_resolution.x) / newWidth);
                double srcY = static_cast<double>(y) * (static_cast<double>(current_resolution.y) / newHeight);

                Spectrum interpolatedLight = nearestNeighborInterpolation(pixelLight, srcX, srcY);

                int pixelIdx = y * newWidth + x;
                resizedLight[pixelIdx] = interpolatedLight;
            }
        }

        // Delete the old image data
        delete[] pixelLight;

        // Set the new image data
        pixelLight = resizedLight;
        current_resolution.x = newWidth;
        current_resolution.y = newHeight;
    }

    void rescale(int newWidth, int newHeight)
    {
        bicubicRescale(newWidth, newHeight);
    }

    // Operator <<
    friend std::ostream& operator<<(std::ostream& os, const Film& film)
    {
        os << "Film: " << film.current_resolution.x << "x" << film.current_resolution.y << ", nSamples: " << film.samplesComputedPixel0 + film.samplesLoadedFromFile << std::endl;
        
        for (int i = 0; i < film.current_resolution.x * film.current_resolution.y; i += film.current_resolution.x)
        {
            for (int j = 0; j < film.current_resolution.y; j++)
            {
                os << film.pixelLight[i*film.current_resolution.x + j] / (film.samplesComputedPixel0+film.samplesLoadedFromFile) << " ";
            }
            os << std::endl;
        }

        return os;
    }

    friend Float computeRMSE(Film film1, Film film2);
    friend Float computePercentajeError(Film film1, Film film2);
};

Float computeRMSE(Film film1, Film film2)
{
    if (film1.current_resolution.x != film2.current_resolution.x 
        || film1.current_resolution.y != film2.current_resolution.y)
    {
        return -1;
    }

    Float rmse = 0;

    for (int i = 0; i < film1.current_resolution.x * film1.current_resolution.y; i++)
    {
        Spectrum pixel = film1.pixelLight[i] / (film1.samplesComputedPixel0 + film1.samplesLoadedFromFile);
        Spectrum otherPixel = film2.pixelLight[i] / (film2.samplesComputedPixel0 + film2.samplesLoadedFromFile);

        rmse += (pixel - otherPixel).squaredNorm();
    }

    rmse /= film1.current_resolution.x * film1.current_resolution.y;

    return sqrt(rmse);
}

Float computePercentajeError(Film film1, Film film2)
{
    if (film1.current_resolution.x != film2.current_resolution.x 
        || film1.current_resolution.y != film2.current_resolution.y)
    {
        return -1;
    }

    Float error = 0;

    for (int i = 0; i < film1.current_resolution.x * film1.current_resolution.y; i++)
    {
        Spectrum pixel = film1.pixelLight[i] / (film1.samplesComputedPixel0 + film1.samplesLoadedFromFile);
        Spectrum otherPixel = film2.pixelLight[i] / (film2.samplesComputedPixel0 + film2.samplesLoadedFromFile);

        error += (abs(pixel - otherPixel) / pixel).norm();
    }

    error /= film1.current_resolution.x * film1.current_resolution.y;

    return error;
}