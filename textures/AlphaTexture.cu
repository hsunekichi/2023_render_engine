#pragma once

#include <string>
#include "../radiometry/Spectrum.cu"
#include "../geometry/Point.cu"

#include <fstream>



class AlphaTexture
{
    protected:

    bool isLoaded = false;
    std::string img_path;
    std::vector<std::vector<Float>> pixelMatrix;

    #pragma pack(push, 1) // Disable padding in the BMP header
    struct BMPHeader {
        char signature[2];  // "BM"
        uint32_t fileSize;  // Total file size
        uint16_t reserved1; // Reserved (0)
        uint16_t reserved2; // Reserved (0)
        uint32_t offset;    // Offset to pixel data
        uint32_t headerSize;// Header size (40 bytes)
        int32_t width;      // Image width
        int32_t height;     // Image height
        uint16_t planes;    // Number of color planes (1)
        uint16_t bpp;       // Bits per pixel (usually 24 for RGB)
        uint32_t compression; // Compression method (usually 0 for uncompressed)
        uint32_t imageSize; // Size of the raw bitmap data
        int32_t xPixelsPerMeter; // Horizontal resolution
        int32_t yPixelsPerMeter; // Vertical resolution
        uint32_t colorsUsed; // Number of colors in the palette (0 for 24-bit)
        uint32_t colorsImportant; // Important colors (0)
    };
    #pragma pack(pop)

    bool loadBmpImage(const std::string& imagePath, std::vector<std::vector<Float>>& pixelMatrix) 
    {
        throw std::runtime_error("Alpha texture bmp loading not implemented");

        /*
        std::ifstream file(imagePath, std::ios::binary);

        if (!file) 
        {
            std::cerr << "Error: Failed to open the BMP file." << std::endl;
            return false;
        }

        BMPHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));

        if (header.signature[0] != 'B' || header.signature[1] != 'M') 
        {
            std::cerr << "Error: Not a valid BMP file." << std::endl;
            return false;
        }

        if (header.bpp != 24) 
        {
            std::cerr << "Error: Only 24-bit BMP files are supported." << std::endl;
            return false;
        }

        int width = header.width;
        int height = header.height;

        // Move the file pointer to the beginning of the pixel data
        file.seekg(header.offset, std::ios::beg);

        // Allocate memory for the pixel data
        //pixelMatrix.resize(height, std::vector<Spectrum>(width, Spectrum(0.0f)));
        
        for (int i = 0; i < height; i++) 
        {
            for (int j = 0; j < width; j++) 
            {
                char rgb[3];
                file.read(&rgb[2], 1); // Blue
                file.read(&rgb[1], 1); // Green
                file.read(&rgb[0], 1); // Red

                // Convert signed char to unsigned byte
                uint8_t uRGB[3];
                uRGB[0] = rgb[0];
                uRGB[1] = rgb[1];
                uRGB[2] = rgb[2];

                // Invert the pixel channels, bmp stores them in BGR order
                pixelMatrix[i][j] = Spectrum(
                        static_cast<Float>(uRGB[0]) / 255.0f,   // Red
                        static_cast<Float>(uRGB[1]) / 255.0f,   // Green
                        static_cast<Float>(uRGB[2]) / 255.0f    // Blue
                    );
            }
        }
        */

        return true;
    }

    public:

        AlphaTexture(std::string img_path)
        {
            this->img_path = img_path;
            
            if (!loadBmpImage(img_path, pixelMatrix))
                throw std::runtime_error("Error loading texture");
            
            isLoaded = true;
        }

        AlphaTexture() {}
        
        ~AlphaTexture() {}

        bool loadFromFile(const char* path)
        {
            img_path = path;
            
            if (!loadBmpImage(path, pixelMatrix))
                throw std::runtime_error("Error loading texture");
        }

        Point2f projectSphere(Point3f point) const
        {
            Float theta = acos(point.z);
            Float phi = atan2(point.y, point.x);

            Float u = phi / (2 * M_PI);
            Float v = theta / M_PI;

            return Point2f(u, v);
        }

        // Points between 0 and 1
        Float evaluate(Point2f samplePoint) const
        {            
            unsigned int x = samplePoint.x * getWidth();
            unsigned int y = samplePoint.y * getHeight();

            x = clamp(x, 0, getWidth() - 1);
            y = clamp(y, 0, getHeight() - 1);

            return pixelMatrix[y][x];
        }

        bool loaded() const
        {
            return isLoaded;
        }


        unsigned int getWidth() const {
            return pixelMatrix[0].size();
        }

        unsigned int getHeight() const {
            return pixelMatrix.size();
        }
};