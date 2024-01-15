#pragma once

#include <string>
#include "../radiometry/Spectrum.cu"
#include "../geometry/Point.cu"

#include <fstream>

class Texture
{
    protected:

    std::string img_path;
    std::vector<std::vector<Spectrum>> pixelMatrix;
    std::vector<std::vector<uint8_t>> alphaLayer;

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

    bool loadBmpImage(const std::string& imagePath) 
    {
        std::ifstream file(imagePath, std::ios::binary);

        if (!file) 
        {
            std::string err = "Error loading " + imagePath + ": Failed to open the texture file";
            throw std::runtime_error(err);
            return false;
        }

        BMPHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));

        if (header.signature[0] != 'B' || header.signature[1] != 'M') 
        {
            std::string err = "Error loading " + imagePath + ": Not a valid BMP file.";
            throw std::runtime_error(err);
            return false;
        }

        if (header.bpp != 24 && header.bpp != 32) 
        {
            std::string err = "Error loading " + imagePath + ": Only 24 or 32 bit BMP files are supported, file is " + std::to_string(header.bpp) + " bit";
            throw std::runtime_error(err);
            return false;
        }

        int width = header.width;
        int height = header.height;

        // Move the file pointer to the beginning of the pixel data
        file.seekg(header.offset, std::ios::beg);

        // Allocate memory for the pixel data
        pixelMatrix.resize(height, std::vector<Spectrum>(width, Spectrum(0.0f)));

        if (header.bpp == 32)
        {
            // Allocate memory for the alpha layer
            alphaLayer.resize(height, std::vector<uint8_t>(width, 255));
        }

        for (int i = 0; i < height; i++) 
        {
            for (int j = 0; j < width; j++) 
            {
                char bgra[4];

                // Read the pixel components (in ABGR order)
                file.read(bgra, header.bpp / 8);

                // Convert signed char to unsigned byte
                uint8_t uRGB[3];

                if (header.bpp == 24) 
                {
                    uRGB[0] = static_cast<uint8_t>(bgra[2]); // Red
                    uRGB[1] = static_cast<uint8_t>(bgra[1]); // Green
                    uRGB[2] = static_cast<uint8_t>(bgra[0]); // Blue
                } 
                else if (header.bpp == 32)
                {
                    // Alpha
                    alphaLayer[i][j] = static_cast<uint8_t>(bgra[3]);

                    uRGB[0] = static_cast<uint8_t>(bgra[2]); // Red
                    uRGB[1] = static_cast<uint8_t>(bgra[1]); // Green
                    uRGB[2] = static_cast<uint8_t>(bgra[0]); // Blue
                }

                // Invert the pixel channels, bmp stores them in BGR order
                pixelMatrix[i][j] = Spectrum(
                        static_cast<Float>(uRGB[0]) / 255.0f,   // Red
                        static_cast<Float>(uRGB[1]) / 255.0f,   // Green
                        static_cast<Float>(uRGB[2]) / 255.0f    // Blue
                    );
            }
        }

        return true;
    }

    public:

        Texture(std::string img_path)
        {
            // Get extension
            std::string ext = img_path.substr(img_path.find_last_of(".") + 1);

            // Change jpg and png to bmp
            if (ext == "jpg" || ext == "png")
            {
                std::cout << "Warning: " << img_path << " is not a bmp file. Changing extension to bmp." << std::endl;
                img_path = img_path.substr(0, img_path.find_last_of(".")) + ".bmp";
            }

            // Get last . position
            int last_dot = img_path.find_last_of(".");

            // Change . to _ except for the first and last one
            for (int i = 1; i < img_path.size(); i++)
            {
                if (img_path[i] == '.' && i != last_dot)
                {
                    img_path[i] = '_';
                }
            }

            this->img_path = img_path;
            
            loadBmpImage(img_path);
        }
        
        ~Texture() {}

        bool loadFromFile(const char* path)
        {
            img_path = path;
            
            return loadBmpImage(img_path);
        }

        Point2f projectSphere(Point3f point) const
        {
            Float theta = acos(point.z);
            Float phi = atan2(point.y, point.x);

            Float u = phi / (2 * M_PI);
            Float v = theta / M_PI;

            return Point2f(u, v);
        }

        virtual Spectrum getColor(Point2f point) const
        {
            unsigned int x = point.x * getWidth();
            unsigned int y = point.y * getHeight();

            //x = x % getWidth();
            //y = y % getHeight();

            x = clamp(x, 0, getWidth() - 1);
            y = clamp(y, 0, getHeight() - 1);

            return pixelMatrix[y][x];
        }

        unsigned int getWidth() const {
            return pixelMatrix[0].size();
        }

        unsigned int getHeight() const {
            return pixelMatrix.size();
        }

        bool hasAlpha() const {
            return alphaLayer.size() > 0;
        }

        bool isTransparent(Point2f point) const
        {
            if (!hasAlpha())
                return false;

            unsigned int x = point.x * getWidth();
            unsigned int y = point.y * getHeight();

            x = x % getWidth();
            y = y % getHeight();

            x = clamp(x, 0, getWidth() - 1);
            y = clamp(y, 0, getHeight() - 1);

            return alphaLayer[y][x] < 3;
        }
};