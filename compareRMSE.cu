#include "camera/Film.cu"
#include "cuda_libraries/types.h"
#include "cuda_libraries/FileImporting.cu"

#include <iostream>
#include <vector>


int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: ./compareRMSE reference directory" << std::endl;
        return 1;
    }

    Film film1;
    std::string directory = argv[2];
    directory =  directory;
    
    if (!film1.loadContributionsFromFile(argv[1]))
    {
        std::cerr << "Could not load file: " << argv[1] << std::endl;
        return 1;
    }

    auto files = getHdrBmpFiles(directory);

    for (auto file : files)
    {
        Film film2;

        // Strip extension
        file = file.substr(0, file.find_last_of("."));

        if (!film2.loadContributionsFromFile(directory + "/" + file))
        {
            std::cout << "Could not load file: " << directory + "/" + file << std::endl;
            continue;
        }

        Float rmse = computeRMSE(film1, film2);

        std::cout << file << " RMSE: " << rmse << std::endl;
    }
}