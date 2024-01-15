#include "cuda_libraries/FileImporting.cu"
#include "camera/Film.cu"
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    Film film;

    std::string fileName = argv[1];
    
    if (film.loadContributionsFromFile(fileName))
    {
        std::cout << "Film loaded from file: " << fileName << std::endl;
        film.rescale(1000, 1000);

        film.writeToBMP("bathRescaled", ToneMappingType::gammaClamp);
    }
    else
    {
        std::cout << "Could not load film from file: " << fileName << std::endl;
    }
}