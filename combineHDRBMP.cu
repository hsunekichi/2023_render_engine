#include "cuda_libraries/FileImporting.cu"
#include "camera/Film.cu"

int main(int argc, char** argv)
{
    // get first arg, path to folder
    if (argc < 3)
    {
        std::cout << "Use: combineHDRBMP directoryPath outputFile" << std::endl;
        return 1;
    }

    std::string path = argv[1];
    std::string outputFile = argv[2];

    auto path = "./boxRenders";
    auto outputFile = "boxCombined";

    // get all files in folder
    std::vector<std::string> files = getHdrBmpFiles(path);

    Film film;

    for (std::string file : files)
    {
        file = file.substr(0, file.find_last_of("."));
        std::cout << "Importing " << file << std::endl;
        film.loadContributionsFromFile(path+std::string("/")+file);  
    }

    film.storeContributionsToFile(outputFile);
}