#include "cuda_libraries/FileImporting.cu"
#include "camera/Film.cu"

int main()
{
    Film film;

    film.loadContributionsFromFile("./bathRenders/bath_2");

    film.writeToBMP("./yenBath", ToneMappingType::uncharted2);
}