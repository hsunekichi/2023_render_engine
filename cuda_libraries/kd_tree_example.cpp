/*
Description :   This is an example of usage of the KDTree class. It does not
                compile, but should give you the idea of how to integrate the
                KDTree in your code
*/


#include "kdtree.h"

/* 
    Your Photon class implementation, which stores each 
    photon walk interaction 
*/
class YourPhoton {
    Vec3D position_;    // 3D point of the interaction
    ...
    // It returns the axis i position (x, y or z)
    float position(std::size_t i) const { return position_[i]}
    ...    
};

/* 
    An additional struct that allows the KD-Tree to access your photon position
*/
struct PhotonPositition 
{
    Float operator()(const Photon& ph, std::size_t i) const {
        return ph.location[i];
    }
};

/* 
    The KD-Tree ready to work in 3 dimensions, with YourPhoton s, under a 
    brand-new name: YourPhotonMap 
*/

using PhotonMap = nn::KDTree<Photon,3,PhotonPosition>;

/*
    Example function to generate the photon map with the given photons
*/
PhotonMap generation_of_photon_map(thrust::host_vector<Photon> &thrust_photons)
{
    std::vector<Photon> photons = thrust_photons.toStdVector();
    auto map = PhotonMap(photons)
    return map
}

/*
    Example method to search for the nearest neighbors of the photon map
*/
thrust::host_vector<Photon> search_nearest(PhotonMap &map, Point3f position)
{
    // Maximum number of photons to look for (infinite)
    unsigned long nphotons_estimate = (ulong)-1;

    // nearest is the nearest photons returned by the KDTree
    std::vector<Photon> nearest = map.nearest_neighbors(position,
                                         nphotons_estimate,
                                         PHOTON_SEARCH_RADIUS)

    thrust::host_vector<Photon> thrust_nearest(nearest);

    return thrust_nearest;
}



