#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>

#include "../geometry/Point.cu"
#include "../geometry/Normal.cu"
#include "../light/Light.cu"
#include "../light/PointLight.cu"
#include "../camera/ProjectiveCamera.cu"

#include "../shapes/Triangle.cu"
#include "../materials/PhongMaterial.cu"
#include "../materials/LambertianMaterial.cu"
#include "../materials/GlassMaterial.cu"
#include "../materials/FresnelBlendMaterial.cu"
#include "../materials/BeckmannMaterial.cu"
#include "../materials/TrowbridgeMaterial.cu"
#include "../materials/TrowbridgeBasicMaterial.cu"
#include "../materials/PlainMaterial.cu"
#include "../materials/SSMaterial.cu"

std::unordered_map<std::string, Material*> loadMTL(std::string fileName)
{
    std::ifstream mtlFile(fileName);
    if (!mtlFile) {
        std::cerr << "MTL file not found: " << fileName << std::endl;
        return std::unordered_map<std::string, Material*>();
    }
    
    std::unordered_map<std::string, Material*> materials;
    
    Spectrum ka(0), kd(1), ks(0), ke(0);
    Float ni = 1, ns = 0;
    int illum = 0;
    Texture *texture = nullptr;
    std::string name = "";
    bool transparent = false, hasIndirect = true;
    Spectrum sigmaS, sigmaA;
    Float g;

    Float roughnessX = 0, roughnessY = 0;
    Spectrum eta(1), k(0);

    std::string line;
    while (std::getline(mtlFile, line)) 
    {
        std::istringstream iss(line);
        std::string keyword;
        iss >> keyword;

        if (keyword == "newmtl") 
        {
            // Start parsing a new material
            if (name != "") 
            {
                if (illum == 0)
                    materials[name] = new PlainMaterial(kd, ke, texture, transparent);
                else if (illum == 1)
                    materials[name] = new LambertianMaterial(kd, ka, texture, transparent);
                else if (illum == 2)
                    materials[name] = new PhongMaterial(ka, kd, ks, ke, ni, ns, texture, transparent, hasIndirect);
                else if (illum == 3)
                    materials[name] = new DielectricMaterial(Spectrum(1), ni);
                else if (illum == 4)
                    materials[name] = new BeckmannMaterial(kd, eta, k, roughnessX, roughnessY, texture);
                else if (illum == 5)
                    materials[name] = new TrowbridgeMaterial(kd, eta, k, roughnessX, roughnessY, texture);
                else if (illum == 6)
                    materials[name] = new TrowbridgeBasicMaterial(kd, eta, k, roughnessX, roughnessY, texture);
                else if (illum == 7)
                    materials[name] = new SSMaterial(ka, kd, ks, ni, ns, texture, transparent, sigmaS, sigmaA, g);
            }
            
            ka = Spectrum(0);
            kd = Spectrum(1);
            ks = Spectrum(0);
            ni = 1.0f;
            ns = 0.0f;
            illum = 0;
            texture = nullptr;
            transparent = false;
            hasIndirect = true;

            roughnessX = 0;
            roughnessY = 0;
            eta = Spectrum(1);
            k = Spectrum(0);

            iss >> name;
        }
        else if (keyword == "Ka") 
        {
            Float r, g, b;
            iss >> r >> g >> b;
            ka = Spectrum(r, g, b);
        }
        else if (keyword == "Kd") 
        {
            Float r, g, b;
            iss >> r >> g >> b;
            kd = Spectrum(r, g, b);
        }
        else if (keyword == "Ks") 
        {
            Float r, g, b;
            iss >> r >> g >> b;
            ks = Spectrum(r, g, b);
        }
        else if (keyword == "Ke")
        {
            Float r, g, b;
            iss >> r >> g >> b;
            ke = Spectrum(r, g, b);
        }
        else if (keyword == "Ns") 
        {
            iss >> ns;
        }
        else if (keyword == "Ni") 
        {
            iss >> ni;
        }
        else if (keyword == "Pr")
        {
            iss >> roughnessX;
            roughnessY = roughnessX;
        }
        else if (keyword == "PrX")
        {
            iss >> roughnessX;
        }
        else if (keyword == "PrY")
        {
            iss >> roughnessY;
        }
        else if (keyword == "eta")
        {
            Float r, g, b;
            iss >> r >> g >> b;
            eta = Spectrum(r, g, b);
        }
        else if (keyword == "k")
        {
            Float r, g, b;
            iss >> r >> g >> b;
            k = Spectrum(r, g, b);
        }
        else if (keyword == "illum") 
        {
            iss >> illum;
        }
        else if (keyword == "makesShadow")
        {
            std::string temp;
            iss >> temp;

            if (temp == "1")
                transparent = false;
            else if (temp == "0")
                transparent = true;
            else
                throw std::runtime_error("Error: makesShadow must be 0 or 1.");
        }
        else if (keyword == "hasIndirect")
        {
            std::string temp;
            iss >> temp;

            if (temp == "1")
                hasIndirect = true;
            else if (temp == "0")
                hasIndirect = false;
            else
                throw std::runtime_error("Error: hasIndirect must be 0 or 1.");
        }
        else if (keyword == "sigmaS")
        {
            Float r, g, b;
            iss >> r >> g >> b;
            sigmaS = Spectrum(r, g, b);
        }
        else if (keyword == "sigmaA")
        {
            Float r, g, b;
            iss >> r >> g >> b;
            sigmaA = Spectrum(r, g, b);
        }
        else if (keyword == "g")
        {
            iss >> g;
        }
        else if (keyword == "map_Kd")
        {
            std::string textureFileName, texturePath;
            iss >> textureFileName;

            // Get path of the material
            size_t found = fileName.find_last_of("/\\");
            std::string dirPath = fileName.substr(0, found + 1);
            texturePath = dirPath + textureFileName;

            try {
                texture = new Texture(texturePath);
            }
            catch (std::runtime_error e) 
            {
                std::string exc = e.what();

                if (exc.find("Failed to open the texture file") != std::string::npos)
                {
                    texturePath = dirPath + "textures/" + textureFileName;
                    texture = new Texture(texturePath);
                }
                else {
                    throw e;
                }
            }
        }
    }

    // Add the last material to the vector
    if (name != "")
    {
        if (illum == 0)
            materials[name] = new PlainMaterial(kd, ke, texture, transparent);
        else if (illum == 1)
            materials[name] = new LambertianMaterial(kd, ka, texture, transparent);
        else if (illum == 2)
            materials[name] = new PhongMaterial(ka, kd, ks, ke, ni, ns, texture, transparent);
        else if (illum == 3)
            materials[name] = new DielectricMaterial(Spectrum(1), ni);
        else if (illum == 4)
            materials[name] = new BeckmannMaterial(kd, eta, k, roughnessX, roughnessY, texture);
        else if (illum == 5)
            materials[name] = new TrowbridgeMaterial(kd, eta, k, roughnessX, roughnessY, texture);
        else if (illum == 6)
            materials[name] = new TrowbridgeBasicMaterial(kd, eta, k, roughnessX, roughnessY, texture);
        else if (illum == 7)
            materials[name] = new SSMaterial(ka, kd, ks, ni, ns, texture, transparent, sigmaS, sigmaA, g);
    }

    mtlFile.close();

    return materials;
}

void loadObj(std::string fileName, 
        std::shared_ptr<std::vector<TriangleMesh>> meshes, 
        std::vector<std::vector<Triangle>> &total_triangles)
{
    std::ifstream objFile(fileName);

    if (!objFile.is_open()) 
    {
        throw(std::runtime_error("Failed to open the OBJ file."));
    }

    std::unordered_map<std::string, int> objectIds;
    std::unordered_map<std::string, Material*> materials;
    std::string objectName;
    std::string currentMaterial = "default";

    Material *defaultMaterial = new LambertianMaterial(Spectrum(1));
    materials["default"] = defaultMaterial;

    TriangleMesh mesh;
    std::vector<Triangle> triangles;

    std::string line;
    while (std::getline(objFile, line)) 
    {
        std::istringstream lineStream(line);
        std::string token;
        lineStream >> token;
        
        if (token == "o")
        {
            lineStream >> objectName;
            triangle_meshes_ids++;
        }
        else if (token == "v") 
        {
            Point3f vertex;
            lineStream >> vertex.x >> vertex.z >> vertex.y;            
            mesh.vertices.push_back(vertex);
        } 
        else if (token == "vn") 
        {
            Vector3f normal;
            lineStream >> normal.x >> normal.z >> normal.y;            
            mesh.normals.push_back(normal);
        } 
        else if (token == "vt") 
        {
            Point2f texCoord;
            lineStream >> texCoord.x >> texCoord.y;
            mesh.textureCoords.push_back(texCoord);
        } 
        else if (token == "f") 
        {
            Triangle face;
            face.meshId = triangle_meshes_ids;

            for (int i = 0; i < 3; ++i) 
            {
                lineStream >> face.vertexIndices[i];

                // Obj indices start at 1, not 0
                face.vertexIndices[i] -= 1;

                if (lineStream.peek() == '/') 
                {
                    lineStream.ignore();  // Skip the '/'

                    if (lineStream.peek() != '/') 
                    {
                        lineStream >> face.textureIndices[i];
                        face.textureIndices[i] -= 1;
                    }
                    if (lineStream.peek() == '/') 
                    {
                        lineStream.ignore();  // Skip the '/'
                        lineStream >> face.normalIndices[i];
                        face.normalIndices[i] -= 1;
                    }
                }
            }

            if (materials.find(currentMaterial) == materials.end())
                throw std::runtime_error("Error: The OBJ file has a face with an unknown material " + currentMaterial + ".");
            else
                face.setMaterial(materials[currentMaterial]);
            
            // Check if there are more than 3 vertices per face
            std::string dummy;
            if (lineStream >> dummy) 
            {
                throw std::runtime_error("Error: The OBJ file has polygons, it should only have triangles.");
            }

            triangles.push_back(face);
        }
        else if (token == "usemtl")
        {
            std::string materialName;
            lineStream >> materialName;

            currentMaterial = materialName;
        }
        else if (token == "mtllib")
        {
            std::string mtlFileName, filePath;
            lineStream >> mtlFileName;

            // If there is something still in lineStream (there were spaces in the name)
            if (lineStream.peek() != EOF) {
                throw std::runtime_error("Error: The OBJ file has a material file name with spaces.");
            }

            // Get the route of the MTL file
            size_t found = fileName.find_last_of("/\\");
            filePath = fileName.substr(0, found + 1);
            filePath += mtlFileName;

            auto MTLMaterials = loadMTL(filePath);

            // Combine with existing materials
            materials.insert(MTLMaterials.begin(), MTLMaterials.end());
        }
    }

    objFile.close();

    if (triangles.size() == 0)
    {
        throw std::runtime_error("Error: The OBJ file has no triangles.");
    }

    meshes->push_back(mesh);
    total_triangles.push_back(triangles);
}


void storeObejotaObject(std::vector<Light*> &lights, 
        std::vector<Shape*> &shapes,
        Camera *&camera,
        std::string &type,
        std::string &name,
        Point3f &position,
        Float &intensity,
        Spectrum &color,
        Point3f &lookat,
        Vector3f &cts,
        Vector3f &up,
        Float &frameSizeY,
        Transform &rotation,
        bool &validRotation)
{
    if (type == "camera")
    {
        //if (lookat == Point3f(0))
        //    throw std::runtime_error("Error: Camera " + name + " has no lookat point.");

        if (validRotation)
            up = rotation(up);

        if (cts == Vector3f(0))
            throw std::runtime_error("Error: Camera " + name + " has no camera to screen vector.");

        if (up == Vector3f(0))
            throw std::runtime_error("Error: Camera " + name + " has no up vector.");

        if (frameSizeY == 0)
            throw std::runtime_error("Error: Camera " + name + " has no frame size.");
            

        Transform lookAt = Transformations::lookAt(position,
                lookat, up);
        Transform cameraToScreen = Transformations::translate(cts);

        Film *film = new Film();

        Float verticalFrameSize = frameSizeY;
        verticalFrameSize /= 2;
        Float horizontalFrameSize = (verticalFrameSize/resolutionY) * resolutionX;
        
        camera = new ProjectiveCamera (inverse(lookAt), cameraToScreen, 
                    Bound2f(Point2f(-horizontalFrameSize, -verticalFrameSize), 
                            Point2f(horizontalFrameSize, verticalFrameSize)),
                    0, 0, film, nullptr);
    }
    else if (type == "lightPoint")
    {
        if (color == Spectrum(0))
            std::cerr << "Warning: Light " << name << " has no color." << std::endl;

        if (intensity == 0)
            std::cerr << "Warning: Light " << name << " has no intensity." << std::endl;

        Light *light = new PointLight(
                Transformations::translate(position.toVector()), 
                color, intensity);

        lights.push_back(light);
    }


    type = "";
    name = "";
    position = Point3f(0);
    intensity = 0;
    color = Spectrum(0);
    lookat = Point3f(0);
    cts = Vector3f(0);
    up = Vector3f(0);
    frameSizeY = 0;
}



void loadObejota(std::string fileName, 
        std::vector<Light*> &lights, 
        std::vector<Shape*> &shapes,
        Camera *&camera)
{
    std::ifstream obejotaFile(fileName);

    if (!obejotaFile.is_open()) 
    {
        throw(std::runtime_error("Failed to open the obejota file."));
    }

    std::string objectName, type;
    Point3f position, lookat;
    Vector3f cameraToScreen, up;
    Float intensity, frameSizeY;
    Spectrum color;

    Transform rotation;
    bool validRotation = false;

    bool cameraFound = false;

    std::string line;
    while (std::getline(obejotaFile, line)) 
    {
        std::istringstream lineStream(line);
        std::string token;
        lineStream >> token;
        
        if (token == "lightPoint")
        {
            if (objectName != "")
            {
                storeObejotaObject(lights, shapes, 
                    camera, type, objectName, 
                    position, intensity, color, 
                    lookat, cameraToScreen, 
                    up, frameSizeY,
                    rotation, validRotation);
            }

            lineStream >> objectName;
            type = "lightPoint";
        }
        else if (token == "p") 
        {
            lineStream >> position.x >> position.y >> position.z;
            
            // Blender has inverted y axis
            position.y = -position.y;           
        } 
        else if (token == "i") 
        {
            lineStream >> intensity;
        } 
        else if (token == "c")
        {
            Float r, g, b;
            lineStream >> r >> g >> b;
            color = Spectrum(r, g, b);
        }
        else if (token == "camera") 
        {
            if (objectName != "")
            {
                storeObejotaObject(lights, shapes, 
                    camera, type, objectName, 
                    position, intensity, color,
                    lookat, cameraToScreen, 
                    up, frameSizeY,
                    rotation, validRotation);
            }

            objectName = "camera";
            type = "camera";
            cameraFound = true;
        } 
        else if (token == "lookat") 
        {
            lineStream >> lookat.x >> lookat.y >> lookat.z;
            
            // Blender has inverted y axis
            lookat.y = -lookat.y;           
        } 
        else if (token == "rotation")
        {
            Float angleX, angleY, angleZ;
            lineStream >> angleX >> angleY >> angleZ;

            // Rotate the look point with the same angles as blender does
            Transform rotationX = Transformations::rotate(-angleX, Vector3f(1, 0, 0));
            Transform rotationY = Transformations::rotate(-angleY, Vector3f(0, 1, 0));
            Transform rotationZ = Transformations::rotate(-angleZ, Vector3f(0, 0, 1));

            rotation = rotationZ(rotationY(rotationX));
            Point3f localLookat = rotation(Point3f(0, 0, -1));

            lookat = position + localLookat;
            validRotation = true;
        }
        else if (token == "cts") 
        {
            lineStream >> cameraToScreen.x >> cameraToScreen.y >> cameraToScreen.z;
            
            // Blender has inverted y axis
            cameraToScreen.y = -cameraToScreen.y;           
        } 
        else if (token == "up") 
        {
            lineStream >> up.x >> up.y >> up.z;
            
            // Blender has inverted y axis
            up.y = -up.y;           
        } 
        else if (token == "frameY")
        {
            lineStream >> frameSizeY;
        }
        else if (token != "" && token != "#" && token != " ")
        {
            std::cerr << "Warning: Unknown token in obejota file: " << token << std::endl;
        }
    }

    if (objectName != "")
    {
        storeObejotaObject(lights, shapes, 
            camera, type, objectName, 
            position, intensity, color,
            lookat, cameraToScreen, 
            up, frameSizeY,
            rotation, validRotation);
    }

    if (!cameraFound)
        std::cerr << "Warning: No camera found in obejota file." << std::endl;

    obejotaFile.close();
}



namespace fs = std::filesystem;

std::vector<std::string> getHdrBmpFiles(const std::string& directoryPath) 
{
    std::vector<std::string> hdrBmpFiles;

    try {
        for (const auto& entry : fs::directory_iterator(directoryPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".hdrbmp") {
                hdrBmpFiles.push_back(entry.path().filename().string());
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error accessing directory: " << e.what() << std::endl;
    }

    return hdrBmpFiles;
}


int getNewHdrBmpId (const std::string& directoryPath) 
{
    std::vector<int> numbers;
    int maxElement=0;

    try {
        for (const auto& entry : fs::directory_iterator(directoryPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".hdrbmp") {
                std::string filename = entry.path().filename().stem().string();

                // Check if the filename is of the form "fileX", where X is a number
                if (!filename.empty() && std::any_of(filename.begin(), filename.end(), ::isdigit)) 
                {
                    // Remove everything but the number
                    filename.erase(std::remove_if(filename.begin(), filename.end(), 
                                [](char c) { return !std::isdigit(c); }), filename.end());

                    numbers.push_back(std::stoi(filename));
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        maxElement = -1;

        // If the directory doesn't exist, create it
        if (e.code() == std::errc::no_such_file_or_directory) {
            fs::create_directory(directoryPath);
        }
    }

    // Find the highest number using std::max_element
    if (!numbers.empty())
        maxElement = *std::max_element(numbers.begin(), numbers.end());

    return maxElement + 1;
}