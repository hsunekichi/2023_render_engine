#ifndef BITMAP_H
#define BITMAP_H

#include <cstdint>
#include <fstream>
#include <string>
#include <iostream>
#include <algorithm>
#include <bit>

//Metodos de compresion
#define BI_RGB 0
#define BI_RLE8 1
#define BI_RLE4 2
#define BI_BITFIELDS 3
#define BI_JPEG 4
#define BI_PNG 5
#define BI_ALPHABITFIELDS 6
#define BI_CMYK 11
#define BI_CMYKRLE8 12
#define BI_CMYKRLE4 13

//Espacios de color
#define LCS_WINDOWS_COLOR_SPACE 0x57696E20 //"Win "

union{
    uint8_t fragment;
    uint32_t word;
} endian_tester;

class BitmapImage {

    //Bitmap File Header
    const uint8_t Signature[2] = {'B', 'M'};
    uint32_t File_Size;
    uint16_t Reserved1 = 0;
    uint16_t Reserved2 = 0;
    uint32_t File_Offset_to_PixelArray; //Localizacion del array de pixeles

    //BITMAPV5HEADER
    uint32_t DIB_Header_Size = 40; //Cabecera hasta la parte de colores importantes
    int32_t Image_Width;
    int32_t Image_Height;
    uint16_t Planes = 1; //El numero de planos debe ser 1 obligatoriamente
    uint16_t Bits_per_Pixel = 24; //Numero de bits dedicados a cada pixel
    uint32_t Compression = BI_RGB;
    uint32_t Image_Size;
    uint32_t X_Pixels_Per_Meter = 3780; //Medido en pixeles por metro
    uint32_t Y_Pixels_Per_Meter = 3780; //---------------------------
    uint32_t Colors_in_Color_Table = 0;
    uint32_t Important_Color_Count = 0;
    uint32_t Red_channel_bitmask = 0x0000FF00;  //Mascaras para cada canal de color
    uint32_t Green_channel_bitmask = 0x00FF0000;
    uint32_t Blue_channel_bitmask = 0xFF000000;
    uint32_t Alpha_channel_bitmask = 0x00000000;
    uint32_t Color_Space_Type = LCS_WINDOWS_COLOR_SPACE;
    uint8_t Color_Space_Endpoints [36] = {0};
    uint32_t Gamma_for_Red_channel = 0;
    uint32_t Gamma_for_Green_channel = 0;
    uint32_t Gamma_for_Blue_channel = 0;
    uint32_t Intent = 0;
    uint32_t ICC_Profile_Data = 0;
    uint32_t ICC_Profile_Size = 0;
    uint32_t Reserved = 0;

    uint32_t* ColorTable = nullptr;

    uint32_t GAP1_size = 0;

    
    

    uint32_t GAP2_size = 0;

    uint8_t* ICC_Color_Profile = nullptr;

    public:

    //Image Data
    uint8_t* image_bytes;

    BitmapImage(int32_t Image_Width, int32_t Image_Height);

    ~BitmapImage();

    void write_image(std::ofstream &salida);

};

bool isLittleEndian(){
    endian_tester.word = 1;
    return endian_tester.fragment != 0;
}

BitmapImage::BitmapImage(int32_t Image_Width, int32_t Image_Height){
    this->Image_Width = Image_Width;
    this->Image_Height = Image_Height;
    Image_Size = (Image_Width + 4 - (Image_Width % 4))*Image_Height;
    File_Size = 14 + DIB_Header_Size + Image_Size;
    File_Offset_to_PixelArray = 14 + DIB_Header_Size;

    image_bytes = new uint8_t [Image_Width*Image_Height*3];
}

BitmapImage::~BitmapImage(){
    if(image_bytes != nullptr){
        delete image_bytes;
        image_bytes = nullptr;
    }
}

template<class T>
constexpr T byteswap(T value) noexcept
{
    //No soporta std c++20 asi que usare std c++17 y hare yo la manipulacion de bytes
    uint8_t* valores_internos = (uint8_t*)&value;
    uint8_t* valores_reversos = new uint8_t[sizeof(T)];

    //Revertimos temporalmente los bytes en un buffer externo
    for(uint8_t i = 0; i < sizeof(T); i++){
        valores_reversos[i] = valores_internos[sizeof(T) - i];
    }

    //Los solapamos sobre el dato de tipo T dado
    for(uint8_t i = 0; i < sizeof(T); i++){
        valores_internos[i] = valores_reversos[i];
    }

    //Borramos el buffer
    delete valores_reversos;
    valores_reversos = nullptr;

    //Devolvemos el valor
    return value;
}

template<class T>
bool write_if_possible(std::ostream &salida, T elemento, int32_t &bytes_restantes){
    bool littleEndian = isLittleEndian();
    if((bytes_restantes -= sizeof(elemento)) >= 0){
        if (littleEndian){
            salida.write((char*)&elemento, sizeof(elemento));
        }else{
            T reversed_data = byteswap(elemento);
            salida.write((char*)&reversed_data, sizeof(reversed_data));
        }
        return true;
    }else{
        bytes_restantes = 0;
    }
    return false;
}

void BitmapImage::write_image(std::ofstream &salida){

    if(salida.is_open()){
        //Escribimos la cabecera inicial
        salida.write((char*)Signature, sizeof(Signature));
        salida.write((char*)&File_Size, sizeof(File_Size));
        salida.write((char*)&Reserved1, sizeof(Reserved1));
        salida.write((char*)&Reserved2, sizeof(Reserved2));
        salida.write((char*)&File_Offset_to_PixelArray, sizeof(File_Offset_to_PixelArray));

        //Miramos el tamanyo de nuestra cabecera
        int32_t bytes_to_write = DIB_Header_Size;

        //Escribimos los campos que podamos con el tamanyo del header dado
        write_if_possible(salida, DIB_Header_Size, bytes_to_write);
        write_if_possible(salida, Image_Width, bytes_to_write);
        write_if_possible(salida, Image_Height, bytes_to_write);
        write_if_possible(salida, Planes, bytes_to_write);
        write_if_possible(salida, Bits_per_Pixel, bytes_to_write);
        write_if_possible(salida, Compression, bytes_to_write);
        write_if_possible(salida, Image_Size, bytes_to_write);
        write_if_possible(salida, X_Pixels_Per_Meter, bytes_to_write);
        write_if_possible(salida, Y_Pixels_Per_Meter, bytes_to_write);
        write_if_possible(salida, Colors_in_Color_Table, bytes_to_write);
        write_if_possible(salida, Important_Color_Count, bytes_to_write);
        
        uint8_t bgr[3] = {0};
        uint32_t bytes_written = 0;

        //Almacenamos la imagen en si
        for(int32_t i = 0; i < Image_Height; i++){
            bytes_written = 0;
            for(int32_t j = 0; j < Image_Width; j++){

                bgr[0] = image_bytes[3*(j + i*Image_Width)];
                bgr[1] = image_bytes[3*(j + i*Image_Width) + 1];
                bgr[2] = image_bytes[3*(j + i*Image_Width) + 2];

                salida.write((char*)bgr, sizeof(bgr));

                bytes_written += 3;
            }

            //Anyadimos el padding restante
            while(bytes_written % 4 != 0){
                salida.write("\0", sizeof(uint8_t));
                bytes_written++;
            }
        }

    }
}

#endif
