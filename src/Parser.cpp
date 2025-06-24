#include "Parser.h"
#include "external/tinyxml2.h"
#include "Scenario.h"

#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include <stdexcept>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include<iostream> //TODO remove
#include<iomanip>

using namespace std;
using namespace tinyxml2; //TODO make the implementation of each subtype of parser in different file to avoid namspacing issues

#define RESET       "\033[0m"
#define BOLD        "\033[1m"
#define RED         "\033[31m"
#define GREEN       "\033[32m"
#define YELLOW      "\033[33m"
#define BLUE        "\033[34m"
#define MAGENTA     "\033[35m"
#define CYAN        "\033[36m"
#define WHITE       "\033[37m"

XMLParser::XMLParser(string file_path):
    Parser(file_path)
{
    XMLError e = root.LoadFile(file_path.c_str());
    if (e != XML_SUCCESS){
        throw runtime_error("Failed to load XML file: " + file_path + " (TinyXML2 error code " + std::to_string(e) + ")");
    }
    cout<< "Scenario file "<<file_path<<" correctly loaded!"<<endl;
}

vector<XMLElement*> XMLParser::get_list(XMLNode* node, string field_name){
    vector<XMLElement*> res;
    XMLElement* element = node->FirstChildElement(field_name.c_str());
    if (element == nullptr){cout<<YELLOW<<"[WARN]"<<WHITE<<" Wrong or inexisting field name " + field_name << endl;} 
    while (element != nullptr){
        res.push_back(element);
        element = element->NextSiblingElement(field_name.c_str()); 
    }

    return res; 
}

std::string getROS2PkgPath(std::string pkg_name){
        return ament_index_cpp::get_package_share_directory(pkg_name);
}


void XMLParser::parse_file_recc(XMLNode* node){
    vector<XMLElement*> temp;
    //----- Materials -----//
    XMLElement* materials = node->FirstChildElement("materials");
    if (materials != nullptr){
        temp = get_list(materials, "material");
        data.materials.insert(data.materials.end(), temp.begin(), temp.end());
    }
    
    //---- Looks -----//
    temp = get_list(node, "looks");
    for (auto look : temp){
        vector<XMLElement*> temp_look;
        temp_look = get_list(look, "look");
        data.looks.insert(data.looks.end(), temp_look.begin(), temp_look.end());
    }

    //----- Static -----//
    temp = get_list(node, "static");
    data.static_meshes.insert(data.static_meshes.end(), temp.begin(), temp.end());

    XMLElement* include_file = node->FirstChildElement("include");
    while (include_file != nullptr){
        //Load include file and extract root node 
        string path = include_file->Attribute("file");
        string correct_path;
        if (path.find("$(find " == 0))
        {
            size_t start = 7;
            size_t end = path.find(")", start);
            
            std::string package_name = path.substr(start, end - start);
            std::string relative_path = path.substr(end + 1, path.size());

            correct_path = getROS2PkgPath(package_name) + relative_path;
        }
        else {
            correct_path = path;
        }
        auto doc = make_unique<XMLDocument>();
        XMLError e = doc->LoadFile(correct_path.c_str());
        if (e != XML_SUCCESS){
            throw runtime_error("Failed to load XML file: " + correct_path + " (TinyXML2 error code " + std::to_string(e) + ")");
        }
        cout<<"Included file: " << correct_path <<" correctly loaded!"<<endl;
        
        XMLElement* scenario = doc->FirstChildElement("scenario");
        if (scenario == nullptr){throw runtime_error("No scenario found in this file !");}
        
        files.push_back(std::move(doc));
        parse_file_recc(scenario);

        include_file = include_file->NextSiblingElement("include");
    }

}

void XMLParser::parse_file(){

    XMLElement* scenario = root.FirstChildElement("scenario");
    if (scenario == nullptr){throw runtime_error("No scenario found in this file !");}

    parse_file_recc(scenario);

      cout << '\n' << BOLD << BLUE << left << setw(25) << "Total Summary" << RESET << endl;
        cout << BOLD << WHITE 
             << left << setw(20) << "Name" 
             << right << setw(10) << "Value" 
             << RESET << endl;
        cout << string(30, '-') << '\n';

        // Table rows with alternating colors (optional)
        cout << CYAN << left << setw(20) << "Materials" 
             << right << setw(10) << data.materials.size() << RESET << endl;

        cout << GREEN << left << setw(20) << "Static Meshes" 
             << right << setw(10) << data.static_meshes.size() << RESET << endl;

        cout << YELLOW << left << setw(20) << "Looks" 
             << right << setw(10) << data.looks.size() << RESET << endl;
}
             
float3 str2float3(string vec){
    stringstream ss(vec);
    float x, y, z;
    ss >> x >> y >> z;
    return {x, y, z};
}

float3 gray2float3(float gray){
    return {gray, gray, gray};
}

void XMLParser::get_scenario(Scenario& scenario){
    //----- Look -----//
    for (auto XMLLook : data.looks){
        Look look;
        look.name = XMLLook->Attribute("name");
        const char* val_ptr;

        string correct_path;
        if ((val_ptr = XMLLook->Attribute("texture")) != nullptr){
            string path = val_ptr; 
            if (path.find("$(find " == 0))
            {
                size_t start = 7;
                size_t end = path.find(")", start);
                std::string package_name = path.substr(start, end - start);
                std::string relative_path = path.substr(end+1, path.size());

                correct_path = getROS2PkgPath(package_name) + relative_path;
            }
            else {
                correct_path = path;
            }
        }
        else {
            correct_path = "";
        }
        look.texture = correct_path;

        float3 rgb_;
        if ((val_ptr = XMLLook->Attribute("rgb")) != nullptr){
            string rgb = val_ptr; 
            rgb_ = str2float3(rgb);
        }
        else if ((val_ptr = XMLLook->Attribute("gray")) != nullptr){
            float gray = stof(val_ptr); 
            rgb_ = float3({gray, gray, gray});
        }
        else {
            rgb_ = float3({0.0, 0.0, 0.0});
        }
        look.rgb = rgb_;

        XMLLook->QueryFloatAttribute("metalness", &look.metalness);
        XMLLook->QueryFloatAttribute("roughness", &look.roughness);

        scenario.Looks.push_back(look);
    }
    
    //----- Static meshes -----//
    for (auto XMLMesh : data.static_meshes){
        Mesh mesh;
        string file_name = XMLMesh->FirstChildElement("physical")->FirstChildElement("mesh")->Attribute("filename");

        string correct_path;
        if (file_name.find("$(find " == 0))
        {
            size_t start = 7;
            size_t end = file_name.find(")", start);
            std::string package_name = file_name.substr(start, end - start);
            std::string relative_path = file_name.substr(end+1, file_name.size());

            correct_path = getROS2PkgPath(package_name) + relative_path;
        }
        else {
            correct_path = file_name;
        }

        mesh.file_name = correct_path;

        XMLMesh->FirstChildElement("physical")->FirstChildElement("mesh")->QueryFloatAttribute("scale", &mesh.scale);
        mesh.material = XMLMesh->FirstChildElement("material")->Attribute("name");
        mesh.look = XMLMesh->FirstChildElement("look")->Attribute("name");

        string origin = XMLMesh->FirstChildElement("physical")->FirstChildElement("origin")->Attribute("xyz"); //origin is a point so normally rpy doesnt matter
        mesh.origin = str2float3(origin);

        string translation = XMLMesh->FirstChildElement("world_transform")->Attribute("xyz");
        mesh.world_translation = str2float3(translation);

        string rotation = XMLMesh->FirstChildElement("world_transform")->Attribute("rpy");
        mesh.world_rotation = str2float3(rotation);

        scenario.Static_meshes.push_back(mesh);

        cout<<mesh.world_rotation.x <<
              mesh.world_rotation.y <<
              mesh.world_rotation.z << 
              endl;
    }

    //----- Materials -----//
    for (auto XMLMaterial : data.materials){
        Material material;
        material.name = XMLMaterial->Attribute("name");
        material.type = XMLMaterial->Attribute("type");

        scenario.Materials.push_back(material);
    }
}

