#pragma once
#include "Scenario.h"
#include "external/tinyxml2.h"
#include <memory>
#include <string>
#include <vector>


class Parser{
public: 
    Parser(std::string file_path_, std::string data_path_){file_path = file_path_; data_path = data_path_;};
    virtual void parse_file() = 0;
    virtual void get_scenario(Scenario& scenario) = 0;
protected :
    std::string file_path;
    std::string data_path;
    std::string get_file(std::string path);
};



class XMLParser : public Parser {
public:
    XMLParser(std::string file_path, std::string data_path);

    void parse_file() override;
    void get_scenario(Scenario& scenario) override;
    

private:
   void parse_file_recc(tinyxml2::XMLNode* node);

   std::vector<tinyxml2::XMLElement*> get_list(tinyxml2::XMLNode* node, std::string field_name);

   tinyxml2::XMLDocument root;
   std::vector<std::unique_ptr<tinyxml2::XMLDocument>> files;

   struct XMLData {
       std::vector<tinyxml2::XMLElement*> materials;
       std::vector<tinyxml2::XMLElement*> static_meshes;
       std::vector<tinyxml2::XMLElement*> looks; 
       std::vector<tinyxml2::XMLElement*> robots;
   } data;

};
