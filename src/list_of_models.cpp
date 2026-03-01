
#include "list_of_models.h"
#include "petl.h"

listOfModels::listOfModels()
{
}

listOfModels::~listOfModels()
{
    clear();
}

void listOfModels::clear()
{
    for (int i = 0; i < int(list.size()); i++)
    {
        PETL* p_model = list[i];
        delete p_model;
        list[i] = NULL;
    }
    list.clear();
}

int listOfModels::append()
{
    PETL* p_model = new PETL;
    list.push_back(p_model);
    return int(list.size()-1);
}

int listOfModels::size()
{
    return int(list.size());
}

PETL* listOfModels::get(int i)
{
    if (list.size() == 0)
    {
        append();
        return list[0];
    }
    else
        return list[i % list.size()];
}
