#ifndef __LIST_OF_MODELS_H
#define __LIST_OF_MODELS_H

#ifdef WIN32
#pragma once
#endif

#include <vector>
#include <stdlib.h>
class PETL;

class listOfModels
{
public:
    // Constructor; these do nothing
    listOfModels();

    // Destructor, just called clear()
    ~listOfModels();

    /**
     * \fn          clear
     * \brief       Clears the list member variable and calls the tomographicModels destructor for all elements in the list.
     */
    void clear();

    /**
     * \fn          append
     * \brief       Adds another instance of tomographicModels class to the end of the list
     */
    int append();

    /**
     * \fn          size
     * \return      returns list.size()
     */
    int size();

    /**
     * \fn          get
     * \param[in]   i, the index of the list to get
     * \return      returns list[i % size()]
     */
    PETL* get(int i);
    
private:

    // Tracks a list (as a stack) of tomographicModels instances
    std::vector<PETL*> list;
};

#endif
