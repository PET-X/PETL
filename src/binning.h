#ifndef __BINNING_H
#define __BINNING_H

#ifdef WIN32
#pragma once
#endif

#include "petl_defines.h"
#include <vector>

class parameters;
class listModeQueue;

class binning
{
public:
    binning();
    binning(parameters*);
    ~binning();

    void init(parameters*);
    void clear_queues();

    bool read_xyzlist(char* xyzfile);
    
	bool read_dethist(char* SimSETfile = nullptr);
	bool read_dethistshort(char* SimSETfile = nullptr);
    bool binData(char* Datafile = nullptr);
    bool binSimSETdata(char* fileName = nullptr);
    bool binAcqdata(char* Acqfile = nullptr);

    bool flush_queue(int i);

    parameters* params;
    std::vector<listModeQueue*> queues;
    int num_planograms;
};

#endif
