//#ifndef LISTMODEQUEUE_CPP
//#define LISTMODEQUEUE_CPP

//#include "pch.h"
#include "list_mode_queue.h"
#include "allocate.h"

listModeQueue::listModeQueue()
{
	events = NULL;
	numEvents = 0;
	maxEvents = 10000;
}

listModeQueue::~listModeQueue()
{
	clearAll();
}

bool listModeQueue::clearAll()
{
	if (events != nullptr)
		multifree(events, 2, maxEvents);
	events = nullptr;
	numEvents = 0;
	return true;
}

bool listModeQueue::reset()
{
	numEvents = 0;
	return true;
}

bool listModeQueue::init()
{
	clearAll();
	events = (float**) multialloc(sizeof(float), 2, maxEvents, 5);
	numEvents = 0;
	return true;
}

bool listModeQueue::insert(float v1, float v0, float u1, float u0, float weight)
{
	if (numEvents == maxEvents)
		return false;
	events[numEvents][0] = v1;
	events[numEvents][1] = v0;
	events[numEvents][2] = u1;
	events[numEvents][3] = u0;
	events[numEvents][4] = weight;
	numEvents += 1;
	if (numEvents == maxEvents)
		return false;
	else
		return true;
}

/*
listModeQueue::listModeQueue()
{
	events = NULL;
	weights = NULL;
	N_events = NULL;
	maxEvents = 1000;
}

listModeQueue::~listModeQueue()
{
	clearAll();
}

bool listModeQueue::clearAll()
{
	multifree(events, 3, N_v1, maxEvents);
	events = NULL;
	
	multifree(weights, 2, N_v1);
	weights = NULL;

	free(N_events);
	N_events = NULL;
	
	return true;
}

bool listModeQueue::init(int N)
{
	clearAll();
	N_v1 = N;
	events = (float***) multialloc(sizeof(float), 3, N_v1, maxEvents, 4);
	weights = (float**) multialloc(sizeof(float), 2, N_v1, maxEvents);
	N_events = (int*) calloc(size_t(N_v1), sizeof(int));
	return true;
}

bool listModeQueue::reset()
{
	for (int i = 0; i < N_v1; i++)
		N_events[i] = 0;
	return true;
}

bool listModeQueue::insert(int v1Ind, float v1, float v0, float u1, float u0, float theWeight)
{
	events[v1Ind][N_events[v1Ind]][0] = v1;
	events[v1Ind][N_events[v1Ind]][1] = v0;
	events[v1Ind][N_events[v1Ind]][2] = u1;
	events[v1Ind][N_events[v1Ind]][3] = u0;
	weights[v1Ind][N_events[v1Ind]] = theWeight;
	N_events[v1Ind] += 1;
	if (N_events[v1Ind] == maxEvents)
		return false;
	else
		return true;
}
//*/
