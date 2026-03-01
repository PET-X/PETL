#ifndef LISTMODEQUEUE_H
#define LISTMODEQUEUE_H

class listModeQueue
{
public:
	listModeQueue();
	~listModeQueue();
	bool clearAll();
	bool reset();
	bool init();

	float** events;
	int maxEvents;
	int numEvents;

	bool insert(float v1, float v0, float u1, float u0, float weight = 1.0);
	/*
	bool insert(int, float, float, float, float, float);

	float*** events; // [v1Ind][i][v1,v0,u1,u0]
	float** weights;
	int* N_events;
	int N_v1;
private:
	int maxEvents;
	//*/
};

#endif
