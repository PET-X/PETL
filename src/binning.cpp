#include <omp.h>
#include "parameters.h"
#include "list_mode_queue.h"
#include "vector_ops.h"
#include "binning.h"

using namespace std;

struct Evnt
{		
    float pos[4]; // energy is pos[3]
    uint64_t time;
};

struct Lor
{
    Evnt blue;
    Evnt pink;
};

binning::binning()
{
    params = nullptr;
}

binning::binning(parameters* params_in)
{
    init(params_in);
}

binning::~binning()
{
    params = nullptr;
    clear_queues();
    num_planograms = 0;
}

void binning::clear_queues()
{
    for (int i = 0; i < int(queues.size()); i++)
    {
        queues[i]->clearAll();
        listModeQueue* queue = queues[i];
        queue->clearAll();
        delete queue;
        queues[i] = nullptr;
    }
    queues.clear();
}

void binning::init(parameters* params_in)
{
    params = params_in;
    num_planograms = int(params->planogramSet.size());
    for (int i = 0; i < int(params->planogramSet.size()); i++)
    {
        planogram* plan = params->planogramSet[i];
        plan->set_constant(0.0);

        listModeQueue* queue = new listModeQueue;
        queue->init();
        queues.push_back(queue);
    }
}

bool binning::read_xyzlist(char* xyzfile)
{
    if (params == nullptr)
        return false;
    printf("planogramSet::read_xyzlist: reading file...\n");
	time_t startTime = time(NULL);

	FILE* fdes = NULL;
	if ((fdes = fopen(xyzfile, "rb")) == NULL)
	{
		printf("planogramSet::read_xyzlist: list mode data file: %s not found, quitting!\n", xyzfile);
        return false;
		//exit(1);
	}

	Lor lor;
	unsigned char mem[sizeof(Lor)];
	unsigned char * p = mem;

	float r[3];
	long N = 0;
	float curWeight = 1.0;
	float energyHistogram[2048];

	unsigned char numBlueScatters = 0;
	unsigned char numPinkScatters = 0;
	/*
	long N_mainpanels = 0;
	long N_sidepanels = 0;
	vector<double> pinkXpos;
	//*/

	//int binningType = 0; // all events
	//int binningType = 1; // primary only
	//int binningType = 2; // scatter only
	//int binningType = 3; // first order scatter only
	int binningType = params->binningType;
	printf("binningType = %d\n", binningType);

	while (1)
	{
		p = mem;
		if (fread(p, sizeof(Lor), 1, fdes) == 0)
			break;
		int fn = sizeof(lor.blue.pos);
		int tn = sizeof(lor.blue.time);
		memcpy(lor.blue.pos, p, fn);
		memcpy(&lor.blue.time, p + fn, tn);
		p = p + fn + tn;
		memcpy(lor.pink.pos, p, fn);
		memcpy(&lor.pink.time, p + fn, tn);

		energyHistogram[int(lor.blue.pos[3])] += 1;
		energyHistogram[int(lor.pink.pos[3])] += 1;

		r[0] = lor.blue.pos[0] - lor.pink.pos[0];
		r[1] = lor.blue.pos[1] - lor.pink.pos[1];
		r[2] = lor.blue.pos[2] - lor.pink.pos[2];

        float v1, v0;
        for (int i = 0; i < num_planograms; i++)
        {
            planogram* plan = params->planogramSet[i];
            if (plan->is_within_acceptable_angle(r, v1, v0))
            {
                float u1, u0;
                plan->get_u_coords(lor.blue.pos, v1, v0, u1, u0);
                if (params->listModeOrigin == parameters::CHEST_WALL)
                {
                    u1 -= plan->H;
                }
                if (!queues[i]->insert(v1, v0, u1, u0))
                    flush_queue(i);

                N += 1;
            }
        }
		if (feof(fdes))
			break;
	}
	fclose(fdes);

    for (int i = 0; i < num_planograms; i++)
        flush_queue(i);

	printf("planogramSet::binAcqdata: Binned %ld coincidences...\n", N);
	printf("planogramSet::binAcqdata: elapsed time: %d seconds\n", int(time(NULL) - startTime));

	return true;
}

bool binning::read_dethist(char* SimSETfile)
{
    printf("planogramSet::read_dethist: reading file...\n");
	time_t startTime = time(NULL);
    
    FILE* fdes = NULL;
    if ((fdes = fopen(SimSETfile, "rb")) == NULL)
    {
        printf("planogramSet::read_dethist: SimSET data file: %s not found, quitting!\n", SimSETfile);
        exit(1);
    }

	char dethist_header[32768];
	unsigned char nBlue;
	unsigned char nPink;
	double centroid[3];
	unsigned int nScat;
	double EDetected;
	double decay[3];
	int crystalNum;
	unsigned int nintBlue;
	unsigned int nintPink;
	double pos[3];
	double energy_int;
	unsigned short isActive;
	
	float p[3];
    float r[3];
    long N = 0;
    short N_blue, N_pink;
    float bluePos[3];
    float pinkPos[3];
    float blueE, pinkE;
    float curWeight = 1.0;

	double x_range[2]; x_range[0] = 0.0; x_range[1] = 0.0;
	double y_range[2]; y_range[0] = 0.0; y_range[1] = 0.0;
	double z_range[2]; z_range[0] = 0.0; z_range[1] = 0.0;
	
	fread(dethist_header, sizeof(char), 32768, fdes);
	while (1)
	{
		if (fread(&nBlue, sizeof(unsigned char), 1, fdes) == 0)
			break;
		if (nBlue > 0)
		{
			fread(centroid, sizeof(double), 3, fdes);
			fread(&nScat, sizeof(int), 1, fdes);
			fread(&EDetected, sizeof(double), 1, fdes);
			fread(decay, sizeof(double), 3, fdes);

			fread(&crystalNum, sizeof(int), 1, fdes);

			fread(&nintBlue, sizeof(int), 1, fdes);
			for (int interaction = 0; interaction < nintBlue; interaction++)
			{
				fread(pos, sizeof(double), 3, fdes);
				fread(&energy_int, sizeof(double), 1, fdes);
				fread(&isActive, sizeof(short), 1, fdes);
			}

			if (params->listModeUnits == parameters::MM)
			{
				bluePos[0] = centroid[0];
				bluePos[1] = centroid[1];
				bluePos[2] = centroid[2];
			}
			else
			{
				bluePos[0] = 10.0*centroid[0];
				bluePos[1] = 10.0*centroid[1];
				bluePos[2] = 10.0*centroid[2];
			}
			blueE = EDetected;
		}

		fread(&nPink, sizeof(char), 1, fdes);
		if (nPink > 0)
		{
			fread(centroid, sizeof(double), 3, fdes);
			fread(&nScat, sizeof(int), 1, fdes);
			fread(&EDetected, sizeof(double), 1, fdes);
			fread(decay, sizeof(double), 3, fdes);

			fread(&crystalNum, sizeof(int), 1, fdes);

			fread(&nintPink, sizeof(int), 1, fdes);
			for (int interaction = 0; interaction < nintPink; interaction++)
			{
				fread(pos, sizeof(double), 3, fdes);
				fread(&energy_int, sizeof(double), 1, fdes);
				fread(&isActive, sizeof(short), 1, fdes);
			}

			if (params->listModeUnits == parameters::MM)
			{
				pinkPos[0] = centroid[0];
				pinkPos[1] = centroid[1];
				pinkPos[2] = centroid[2];
			}
			else
			{
				pinkPos[0] = 10.0*centroid[0];
				pinkPos[1] = 10.0*centroid[1];
				pinkPos[2] = 10.0*centroid[2];
			}
			pinkE = EDetected;
		}
		
		if (nBlue > 0 && nPink > 0)
		{
            r[0] = bluePos[0] - pinkPos[0];
			r[1] = bluePos[1] - pinkPos[1];
			r[2] = bluePos[2] - pinkPos[2];

            float v1, v0;
            for (int i = 0; i < num_planograms; i++)
            {
                planogram* plan = params->planogramSet[i];
                if (plan->is_within_acceptable_angle(r, v1, v0))
                {
                    float u1, u0;
                    plan->get_u_coords(bluePos, v1, v0, u1, u0);
                    if (params->listModeOrigin == parameters::CHEST_WALL)
                    {
                        u1 -= plan->H;
                    }
                    if (!queues[i]->insert(v1, v0, u1, u0))
                        flush_queue(i);

                    N += 1;
                    x_range[0] = min(x_range[0], min(bluePos[0], pinkPos[0]));
                    x_range[1] = max(x_range[1], max(bluePos[0], pinkPos[0]));
                    y_range[0] = min(y_range[0], min(bluePos[1], pinkPos[1]));
                    y_range[1] = max(y_range[1], max(bluePos[1], pinkPos[1]));
                    z_range[0] = min(z_range[0], min(bluePos[2], pinkPos[2]));
                    z_range[1] = max(z_range[1], max(bluePos[2], pinkPos[2]));
                }
            }
		}
		if( feof(fdes) ) 
			break;
	}
    fclose(fdes);

    for (int i = 0; i < num_planograms; i++)
        flush_queue(i);
	
	printf("planogramSet::binSimSETdata: Binned %ld coincidences...\n", N);
    printf("planogramSet::binSimSETdata: elapsed time: %d seconds\n", int(time(NULL)-startTime));
	printf("LOR end point range: (%f, %f); (%f, %f); (%f, %f)\n", x_range[0], x_range[1], y_range[0], y_range[1], z_range[0], z_range[1]);
    
    return true;
}

bool binning::read_dethistshort(char* SimSETfile)
{
    if (params == nullptr)
        return false;
    printf("planogramSet::read_dethistshort: reading file...\n");
	time_t startTime = time(NULL);

	FILE* fdes = NULL;
	if ((fdes = fopen(SimSETfile, "rb")) == NULL)
	{
		printf("planogramSet::read_dethistshort: SimSET data file: %s not found, quitting!\n", SimSETfile);
		exit(1);
	}

	unsigned char nBlue;
	float bluePos[3];
	unsigned char numBlueScatters;
	float blueDetectedEnergy;

	unsigned char nPink;
	float pinkPos[3];
	unsigned char numPinkScatters;
	float pinkDetectedEnergy;

	float r[3];
	long N = 0;
	float curWeight = 1.0;
	float energyHistogram[512];

    //int binningType = 0; // all events
	//int binningType = 1; // primary only
	//int binningType = 2; // scatter only
	//int binningType = 3; // first order scatter only
	int binningType = params->binningType;
	printf("binningType = %d\n", binningType);

	char dethist_header[32768];
	fread(dethist_header, sizeof(char), 32768, fdes);
	while (1)
	{
		// read in the blue photon if one is in file
		if (fread(&nBlue, sizeof(unsigned char), 1, fdes) == 0)
			break;
		if (nBlue > 0)
		{
			fread(bluePos, sizeof(float), 3, fdes);
			fread(&numBlueScatters, sizeof(unsigned char), 1, fdes);
			fread(&blueDetectedEnergy, sizeof(float), 1, fdes);

			energyHistogram[int(blueDetectedEnergy)] += 1;

			if (params->listModeUnits != parameters::MM)
			{
				bluePos[0] *= 10.0;
				bluePos[1] *= 10.0;
				bluePos[2] *= 10.0;
			}
		}

		// read in the pink photon if one is in file
		if (fread(&nPink, sizeof(unsigned char), 1, fdes) == 0)
			break;
		if (nPink > 0)
		{
			fread(pinkPos, sizeof(float), 3, fdes);
			fread(&numPinkScatters, sizeof(unsigned char), 1, fdes);
			fread(&pinkDetectedEnergy, sizeof(float), 1, fdes);

			energyHistogram[int(pinkDetectedEnergy)] += 1;

			if (params->listModeUnits != parameters::MM)
			{
				pinkPos[0] *= 10.0;
				pinkPos[1] *= 10.0;
				pinkPos[2] *= 10.0;
			}
		}

        // Bin coincidence event
        bool binEvent = false;
        if (binningType == 0)
            binEvent = true;
        else if (binningType == 1 && numBlueScatters == 0 && numPinkScatters == 0)
            binEvent = true;
        else if (binningType == 2 && (numBlueScatters > 0 || numPinkScatters > 0))
            binEvent = true;
        else if (binningType == 3 && ((numBlueScatters == 1 && numPinkScatters == 0) || (numBlueScatters == 0 && numPinkScatters == 1)))
            binEvent = true;

		if (nBlue > 0 && nPink > 0 && binEvent)
		{
            r[0] = bluePos[0] - pinkPos[0];
			r[1] = bluePos[1] - pinkPos[1];
			r[2] = bluePos[2] - pinkPos[2];

            float v1, v0;
            for (int i = 0; i < num_planograms; i++)
            {
                planogram* plan = params->planogramSet[i];
                if (plan->is_within_acceptable_angle(r, v1, v0))
                {
                    float u1, u0;
                    plan->get_u_coords(bluePos, v1, v0, u1, u0);
                    if (params->listModeOrigin == parameters::CHEST_WALL)
                    {
                        u1 -= plan->H;
                    }
                    if (!queues[i]->insert(v1, v0, u1, u0))
                        flush_queue(i);

                    N += 1;
                }
            }
		}
		if (feof(fdes))
			break;
	}
	fclose(fdes);

    for (int i = 0; i < num_planograms; i++)
        flush_queue(i);

	printf("planogramSet::binSimSETdata: Binned %ld coincidences...\n", N);
	printf("planogramSet::binSimSETdata: elapsed time: %d seconds\n", int(time(NULL) - startTime));

	return true;
}

/*
bool binning::binData(char* Datafile)
{
    char fullPath[STRLENGTH];
	if (Datafile == NULL)
		sprintf(&fullPath[0], "%s%s", &params->dataFolderName[0], &params->listModeFileName[0]);
	else
		sprintf(&fullPath[0], "%s%s", &params->dataFolderName[0], Datafile);
	return true;
}
//*/

bool binning::binSimSETdata(char* fileName)
{
    if (params == nullptr)
        return false;
	if (params->listModeOrigin == parameters::CHEST_WALL)
		printf("assuming z=0 plane of the list mode data is at the chest wall\n");
	else
		printf("assuming z=0 plane of the list mode data is at the center of the field of view\n");

	if (params->listModeUnits == parameters::CM)
		printf("assuming list mode data is in units of cm\n");
	else
		printf("assuming list mode data is in units of mm\n");

	string fullPath_str = fileName;
	if (hasEnding(fullPath_str, ".dethist"))
		return read_dethist(fileName);
	else if (hasEnding(fullPath_str, ".dethistshort"))
		return read_dethistshort(fileName);
	
    time_t startTime = time(NULL);
    	
    FILE* fdes = NULL;
    if ((fdes = fopen(fileName, "rb")) == NULL)
    {
        printf("planogramSet::binSimSETdata: SimSET data file: %s not found, quitting!\n", fileName);
        exit(1);
    }
    
    float p[3];
    float r[3];
    long N;
    short N_blue, N_pink;
    float bluePos[3];
    float pinkPos[3];
    float blueE, pinkE;
    float curWeight = 1.0;
    int baseSkipAmount = sizeof(short) + 4*sizeof(float);
    
    long total_binned_events = 0;

    //fread(&N_dims, sizeof(int), 1, fdes);
    fread(&N, sizeof(long), 1, fdes);
    printf("planogramSet::binSimSETdata: Binning %ld coincidences...\n", N);
    for (int i = 0; i < N; i++)
    {
        // Read blue
        fread(&N_blue, sizeof(short), 1, fdes);
        fread(&bluePos[0], sizeof(float), 3, fdes);
        fread(&blueE, sizeof(float), 1, fdes);
		if (N_blue > 1)
			fseek(fdes, baseSkipAmount*(N_blue-1), SEEK_CUR); // not interested in other interactions
        
        // Read pink
        fread(&N_pink, sizeof(short), 1, fdes);
        fread(&pinkPos[0], sizeof(float), 3, fdes);
        fread(&pinkE, sizeof(float), 1, fdes);
		if (N_pink > 1)
			fseek(fdes, baseSkipAmount*(N_pink-1), SEEK_CUR); // not interested in other interactions
        
        if (params->listModeDataIsWeighted == true)
            fread(&curWeight, sizeof(float), 1, fdes);
        
		if (params->listModeUnits == parameters::CM)
		{
			bluePos[0] *= 10.0;
			bluePos[1] *= 10.0;
			bluePos[2] *= 10.0;
			pinkPos[0] *= 10.0;
			pinkPos[1] *= 10.0;
			pinkPos[2] *= 10.0;
		}

        r[0] = bluePos[0] - pinkPos[0];
        r[1] = bluePos[1] - pinkPos[1];
        r[2] = bluePos[2] - pinkPos[2];

        float v1, v0;
        for (int i = 0; i < num_planograms; i++)
        {
            planogram* plan = params->planogramSet[i];
            if (plan->is_within_acceptable_angle(r, v1, v0))
            {
                float u1, u0;
                plan->get_u_coords(bluePos, v1, v0, u1, u0);
                if (params->listModeOrigin == parameters::CHEST_WALL)
                {
                    u1 -= plan->H;
                }
                if (!queues[i]->insert(v1, v0, u1, u0))
                    flush_queue(i);

                total_binned_events += 1;
            }
        }
    }
    fclose(fdes);

    for (int i = 0; i < num_planograms; i++)
        flush_queue(i);
	
    printf("planogramSet::binSimSETdata: elapsed time: %d seconds\n", int(time(NULL)-startTime));
    
    return true;
}

bool binning::binAcqdata(char* Acqfile)
{
	if (params->listModeOrigin == parameters::CHEST_WALL)
		printf("assuming z=0 plane of the list mode data is at the chest wall\n");
	else
		printf("assuming z=0 plane of the list mode data is at the center of the field of view\n");

	if (params->listModeUnits == parameters::CM)
		printf("assuming list mode data is in units of cm\n");
	else
		printf("assuming list mode data is in units of mm\n");

	string fullPath_str = Acqfile;

	if (hasEnding(fullPath_str, ".lst"))
		return read_xyzlist(Acqfile);

	time_t startTime = time(NULL);

	FILE* fdes = NULL;
	if ((fdes = fopen(Acqfile, "rb")) == NULL)
	{
		printf("planogramSet::binAcqdata: xyz list data file: %s not found, quitting!\n", Acqfile);
        return false;
		//exit(1);
	}

	float p[3];
	float r[3];
	long N;
	short N_blue, N_pink;
	float bluePos[3];
	float pinkPos[3];
	float blueE, pinkE;
	float curWeight = 1.0;
	int baseSkipAmount = sizeof(short) + 4 * sizeof(float);

    long total_binned_events = 0;

	//fread(&N_dims, sizeof(int), 1, fdes);
	fread(&N, sizeof(long), 1, fdes);
	printf("planogramSet::binAcqdata: Binning %ld coincidences...\n", N);
	for (int i = 0; i < N; i++)
	{
		// Read blue
		fread(&N_blue, sizeof(short), 1, fdes);
		fread(&bluePos[0], sizeof(float), 3, fdes);
		fread(&blueE, sizeof(float), 1, fdes);
		if (N_blue > 1)
			fseek(fdes, baseSkipAmount * (N_blue - 1), SEEK_CUR); // not interested in other interactions

		// Read pink
		fread(&N_pink, sizeof(short), 1, fdes);
		fread(&pinkPos[0], sizeof(float), 3, fdes);
		fread(&pinkE, sizeof(float), 1, fdes);
		if (N_pink > 1)
			fseek(fdes, baseSkipAmount * (N_pink - 1), SEEK_CUR); // not interested in other interactions

		if (params->listModeDataIsWeighted == true)
			fread(&curWeight, sizeof(float), 1, fdes);

		if (params->listModeUnits == parameters::CM)
		{
			bluePos[0] *= 10.0;
			bluePos[1] *= 10.0;
			bluePos[2] *= 10.0;
			pinkPos[0] *= 10.0;
			pinkPos[1] *= 10.0;
			pinkPos[2] *= 10.0;
		}

        r[0] = bluePos[0] - pinkPos[0];
		r[1] = bluePos[1] - pinkPos[1];
		r[2] = bluePos[2] - pinkPos[2];

        float v1, v0;
        for (int i = 0; i < num_planograms; i++)
        {
            planogram* plan = params->planogramSet[i];
            if (plan->is_within_acceptable_angle(r, v1, v0))
            {
                float u1, u0;
                plan->get_u_coords(bluePos, v1, v0, u1, u0);
                if (params->listModeOrigin == parameters::CHEST_WALL)
                {
                    u1 -= plan->H;
                }
                if (!queues[i]->insert(v1, v0, u1, u0))
                    flush_queue(i);

                total_binned_events += 1;
            }
        }
	}
	fclose(fdes);

    for (int i = 0; i < num_planograms; i++)
        flush_queue(i);

	printf("planogramSet::binAcqdata: elapsed time: %d seconds\n", int(time(NULL) - startTime));

	return true;
}

bool binning::flush_queue(int i)
{
    listModeQueue* queue = queues[i];
    if (queue->numEvents > 0)
    {
        planogram* plan = params->planogramSet[i];
        uint64 proj_size = uint64(plan->N_u1) * uint64(plan->N_u0);
        uint64 oblique_size = uint64(plan->N_v0) * proj_size;

        omp_set_num_threads(omp_get_num_procs());
        #pragma omp parallel for
        for (int iv1 = 0; iv1 < plan->N_v1; iv1++)
        {
            int v0_ind, u1_ind, u0_ind;
            float iu1_val, iu0_val;
            int iu1_lo, iu1_hi;
            int iu0_lo, iu0_hi;
            float du1, du0;

            float* oblique_projection = nullptr;
            for (int ievent = 0; ievent < queue->numEvents; ievent++)
            {
                float* event = queue->events[ievent];
                float iv1_val = plan->v1_inv(event[0]);

                if (int(iv1_val+0.5) == iv1)
                {
                    if (oblique_projection == nullptr)
                        oblique_projection  = &(plan->data[uint64(iv1)*oblique_size]);
                    v0_ind = int(plan->v0_inv(event[1])+0.5);
                    if (0 <= v0_ind && v0_ind < plan->N_v0)
                    {
                        float* proj = &oblique_projection[uint64(v0_ind)*proj_size];
                        if (params->binningType == 0) // nearest neighbor
                        {
                            u1_ind = int(plan->u1_inv(event[2])+0.5);
                            u0_ind = int(plan->u0_inv(event[3])+0.5);
                            if (u1_ind >= 0 && u1_ind < plan->N_u1 && u0_ind >= 0 && u0_ind < plan->N_u0)
                                proj[uint64(u1_ind*plan->N_u0 + u0_ind)] += event[4];
                        }
                        else // linear interpolation
                        {
                            iu1_val = plan->u1_inv(event[2]);
                            if (iu1_val <= 0.0)
                            {
                                iu1_lo = 0;
                                iu1_hi = 0;
                                du1 = 0.0;
                            }
                            else if (iu1_val >= plan->N_u1-1)
                            {
                                iu1_lo = plan->N_u1-1;
                                iu1_hi = iu1_lo;
                                du1 = 0.0;
                            }
                            else
                            {
                                iu1_lo = int(iu1_val);
                                iu1_hi = iu1_lo + 1;
                                du1 = iu1_val - float(iu1_lo);
                            }

                            iu0_val = plan->u0_inv(event[3]);
                            if (iu0_val <= 0.0)
                            {
                                iu0_lo = 0;
                                iu0_hi = 0;
                                du0 = 0.0;
                            }
                            else if (iu0_val >= plan->N_u0-1)
                            {
                                iu0_lo = plan->N_u0-1;
                                iu0_hi = iu0_lo;
                                du0 = 0.0;
                            }
                            else
                            {
                                iu0_lo = int(iu0_val);
                                iu0_hi = iu0_lo + 1;
                                du0 = iu0_val - float(iu0_lo);
                            }
                            
                            proj[iu1_lo*plan->N_u0 + iu0_lo] += (1.0-du1)*(1.0-du0)*event[4];
                            proj[iu1_hi*plan->N_u0 + iu0_lo] += du1*(1.0-du0)*event[4];
                            proj[iu1_lo*plan->N_u0 + iu0_hi] += (1.0-du1)*du0*event[4];
                            proj[iu1_hi*plan->N_u0 + iu0_hi] += du1*du0*event[4];
                        }
                    }
                }

                /*
                if (params->binningType == 0) // nearest neighbor
                {
                    if (int(iv1_val+0.5) == iv1)
                    {
                        if (oblique_projection == nullptr)
                            oblique_projection  = &(plan->data[uint64(iv1)*oblique_size]);

                        v0_ind = int(plan->v0_inv(event[1])+0.5);
                        u1_ind = int(plan->u1_inv(event[2])+0.5);
                        u0_ind = int(plan->u0_inv(event[3])+0.5);
                        if (v0_ind >= 0 && v0_ind < plan->N_v0 && u1_ind >= 0 && u1_ind < plan->N_u1 && u0_ind >= 0 && u0_ind < plan->N_u0)
                            oblique_projection[uint64(v0_ind)*proj_size + uint64(u1_ind*plan->N_u0 + u0_ind)] += event[4];
                    }
                }
                else // linear interpolation
                {
                    if (fabs(iv1_val - float(iv1)) < 1.0)
                    {
                        if (oblique_projection == nullptr)
                            oblique_projection  = &(plan->data[uint64(iv1)*oblique_size]);

                        v1_ind = int(v1_val); dv1 = v1_val - double(v1_ind); dv1 = max(0.0, min(dv1, 1.0));
                        v0_ind = int(v0_val); dv0 = v0_val - double(v0_ind); dv0 = max(0.0, min(dv0, 1.0));
                        u1_ind = int(u1_inv(u1_in, v1_ind) + 0.5);
                        u0_ind = int(u0_inv(u0_in, v0_ind) + 0.5);
                        if (v1_ind >= 0 && v1_ind < N_v1 && v0_ind >= 0 && v0_ind < N_v0 && u1_ind >= 0 && u1_ind < N_u1[v1_ind] && u0_ind >= 0 && u0_ind < N_u0[v0_ind])
                        {
                            data[v1_ind][v0_ind][u1_ind][u0_ind] += (1.0 - dv1) * (1.0 - dv0) * theWeight;
                            retVal = true;
                        }
                    }
                }
                //*/
            }
        }
    }
    queue->reset();
    return true;
}
