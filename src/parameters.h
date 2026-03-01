#ifndef __PARAMETERS_H
#define __PARAMETERS_H

#ifdef WIN32
#pragma once
#endif

#include "petl_defines.h"
#include "planogram.h"
#include <vector>

/**
 *  parameters class
 * This class tracks all the PETL parameters including: the PET geometry parameters, PET volume parameters, which GPUs to use, etc.
 * A pointer to an instance of this class is usually passed with the input/output arrays so that the algorithms are aware of the
 * CT parameters.  For dividing parameters into chunks, usually copies of the original object are made and certain parameters
 * updated to hande the sub-job.
 */

class parameters
{
public:
	parameters();
    parameters(const parameters& other);
	~parameters();

	/**
	 * \fn          operator =
	 * \brief       makes a deep copy of the given parameter object
	 */
    parameters& operator = (const parameters& other);

	/**
	 * \fn          initialize
	 * \brief       initialize all CT geometry and CT volume parameter values
	 */
	void initialize();

	/**
	 * \fn          printAll
	 * \brief       prints all CT geometry and CT volume parameter values
	 */
	void printAll();

	/**
	 * \fn          clearAll
	 * \brief       clears all (including memory) CT geometry and CT volume parameter values
	 */
	void clearAll();

	void clearPlanograms();

    /**
	 * \fn          assign
	 * \brief       makes a deep copy of the given parameter object
	 */
	void assign(const parameters& other, float** g = nullptr);

	/**
	 * \fn          allDefined
	 * \brief       returns whether all PET geometry and PET volume parameter values are defined and valid
	 * \return      returns true if all PET geometry and PET volume parameter values are defined and valid, false otherwise
	 */
	bool allDefined(bool doPrint = true);

	/**
	 * \fn          geometryDefined
	 * \brief       returns whether all PET geometry parameter values are defined and valid
	 * \return      returns true if all PET geometry parameter values are defined and valid, false otherwise
	 */
	bool geometryDefined(bool doPrint = true);

	/**
	 * \fn          volumeDefined
	 * \brief       returns whether all PET volume parameter values are defined and valid
	 * \return      returns true if all PET volume parameter values are defined and valid, false otherwise
	 */
	bool volumeDefined(bool doPrint = true);

	int num_planograms();

	bool add_planogram(float psi, float R, float L, float H, float v_m0, float v_m1, float T);
	bool remove_planogram(int);
	bool keep_only_planogram(int);

	bool set_default_volume(float scale = 1.0);
	bool set_volume(int numX_in, int numY_in, int numZ_in, float voxelWidth_in, float voxelHeight_in, float offsetX_in = 0.0, float offsetY_in = 0.0, float offsetZ_in = 0.0);

	float x_0();
	float y_0();
	float z_0();

	uint64 projectionData_numberOfElements(int iplan = 0);
	uint64 volumeData_numberOfElements();

    std::vector<planogram*> planogramSet;

    // Volume Parameters
	int numX, numY, numZ;
	float voxelWidth, voxelHeight;
	float offsetX, offsetY, offsetZ;

	int whichGPU;
	std::vector<int> whichGPUs;

	// binning parameters
	int listModeUnits;
	enum listModeUnits_list {MM,CM};
	int listModeOrigin;
	enum listModeOrigin_list {CHEST_WALL,FOV_CENTER};
	int binningType;
	enum binningType_list {ALL=0,PRIMARY=1,SCATTER=2,FIRST_ORDER_SCATTER=3};
	bool listModeDataIsWeighted;
	int binningOrder;
};

#endif
