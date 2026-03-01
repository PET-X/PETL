//#include "pch.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#include "allocate.h"

void* SwapEndian(void* Addr, const int Nb)
{
	static char Swapped[16];
	switch (Nb)
	{
		case 2:	Swapped[0]=*((char*)Addr+1);
				Swapped[1]=*((char*)Addr  );
				break;
		case 4:	Swapped[0]=*((char*)Addr+3);
				Swapped[1]=*((char*)Addr+2);
				Swapped[2]=*((char*)Addr+1);
				Swapped[3]=*((char*)Addr  );
				break;
		case 8:	Swapped[0]=*((char*)Addr+7);
				Swapped[1]=*((char*)Addr+6);
				Swapped[2]=*((char*)Addr+5);
				Swapped[3]=*((char*)Addr+4);
				Swapped[4]=*((char*)Addr+3);
				Swapped[5]=*((char*)Addr+2);
				Swapped[6]=*((char*)Addr+1);
				Swapped[7]=*((char*)Addr  );
				break;
		case 16:Swapped[0]=*((char*)Addr+15);
				Swapped[1]=*((char*)Addr+14);
				Swapped[2]=*((char*)Addr+13);
				Swapped[3]=*((char*)Addr+12);
				Swapped[4]=*((char*)Addr+11);
				Swapped[5]=*((char*)Addr+10);
				Swapped[6]=*((char*)Addr+9);
				Swapped[7]=*((char*)Addr+8);
				Swapped[8]=*((char*)Addr+7);
				Swapped[9]=*((char*)Addr+6);
				Swapped[10]=*((char*)Addr+5);
				Swapped[11]=*((char*)Addr+4);
				Swapped[12]=*((char*)Addr+3);
				Swapped[13]=*((char*)Addr+2);
				Swapped[14]=*((char*)Addr+1);
				Swapped[15]=*((char*)Addr  );
				break;
	}
	return (void*)Swapped;
}

void *get_spc(int num, size_t size)
{
	void *pt;

	if( (pt=calloc((size_t)num,size)) == NULL ) {
		fprintf(stderr, "==> calloc() error\n");
		exit(-1);
		}
	return(pt);
}

void *mget_spc(int num,size_t size)
{
	void *pt;

	if( (pt=malloc((size_t)(num*size))) == NULL ) {
		fprintf(stderr, "==> malloc() error\n");
		exit(-1);
		}
	return(pt);
}


void **get_img(int wd,int ht,size_t size)
{
	void  *pt;

	if( (pt=multialloc(size,2,ht,wd))==NULL) {
          fprintf(stderr, "get_img: out of memory\n");
          exit(-1);
          }
	return((void **)pt);
}

void ***get_3D(int N, int M, int A, size_t size)
{
	void  *pt;

	if( (pt=multialloc(size,3,N,M,A))==NULL) {
          fprintf(stderr, "get_3D: out of memory\n");
          exit(-1);
          }
	return((void ***)pt);
}


void free_img(void **pt)
{
	multifree((void *)pt,2);
}

void free_3D(void ***pt)
{
	multifree((void *)pt,3);
}



/* modified from dynamem.c on 4/29/91 C. Bouman                           */
/* Converted to ANSI on 7/13/93 C. Bouman         	                  */
/* multialloc( s, d,  d1, d2 ....) allocates a d dimensional array, whose */
/* dimensions are stored in a list starting at d1. Each array element is  */
/* of size s.                                                             */



void* multialloc(size_t s, int N_dims, ...)
{
	if (N_dims == 0 || N_dims > 5 || s == 0)
		return NULL;

	va_list ap;
	va_start(ap,N_dims);
	int* Ns = (int*) malloc(sizeof(int)*N_dims);
	for(int i = 0; i < N_dims; i++)
    	Ns[i] = va_arg(ap,int);

	void* retVal = NULL;
	if (N_dims == 1)
	{
		retVal = get_spc(Ns[0], s);
	}
	else if (N_dims == 2)
	{
		void** data = (void**) malloc(sizeof(void*)*Ns[0]);
		if (data == NULL)
			return NULL;
		for (int i = 0; i < Ns[0]; i++)
		{
			data[i] = get_spc(Ns[1], s);
			if (data[i] == NULL)
			{
				multifree(data, 2, Ns[0]);
				return NULL;
			}
		}
		retVal = (void*) data;
	}
	else if (N_dims == 3)
	{
		void*** data = (void***) malloc(sizeof(void**)*Ns[0]);
		if (data == NULL)
			return NULL;

        #ifdef MULTIALLOC_WITH_OMP
        for (int i = 0; i < Ns[0]; i++)
        {
            data[i] = (void**) malloc(sizeof(void*)*Ns[1]);
            void** dataSlice = data[i];
            if (dataSlice == NULL)
            {
                multifree(data, 3, Ns[0], Ns[1]);
                return NULL;
            }
            //omp_set_num_threads(std::min(omp_get_num_procs(), 8));
            //#pragma omp parallel for
            for (int j = 0; j < Ns[1]; j++)
            {
                //dataSlice[j] = get_spc(Ns[2], s);
                dataSlice[j] = mget_spc(Ns[2], s);
                //*
                if (dataSlice[j] == NULL)
                {
                    multifree(data, 3, Ns[0], Ns[1]);
                    return NULL;
                }
                //*/
                //memset(dataSlice[j], 0, s * Ns[2]);
            }
        }
		//*
		//printf("HERE!\n");
		omp_set_num_threads(std::min(omp_get_num_procs(), 8));
		#pragma omp parallel for
		for (int i = 0; i < Ns[0]; i++)
		{
			void** dataSlice = data[i];
			for (int j = 0; j < Ns[1]; j++)
			{
				void* dataLine = dataSlice[j];
				memset(dataLine, 0, s * Ns[2]);
			}
		}
		//*/
        retVal = (void*) data;
        #else
        //double count = 0;
        for (int i = 0; i < Ns[0]; i++)
        {
            data[i] = (void**) malloc(sizeof(void*)*Ns[1]);
            void** dataSlice = data[i];
            if (dataSlice == NULL)
            {
                multifree(data, 3, Ns[0], Ns[1]);
                return NULL;
            }
            for (int j = 0; j < Ns[1]; j++)
            {
                //dataSlice[j] = mget_spc(Ns[2], s);
				dataSlice[j] = get_spc(Ns[2], s);
                //count += Ns[2];
                if (dataSlice[j] == NULL)
                {
                    multifree(data, 3, Ns[0], Ns[1]);
                    return NULL;
                }
            }
        }
        retVal = (void*) data;
        //printf("MB allocated: %f\n", 4.0*count / pow(2.0,20.0));
        #endif
	}
	else if (N_dims == 4)
	{
		void**** data = (void****) malloc(sizeof(void***)*Ns[0]);
		if (data == NULL)
			return NULL;
		for (int i = 0; i < Ns[0]; i++)
		{
			data[i] = (void***) malloc(sizeof(void**)*Ns[1]);
			if (data[i] == NULL)
			{
				multifree(data, 4, Ns[0], Ns[1], Ns[2]);
				return NULL;
			}
			for (int j = 0; j < Ns[1]; j++)
			{
				data[i][j] = (void**) malloc(sizeof(void*)*Ns[2]);
				if (data[i][j] == NULL)
				{
					multifree(data, 4, Ns[0], Ns[1], Ns[2]);
					return NULL;
				}
				for (int k = 0; k < Ns[2]; k++)
				{
					data[i][j][k] = get_spc(Ns[3], s);
					if (data[i][j][k] == NULL)
					{
						multifree(data, 4, Ns[0], Ns[1], Ns[2]);
						return NULL;
					}
				}
			}
		}
		retVal = (void*) data;
	}
	else if (N_dims == 5)
	{
		void***** data = (void*****) malloc(sizeof(void****)*Ns[0]);
		if (data == NULL)
			return NULL;
		for (int i = 0; i < Ns[0]; i++)
		{
			data[i] = (void****) malloc(sizeof(void***)*Ns[1]);
			if (data[i] == NULL)
			{
				multifree(data, 5, Ns[0], Ns[1], Ns[2], Ns[3]);
				return NULL;
			}
			for (int j = 0; j < Ns[1]; j++)
			{
				data[i][j] = (void***) malloc(sizeof(void**)*Ns[2]);
				if (data[i][j] == NULL)
				{
					multifree(data, 5, Ns[0], Ns[1], Ns[2], Ns[3]);
					return NULL;
				}
				for (int k = 0; k < Ns[2]; k++)
				{
					data[i][j][k] = (void**) malloc(sizeof(void*)*Ns[3]);
					if (data[i][j][k] == NULL)
					{
						multifree(data, 5, Ns[0], Ns[1], Ns[2], Ns[3]);
						return NULL;
					}
					for (int l = 0; l < Ns[3]; l++)
					{
						data[i][j][k][l] = get_spc(Ns[4], s);
						if (data[i][j][k][l] == NULL)
						{
							multifree(data, 5, Ns[0], Ns[1], Ns[2], Ns[3]);
							return NULL;
						}
					}
				}
			}
		}
		retVal = (void*) data;
	}
	free(Ns);

	return retVal;
}

bool multifree(void* data, int N_dims, ...)
{
	if (N_dims == 0 || N_dims > 5 || data == NULL)
		return NULL;

	va_list ap;
	va_start(ap,N_dims);
	int* Ns = (int*) malloc(sizeof(int)*N_dims);
	for(int i = 0; i < N_dims-1; i++)
    	Ns[i] = va_arg(ap,int);

	if (N_dims == 1)
	{
		free(data);
	}
	else if (N_dims == 2)
	{
		void** data2D = (void**) data;
		for (int i = 0; i < Ns[0]; i++)
		{
			if (data2D[i] != NULL)
				free(data2D[i]);
			data2D[i] = NULL;
		}
		free(data2D);
	}
	else if (N_dims == 3)
	{
		//long count = 0;
		void*** data3D = (void***) data;
        //omp_set_num_threads(std::min(omp_get_num_procs(), 8));
        //#pragma omp parallel for
		for (int i = 0; i < Ns[0]; i++)
		{
			void** data2D = data3D[i];
			if (data2D != NULL)
			{
				for (int j = 0; j < Ns[1]; j++)
				{
					if (data2D[j] != NULL)
					{
						free(data2D[j]);
						//count += 1;
					}
					//data2D[j] = NULL;
				}
				free(data3D[i]);
			}
			//data3D[i] = NULL;
		}
		free(data3D);
		//printf("total units free: %ld\n", count);
	}
	else if (N_dims == 4)
	{
		void**** data4D = (void****) data;
		for (int i = 0; i < Ns[0]; i++)
		{
			if (data4D[i] != NULL)
			{
				for (int j = 0; j < Ns[1]; j++)
				{
					if (data4D[i][j] != NULL)
					{
						for (int k = 0; k < Ns[2]; k++)
						{
							if (data4D[i][j][k] != NULL)
								free(data4D[i][j][k]);
							data4D[i][j][k] = NULL;
						}
						free(data4D[i][j]);
					}
					data4D[i][j] = NULL;
				}
				free(data4D[i]);
			}
			data4D[i] = NULL;
		}
		free(data4D);
	}
	else if (N_dims == 5)
	{
		void***** data5D = (void*****) data;
		for (int i = 0; i < Ns[0]; i++)
		{
			if (data5D[i] != NULL)
			{
				for (int j = 0; j < Ns[1]; j++)
				{
					if (data5D[i][j] != NULL)
					{
						for (int k = 0; k < Ns[2]; k++)
						{
							if (data5D[i][j][k] != NULL)
							{
								for (int l = 0; l < Ns[3]; l++)
								{
									if (data5D[i][j][k][l] != NULL)
										free(data5D[i][j][k][l]);
									data5D[i][j][k][l] = NULL;
								}
								free(data5D[i][j][k]);
							}
							data5D[i][j][k] = NULL;
						}
						free(data5D[i][j]);
					}
					data5D[i][j] = NULL;
				}
				free(data5D[i]);
			}
			data5D[i] = NULL;
		}
		free(data5D);
	}
	data = NULL;
	free(Ns);
	
	return true;
}


