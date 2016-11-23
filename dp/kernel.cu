#include "stdafx.h"
#include <vector>
#include <string>
#include <iostream>
#include "fstream"
#include "iomanip" 
#include "traj.h"
#include "dp.h"
#include "proj_api.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>
#define TRAJ_NUM 10
#define POINTSUM 11731
//#define POINTSUM 120296
//#define POINTSUM 244982
//#define POINTSUM 359336
//#define POINTSUM 477684
//#define POINTSUM 605316
using namespace std;

void split(std::string& s, std::string& delim,std::vector< std::string >* ret)  
{  
	size_t last = 0;  
	size_t index=s.find_first_of(delim,last);  
	while (index!=std::string::npos)  
	{  
		ret->push_back(s.substr(last,index-last));  
		last=index+1;  
		index=s.find_first_of(delim,last);  
	}  
	if (index-last>0)  
	{  
		ret->push_back(s.substr(last,index-last));  
	}  
} 
__global__ void dp_kenel(float *x,float *y,bool *flag,int *count)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	int k = 0;
	for (int j = 0;j < i;j++)
		k+=count[j];
	DP(x,y,flag,k,k + count[i] - 1,50);
}

int main(void) {

	projPJ pj_merc, pj_latlong;

	if (!(pj_merc = pj_init_plus("+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")))
		exit(1);
	if (!(pj_latlong = pj_init_plus("+proj=longlat +datum=WGS84 +no_defs")))
		exit(1);

	ifstream fin("C:\\Users\\Constantine\\Desktop\\out1.txt",ios::in);
	if(!fin)
	{
		cout<<"Cannot open input file!"<<endl;
		system("pause");
		return 1;
	}
	
	float x[POINTSUM] = {0.0};
	float y[POINTSUM] = {0.0};
	bool flag[POINTSUM] = {false};
	int traj_num[TRAJ_NUM];
	vector<std::string> vec_id;

	string line;
	string delim=" ";
	int trajIdx = 0;
	int point_Idx = 0;
	while(getline(fin,line)) {

		vector<string> result;
		split(line,delim,&result);

		vec_id.push_back(result.at(0).c_str());
		traj_num[trajIdx] = atoi(result.at(1).c_str());
		
		for ( vector<string>::iterator it = result.begin()+2;it != result.end(); it++) {

			double lng = atof(&(*it->c_str()));
			it++;
			double lat = atof(&(*it->c_str()));

			lat *= DEG_TO_RAD;
			lng *= DEG_TO_RAD;
			pj_transform(pj_latlong, pj_merc, 1, 1, &lng, &lat, NULL);

			
			x[point_Idx] = lng;
			y[point_Idx] = lat;
			point_Idx++;
		}
		trajIdx++;
		if (trajIdx == 10)
		    break;
	}
	fin.close();

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	float *d_x ;
	float *d_y ;
    bool *d_flag;
	int *d_count;

	cudaStatus = cudaMalloc((void**)&d_x, POINTSUM * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_y, POINTSUM * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_flag, POINTSUM * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&d_count, TRAJ_NUM * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_x, x, POINTSUM * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}
	cudaStatus = cudaMemcpy(d_y, y, POINTSUM * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}
	cudaStatus = cudaMemcpy(d_flag, flag, POINTSUM * sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}
	cudaStatus = cudaMemcpy(d_count, traj_num, TRAJ_NUM * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	clock_t starttime = clock();
	dp_kenel<<<2,5>>>(d_x,d_y,d_flag,d_count);   //ÔËÐÐºËº¯Êý
	cudaDeviceSynchronize();
	clock_t endtime = clock();

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	float o_x[POINTSUM] ={0.0};
	float o_y[POINTSUM] ={0.0};
	bool o_flag[POINTSUM] = {false};

	cudaStatus = cudaMemcpy(o_x, d_x, POINTSUM * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}
	cudaStatus = cudaMemcpy(o_y, d_y, POINTSUM * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}
	cudaStatus = cudaMemcpy(o_flag, d_flag, POINTSUM * sizeof(bool), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}
Error:
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_flag);
	cudaFree(d_count);
	cudaFree(o_x);
	cudaFree(o_y);
	cudaFree(o_flag);

	/*int simple_Num = 0;
	for (int i=0;i< POINTSUM;i++)
	{
	if (o_flag[i]==true)
	{
	cout<<o_y[i]<<endl;
	simple_Num++;
	}
	}
	cout << simple_Num << endl;*/

	ofstream fout("C:\\Users\\Constantine\\Desktop\\10.txt",ios::out);
	int j = 0;
	int i ;
	int k = 0;
	for (i = 0;i<10;i++)
	{
		for(;j<traj_num[i]+k;j++)
		{
			if(o_flag[j] == true)
			{
				double x = (double)o_x[j];
				double y = (double)o_y[j];
				pj_transform(pj_merc, pj_latlong, 1, 1, &x, &y, NULL);
				x /= DEG_TO_RAD;
				y /= DEG_TO_RAD;
				fout<<fixed<<setprecision(6)<<vec_id.at(i)<<","<<x<<","<<y<<endl;
			}
		}
		k = j;
	}
	fout.close();
		
	printf("totals time is %lf s\n",(double)(endtime-starttime)/CLOCKS_PER_SEC);
	system("pause");
}