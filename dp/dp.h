#pragma once
#ifndef DP_H
#define DP_H
#include <math.h>
__device__ double  perpendicularDistance(float x,float y,float start_x,float start_y,float end_x,float end_y) {


	double h = 0.0;
	double a = sqrt(pow(y-start_y,2)+pow(x-start_x,2));
	double b = sqrt(pow(y-end_y,2)+pow(x-end_x,2));
	double c = sqrt(pow(start_y-end_y,2)+pow(start_x-end_x,2));
	double p = (a+b+c)/2.0;
	double s = sqrt(p*(p-a)*(p-b)*(p-c));
	h = 2*s/c;
	return h;
}


__device__ void  DP(float* x,float* y,bool* flag,int start,int end,float tol) {

	flag[start] = true;
	flag[end] = true;

	float dmax = 0.0;
	float dtmp = 0.0;

	long index;

	int i;
	for (i=start+1;i<end;i++) {
		dtmp = perpendicularDistance( x[i],y[i],x[start],y[start],x[end],y[end] );
		if (dtmp > dmax) {
			dmax = dtmp;
			index = i;
		}
	}
	//std::cout<< dmax <<',' << tol<< std::endl;

	if (dmax < tol) {
	}

	else {
		flag[index]  = true;
		DP(x,y,flag,start,index,tol);
		//DP(x,y,flag,index+1,end,tol);
		DP(x,y,flag,index,end,tol);
	}

}
#endif