#pragma once
#ifndef TRAJ_H
#define TRAJ_H
class Traj
{
public:
	int id;
	int point_Num;
	float* x;
	float* y;
	bool* flag;

public:
	Traj(void);
	~Traj(void);
};
#endif