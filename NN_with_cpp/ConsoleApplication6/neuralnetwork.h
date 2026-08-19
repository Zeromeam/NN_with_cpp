#pragma once
#include "matrx.h"

class neuralnetwork
{
public:
	neuralnetwork(int inputnodes, int hiddennodes, int outputnodes, double learningrate);

	void train(const Matrix<double>& inputs, const Matrix<double>& targets);
	Matrix<double> query(const Matrix<double>& inputs) const;

	int inodes, hnodes, onodes;
	double lr;
	Matrix<double> wih, who;
};
