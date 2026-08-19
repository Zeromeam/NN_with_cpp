#include "neuralnetwork.h"

neuralnetwork::neuralnetwork(int inputnodes, int hiddennodes, int outputnodes, double learningrate)
{
	inodes = inputnodes;
	hnodes = hiddennodes;
	onodes = outputnodes;
	lr = learningrate;

	wih.resize(inodes, hnodes);
	who.resize(hnodes, onodes);
	wih.randomize(pow((double)inodes, -0.5));
	who.randomize(pow((double)hnodes, -0.5));
}

void neuralnetwork::train(const Matrix<double>& inputs, const Matrix<double>& targets)
{
	Matrix<double> hidden_outputs = inputs.dot(wih).sigmoid();
	Matrix<double> final_outputs = hidden_outputs.dot(who).sigmoid();

	Matrix<double> output_errors = targets - final_outputs;
	Matrix<double> hidden_errors = output_errors.dot(who.transpose());

	who = who + hidden_outputs.transpose()
		.dot(output_errors.mul(final_outputs.sigmoidprime()))
		.scale(lr);

	wih = wih + inputs.transpose()
		.dot(hidden_errors.mul(hidden_outputs.sigmoidprime()))
		.scale(lr);
}

Matrix<double> neuralnetwork::query(const Matrix<double>& inputs) const
{
	return inputs.dot(wih).sigmoid().dot(who).sigmoid();
}
