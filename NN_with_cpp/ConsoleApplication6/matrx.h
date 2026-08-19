#pragma once
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>

template <class T>
class Matrix
{
public:
	Matrix() : rows(0), cols(0) {}
	Matrix(int r, int c) { resize(r, c); }

	void resize(int r, int c)
	{
		rows = r;
		cols = c;
		data.assign(r * c, T());
	}

	void fill(T v)
	{
		for (size_t i = 0; i < data.size(); i++)
			data[i] = v;
	}

	void randomize(double range)
	{
		for (size_t i = 0; i < data.size(); i++)
			data[i] = T(((double)rand() / RAND_MAX) * 2 * range - range);
	}

	void put(int r, int c, T v) { data[r * cols + c] = v; }
	T get(int r, int c) const { return data[r * cols + c]; }

	int getrows() const { return rows; }
	int getcols() const { return cols; }

	Matrix<T> dot(const Matrix<T>& b) const
	{
		Matrix<T> out(rows, b.cols);
		for (int i = 0; i < rows; i++)
			for (int k = 0; k < cols; k++)
			{
				T v = data[i * cols + k];
				for (int j = 0; j < b.cols; j++)
					out.data[i * b.cols + j] += v * b.data[k * b.cols + j];
			}
		return out;
	}

	Matrix<T> transpose() const
	{
		Matrix<T> out(cols, rows);
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
				out.data[j * rows + i] = data[i * cols + j];
		return out;
	}

	Matrix<T> operator+(const Matrix<T>& b) const
	{
		Matrix<T> out(rows, cols);
		for (size_t i = 0; i < data.size(); i++)
			out.data[i] = data[i] + b.data[i];
		return out;
	}

	Matrix<T> operator-(const Matrix<T>& b) const
	{
		Matrix<T> out(rows, cols);
		for (size_t i = 0; i < data.size(); i++)
			out.data[i] = data[i] - b.data[i];
		return out;
	}

	Matrix<T> mul(const Matrix<T>& b) const
	{
		Matrix<T> out(rows, cols);
		for (size_t i = 0; i < data.size(); i++)
			out.data[i] = data[i] * b.data[i];
		return out;
	}

	Matrix<T> scale(T k) const
	{
		Matrix<T> out(rows, cols);
		for (size_t i = 0; i < data.size(); i++)
			out.data[i] = data[i] * k;
		return out;
	}

	Matrix<T> sigmoid() const
	{
		Matrix<T> out(rows, cols);
		for (size_t i = 0; i < data.size(); i++)
			out.data[i] = T(1) / (T(1) + exp(-data[i]));
		return out;
	}

	// takes the sigmoid output, not the pre-activation
	Matrix<T> sigmoidprime() const
	{
		Matrix<T> out(rows, cols);
		for (size_t i = 0; i < data.size(); i++)
			out.data[i] = data[i] * (T(1) - data[i]);
		return out;
	}

	int argmax() const
	{
		int best = 0;
		for (size_t i = 1; i < data.size(); i++)
			if (data[i] > data[best])
				best = (int)i;
		return best;
	}

	void printm() const
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
				std::cout << data[i * cols + j] << "  ";
			std::cout << std::endl;
		}
	}

private:
	std::vector<T> data;
	int rows, cols;
};
