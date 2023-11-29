// ConsoleApplication6.cpp : Defines the entry point for the console application.
//
#include "iostream"
#include "windows.h"
#include "vector"
#include "neuralnetwork.h"
#include <fstream>
#include <sstream>
using namespace::std;
void draw(vector<double>rgb)
{

		HDC hdc = GetDC(GetConsoleWindow());
		for (int y = 0; y < 280; ++y)
			for (int x = 0; x < 280; ++x)\
			{
				if (x == 0 && y == 0)
					continue;
				SetPixel(hdc, x, y, RGB(rgb[(y / 10) * 28 + x/10], rgb[(y / 10) * 28 + x/10], rgb[(y / 10) * 28 + x/10]));
			}
}
Matrix<double> rescale(vector<double> mat)
{
	Matrix<double> t;
	t.resize(1, mat.size() - 1);
	for (int i = 1; i < mat.size(); i++)
		t.put(0, i-1, ((mat[i] / 255)*.99) + .01);
	return t;
}
int main()
{
	int val;
	vector<string> row;
	string line,word;
	Matrix <double> a(1,10);
	a.fill(.01);
	vector<vector<double>> cl;
	vector <double > tem;
	// open the file 
	fstream myfile;
	myfile.open("mnist_test.csv");
	getline(myfile, line);
	int count = 0;
	vector<Matrix<double>>re;
	vector<Matrix<double>>out;
	while (getline(myfile, line, '\n'))
	{
		istringstream templine(line);
		string data;
		while (getline(templine, data, ','))
		{
			tem.push_back(atof(data.c_str()));
		}
		cl.push_back(tem);
		re.push_back(a);
		a.put(0, int(cl[count][0]), .99);
		out.push_back(a);
		re[count] = rescale(cl[count]);
		count++;
		tem.clear();
		if (!(count % 100))
			cout << count << "  " << endl;
		a.fill(.01);
	}
	// neralnetwork
	cout << "installed\n";
	system("pause");
	neuralnetwork n(784, 100, 10, 0.3);
	for (int i = 0; i < count; i++)
	{
		n.train(re[i], out[i]);
		if (!(i % 10))
			cout << i;
	}
	cout << "\n\n continue \n ";
	system("pause");
	// test
	char end = 0;
	int testn = 0;
	while (!(end == 'y'))
	{
		cout << "\n the targt    " << cl[testn][0] <<endl;
		n.query(re[testn]).printm();
		cin >> end;
		testn++;
	}
	system("pause");
	return 0;
}