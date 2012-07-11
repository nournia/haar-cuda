#include <fstream>
#include <iostream>
#include <cutil_inline.h>
#include <shrQATest.h>

#include "Timer.h"

using namespace std;

const double sqrt_2 = 1.4142135;

__global__ void cal_haar(float input[], float output [], int o_width, int o_height)
{ 
	int x_index = blockIdx.x*blockDim.x+threadIdx.x; 
	int y_index = blockIdx.y*blockDim.y+threadIdx.y; 

	if(x_index>=(o_width+1)/2 || y_index>=o_height) return; 

	int i_thread_id = y_index*o_width + 2*x_index;
	int o_thread_id = y_index*o_width + x_index;

	const double sqrt_2 = 1.4142135;
	output[o_thread_id] = (input[i_thread_id]+input[i_thread_id+1])/sqrt_2;
	output[o_thread_id+o_width/2] = (input[i_thread_id]-input[i_thread_id+1])/sqrt_2;
} 
 
void haar(float input[], float output [], int o_width, int o_height)
{ 
	float* d_input; 
	float* d_output; 

	int widthstep = o_width*sizeof(float);

	cudaMalloc(&d_input, widthstep*o_height); 
	cudaMalloc(&d_output, widthstep*o_height); 

	cudaMemcpy(d_input, input, widthstep*o_height, cudaMemcpyHostToDevice); 

	dim3 blocksize(16,16);
	dim3 gridsize;
	gridsize.x=(o_width+blocksize.x-1)/blocksize.x;
	gridsize.y=(o_height+blocksize.y-1)/blocksize.y;

	cal_haar<<<gridsize,blocksize>>>(d_input,d_output,o_width,o_height);

	cudaMemcpy(output,d_output,widthstep*o_height,cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);
}

void haar2d_gpu(float** matrix, int size)
{
	int w = size;
	float* input = new float[size*size];
	float* output = new float[size*size];

	while(w>1)
	{
		// horizontal
		for (int i = 0; i < w; i++)
			for (int j = 0; j < w; j++)
				input[i*w+j] = matrix[i][j];

		haar(input, output, w, w);

		for (int i = 0; i < w; i++)
			for (int j = 0; j < w; j++)
				matrix[i][j] = output[i*w+j];

		// vertical
		for (int i = 0; i < w; i++)
			for (int j = 0; j < w; j++)
				input[i*w+j] = matrix[j][i];

		haar(input, output, w, w);

		for (int i = 0; i < w; i++)
			for (int j = 0; j < w; j++)
				matrix[j][i] = output[i*w+j];

		w/=2;
	}
}

void printMatrix(float** mat, int size)
{
	ofstream fout("gpu.txt");
	for(int i = 0; i < size; i++)
	{
		for(int j = 0; j < size; j++)
			fout << mat[i][j] << " ";
		fout << endl;
	}
	fout << endl;
	fout.close();
}

void haar1d_cpu(float *vec, int n, int w)
{
	int i=0;
	float *vecp = new float[n];
	for(i=0;i<n;i++)
		vecp[i] = 0;

		w/=2;
		for(i=0;i<w;i++)
		{
			vecp[i] = (vec[2*i] + vec[2*i+1])/sqrt_2;
			vecp[i+w] = (vec[2*i] - vec[2*i+1])/sqrt_2;
		}
		
		for(i=0;i<(w*2);i++)
			vec[i] = vecp[i];

		delete [] vecp;
}

void haar2d_cpu(float **matrix, int size)
{
	float *temp_col = new float[size];

	int i = 0, j = 0;
	int w = size;

	while(w>1)
	{
		for(i=0;i<w;i++)
			haar1d_cpu(matrix[i], w, w);

		for(i=0;i<w;i++)
		{
			for(j=0;j<w;j++)
				temp_col[j] = matrix[j][i];
			
			haar1d_cpu(temp_col, w, w);
			
			for(j=0;j<w;j++)
				matrix[j][i] = temp_col[j];
		}

		w/=2;
	}

	delete [] temp_col;
}

int main(int argc, char **argv)
{
	Timer timer;

	ifstream fin;
	fin.open("img.txt");
	if(! fin)
		cout <<"Input File Error!";

	// Input matrix
	int size = 2048;
	float** mat = new float*[size];
	float** mat2 = new float*[size];
	for(int m = 0; m < size; m++) {
		mat[m] = new float[size];
		mat2[m] = new float[size];
	}

	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++) {
			fin >> mat[i][j];
			mat2[i][j] = mat[i][j];
		}
	fin.close();

	// Haar transform with CPU
	timer.start();
	haar2d_cpu(mat, size);
	timer.stop();
	cout << "CPU Time: " << timer.getElapsedTimeInMilliSec() << " ms" << endl;
	// printMatrix(mat, size);

	// Haar transform with GPU
	timer.start();
	haar2d_gpu(mat2, size);
	timer.stop();
	cout << "GPU Time: " << timer.getElapsedTimeInMilliSec() << " ms" << endl;
	//printMatrix(mat2, size);

	cin.get();

	return 0;
}