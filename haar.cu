#include <fstream>
#include <iostream>
#include <cutil_inline.h>
#include <shrQATest.h>

#include "Timer.h"

using namespace std;

const double sqrt_2 = 1.4142135;

__global__ void haar_horizontal(float input[], float output [], int o_width, int w)
{ 
	int x_index = blockIdx.x*blockDim.x+threadIdx.x; 
	int y_index = blockIdx.y*blockDim.y+threadIdx.y; 

	if(x_index>=(w+1)/2 || y_index>=w) return; 

	int i_thread_id = y_index*o_width + 2*x_index;
	int o_thread_id = y_index*o_width + x_index;

	const double sqrt_2 = 1.4142135;
	output[o_thread_id] = (input[i_thread_id]+input[i_thread_id+1])/sqrt_2;
	output[o_thread_id+w/2] = (input[i_thread_id]-input[i_thread_id+1])/sqrt_2;
} 
 
__global__ void haar_vertical(float input[], float output [], int o_width, int w)
{ 
	int x_index = blockIdx.x*blockDim.x+threadIdx.x; 
	int y_index = blockIdx.y*blockDim.y+threadIdx.y; 

	if(y_index>=(w+1)/2 || x_index>=w) return; 

	int p1 = 2*y_index*o_width + x_index;
	int p2 = (2*y_index+1)*o_width + x_index;
	int p3 = y_index*o_width + x_index;

	const double sqrt_2 = 1.4142135;
	output[p3] = (input[p1]+input[p2])/sqrt_2;
	output[p3+o_width*w/2] = (input[p1]-input[p2])/sqrt_2;
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

	int w = o_width;
	gridsize.x=(w+blocksize.x-1)/blocksize.x;
	gridsize.y=(w+blocksize.y-1)/blocksize.y;

	while(w>1)
	{
		haar_horizontal<<<gridsize,blocksize>>>(d_input,d_output,o_width,w);
		haar_vertical<<<gridsize,blocksize>>>(d_output,d_input,o_width,w);
		w /= 2;
	}

	cudaMemcpy(output,d_input,widthstep*o_height,cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);
}

float* haar2d_gpu(float* input, int size)
{
	int w = size;
	float* output = new float[size*size];

	haar(input, output, w, w);

	return output;
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
	float* vec = new float[size*size];
	for (int i = 0; i < size*size; i++)
		fin >> vec[i];
	fin.close();

	// Haar transform with GPU
	timer.start();
	vec = haar2d_gpu(vec, size);
	timer.stop();
	cout << "GPU Time: " << timer.getElapsedTimeInMilliSec() << " ms" << endl;
	//printVector(vec, size);

	float** mat = new float*[size];
	for(int m = 0; m < size; m++)
		mat[m] = new float[size];

	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			mat[i][j] = vec[i*size+j];

	// Haar transform with CPU
	timer.start();
	haar2d_cpu(mat, size);
	timer.stop();
	cout << "CPU Time: " << timer.getElapsedTimeInMilliSec() << " ms" << endl;
	//printMatrix(mat, size);

	cin.get();

	return 0;
}