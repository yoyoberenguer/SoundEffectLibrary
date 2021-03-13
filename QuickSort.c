/* C implementation */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <time.h>


// Quicksort algorithm
void swap(int* a, int* b);
int partition (int arr[], int low, int high);
int * quickSort(int arr[], int low, int high);
double lowpass (double *x, double *y, int M, double xm1);


/* C function implementing the simplest lowpass:
 *
 *      y(n) = x(n) + x(n-1)
 *
 */
double lowpass (double *x, double *y,
               int M, double xm1)
{
  int n;
  y[0] = x[0] + xm1;
  for (n=1; n < M ; n++) {
    y[n] =  x[n]  + x[n-1];
  }
  return x[M-1];
}



// A utility function to swap two elements
inline void swap(int* a, int* b)
{
	int t = *a;
	*a = *b;
	*b = t;
}

/* This function takes last element as pivot, places
the pivot element at its correct position in sorted
	array, and places all smaller (smaller than pivot)
to left of pivot and all greater elements to right
of pivot */
inline int partition (int arr[], int low, int high)
{
	int pivot = arr[high]; // pivot
	int i = (low - 1); // Index of smaller element

	for (int j = low; j <= high- 1; j++)
	{
		// If current element is smaller than the pivot
		if (arr[j] < pivot)
		{
			i++; // increment index of smaller element
			swap(&arr[i], &arr[j]);
		}
	}
	swap(&arr[i + 1], &arr[high]);
	return (i + 1);
}

/* The main function that implements QuickSort
arr[] --> Array to be sorted,
low --> Starting index,
high --> Ending index */
int * quickSort(int arr[], int low, int high)
{
	if (low < high)
	{
		/* pi is partitioning index, arr[p] is now
		at right place */
		int pi = partition(arr, low, high);

		// Separately sort elements before
		// partition and after partition
		quickSort(arr, low, pi - 1);
		quickSort(arr, pi + 1, high);
	}
return arr;
}

float f_max(float arr[], int element)
{

 int i = 1;
 float maximum = fabs(arr[0]);
 for (i;i<element;i++) {
    if (fabs(arr[i]) > maximum) {
        maximum = fabs(arr[i]);}
 }
 return maximum;
}




int main(){

  double x[10] = {1,2,3,4,5,6,7,8,9,10};
  double y[10];

  int i;
  int N=10;
  int M=N/2; /* block size */
  double xm1 = 0;

  xm1 = lowpass(x, y, M, xm1);
  xm1 = lowpass(&x[M], &y[M], M, xm1);

  for (i=0;i<N;i++) {
    printf("x[%d]=%f\ty[%d]=%f\n",i,x[i],i,y[i]);
  }


  float z[10] = {10,-8000000,150,400,250,13.5,8000001,523,-15.054,100.0};
  printf("RESULT : %f ", f_max(z, 10));
return 0;
}
