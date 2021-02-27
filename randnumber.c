
// C program for generating a 
// random number in a given range 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void init_clock(){
    // clock_t t;
    srand(clock());
}

float randRangeFloat(float lower, float upper){
    return lower + ((float)rand()/(float)(RAND_MAX)) * (upper - lower);
}

int randRange(int lower, int upper)
{
    return (rand() % (upper - lower  + 1)) + lower;
}


//
//int main(){
//return 0;
//}