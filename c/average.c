#include "prototype.h"

/*
 * Function:  average
 * --------------------
 * computes the average value of a given array.
 *
 *  list[] (double) : target array
 *  len    (double)    : total count of elements in target array  
 *
 *  returns: average of array.
 */
double average(double list[], double len){
  double result;
  double sum;
  for(int i = 0; i < len; i++){
    sum = list[i] + sum;
  }
  result = sum/len;
  return result;
}
