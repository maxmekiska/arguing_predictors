#include <stdio.h>
#include <math.h>
#include "prototype.h"

int main () {

  /*disagreement function test*/
  int size;
  double preds[] = {34, 2, 1, 1};
  double result;

  size = sizeof preds / sizeof preds[0]; 

  result = disagreement(preds, size);

  printf("%f\n", result);


  /*consensus function test*/
  int size2;
  double old_[] = {3, 7, 5, 10};
  double new_[] = {4, 5, 7, 7};
  double real = 10;
  double result2;

  size2 = sizeof old_ / sizeof new_[0]; 

  result2 = consensus(old_, new_, real, size);

  printf("%f\n", result2);

  return 0;
}
