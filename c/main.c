#include <stdio.h>

double consensus(double old_pred[], double new_pred[], double real, int len);

int main () {
  int size;
  double old_[] = {10, 22, 30, 23};
  double new_[] = {13, 20, 30, 4};
  double real = 10;
  double result;

  size = sizeof old_ / sizeof new_[0]; 

  result = consensus(old_, new_, real, size);

  printf("%f\n", result);

  return 0;
}
 
double consensus(double old_pred[], double new_pred[], double real, int len){
  double con;
  double weights[len];
  double adjusted_pred[len];
  double sum;

  for(int i = 0; i < len; i++){
    weights[i] = (real/old_pred[i]);
   }

  for(int j = 0; j < len; j++){
    adjusted_pred[j] = weights[j] * new_pred[j];
  }

  for(int k = 0; k< len; k++){
    sum = adjusted_pred[k] + sum;
  }

  con = sum/len;

  return con;
}

