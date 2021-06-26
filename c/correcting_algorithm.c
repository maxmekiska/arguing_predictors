#include <stdio.h>

double average(double list[], double len);
double consensus(double old_pred[], double new_pred[], double real, int len);

int main () {
  int size;
  double old_[] = {3, 7, 5, 10};
  double new_[] = {4, 5, 7, 7};
  double real = 10;
  double result;

  size = sizeof old_ / sizeof new_[0]; 

  result = consensus(old_, new_, real, size);

  printf("%f\n", result);

  return 0;
}

double average(double list[], double len){
  double result;
  double sum;
  for(int i = 0; i < len; i++){
    sum = list[i] + sum;
  }
  result = sum/len;
  return result;
}

/*
 * Function:  consensus 
 * --------------------
 * computes the consensus value by using the following correction algorithm:
 * - look at past individual predictor forecasts (t-1)
 * - compare them with the real value
 * - compute correction weight by: (real value) / (past individual predictor forecast)
 * - apply these correction weights to current individual predictor forecasts (t) for real value at (t+1)
 * - take the average of all current correction weigth adjusted individual predictor forecasts 
 *
 *  old_pred[] (double) : individual predictor forecasts at time (t-1)
 *  new_pred[] (double) : current individual predictor forecasts (t) 
 *  real       (double) : real value known at time t
 *  len	       (int)    : number of individual predictors in system
 *
 *  returns: the consensus value of the system prediction for (t+1) which represents the average of all
 *           correction-weight adjusted individual predictor forecasts.           
 */
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

  con = average(adjusted_pred, len);

  return con;
}
