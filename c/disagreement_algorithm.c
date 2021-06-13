#include <stdio.h>
#include <math.h>

double disagreement(double predictions[], int len);

int main () {
  int size;
  double preds[] = {34, 2, 1, 1};
  double result;

  size = sizeof preds / sizeof preds[0]; 

  result = disagreement(preds, size);

  printf("%f\n", result);

  return 0;
}

/*
 * Function:  disagreement 
 * -----------------------
 * computes the system disagreement value by using the following algorithm:
 * - compute the absolute value of individual predictor forecast pair
 * - take the average of all absolute value pairs generated
 *
 *  predictions[] (double) : list containing forecasts of individual predictors in the system
 *  len	          (int)    : number of individual predictors in system
 *
 *  returns: the overall disagreement level in the system.
 */
double disagreement(double predictions[], int len){
  int mod_len = len*len;
  double result;
  double sum;
  double disagreement_scores[mod_len];
  int k = 0;

  for(int i = 0; i < len; i++){
    for(int j = 0; j < len; j++){
      disagreement_scores[k] = fabs(predictions[i]-predictions[j]);
      k++;
    }
   }
  for(int i = 0; i< mod_len; i++){
    sum = disagreement_scores[i] + sum;
  }
  return (sum/mod_len);
}

