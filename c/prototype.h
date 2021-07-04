#ifndef   PROTOYPE_HEADER
#define   PROTOYPE_HEADER

double average(double list[], double len);

double disagreement(double predictions[], int len);

double consensus(double old_pred[], double new_pred[], double real, int len);

#endif