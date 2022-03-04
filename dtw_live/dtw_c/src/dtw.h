#ifndef DTW_H
#define DTW_H

void update_cost(double *frame, double *tempt, int n, int m, double *dist);

void update_cost_width(double *frame, double *tempt, int n, int m, double *dist, unsigned short *width);

void cost_matrix(double *x, double *y, int nx, int ny, int m, int psi_x, int psi_y, double *mat);

#endif