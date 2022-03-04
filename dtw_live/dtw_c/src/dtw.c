#include "dtw.h"
#include <math.h>

#define MIN2(a,b) (((a) < (b)) ? (a) : (b))
#define MIN3(a,b,c) (MIN2(a, MIN2(b, c)))

void update_cost(double *frame, double *tempt, int n, int m, double *dist)
{
    double new_dist[n];

    for (int i=0; i<n; i++)
    {
        double cost = 0.0;
        for (int j=0; j<m; j++)
            cost += pow(tempt[i*m+j] - frame[j], 2);

        double top = dist[i];
        double mid = dist[i-1];
        double bot = new_dist[i-1];
        new_dist[i] = cost + MIN3(top, mid, bot);
    }

    for (int i=0; i<n; i++)
        dist[i] = new_dist[i];
}

void update_cost_width(double *frame, double *tempt, int n, int m, double *dist, unsigned short *width)
{
    double new_dist[n];
    unsigned short new_width[n];

    for (int i=0; i<n; i++)
    {
        double cost = 0.0;
        for (int j=0; j<m; j++)
            cost += pow(tempt[i*m+j] - frame[j], 2);

        double top = dist[i];
        double mid = dist[i-1];
        double bot = new_dist[i-1];

        double min_val = MIN3(top, mid, bot);
        new_dist[i] = cost + min_val;

        if (min_val == top)
            new_width[i] = width[i] + 1;
        else if (min_val == mid)
            new_width[i] = width[i-1] + 1;
        else // always bot
            new_width[i] = new_width[i-1];
    }

    for (int i=0; i<n; i++)
    {
        dist[i] = new_dist[i];
        width[i] = new_width[i];
    }
}

void cost_matrix(double *x, double *y, int nx, int ny, int m, int psi_x, int psi_y, double *mat)
{
    for (int i=0; i<nx; i++)
        for (int j=0; j<ny; j++)
        {
            // euclidian distance
            double cost = 0.0;
            for (int k=0; k<m; k++)
                cost += pow(x[i*m + k] - y[j*m + k], 2);

            // get previous costs
            double top = (i > 0) ? mat[(i-1)*ny + j]
                : (j <= psi_y) ? 0.0
                : INFINITY;
            double mid = (i > 0 && j > 0) ? mat[(i-1)*ny + (j-1)]
                : ((i == 0 && j <= psi_y) || (j == 0 && i <= psi_x)) ? 0.0
                : INFINITY;
            double bot = (j > 0) ? mat[i*ny + (j-1)]
                : (i <= psi_x) ? 0.0
                : INFINITY;

            mat[i*ny + j] = cost + MIN3(top, mid, bot);
        }

    for (int i=0; i<nx; i++)
        for (int j=0; j<ny; j++)
            mat[i*ny + j] = sqrt(mat[i*ny + j]);
}
