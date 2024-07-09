#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include "headers/qsort.h"
#include "headers/bfKNN.h"

int main(int argc, char **argv)
{
    int num_reference_points = 10000;
    int dimension = 64;
    int num_query_points = 10000;
    int k = 10;
    srand(time(NULL));

    int num_threads = omp_get_num_procs();
    omp_set_num_threads(num_threads);
    if (argc != 3)
    {
        printf("Usage error!\n");
        printf("Knearest <Array Size> <Dimension>\n");
        exit(1);
    }
    else
    {
        char *val = argv[1];
        num_reference_points = atoi(val);
        val = argv[2];
        dimension = atoi(val);
    }
    // printf("Generate array of %d points in dimension %d\n",num_reference_points, dimension);
    point_ptr_t *reference_points = (point_ptr_t *)malloc(sizeof(point_ptr_t) * num_reference_points);
    for (int i = 0; i < num_reference_points; i++)
    {
        reference_points[i] = malloc(sizeof(point_t));
        reference_points[i]->arr = malloc(sizeof(double) * dimension);
        for (int j = 0; j < dimension; j++)
        {
            reference_points[i]->arr[j] = (double)rand() * 5000 / RAND_MAX;
            reference_points[i]->dimension = dimension;
        }
    }

    // printf("Generate array of %d points in dimension %d\n",num_query_points, dimension);
    point_ptr_t *query_points = (point_ptr_t *)malloc(sizeof(point_ptr_t) * num_query_points);
    for (int i = 0; i < num_query_points; i++)
    {
        query_points[i] = malloc(sizeof(point_t));
        query_points[i]->arr = malloc(sizeof(double) * dimension);
        for (int j = 0; j < dimension; j++)
        {
            query_points[i]->arr[j] = (double)rand() * 5000 / RAND_MAX;
            query_points[i]->dimension = dimension;
        }
    }

    point_ptr_t **result = NULL;
    double start_time = omp_get_wtime();
#ifdef PARALLEL
    // printf("Parallel Algorithm!\n");
    result = parallel_bruteforce_knearest(num_reference_points, reference_points, num_query_points, query_points, k);
#endif

#ifdef SERIAL
    // printf("Serial Algorithm!\n");
    result = serial_bruteforce_knearest(num_reference_points, reference_points, num_query_points, query_points, k);
#endif
    double end_time = omp_get_wtime() - start_time;
    printf("Completed program in %f seconds!\n", end_time);

    for (int i = 0; i < num_reference_points; i++)
    {
        free(reference_points[i]->arr);
        free(reference_points[i]);
    }
    free(reference_points);

    for (int i = 0; i < num_query_points; i++)
    {
        free(query_points[i]->arr);
        free(query_points[i]);
    }
    free(query_points);

    for (int i = 0; i < num_query_points; i++)
    {
        free(result[i]);
    }
    free(result);

    return 0;
}
