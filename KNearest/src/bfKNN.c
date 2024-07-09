//
// Created by akshar on 2024/03/16.
//
#include "../headers/bfKNN.h"

point_ptr_t **serial_bruteforce_knearest(int size_reference, point_ptr_t *reference_points, int size_query, point_ptr_t *query_points, int k)
{
    point_ptr_t **query_nearest = (point_ptr_t **)malloc(sizeof(point_ptr_t) * size_query);

    dist_ptr_t *distances = (dist_ptr_t *)malloc(sizeof(dist_ptr_t) * size_reference);
    for (int j = 0; j < size_reference; j++)
    {
        distances[j] = (dist_ptr_t)malloc(sizeof(dist_t));
    }

    double distance_run_time = 0;
    double sort_run_time = 0;
    double distance_time_start = 0;
    double sort_time_start = 0;

    int valid_sort = 1;
    double dist;
    for (int i = 0; i < size_query; i++)
    {
        distance_time_start = omp_get_wtime();
        for (int j = 0; j < size_reference; j++)
        {
            distances[j]->point = reference_points[j];
            dist = 0;
            int dim = query_points[i]->dimension;
            for (int i = 0; i < dim; i++)
            {
                dist += ((reference_points[j]->arr[i] - query_points[i]->arr[i])) * ((reference_points[j]->arr[i] - query_points[i]->arr[i]));
            }
            dist = sqrt(dist);
            distances[j]->dist = dist;
        }
        distance_run_time += omp_get_wtime() - distance_time_start;

        sort_time_start = omp_get_wtime();
        sequential_qsort(0, size_reference - 1, distances);
        sort_run_time += omp_get_wtime() - sort_time_start;

        if (valid_sort != 0)
        {
            valid_sort = validateSort(size_reference, distances);
        }

        // Create an array to store the k-nearest neighbours points
        point_ptr_t *nearest = (point_ptr_t *)malloc(sizeof(point_ptr_t) * k);
        for (int idx = 0; idx < k; idx++)
        {
            nearest[idx] = distances[idx]->point;
        }

        // Place array in set for all points
        query_nearest[i] = nearest;
    }

    if (valid_sort == 1)
    {
        printf("Validation passed!\n");
    }
    else
    {
        printf("Validation failed!\n");
    }
    printf("Distance computation time: %f\n", distance_run_time);
    printf("Sort computation time: %f\n", sort_run_time);

    for (int j = 0; j < size_reference; j++)
    {
        free(distances[j]);
    }
    free(distances);

    return query_nearest;
}

point_ptr_t **parallel_bruteforce_knearest(int size_reference, point_ptr_t *reference_points, int size_query, point_ptr_t *query_points, int k)
{
    point_ptr_t **query_nearest = (point_ptr_t **)malloc(sizeof(point_ptr_t) * size_query);

    dist_ptr_t *distances = (dist_ptr_t *)malloc(sizeof(dist_ptr_t) * size_reference);
    for (int j = 0; j < size_reference; j++)
    {
        distances[j] = (dist_ptr_t)malloc(sizeof(dist_t));
    }

    double distance_run_time = 0;
    double sort_run_time = 0;
    double distance_time_start = 0;
    double sort_time_start = 0;

    int valid_sort = 1;

    double dist;
    for (int i = 0; i < size_query; i++)
    {
        // printf("Calculating for query %d\n", i);
        distance_time_start = omp_get_wtime();
#pragma omp parallel for num_threads(8) shared(distances, query_points, reference_points)
        for (int j = 0; j < size_reference; j++)
        {
            distances[j]->point = reference_points[j];
            dist = 0;
            int dim = query_points[i]->dimension;
            for (int i = 0; i < dim; i++)
            {
                dist += ((reference_points[j]->arr[i] - query_points[i]->arr[i])) * ((reference_points[j]->arr[i] - query_points[i]->arr[i]));
            }
            dist = sqrt(dist);
            distances[j]->dist = dist;
            // printf("Distance from query %d to ref %d is: %f\n",i, j, distances[j]->dist);
        }
        distance_run_time += omp_get_wtime() - distance_time_start;

        sort_time_start = omp_get_wtime();
#ifdef SECTION
        parallel_qsort_section(0, size_reference - 1, distances, 5000);
#endif

#ifdef TASK
#pragma omp parallel num_threads(8)
        {
#pragma omp single
            parallel_qsort_task(0, size_reference - 1, distances, 1000);
#pragma omp taskwait
        };
#endif
        sort_run_time += omp_get_wtime() - sort_time_start;

        if (valid_sort != 0)
        {
            valid_sort = validateSort(size_reference, distances);
        }

        // Create an array to store the k-nearest neighbours points
        point_ptr_t *nearest = (point_ptr_t *)malloc(sizeof(point_ptr_t) * k);
#pragma omp parallel for num_threads(8) shared(nearest, distances)
        for (int idx = 0; idx < k; idx++)
        {
            nearest[idx] = distances[idx]->point;
        }

        // Place array in set for all points
        query_nearest[i] = nearest;
    }

    if (valid_sort == 1)
    {
        printf("Validation passed!\n");
    }
    else
    {
        printf("Validation failed!\n");
    }
    printf("Distance computation time: %f\n", distance_run_time);
    printf("Sort computation time: %f\n", sort_run_time);
    for (int j = 0; j < size_reference; j++)
    {
        free(distances[j]);
    }
    free(distances);

    return query_nearest;
}