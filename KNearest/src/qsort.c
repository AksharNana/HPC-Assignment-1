//
// Created by akshar on 2024/03/15.
//
#include "../headers/qsort.h"

/**
 * Parallel Quicksort Recursive algorithm (until specified depth)
 * This function is implemented using the sections work sharing construct
 * @param p
 * @param r
 * @param data
 * @param depth
 */
void parallel_qsort_section(int p, int r, dist_ptr_t *data, int limit)
{
    int i = p;
    int j = r;
    dist_ptr_t temp;
    dist_ptr_t pivot = data[(p + r) / 2];

    while (i <= j)
    {
        while (data[i]->dist < pivot->dist)
        {
            i++;
        }

        while (data[j]->dist > pivot->dist)
        {
            j--;
        }

        if (i <= j)
        {
            temp = data[i];
            data[i] = data[j];
            data[j] = temp;
            i++;
            j--;
        }
    }

    if ((r - p) < limit)
    {
        if (p < j)
        {
            parallel_qsort_section(p, j, data, limit);
        }
        if (i < r)
        {
            parallel_qsort_section(i, r, data, limit);
        }
    }
    else
    {
#pragma omp parallel sections num_threads(2)
        {
#pragma omp section
            parallel_qsort_section(p, j, data, limit);

#pragma omp section
            parallel_qsort_section(i, r, data, limit);
        }
    }
}

/**
 * Parallel Quicksort Recursive algorithm (until specified depth)
 * This function is implemented using the task work sharing construct
 * @param p
 * @param r
 * @param data
 * @param depth
 */
void parallel_qsort_task(int p, int r, dist_ptr_t *data, int limit)
{
    int i = p;
    int j = r;
    dist_ptr_t temp;
    dist_ptr_t pivot = data[(p + r) / 2];

    while (i <= j)
    {
        while (data[i]->dist < pivot->dist)
        {
            i++;
        }

        while (data[j]->dist > pivot->dist)
        {
            j--;
        }

        if (i <= j)
        {
            temp = data[i];
            data[i] = data[j];
            data[j] = temp;
            i++;
            j--;
        }
    }

    if ((r - p) < limit)
    {
        if (p < j)
        {
            parallel_qsort_task(p, j, data, limit);
        }
        if (i < r)
        {
            parallel_qsort_task(i, r, data, limit);
        }
    }
    else
    {
#pragma omp task
        parallel_qsort_task(p, j, data, limit);

#pragma omp task
        parallel_qsort_task(i, r, data, limit);
    }
}

/*
 * Sequential Quicksort Recursive Algorithm.
 * Takes position p, r and data.
 * @param p
 * @param r
 * @param data
 */
void sequential_qsort(int p, int r, dist_ptr_t *data)
{
    int i = p, j = r;
    dist_ptr_t temp;
    dist_ptr_t pivot = data[(p + r) / 2];

    while (i <= j)
    {
        while (data[i]->dist < pivot->dist)
            i++;
        while (data[j]->dist > pivot->dist)
            j--;
        if (i <= j)
        {
            temp = data[i];
            data[i] = data[j];
            data[j] = temp;
            i++;
            j--;
        }
    }

    if (p < j)
    {
        sequential_qsort(p, j, data);
    }
    if (i < r)
    {
        sequential_qsort(i, r, data);
    }
}

/**
 * @brief Function validates sorted array: returns 1 if valid, 0 if invalid sort.
 *
 * @param size
 * @param data
 * @return int
 */
int validateSort(int size, dist_ptr_t *data)
{
    for (int i = 0; i < size - 1; i++)
    {
        if (data[i]->dist > data[i + 1]->dist)
        {
            return 0;
        }
    }
    return 1;
}