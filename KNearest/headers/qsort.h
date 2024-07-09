//
// Created by akshar on 2024/03/15.
//

#ifndef KNEAREST_QSORT_H
#define KNEAREST_QSORT_H
#include <stdio.h>
#include "../headers/type.h"
#include <omp.h>

void parallel_qsort_section(int p, int r, dist_ptr_t *data, int limit);
void parallel_qsort_task(int p, int r, dist_ptr_t *data, int limit);
void sequential_qsort(int p, int r, dist_ptr_t *data);
int validateSort(int size, dist_ptr_t *data);
#endif // KNEAREST_QSORT_H
