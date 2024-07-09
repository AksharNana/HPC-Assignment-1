//
// Created by akshar on 2024/03/16.
//

#ifndef KNEAREST_BFKNN_H
#define KNEAREST_BFKNN_H
#include <stdlib.h>
#include <math.h>
#include "../headers/qsort.h"
#include "../headers/type.h"

point_ptr_t **serial_bruteforce_knearest(int size_reference, point_ptr_t *reference_points, int size_query, point_ptr_t *query_points, int k);
point_ptr_t **parallel_bruteforce_knearest(int size_reference, point_ptr_t *reference_points, int size_query, point_ptr_t *query_points, int k);

#endif // KNEAREST_BFKNN_H
