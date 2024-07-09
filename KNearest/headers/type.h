//
// Created by akshar on 2024/03/16.
//

#ifndef KNEAREST_TYPE_H
#define KNEAREST_TYPE_H

struct point
{
    double *arr;
    int dimension;
};

typedef struct point point_t;
typedef point_t *point_ptr_t;

struct dist
{
    point_ptr_t point;
    double dist;
};

typedef struct dist dist_t;
typedef dist_t *dist_ptr_t;

#endif // KNEAREST_TYPE_H
