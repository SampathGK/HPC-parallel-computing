/* 
Course  : CS566 Parallel Computing
Project : Sum of Large Array
Desc    : imposing Hypercube topology for parallel computation of sum of array on UIC ACER 
          cluster with S/F routing using MPI_Send and MPI_Recv primitives.
Author  : Sampath GK
*/

#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>

/*
This method does one to all broadcast with Store and Forward Routing
Decomposition of a large array is achieved using MPI_SEND & MPI_RECV primitives.
*/
int* one_to_all_bc(int d, int my_rank, int *arr, int *arr_size)
{
    // create array at source node
    //  creating array of numbers at the source node
    int i, j;

    if (my_rank == 0)
    {
        int numbers[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        arr = (int *)malloc(*arr_size * sizeof(int));
        for (i = 0; i < *arr_size; ++i)
            arr[i] = i;
    }

    int mask = ((int)pow(2, d)) - 1;
    for (i = d - 1; i >= 0; --i)
    {
        mask = mask ^ ((int)pow(2, i));
        if ((my_rank & mask) == 0)
        {
            if ((my_rank & ((int)pow(2, i))) == 0)
            {
                // send half of array to a single neighbor along one of the dimension
                int rank_destn = my_rank ^ ((int)pow(2, i));
                *arr_size = (int)(*arr_size / 2);
                MPI_Send(arr_size,
                         1, MPI_INT, rank_destn, 0,
                         MPI_COMM_WORLD);

                MPI_Send(&arr[*arr_size],
                         *arr_size,
                         MPI_INT, rank_destn, 0,
                         MPI_COMM_WORLD);
            }
            else
            {
                int rank_source = my_rank ^ ((int)pow(2, i));
                MPI_Status status;
                // receive array
                MPI_Recv(arr_size,
                         1, MPI_INT, rank_source, 0,
                         MPI_COMM_WORLD,
                         &status);
                arr = (int *)malloc(*arr_size * sizeof(int));
                // stores the received array segment
                // in local array a2
                MPI_Recv(arr, *arr_size,
                         MPI_INT, rank_source, 0,
                         MPI_COMM_WORLD,
                         &status);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    return arr;
}

/*
This method does single node accumulation of intermediate sums with Store and Forward Routing
Reduction of intermediate sums is achieved using MPI_SEND & MPI_RECV primitives.
*/
void single_node_acc(int d, int my_rank, int *arr, int arr_size)
{ 
    int i, j, sum, mask, rank_source, rank_recv, inter_sum;
    MPI_Status status;

    // calculate the sum
    sum = 0;
    for(i = 0; i < arr_size; ++i) sum += arr[i];

    mask = 0;
    for (i = 0; i < d; ++i)
    {
        if ((my_rank & mask) == 0)
        {
            if ((my_rank & ((int)pow(2, i))) != 0)
            {
                //send sum to the appropriate node
                rank_recv = (my_rank ^ ((int)pow(2, i)));
                MPI_Send(&sum,
                         1, MPI_INT, rank_recv, 0,
                         MPI_COMM_WORLD);
            }
            else
            {
                //block to receive sum from the appropriate node
                rank_source = (my_rank ^ ((int)pow(2, i)));
                MPI_Recv(&inter_sum, 1,
                         MPI_INT, rank_source, 0,
                         MPI_COMM_WORLD,
                         &status);
                sum += inter_sum;
            }
        }
        mask = (mask ^ ((int)pow(2, i)));
    }

    //all reduction done
    if(my_rank == 0)
        printf("Result of array addition : %d\n", sum);
}

int main(int argc, char **argv)
{
    int rank, *arr, arr_size, dim;
    arr_size = 40000; dim = 3;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Decomposition
    arr = one_to_all_bc(dim, rank, arr, &arr_size);

    // Reduction
    single_node_acc(dim, rank, arr, arr_size);

    MPI_Finalize();
    return 0;
}