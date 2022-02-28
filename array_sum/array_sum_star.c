/* 
Course  : CS566 Parallel Computing
Project : Sum of Large Array
Desc    : imposing Hypercube topology for parallel computation of sum of array on UIC ACER 
          cluster with C/T routing using MPI_Scatter and MPI_Gather primitives.
Author  : Sampath GK
*/

#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>

/*
method scatters pieces of a large array(called mother array) from source node to every node,
and returns the pointer to array containing a part of the mother array sent by the source node.
*/
int* one_to_all_bc(int k, int my_rank, int arr_size)
{
    int i, *mother_arr, mother_arr_size, rank_source = 0, *arr;
    mother_arr_size = (int)(arr_size * k);
    arr = (int *)malloc(arr_size * sizeof(int));

    //Source Node Only : creates and initiates the mother array
    if (my_rank == rank_source)
    {
        mother_arr = (int *)malloc(mother_arr_size * sizeof(int));
        for (i = 0; i < mother_arr_size; ++i)
            mother_arr[i] = i;
    } 
    
    MPI_Scatter(
        mother_arr, 
        arr_size,
        MPI_INT,
        arr,
        arr_size,
        MPI_INT,
        rank_source,
        MPI_COMM_WORLD
    );

    return arr;
}

/*
method calculates sum of local array at each node and gathers these sums in the source node
and at source node, calculates the final sum before printing it.
*/
void single_node_acc(int k, int my_rank, int *arr, int arr_size)
{ 
    int i, sum, rank_source = 0, *inter_sum_arr;

    //All Nodes : calculate the sum of the (local) array piece
    sum = 0;
    for(i = 0; i < arr_size; ++i) sum += arr[i];

    //Source Node Only : create an array for receiving from each node, a sum of a array piece
    if (my_rank == rank_source)
    {
        inter_sum_arr = (int*)malloc(sizeof(int)*k);
    }

    MPI_Gather(
        &sum,
        1,
        MPI_INT,
        inter_sum_arr,
        1,
        MPI_INT,
        rank_source,
        MPI_COMM_WORLD
    );

    //Source Node Only : add up all the array-piece sum gathered from all nodes
    if(my_rank == rank_source){
        sum = 0;
        for( i = 0; i < k; ++i)
            sum += inter_sum_arr[i];
        printf("Result of array addition : %d\n", sum);
    }
}

int main(int argc, char **argv)
{
    int *arr, arr_size = 1000, k = 8, my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    double t1 = MPI_Wtime();

    // Decomposition of a big array
    arr = one_to_all_bc(k, my_rank, arr_size);

    // Reduction
    single_node_acc(k, my_rank, arr, arr_size);

    double t2 = MPI_Wtime();
    if(my_rank == 0)
        printf("Execution time : %f\n", (t2-t1));

    MPI_Finalize();
    return 0;
}