// 测试矩阵行主序与列主序存储
#include <iostream>

int m = 2, k = 2, n = 3;
int A[2][2] = {{1, 2}, {3, 4}};
int B[2][3] = {{1, 1, 1}, {1, 1, 1}};
int C[2][3] = {0};

void print_by_row_order(int *matrix, int m, int n)
{
    // C++中二维矩阵底层按行主序存储
    for (int i = 0; i < m * n; i++)
    {
        std::cout << *matrix++ << " ";
    }
    puts("");
}
void print_C()
{
    // 二维矩阵
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            std::cout << C[i][j] << " ";
        }
        puts("");
    }
}

void row_order_multi(int *A, int *B, int *C, int M, int K, int N)
{
    // 按照行主序进行矩阵乘法
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < K; k++)
            {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

int main()
{
    std::cout << "start row order GEMM" << std::endl;
    print_by_row_order((int *)C, m, n);
    row_order_multi((int *)A, (int *)B, (int *)C, m, k, n);
    std::cout << "finish row order GEMM" << std::endl;
    print_by_row_order((int *)C, m, n);
    print_C();

    return 0;
}