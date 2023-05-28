#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <string.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <iostream>
#include <iomanip> // for stew & left
#define isprint 1

/*
    GEMM的二维与一维实现,矩阵维度不超过2048
    C = A * B
        - A(m x k)
        - B(k x n)
        - C(m x n)
*/

using namespace std;

const int MAX_SIZE = 2048;

int m, n, k;
int a[MAX_SIZE][MAX_SIZE] = {0},
    b[MAX_SIZE][MAX_SIZE] = {0},
    c[MAX_SIZE][MAX_SIZE] = {0};

void gemm(int *matA, int *matB, int *matC, const int M, const int N, const int K)
{
    // cout << "展开为行主序一维数组计算..." << endl;
    // for (int i = 0; i < M; i++)
    // {
    //     for (int j = 0; j < N; j++)
    //     {
    //         // int sum = 0;
    //         matC[i * N + j] = 0;
    //         for (int k = 0; k < K; k++)
    //         {
    //             // bug...
    //             matC[i * N + j] += matA[i * K + k] * matB[k * N + j];
    //         }
    //         // printf("C:%.2f \n", sum);
    //         // matC[i * N + j] = sum;
    //     }
    // }

    // debug..
    for (int i = 0; i < m * k; i++)
        std::cout << *matA++ << " ";
}

void gemm_naive(int M, int N, int K)
{
    // GEMM_2D_NAIVE
    // C = A * B
    cout << "正在使用二维矩阵朴素乘法计算..." << endl;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int temp = 0;
            for (int l = 0; l < K; l++)
            {
                temp += a[i][l] * b[l][j];
            }
            c[i][j] = temp;
        }
    }
}

void Gen_Matrix(int m, int n, int k)
{
    srand(time(NULL));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            a[i][j] = rand() % 11; // 随机产生0～10之间的浮点数
        }
    }
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < n; j++)
        {
            b[i][j] = rand() % 11;
        }
    }
}

void print_A(int m, int k)
{
    printf("=======================================================================\n");
    printf("矩阵A(%d行%d列)\n", m, k);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            printf("%d\t", a[i][j]);
        }
        printf("\n");
    }
}
void print_B(int k, int n)
{
    printf("矩阵B(%d行%d列)\n", k, n);
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%d\t", b[i][j]);
        }
        printf("\n");
    }
}
void print_C(int m, int n)
{
    printf("=======================================================================\n");
    printf("矩阵C(%d行%d列)\n", m, n);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%d\t", c[i][j]);
        }
        printf("\n");
    }
}

int main()
{
    clock_t start, end;

    printf("请依次输入m,n,k的值（不超过2048，m和n分别为C的行列数，k为A的列数和B的行数）:");
    scanf("%d%d%d", &m, &n, &k);
    while (m > 2048 || n > 2048 || k > 2048)
    {
        printf("超出范围，请输入小于2048的值");
        scanf("%d%d%d", &m, &n, &k);
    }

    // 矩阵初始化
    Gen_Matrix(m, n, k);
    // print A & B
    if (isprint)
    {
        print_A(m, k);
        print_B(k, n);
    }
    //  选择GEMM方法
    int choice = 0;
    while (choice != 1 && choice != 2)
    {
        cout << "请选择GEMM方法：（1. 二维矩阵朴素乘法 2. 二维矩阵朴素乘法一维优化）" << endl;
        cin >> choice;
    }
    start = clock();
    if (choice == 1)
        gemm_naive(m, n, k);
    else
        // bug fix: 展平为一维剧矩阵时不应该取地址
        gemm((int *)a, (int *)b, (int *)c, m, n, k);
    end = clock();
    double endtime = (double)(end - start) / CLOCKS_PER_SEC;
    // print matrix C
    if (isprint)
    {
        // print_C(m, n);
    }
    printf("GEMM通用矩阵乘法已完成,用时:%f ms.\n", endtime * 1000);

    return 0;
}