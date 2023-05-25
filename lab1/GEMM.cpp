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
double a[MAX_SIZE][MAX_SIZE], b[MAX_SIZE][MAX_SIZE], c[MAX_SIZE][MAX_SIZE];

void gemm(double *matA, double *matB, double *matC, const int M, const int N, const int K)
{
    cout << "正在使用二维矩阵乘法一维优化计算..." << endl;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double sum = 0;
            for (int l = 0; l < K; l++)
            {
                // printf("A:%.2f * B:%.2f  ", matA[i * K + l], matB[l * N + j]);
                //  c[i][j] += a[i][l] * b[l][j];  照着二维矩阵按行主序做映射
                // bug...
                sum += matA[i * K + l] * matB[l * N + j];
            }
            // printf("C:%.2f \n", sum);
            matC[i * N + j] = sum;
        }
    }
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
            c[i][j] = 0;
            for (int l = 0; l < K; l++)
            {
                c[i][j] += a[i][l] * b[l][j];
            }
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
            a[i][j] = (double)rand() / (double)(RAND_MAX)*10; // 随机产生0～10之间的浮点数
        }
    }
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < n; j++)
        {
            b[i][j] = (double)rand() / (double)(RAND_MAX)*10;
        }
    }
}

void print_A(int m, int k)
{
    printf("=======================================================================\n");
    printf("矩阵A有%d行%d列 :\n", m, k);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            printf("%-6.2f\t", a[i][j]);
        }
        printf("\n");
    }
}
void print_B(int k, int n)
{
    printf("矩阵B有%d行%d列 :\n", k, n);
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%-6.2f\t", b[i][j]);
        }
        printf("\n");
    }
}
void print_C(int m, int n)
{
    printf("=======================================================================\n");
    printf("矩阵C有%d行%d列 :\n", m, n);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {

            printf("%-6.2f\t", c[i][j]);
        }
        printf("\n");
    }
}

int main()
{
    clock_t start, end;

    printf("请依次输入m,n,k的值（不超过2048，m和n分别为C的行列数，k为A的列数B的行数）:");
    scanf("%d%d%d", &m, &n, &k);
    while (m > 2048 || n > 2048 || k > 2048)
    {
        printf("超出范围，请输入小于2048的值");
        scanf("%d%d%d", &m, &n, &k);
    }
    // double a[m][m],b[m][m],c[m][m];

    // 矩阵初始化
    Gen_Matrix(m, n, k);
    // print A & B
    if (isprint)
    {
        print_A(m, k);
        print_B(k, n);
    }
    // 控制选择哪种方法
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
        gemm((double *)&a, (double *)&b, (double *)&c, m, n, k);
    end = clock();
    double endtime = (double)(end - start) / CLOCKS_PER_SEC;
    // print matrix C
    if (isprint)
    {
        print_C(m, n);
    }
    printf("GEMM通用矩阵乘法已完成,用时:%f ms.\n", endtime * 1000);

    return 0;
}