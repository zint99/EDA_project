#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <string.h>
#include <random> // 引入随机数生成器和浮点数分布的头文件
#include "common.h"
#define isprint 0
#define MAX_SIZE 2048

double a[MAX_SIZE][MAX_SIZE], b[MAX_SIZE][MAX_SIZE], c[MAX_SIZE][MAX_SIZE];

int m, n, k;
clock_t start, end;
double endtime;
void test_gemm() {
    start = clock();
    gemm(a, b, c, m, k, n);
    end = clock();
    endtime = (double)(end - start) / CLOCKS_PER_SEC;

    if (isprint) {
        printf("=======================================================================\n");
        printf("矩阵C有%d行%d列 ：\n", m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                printf("%.2f\t", c[i][j]);
            }
            printf("\n");
        }
    }
    printf("GEMM通用矩阵乘法已完成，用时：%f ms.\n", endtime * 1000);
}

void test_gemm_changeOrder() {
    start = clock();
    gemm_order(a, b, c, m, k, n);
    end = clock();
    endtime = (double)(end - start) / CLOCKS_PER_SEC;

    if (isprint) {
        printf("=======================================================================\n");
        printf("矩阵C有%d行%d列 ：\n", m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                printf("%.2f\t", c[i][j]);
            }
            printf("\n");
        }
    }
    printf("GEMM改变循环顺序优化矩阵乘法已完成，用时：%f ms.\n", endtime * 1000);
}

void test_gemm_split() {
    start = clock();
    if (m != n || m != k || n != k || m % 4 != 0) {
        gemm(a, b, c, m, k, n);
    } else {
        gemm_split(a, b, c, m, k, n);
    }
    end = clock();
    endtime = (double)(end - start) / CLOCKS_PER_SEC;

    if (isprint) {
        printf("=======================================================================\n");
        printf("矩阵C有%d行%d列 ：\n", m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                printf("%.2f\t", c[i][j]);
            }
            printf("\n");
        }
    }
    printf("GEMM循环拆分优化矩阵乘法已完成，用时：%f ms.\n", endtime * 1000);
}

void test_strassen() {
    // Strassen
    printf("\nStrassen优化GEMM矩阵乘法开始\n");
    // memset(c, 0, sizeof(double) * MAX_SIZE * MAX_SIZE);
    start = clock();
    strassen(a, b, c, m, k, n);
    end = clock();
    endtime = (double)(end - start) / CLOCKS_PER_SEC;

    if (isprint) {
        printf("=======================================================================\n");
        printf("矩阵C有%d行%d列 ：\n", m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                printf("%.2f\t", c[i][j]);
            }
            printf("\n");
        }
    }
    printf("Strassen优化GEMM矩阵乘法已完成，用时：%f ms.\n", endtime * 1000);
}

void test_winograd() {
    // CW
    printf("\nCoppersmith-Winograd优化GEMM矩阵乘法开始\n");
    // memset(c, 0, sizeof(double) * MAX_SIZE * MAX_SIZE);
    start = clock();
    Winograd(a, b, c, m, k, n);
    end = clock();
    endtime = (double)(end - start) / CLOCKS_PER_SEC;

    if (isprint) {
        printf("=======================================================================\n");
        printf("矩阵C有%d行%d列 ：\n", m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                printf("%.2f\t", c[i][j]);
            }
            printf("\n");
        }
    }
    printf("Coppersmith-Winograd优化GEMM矩阵乘法已完成，用时：%f ms.\n", endtime * 1000);
}

int main() {
    printf("请依次输入m，k，n的值（不超过2048）：");
    scanf("%d%d%d", &m, &k, &n);
    while (m > 2048 || n > 2048 || k > 2048) {
        printf("请输入小于2048的值");
        scanf("%d%d%d", &m, &k, &n);
    }

    initMatrix(a, b, m, k, n);

    print_A_B(a, b, m, k, n);

    test_gemm();
    // test_strassen();
    // test_winograd();

    /* 循环拆分向量化avx */
    // 1. 测试调换循环顺序对GEMM的影响
    // test_gemm_changeOrder();

    // 2. 循环拆分
    test_gemm_split();
    return 0;
}
