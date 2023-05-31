#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <string.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <random> // 引入随机数生成器和浮点数分布的头文件
#define isprint 1

const int MAX_SIZE = 2048;

int m, n, k;
// 编译时分配内存
double a[MAX_SIZE][MAX_SIZE], b[MAX_SIZE][MAX_SIZE], c[MAX_SIZE][MAX_SIZE];

void gemm(double (*a)[MAX_SIZE], double (*b)[MAX_SIZE], double (*c)[MAX_SIZE], const int m, const int k, const int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int l = 0; l < k; l++)
                sum += a[i][l] * b[l][j];
            c[i][j] = sum;
        }
    }
}

// 使用随机浮点数初始化矩阵
void initMatrix(double (*a)[MAX_SIZE], double (*b)[MAX_SIZE], const int m, const int k, const int n) {
    std::random_device rd;                           // 随机数种子
    std::mt19937 gen(rd());                          // 随机数生成器
    std::uniform_real_distribution<> dis(0.0, 10.0); // 定义浮点数分布
    // init a;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            a[i][j] = dis(gen);
        }
    }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            b[i][j] = dis(gen);
        }
    }
}

int main() {
    clock_t start, end;

    printf("请依次输入m，n，k的值（不超过2048）：");
    scanf("%d%d%d", &m, &n, &k);
    while (m > 2048 || n > 2048 || k > 2048) {
        printf("请输入小于2048的值");
        scanf("%d%d%d", &m, &n, &k);
    }

    initMatrix(a, b, m, k, n);

    if (isprint) {
        printf("=======================================================================\n");
        printf("矩阵A有%d行%d列 ：\n", m, k);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                printf("%.2f\t", a[i][j]);
            }
            printf("\n");
        }
        printf("矩阵B有%d行%d列 ：\n", k, n);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                printf("%.2f\t", b[i][j]);
            }
            printf("\n");
        }
    }

    start = clock();
    gemm(a, b, c, m, n, k);

    end = clock();
    double endtime = (double)(end - start) / CLOCKS_PER_SEC;
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

    return 0;
}
