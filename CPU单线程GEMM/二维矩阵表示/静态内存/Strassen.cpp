/*
    Strassen算法从数学上优化GEMM
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <string.h>
#include <immintrin.h>
#include <emmintrin.h>
#include <random> // 引入随机数生成器和浮点数分布的头文件
#define isprint 0

const int MAX_SIZE = 2048;
double a[MAX_SIZE][MAX_SIZE], b[MAX_SIZE][MAX_SIZE], c[MAX_SIZE][MAX_SIZE];
// 中间数组
double a11[MAX_SIZE][MAX_SIZE], a12[MAX_SIZE][MAX_SIZE], a21[MAX_SIZE][MAX_SIZE], a22[MAX_SIZE][MAX_SIZE];
double b11[MAX_SIZE][MAX_SIZE], b12[MAX_SIZE][MAX_SIZE], b21[MAX_SIZE][MAX_SIZE], b22[MAX_SIZE][MAX_SIZE];
double c11[MAX_SIZE][MAX_SIZE], c12[MAX_SIZE][MAX_SIZE], c21[MAX_SIZE][MAX_SIZE], c22[MAX_SIZE][MAX_SIZE];
double M1[MAX_SIZE][MAX_SIZE], M2[MAX_SIZE][MAX_SIZE], M3[MAX_SIZE][MAX_SIZE], M4[MAX_SIZE][MAX_SIZE], M5[MAX_SIZE][MAX_SIZE], M6[MAX_SIZE][MAX_SIZE], M7[MAX_SIZE][MAX_SIZE];
double AResult[MAX_SIZE][MAX_SIZE], BResult[MAX_SIZE][MAX_SIZE];

int m, n, k;

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

void gemm(double (*a)[MAX_SIZE], double (*b)[MAX_SIZE], double (*c)[MAX_SIZE], const int m, const int k, const int n) {
    // printf("call gemm\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int l = 0; l < k; l++)
                sum += a[i][l] * b[l][j];
            c[i][j] = sum;
        }
    }
}

/*
    FOR STRASSEN
        - strassen
        - matrixAdd
        - matrixSub
*/
/*This function will add two square matrix*/
void MatrixAdd(double (*a)[MAX_SIZE], double (*b)[MAX_SIZE], double (*Result)[MAX_SIZE], int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Result[i][j] = a[i][j] + b[i][j];
        }
    }
}
/*This function will subtract one  square matrix from another*/
void MatrixSub(double (*a)[MAX_SIZE], double (*b)[MAX_SIZE], double (*Result)[MAX_SIZE], int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Result[i][j] = a[i][j] - b[i][j];
        }
    }
}

void strassen(double (*a)[MAX_SIZE], double (*b)[MAX_SIZE], double (*c)[MAX_SIZE], const int m, const int k, const int n) {
    // strassen算法通过将矩阵划分为更小的子矩阵来实现。需要确定矩阵尺寸的阈值决定使用传统GEMM还是strassen算法
    // 如果阶数小于64或不为2的次幂，则直接调用朴素GEMM计算
    // REFS:https://zhenhuaw.me/blog/2019/gemm-optimization.html
    // printf("call strassen\n");
    if ((m <= 64) || (m % 2 != 0 || n % 2 != 0 || k % 2 != 0)) {
        // printf("call gemm from strassen\n");
        return gemm(a, b, c, m, k, n);
    } else {
        int Divide = m / 2; // 非方阵直接调用gemm

        // dividing the matrices in 4 sub-matrices:
        for (int i = 0; i < Divide; i++) {
            for (int j = 0; j < Divide; j++) {
                a11[i][j] = a[i][j];
                a12[i][j] = a[i][j + Divide];
                a21[i][j] = a[i + Divide][j];
                a22[i][j] = a[i + Divide][j + Divide];

                b11[i][j] = b[i][j];
                b12[i][j] = b[i][j + Divide];
                b21[i][j] = b[i + Divide][j];
                b22[i][j] = b[i + Divide][j + Divide];
            }
        }
        // Calculating M1 to M7:
        MatrixAdd(a11, a22, AResult, Divide);                   // a11 + a22
        MatrixAdd(b11, b22, BResult, Divide);                   // b11 + b22
        strassen(AResult, BResult, M1, Divide, Divide, Divide); // M1 = (a11+a22) * (b11+b22)

        MatrixAdd(a21, a22, AResult, Divide);                   // a21 + a22
        strassen(AResult, b11, M2, Divide, Divide, Divide);     // M2 = (a21+a22) * (b11)

        MatrixSub(b12, b22, BResult, Divide);                   // b12 - b22
        strassen(a11, BResult, M3, Divide, Divide, Divide);     // M3 = (a11) * (b12 - b22)

        MatrixSub(b21, b11, BResult, Divide);                   // b21 - b11
        strassen(a22, BResult, M4, Divide, Divide, Divide);     // M4 = (a22) * (b21 - b11)

        MatrixAdd(a11, a12, AResult, Divide);                   // a11 + a12
        strassen(AResult, b22, M5, Divide, Divide, Divide);     // M5 = (a11+a12) * (b22)

        MatrixSub(a21, a11, AResult, Divide);                   // a21 - a11
        MatrixAdd(b11, b12, BResult, Divide);                   // b11 + b12
        strassen(AResult, BResult, M6, Divide, Divide, Divide); // M6 = (a21-a11) * (b11+b12)

        MatrixSub(a12, a22, AResult, Divide);                   // a12 - a22
        MatrixAdd(b21, b22, BResult, Divide);                   // b21 + b22
        strassen(AResult, BResult, M7, Divide, Divide, Divide); // M7 = (a12-a22) * (b21+b22)

        // calculating c21, c21, c11 e c22:
        MatrixAdd(M3, M5, c12, Divide);          // c12 = M3 + M5
        MatrixAdd(M2, M4, c21, Divide);          // c21 = M2 + M4

        MatrixAdd(M1, M4, AResult, Divide);      // M1 + M4
        MatrixAdd(AResult, M7, BResult, Divide); // M1 + M4 + M7
        MatrixSub(BResult, M5, c11, Divide);     // c11 = M1 + M4 - M5 + M7

        MatrixAdd(M1, M3, AResult, Divide);      // M1 + M3
        MatrixAdd(AResult, M6, BResult, Divide); // M1 + M3 + M6
        MatrixSub(BResult, M2, c22, Divide);     // c22 = M1 + M3 - M2 + M6

        // Grouping the results obtained in a single matrice:

        for (int i = 0; i < Divide; i++) {
            for (int j = 0; j < Divide; j++) {
                c[i][j] = c11[i][j];
                c[i][j + Divide] = c12[i][j];
                c[i + Divide][j] = c21[i][j];
                c[i + Divide][j + Divide] = c22[i][j];
            }
        }
    }
}

int main() {
    clock_t start, end;
    // 编译时分配内存
    printf("请依次输入m，k，n的值（不超过2048）：");
    scanf("%d%d%d", &m, &k, &n);
    while (m > 2048 || n > 2048 || k > 2048) {
        printf("请输入小于2048的值");
        scanf("%d%d%d", &m, &k, &n);
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
    gemm(a, b, c, m, k, n);
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

    // Strassen
    printf("Strassen优化GEMM矩阵乘法开始\n");
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

    return 0;
}
