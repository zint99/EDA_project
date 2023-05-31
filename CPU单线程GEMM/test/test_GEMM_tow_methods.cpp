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
#include <random>  // 引入随机数生成器和浮点数分布的头文件
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
// bug fix:动态分配矩阵
// int a[MAX_SIZE][MAX_SIZE],
//     b[MAX_SIZE][MAX_SIZE],
//     c[MAX_SIZE][MAX_SIZE];

void gemm(double *matA, double *matB, double *matC, const int M, const int N, const int K) {
    cout << "展开为行主序一维数组计算..." << endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0;
            matC[i * N + j] = 0;
            for (int k = 0; k < K; k++) {
                // debug...
                sum += matA[i * K + k] * matB[k * N + j];
            }
            matC[i * N + j] = sum;
        }
    }
}

void gemm_naive(double **a, double **b, double **c, int M, int N, int K) {
    // GEMM_2D_NAIVE
    // C = A * B
    cout << "正在使用二维矩阵朴素乘法计算..." << endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double temp = 0;
            for (int l = 0; l < K; l++) {
                temp += a[i][l] * b[l][j];
            }
            c[i][j] = temp;
        }
    }
}

void Gen_Matrix(double *a, double *b, int m, int n, int k) {
    //
    std::random_device rd;                           // 随机数种子
    std::mt19937 gen(rd());                          // 随机数生成器
    std::uniform_real_distribution<> dis(1.0, 10.0); // 定义浮点数分布
    for (int i = 0; i < m * k; i++) {
        *a++ = dis(gen);                             // 生成1~10的随机浮点数
    }
    for (int i = 0; i < k * n; i++) {
        *b++ = dis(gen); // 生成1~10的随机浮点数
    }
}
void print_A(double **a, int m, int k) {
    printf("=======================================================================\n");
    printf("矩阵A(%d行%d列)\n", m, k);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            printf("%.2f\t", a[i][j]);
        }
        printf("\n");
    }
}
void print_B(double **b, int k, int n) {
    printf("矩阵B(%d行%d列)\n", k, n);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f\t", b[i][j]);
        }
        printf("\n");
    }
}
void print_C(double **c, int m, int n) {
    printf("=======================================================================\n");
    printf("矩阵C(%d行%d列)\n", m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f\t", c[i][j]);
        }
        printf("\n");
    }
}

/*
    1.编译时分配给矩阵最大维度的内存空间，然后运行时接受参数进行矩阵的初始化
        - 矩阵初始化为最大维度内存，所以当运行时指定的矩阵维度小于分配好的内存空间时，无法通过将矩阵转换为一维数组行主序相乘。
        - 每次测试不同维度矩阵GEMM时，需要先修改代码中编译时给矩阵分配的内存大小，再在运行时给程序传入相同维度的参数才能正确执行行主序。
    2.矩阵通过运行时输入参数动态分配指定大小的内存空间，所以会导致矩阵元素在内存中并非连续存储。
        - 通过运行时动态分配内存，可以分配指定的内存空间大小
        - 若分配为二维矩阵，那行指针在内存中是连续的，但列可能随机分配在不连续的内存中。所以只能使用朴素GEMM
        - 若分配为一维数组，则矩阵的所有元素在内存中连续排列，可以使用行主序GEMM
    3.以后都采用编译时分配最大内存 + 朴素GEMM乘法
    总结：
    1.编译时分配矩阵内存会导致将矩阵转换为一维数组后相乘的方法失败，即只能使用朴素GEMM。
    2.运行时分配矩阵内存，有两种方法。
        - 第一种是先分配行再分配列，此时可以使用M[i][j]方式访问矩阵元素，但此时矩阵元素在内存中并非线性连续分布，所以只能使用朴素GEMM
        - 第二种是将矩阵展平，分配为元素个数大小的一维数组。此时只能通过行主序的方式访问矩阵元素，元素在矩阵中连续线性分布，只能使用行主序进行GEMM。
*/

int main() {
    clock_t start, end;
    std::random_device rd;                           // 随机数种子
    std::mt19937 gen(rd());                          // 随机数生成器
    std::uniform_real_distribution<> dis(1.0, 10.0); // 定义浮点数分布

    printf("请依次输入m,n,k的值（不超过2048，m和n分别为C的行列数，k为A的列数和B的行数）:");
    scanf("%d%d%d", &m, &n, &k);
    while (m > 2048 || n > 2048 || k > 2048) {
        printf("超出范围，请输入小于2048的值");
        scanf("%d%d%d", &m, &n, &k);
    }

    // chose layout of matrix
    printf("矩阵内存动态分配，请选择矩阵内存方式：1. 二维非线性（朴素GEMM） 2. 一维线性（行主序GEMM）\n");
    int layout;
    cin >> layout;

    if (layout == 1) {
        // BUG FIX：动态分配矩阵
        double **a = new double *[m]; // 创建m行
        for (int i = 0; i < m; i++) {
            a[i] = new double[k];     // 创建k列
        }
        double **b = new double *[k]; // 创建k行
        for (int i = 0; i < k; i++) {
            b[i] = new double[n];     // 创建n列
        }
        double **c = new double *[m]; // 创建m行
        for (int i = 0; i < m; i++) {
            c[i] = new double[n];     // 创建n列
        }
        // 矩阵初始化
        // Gen_Matrix((double *)a, (double *)b, m, n, k);
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

        // print A & B
        if (isprint) {
            // print_A(a, m, k);
            // print_B(b, k, n);
            printf("=======================================================================\n");
            printf("矩阵A(%d行%d列)\n", m, k);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < k; j++) {
                    printf("%.2f\t", a[i][j]);
                }
                puts("");
            }
            printf("=======================================================================\n");
            printf("矩阵B(%d行%d列)\n", k, n);
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < n; j++) {
                    printf("%.2f\t", b[i][j]);
                }
                puts("");
            }
        }

        // GEMM
        start = clock();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double temp = 0;
                for (int l = 0; l < k; l++) {
                    temp += a[i][l] * b[l][j];
                }
                c[i][j] = temp;
            }
        }
        end = clock();
        if (isprint) {
            // print_C(c, m, n);
            printf("=======================================================================\n");
            printf("矩阵C(%d行%d列)\n", m, n);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    printf("%.2f\t", c[i][j]);
                }
                puts("");
            }
        }
        delete[] a;
        delete[] b;
        delete[] c;
    } else {
        double *a = new double[m * k]; //
        double *b = new double[k * n]; //
        double *c = new double[m * n]; // 创建m行
        // 矩阵初始化
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                a[i * m + j] = dis(gen);
            }
        }
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) {
                b[i * k + j] = dis(gen);
            }
        }

        // print A & B
        if (isprint) {
            // print_A(a, m, k);
            // print_B(b, k, n);
            printf("=======================================================================\n");
            printf("矩阵A(%d行%d列)\n", m, k);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < k; j++) {
                    printf("%.2f\t", a[i * m + j]);
                }
                puts("");
            }
            printf("=======================================================================\n");
            printf("矩阵B(%d行%d列)\n", k, n);
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < n; j++) {
                    printf("%.2f\t", b[i * k + j]);
                }
                puts("");
            }
        }

        // 行主序GEMM
        start = clock();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double temp = 0;
                for (int l = 0; l < k; l++) {
                    temp += a[i * m + l] * b[l * k + j];
                }
                c[i * m + j] = temp;
            }
        }
        end = clock();

        if (isprint) {
            // print_C(c, m, n);
            printf("=======================================================================\n");
            printf("矩阵C(%d行%d列)\n", m, n);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    printf("%.2f\t", c[i * m + j]);
                }
                puts("");
            }
        }
        delete[] a;
        delete[] b;
        delete[] c;
    }
    double endtime = (double)(end - start) / CLOCKS_PER_SEC;

    printf("GEMM通用矩阵乘法已完成,用时:%f ms.\n", endtime * 1000);

    return 0;
}