#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <string.h>
#include <random> // 引入随机数生成器和浮点数分布的头文件
#define isprint 0
#define MAX_SIZE 2048

double a[MAX_SIZE][MAX_SIZE], b[MAX_SIZE][MAX_SIZE], c[MAX_SIZE][MAX_SIZE];

int m, n, k;
clock_t start, end;
double endtime;

/**
 *  GEMM优化方法
 *      1. 基于算法优化：借助辅助矩阵，减少矩阵乘法次数
 *          1.1 strassen(O^2.83)
 *          1.2 coppersmith-winograd(O^2.3)
 *      2. 基于软件优化：优化访存，增加cache hit rate，并行计算提速
 *          2.1 改变循环顺序(ijk)-> (ikj)，在一千阶矩阵下，GEMM提升五倍
 *          2.2 循环拆分
 *          2.3 SIMD向量化指令
 */
// 中间数组
// ab_{ij}分别表示乘数矩阵的四个子矩阵
double a11[MAX_SIZE][MAX_SIZE], a12[MAX_SIZE][MAX_SIZE], a21[MAX_SIZE][MAX_SIZE], a22[MAX_SIZE][MAX_SIZE];
double b11[MAX_SIZE][MAX_SIZE], b12[MAX_SIZE][MAX_SIZE], b21[MAX_SIZE][MAX_SIZE], b22[MAX_SIZE][MAX_SIZE];
double c11[MAX_SIZE][MAX_SIZE], c12[MAX_SIZE][MAX_SIZE], c21[MAX_SIZE][MAX_SIZE], c22[MAX_SIZE][MAX_SIZE];
double M1[MAX_SIZE][MAX_SIZE], M2[MAX_SIZE][MAX_SIZE], M3[MAX_SIZE][MAX_SIZE], M4[MAX_SIZE][MAX_SIZE], M5[MAX_SIZE][MAX_SIZE], M6[MAX_SIZE][MAX_SIZE], M7[MAX_SIZE][MAX_SIZE];
double AResult[MAX_SIZE][MAX_SIZE], BResult[MAX_SIZE][MAX_SIZE];

// Winograd辅助矩阵
double S1[MAX_SIZE][MAX_SIZE], S2[MAX_SIZE][MAX_SIZE], S3[MAX_SIZE][MAX_SIZE], S4[MAX_SIZE][MAX_SIZE];
double T1[MAX_SIZE][MAX_SIZE], T2[MAX_SIZE][MAX_SIZE], T3[MAX_SIZE][MAX_SIZE], T4[MAX_SIZE][MAX_SIZE];
// double S1[MAX_SIZE][MAX_SIZE], S2[MAX_SIZE][MAX_SIZE], S3[MAX_SIZE][MAX_SIZE], S4[MAX_SIZE][MAX_SIZE];
double P1[MAX_SIZE][MAX_SIZE], P2[MAX_SIZE][MAX_SIZE], P3[MAX_SIZE][MAX_SIZE], P4[MAX_SIZE][MAX_SIZE], P5[MAX_SIZE][MAX_SIZE], P6[MAX_SIZE][MAX_SIZE], P7[MAX_SIZE][MAX_SIZE];
double U1[MAX_SIZE][MAX_SIZE], U2[MAX_SIZE][MAX_SIZE], U3[MAX_SIZE][MAX_SIZE], U4[MAX_SIZE][MAX_SIZE];

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

void print_A_B(double (*a)[MAX_SIZE], double (*b)[MAX_SIZE], int m, int k, int n) {
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
}

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

// naive gemm
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
// ikj for gemm
void gemm_order(double (*a)[MAX_SIZE], double (*b)[MAX_SIZE], double (*c)[MAX_SIZE], const int m, const int k, const int n) {
    // printf("call gemm\n");
    for (int i = 0; i < m; i++) {
        for (int l = 0; l < k; l++) {
            // 由于内层循环a[i][l]不变，所以使用变量保存a[i][l]，减少内存访问
            double num_a = a[i][l];
            for (int j = 0; j < n; j++) {
                c[i][j] += num_a * b[l][j];
            }
        }
    }
}

// 循环拆分，每次同时计算C的4 * 4 = 16个元素，同时使用到了A的连续4行和B的连续4列
void gemm_split(double (*a)[MAX_SIZE], double (*b)[MAX_SIZE], double (*c)[MAX_SIZE], const int m, const int k, const int n) {
    for (int i = 0; i < m; i += 4) {
        for (int j = 0; j < n; j += 4) {
            // 循环展开，GEMM是每次计算C中的一个元素，展开后是每次计算C中16个元素
            // 使用寄存器保存c矩阵中的元素值，减少访存
            // c00 == c[i+0][j+0]，下标为偏移量
            register double c00 = 0;
            register double c01 = 0;
            register double c02 = 0;
            register double c03 = 0;
            register double c10 = 0;
            register double c11 = 0;
            register double c12 = 0;
            register double c13 = 0;
            register double c20 = 0;
            register double c21 = 0;
            register double c22 = 0;
            register double c23 = 0;
            register double c30 = 0;
            register double c31 = 0;
            register double c32 = 0;
            register double c33 = 0;
            for (int l = 0; l < k; l += 4) {
                // 在每轮计算核中，a和b某些元素会重复访存，所以可以先用寄存器register保存起来，减少访存；
                // 如果不写下面这个循环，则要写64个计算操作。
                for (int t = 0; t < 4; t++) {
                    // 每轮循环拿出A一列和B一行，计算1/4个C
                    // 每轮有16条计算指令，总共64条
                    // 循环拆分的作用是减少访存，而非减少计算量。通过向量化操作可以一条计算指令操作4次计算，将总计算指令降低到16条。
                    register double a0t = a[i + 0][l + t];
                    register double a1t = a[i + 1][l + t];
                    register double a2t = a[i + 2][l + t];
                    register double a3t = a[i + 3][l + t];
                    register double bt0 = b[l + t][j + 0];
                    register double bt1 = b[l + t][j + 1];
                    register double bt2 = b[l + t][j + 2];
                    register double bt3 = b[l + t][j + 3];
                    c00 += a0t * bt0;
                    c01 += a0t * bt1;
                    c02 += a0t * bt2;
                    c03 += a0t * bt3;
                    c10 += a1t * bt0;
                    c11 += a1t * bt1;
                    c12 += a1t * bt2;
                    c13 += a1t * bt3;
                    c20 += a2t * bt0;
                    c21 += a2t * bt1;
                    c22 += a2t * bt2;
                    c23 += a2t * bt3;
                    c30 += a3t * bt0;
                    c31 += a3t * bt1;
                    c32 += a3t * bt2;
                    c33 += a3t * bt3;
                }
            }
            c[i + 0][j + 0] = c00;
            c[i + 0][j + 1] = c01;
            c[i + 0][j + 2] = c02;
            c[i + 0][j + 3] = c03;
            c[i + 1][j + 0] = c10;
            c[i + 1][j + 1] = c11;
            c[i + 1][j + 2] = c12;
            c[i + 1][j + 3] = c13;
            c[i + 2][j + 0] = c20;
            c[i + 2][j + 1] = c21;
            c[i + 2][j + 2] = c22;
            c[i + 2][j + 3] = c23;
            c[i + 3][j + 0] = c30;
            c[i + 3][j + 1] = c31;
            c[i + 3][j + 2] = c32;
            c[i + 3][j + 3] = c33;
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

void Winograd(double (*a)[MAX_SIZE], double (*b)[MAX_SIZE], double (*c)[MAX_SIZE], const int m, const int k, const int n) {
    if ((m <= 64) || (m % 2 != 0 || n % 2 != 0 || k % 2 != 0))
        // printf("call gemm from strassen\n");
        return gemm(a, b, c, m, k, n);
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
    // Calculate S, T, P and U
    MatrixAdd(a21, a22, S1, Divide); // A11 + A22
    MatrixSub(S1, a11, S2, Divide);  // S2 = S1 - A11
    MatrixSub(a11, a21, S3, Divide); // S3 = A11 - A21
    MatrixSub(a12, S2, S4, Divide);  // S4 = A12 - S2

    MatrixSub(b12, b11, T1, Divide); // T1 = B12 - B11
    MatrixSub(b22, T1, T2, Divide);  // T2 = B22 - T1
    MatrixSub(b22, b12, T3, Divide); // T3 = B22 - B12
    MatrixSub(T2, b21, T4, Divide);  // T4 = T2 - B21

    Winograd(a11, b11, P1, Divide, Divide, Divide);
    Winograd(a12, b21, P2, Divide, Divide, Divide);
    Winograd(S4, b22, P3, Divide, Divide, Divide);
    Winograd(a22, T4, P4, Divide, Divide, Divide);
    Winograd(S1, T1, P5, Divide, Divide, Divide);
    Winograd(S2, T2, P6, Divide, Divide, Divide);
    Winograd(S3, T3, P7, Divide, Divide, Divide);

    MatrixAdd(P1, P2, c11, Divide); //
    MatrixAdd(P1, P6, U2, Divide);  //
    MatrixAdd(U2, P7, U3, Divide);  //
    MatrixAdd(U2, P5, U4, Divide);  //

    MatrixAdd(U4, P3, c12, Divide); //
    MatrixSub(U3, P4, c21, Divide); //
    MatrixAdd(U3, P5, c22, Divide); //

    // put results together
    for (int i = 0; i < Divide; ++i) {
        for (int j = 0; j < Divide; ++j) {
            c[i][j] = c11[i][j];
            c[i][j + Divide] = c12[i][j];
            c[i + Divide][j] = c21[i][j];
            c[i + Divide][j + Divide] = c22[i][j];
        }
    }
}

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
