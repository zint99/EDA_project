// 测试矩阵行主序与列主序存储
#include <iostream>

int m = 2, k = 2, n = 3;
int A[10][10];
int B[10][10];
int C[10][10];

void print_by_row_order(int *matrix, int m, int n) {
    // C++中二维矩阵底层按行主序存储
    for (int i = 0; i < m * n; i++) {
        std::cout << *matrix++ << " ";
    }
    puts("");
}
void print_C() {
    // 二维矩阵
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << C[i][j] << " ";
        }
        puts("");
    }
}

void row_order_multi(int *A, int *B, int *C, int M, int K, int N) {
    // 按照行主序进行矩阵乘法
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

int main() {
    // C在全局变量定义为10 * 10大小的二维矩阵
    // 这里初始化为 2 * 3 矩阵，再通过row-order打印，验证debug
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            C[i][j] = 1;
        }
    }

    std::cout << "start row order GEMM" << std::endl;
    print_by_row_order((int *)C, m, n);
    // row_order_multi((int *)A, (int *)B, (int *)C, m, k, n);
    // std::cout << "finish row order GEMM" << std::endl;
    // print_by_row_order((int *)C, m, n);
    // print_C();

    return 0;
}