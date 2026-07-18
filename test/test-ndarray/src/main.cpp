/**
 * @file main.cpp
 * @brief ndarray 测试总入口
 *
 * 依次运行 M1–M6 与最终业务测试。
 */

#include <cstdio>

// 各模块测试入口（定义在各自的 .cpp 中）
void run_test_m1();
void run_test_m2();
void run_test_m3();
void run_test_m4();
void run_test_m5();
void run_test_m6();
void run_test_final();

int main() {
    std::puts("========== ndarray test suite ==========");

    run_test_m1();
    run_test_m2();
    run_test_m3();
    run_test_m4();
    run_test_m5();
    run_test_m6();
    run_test_final();

    std::puts("========== all ndarray tests passed ==========");
    return 0;
}
