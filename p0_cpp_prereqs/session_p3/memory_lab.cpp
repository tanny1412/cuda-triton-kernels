#include <stdio.h>

// A struct that represents a 2D matrix
struct Matrix {
    float* data;  // pointer to flat array on the heap
    int rows;
    int cols;
};

// Allocate a matrix on the heap
Matrix alloc_matrix(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = new float[rows * cols];
    return m;
}

// Fill matrix with a value
void fill(Matrix* m, float val) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = val;
    }
}

// Print one element using 2D indexing
void print_element(Matrix* m, int r, int c) {
    float val = m->data[r * m->cols + c];
    printf("m[%d][%d] = %.1f\n", r, c, val);
}

// Free the matrix
void free_matrix(Matrix* m) {
    delete[] m->data;
    m->data = nullptr;
}

int main() {
    Matrix m = alloc_matrix(3, 3);
    fill(&m, 7.0f);

    print_element(&m, 0, 0);
    print_element(&m, 1, 2);
    print_element(&m, 2, 2);

    free_matrix(&m);
    printf("after free, data ptr = %p\n", m.data);  // should be 0x0

    return 0;
}
