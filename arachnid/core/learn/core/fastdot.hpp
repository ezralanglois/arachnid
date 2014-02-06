typedef long size_type;

inline void x_gemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transa, const enum CBLAS_TRANSPOSE transb, const int m, const int n, const int k, const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta, float* C, const int ldc)
{
	cblas_sgemm(order, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline void x_gemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transa, const enum CBLAS_TRANSPOSE transb, const int m, const int n, const int k, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc)
{
	cblas_dgemm(order, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline void x_gemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transa, const enum CBLAS_TRANSPOSE transb, const int m, const int n, const int k, const std::complex<float> alpha, const std::complex<float>* A, const int lda, const std::complex<float>* B, const int ldb, const std::complex<float> beta, std::complex<float>* C, const int ldc)
{
	cblas_cgemm(order, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}
inline void x_gemm(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE transa, const enum CBLAS_TRANSPOSE transb, const int m, const int n, const int k, const std::complex<double> alpha, const std::complex<double>* A, const int lda, const std::complex<double>* B, const int ldb, const std::complex<double> beta, std::complex<double>* C, const int ldc)
{
	cblas_zgemm(order, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

template<class T>
void gemm(T* samp1, int n1, int m1, T* samp2, int n2, int m2, T* distm, int n3, int m3, T alpha, T beta)
{
	//x_gemm(CblasRowMajor, CblasNoTrans, CblasTrans,      n1, n2, m1, T(alpha), samp1, m1, samp2, m1, T(beta), distm, n2);
	x_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, n1, n2, m1, T(alpha), samp1, m1, samp2, m2, T(beta), distm, m3);
}

/*
 SUBROUTINE xGEMM ( TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC )
 where TRANSA and TRANSB determines
 if the matrices A and B are to be transposed.
 M is the number of rows in matrix C and, depending on TRANSA, the number of rows in the original matrix A or its transpose.
 N is the number of columns in matrix C and, depending on TRANSB, the number of columns in the matrix B or its transpose.
 K is the number of columns in matrix A (or its transpose) and rows in matrix B (or its transpose).
 LDA, LDB and LDC specifies the size of the first dimension of the matrices, as laid out in memory; meaning the memory distance between the start of each row/column, depending on the memory structure
 */

template<class T>
void gemm_t1(T* samp1, int n1, int m1, T* samp2, int n2, int m2, T* distm, int n3, int m3, double alpha, double beta)
{
	x_gemm(CblasRowMajor, CblasTrans, CblasNoTrans, m1, m2, n1, T(alpha), samp1, m1, samp2, m2, T(beta), distm, m3);
}

