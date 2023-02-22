/*
 * --------------------------------------------------------------------------
 * BLISLAB 
 * --------------------------------------------------------------------------
 * Copyright (C) 2016, The University of Texas at Austin
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of The University of Texas nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * bl_dgemm.c
 *
 *
 * Purpose:
 * this is the main file of blislab dgemm.
 *
 * Todo:
 *
 *
 * Modification:
 *
 * 
 * */


#include "bl_dgemm.h"

#define USE_POINTER_OPT 1
#define USE_UNROLLING_OPT 0
#define USE_REGISTER_OPT 0

#if USE_POINTER_OPT

void AddDot(int k, double *A, int lda, double *B, int ldb, double *result) {
    int p;
    for (p = 0; p < k; p++) {
        // *result += A(0, p) * B(p, 0);
        *result += (*A) * (*B);

        A += lda;   // A points to next column
        B++;        // B points to next row
    }
}

#elif USE_UNROLLING_OPT
void AddDot(int k, double *A, int lda, double *B, int ldb, double *result) {
    int p;
    int step = 4;
    for (p = 0; p < k; p += step) {
        // *result += A(0, p) * B(p, 0);
        *result += (*(A)) * (*(B + 0));
        *result += (*(A + lda)) * (*(B + 1));
        *result += (*(A + 2 * lda)) * (*(B + 2));
        *result += (*(A + 3 * lda)) * (*(B + 3));

        A += step * lda;
        B += step;
    }
}
#elif USE_REGISTER_OPT
void AddDot(int k, double *A, int lda, double *B, int ldb, double *result) {
    int p;
    int step = 4;
    for (p = 0; p < k; p += step) {
        // 2.4.3: Register variables
        register double c0=(*(A)) * (*(B + 0)),
                        c1=(*(A + lda)) * (*(B + 1)),
                        c2=(*(A + 2 * lda)) * (*(B + 2)),
                        c3=(*(A + 3 * lda)) * (*(B + 3));
        // *result += A(0, p) * B(p, 0);
        *result += c0;
        *result += c1;
        *result += c2;
        *result += c3;

        A += step * lda;
        B += step;
    }
}
#endif

#if USE_POINTER_OPT

void AddDot_MRxNR(int k, double *A, int lda, double *B, int ldb, double *C, int ldc) {
    int ir, jr;
    int p;
    for (jr = 0; jr < DGEMM_NR; jr++) {
        for (ir = 0; ir < DGEMM_MR; ir++) {

            // AddDot(k, &A(ir, 0), lda, &B(0, jr), ldb, &C(ir, jr));
            AddDot(k, A++, lda, B, ldb, C++);

        }

        A -= DGEMM_MR;  // A goes back to the initial row
        B += ldb;   // B goes to next column
        C = C + ldc - DGEMM_MR; // C goes to the next column, but to the initial row.
    }
}

#elif USE_UNROLLING_OPT || USE_REGISTER_OPT
void AddDot_MRxNR(int k, double *A, int lda, double *B, int ldb, double *C, int ldc) {
    int ir, jr;
    int p;
    // 2.4.2: Loop unrolling
    // jr = 0
    AddDot(k, A + 0, lda, B, ldb, C + 0);
    AddDot(k, A + 1, lda, B, ldb, C + 1);
    AddDot(k, A + 2, lda, B, ldb, C + 2);
    AddDot(k, A + 3, lda, B, ldb, C + 3);
    // jr = 1
    B += ldb;
    C += ldc;
    AddDot(k, A + 0, lda, B, ldb, C + 0);
    AddDot(k, A + 1, lda, B, ldb, C + 1);
    AddDot(k, A + 2, lda, B, ldb, C + 2);
    AddDot(k, A + 3, lda, B, ldb, C + 3);
    // jr = 2
    B += ldb;
    C += ldc;
    AddDot(k, A + 0, lda, B, ldb, C + 0);
    AddDot(k, A + 1, lda, B, ldb, C + 1);
    AddDot(k, A + 2, lda, B, ldb, C + 2);
    AddDot(k, A + 3, lda, B, ldb, C + 3);
    // jr = 3
    B += ldb;
    C += ldc;
    AddDot(k, A + 0, lda, B, ldb, C + 0);
    AddDot(k, A + 1, lda, B, ldb, C + 1);
    AddDot(k, A + 2, lda, B, ldb, C + 2);
    AddDot(k, A + 3, lda, B, ldb, C + 3);
}
#endif

#if USE_POINTER_OPT || USE_UNROLLING_OPT || USE_REGISTER_OPT

void bl_dgemm(
        int m,
        int n,
        int k,
        double *A,
        int lda,
        double *B,
        int ldb,
        double *C,        // must be aligned
        int ldc        // ldc must also be aligned
) {
    int i, j, p;
    int ir, jr;

    // Early return if possible
    if (m == 0 || n == 0 || k == 0) {
        printf("bl_dgemm(): early return\n");
        return;
    }

    // 2.4.1 Using pointers
    double *cp, *ap, *bp;

    for (j = 0; j < n; j += DGEMM_NR) {          // Start 2-nd loop
        cp = &C[j * ldc];   // C(i, j): point to j-th column of C
        bp = &B[j * ldb];   // B(0, j): point to the j-th column, 0-th row of B

        for (i = 0; i < m; i += DGEMM_MR) {      // Start 1-st loop
            ap = &A[i];         // A(i, 0): point to the i-th row, 0-th column of A

            // AddDot_MRxNR( k, &A( i, 0 ), lda, &B( 0, j ), ldb, &C( i, j ), ldc );
            AddDot_MRxNR(k, ap, lda, bp, ldb, cp + i, ldc);

        }                                          // End   1-st loop
    }                                              // End   2-nd loop
}

#endif


