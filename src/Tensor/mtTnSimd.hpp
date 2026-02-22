// Created by Unium on 22.02.26

#pragma once

#include "mtTnTnsr.hpp"

namespace MT {
namespace SM {
// <<<s_start(utils)
// --- cpu detection stuff
/*---------------------------------------------------------
 * FN: bHasAvx2
 * DESC: checks if the cpu supports avx2 instructions
 * PARMS: none
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto bHasAvx2() -> bool;

/*---------------------------------------------------------
 * FN: bHasAvx512
 * DESC: checks if the cpu supports avx512f instructions
 * PARMS: none
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto bHasAvx512() -> bool;

/*---------------------------------------------------------
 * FN: PCF
 * DESC: prints detected simd and threading info to stdout
 * PARMS: none
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
void PCF();
// >>>s_end(utils)

// <<<s_start(element)
// --- element wise opers
/*---------------------------------------------------------
 * FN: Add
 * DESC: performs element wise addition of two tensors
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto Add(const CTensor &tA, const CTensor &tB) -> CTensor;

/*---------------------------------------------------------
 * FN: Sub
 * DESC: performs element wise subtraction of two tensors
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto Sub(const CTensor &tA, const CTensor &tB) -> CTensor;

/*---------------------------------------------------------
 * FN: Mul
 * DESC: performs element wise multiplication of two tensors
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto Mul(const CTensor &tA, const CTensor &tB) -> CTensor;

/*---------------------------------------------------------
 * FN: AddScalar
 * DESC: adds a scalar value to every element in the tensor
 * PARMS: tA (tensor), fS (scalar value)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto AddScalar(const CTensor &tA, float fS) -> CTensor;

/*---------------------------------------------------------
 * FN: MulScalar
 * DESC: multiplies every element in the tensor by a scalar
 * PARMS: tA (tensor), fS (scalar value)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto MulScalar(const CTensor &tA, float fS) -> CTensor;
// >>>s_end(element)

// <<<s_start(matrix)
// --- matrix opers
/*---------------------------------------------------------
 * FN: Matmul
 * DESC: performs matrix multiplication between two tensors
 * PARMS: tA (matrix A), tB (matrix B)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto Matmul(const CTensor &tA, const CTensor &tB) -> CTensor;

/*---------------------------------------------------------
 * FN: Matvec
 * DESC: performs matrix vector multiplicstion
 * PARMS: tMat (matrix), tVec (vector)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto Matvec(const CTensor &tMat, const CTensor &tVec) -> CTensor;
// >>>s_end(matrixu)

// <<<s_start(reductions)
// --- reductions
/*---------------------------------------------------------
 * FN: fDot
 * DESC: calculates the dot product of two tensors
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto fDot(const CTensor &tA, const CTensor &tB) -> float;

/*---------------------------------------------------------
 * FN: fSum
 * DESC: calculates the sum of all elements in the tensor
 * PARMS: tA (tensor)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto fSum(const CTensor &tA) -> float;
// >>>s_end(reductions)

// <<<s_start(activations)
// --- activations
/*---------------------------------------------------------
 * FN: Silu
 * DESC: applies the sigmoid linear unit activation function
 * PARMS: tA (input tensor)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto Silu(const CTensor &tA) -> CTensor;

/*---------------------------------------------------------
 * FN: Relu
 * DESC: applies the rectified linear unit activation function
 * PARMS: tA (input tensor)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto Relu(const CTensor &tA) -> CTensor;
// >>>s_end(activations)

// <<<s_start(normalization)
// --- normalization
/*---------------------------------------------------------
 * FN: RmsNorm
 * DESC: applies root mean square normalization
 * PARMS: tX (input), tW (weights), fEps (epsilon)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto RmsNorm(const CTensor &tX, const CTensor &tW, float fEps = 1e-5f) -> CTensor;

/*---------------------------------------------------------
 * FN: Softmax
 * DESC: apolies the softmax function to the input tensor
 * PARMS: tA (input tensor)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto Softmax(const CTensor &tA) -> CTensor;
// >>>s_end(normalization)
} // namespace SM
} // namespace MT
