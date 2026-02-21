// Created by Unium on 2.02.26

#pragma once

#include "mtTnTnsr.hpp"

namespace MT {
namespace OP {
// <<<s_start(element_wb)
// --- element wise binary
/*---------------------------------------------------------
 * FN: Add
 * DESC: element wise add of two same shape tensoir
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (12.02.26 R: 21.02.26)
 *-------------------------------------------------------*/
auto Add(const CTensor &tA, const CTensor &tB) -> CTensor;

/*---------------------------------------------------------
 * FN: Sub
 * DESC: element wise subtraction of two same shape tensors
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (12.02.26 R: 21.02.26)
 *-------------------------------------------------------*/
auto Sub(const CTensor &tA, const CTensor &tB) -> CTensor;

/*---------------------------------------------------------
 * FN: Mul
 * DESC: element wise multiplication of two same shape tensors
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (12.02.26 R: 21.02.26)
 *-------------------------------------------------------*/
auto Mul(const CTensor &tA, const CTensor &tB) -> CTensor;

/*---------------------------------------------------------
 * FN: Div
 * DESC: element wise division of two same shape tensors
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (12.02.26 R: 21.02.26)
 *-------------------------------------------------------*/
auto Div(const CTensor &tA, const CTensor &tB) -> CTensor;
// >>>s_end(element_wb)

// <<<s_start(broadcast)
// --- binary broadcast stuff
/*---------------------------------------------------------
 * FN: AddBroadcast
 * DESC: element wise add with bcasting
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (13.02.26 R: 19.02.26)
 *-------------------------------------------------------*/
auto AddBroadcast(const CTensor &tA, const CTensor &tB) -> CTensor;

/*---------------------------------------------------------
 * FN: SubBroadcast
 * DESC: element wise sub with bcasting
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (13.02.26 R: 19.02.26)
 *-------------------------------------------------------*/
auto SubBroadcast(const CTensor &tA, const CTensor &tB) -> CTensor;

/*---------------------------------------------------------
 * FN: MulBroadcast
 * DESC: element wise mul with bcasting
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (13.02.26 R: 19.02.26)
 *-------------------------------------------------------*/
auto MulBroadcast(const CTensor &tA, const CTensor &tB) -> CTensor;

/*---------------------------------------------------------
 * FN: DivBroadcast
 * DESC: element wise div with bcasting
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (13.02.26 R: 19.02.26)
 *-------------------------------------------------------*/
auto DivBroadcast(const CTensor &tA, const CTensor &tB) -> CTensor;
// >>>s_end(broadcast)

// <<<s_start(scalar)
// --- scalar opers
/*---------------------------------------------------------
 * FN: AddScalar
 * DESC: adds a scalar to every element
 * PARMS: tA (tensor), fS (scalar)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto AddScalar(const CTensor &tA, float fS) -> CTensor;

/*---------------------------------------------------------
 * FN: MulScalar
 * DESC: multiplies every element by a scalar
 * PARMS: tA (tensor), fS (scalar)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto MulScalar(const CTensor &tA, float fS) -> CTensor;
// >>>s_end(scalar)

// <<<s_start(matrix)
// --- matrix mult
/*---------------------------------------------------------
 * FN: Matmul
 * DESC: 2d matrix multiply
 *       [i.e., (m, k) x (k, n) --> (m, n)]
 * PARMS: tA (left matrix), tB (right matrix)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto Matmul(const CTensor &tA, const CTensor &tB) -> CTensor;

/*---------------------------------------------------------
 * FN: Bmm
 * DESC: batched matrix multiply
 *       [i.e., (b, m, k) x (b, k, n) --> (b, m, n)]
 * PARMS: tA (left batch), tB (right batch)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto Bmm(const CTensor &tA, const CTensor &tB) -> CTensor;

/*---------------------------------------------------------
 * FN: Matvec
 * DESC: matrix-vector multiply
 *       [i.e., (m, k) x (k) -> (m)]
 * PARMS: tMat (matrix), tVec (vector)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto Matvec(const CTensor &tMat, const CTensor &tVec) -> CTensor;
// >>>s_end(matrix)

// <<<s_start(reduction)
// --- reductions
/*---------------------------------------------------------
 * FN: Sum
 * DESC: sum elements (o: dimension)
 * PARMS: tA (tensor), iDim (dimension, -1 for all)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto Sum(const CTensor &tA, int iDim = -1) -> CTensor;

/*---------------------------------------------------------
 * FN: Mean
 * DESC: mean of elements (o: dimension)
 * PARMS: tA (tensor), iDim (dimension, -1 for all)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto Mean(const CTensor &tA, int iDim = -1) -> CTensor;

/*---------------------------------------------------------
 * FN: Max
 * DESC: max of elements (o: dimension)
 * PARMS: tA (tensor), iDim (dimension, -1 for all)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto Max(const CTensor &tA, int iDim = -1) -> CTensor;

/*---------------------------------------------------------
 * FN: Argmaxxing
 * DESC: index of max element along dimension
 * PARMS: tA (tensor), iDim (dimension, -1 for flat)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Argmaxxing(const CTensor &tA, int iDim = -1) -> CTensor;
// >>>s_end(reductions)

// <<<s_start(activiation)
// --- activations
/*---------------------------------------------------------
 * FN: Relu
 * DESC: rectified linear unit, max(0, x)
 * PARMS: tA (input tensor)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto Relu(const CTensor &tA) -> CTensor;

/*---------------------------------------------------------
 * FN: Silu
 * DESC: sigmoid linear unit, x * sigmoid(x)
 * PARMS: tA (input tensor)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto Silu(const CTensor &tA) -> CTensor;

/*---------------------------------------------------------
 * FN: Gelu
 * DESC: gaussian error linear unit (approximate)
 * PARMS: tA (input tensor)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto Gelu(const CTensor &tA) -> CTensor;

/*---------------------------------------------------------
 * FN: Softmax
 * DESC: softmax along a dimension (default last)
 * PARMS: tA (input tensor), iDim (dimension)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto Softmax(const CTensor &tA, int iDim = -1) -> CTensor;
// >>>s_end(activiation)

// <<<s_start(normalization)
// --- literally just rms norm
/*---------------------------------------------------------
 * FN: RmsNorm
 * DESC: root mean square normalization with learned weight
 * PARMS: tX (input), tW (weight), fEps (epsilon)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto RmsNorm(const CTensor &tX, const CTensor &tW, float fEps = 1e-5f) -> CTensor;
// >>>s_end(normalization)

// <<<s_start(unary)
// --- unary maths
/*---------------------------------------------------------
 * FN: Neg
 * DESC: negates every element
 * PARMS: tA (input tensor)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Neg(const CTensor &tA) -> CTensor;

/*---------------------------------------------------------
 * FN: Sqrt
 * DESC: square root of every element
 * PARMS: tA (input tensor)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Sqrt(const CTensor &tA) -> CTensor;

/*---------------------------------------------------------
 * FN: Rsqrt
 * DESC: reciprocal square root (1/sqrt) of every element
 * PARMS: tA (input tensor)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Rsqrt(const CTensor &tA) -> CTensor;

/*---------------------------------------------------------
 * FN: Abs
 * DESC: absolute value of every element
 * PARMS: tA (input tensor)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Abs(const CTensor &tA) -> CTensor;

/*---------------------------------------------------------
 * FN: Exp
 * DESC: e^x of every element
 * PARMS: tA (input tensor)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Exp(const CTensor &tA) -> CTensor;

/*---------------------------------------------------------
 * FN: Log
 * DESC: natural log of every element
 * PARMS: tA (input tensor)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Log(const CTensor &tA) -> CTensor;

/*---------------------------------------------------------
 * FN: Clamp
 * DESC: clamps every element to [fMin, fMax]
 * PARMS: tA (input), fMin (lower bound), fMax (upper bound)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Clamp(const CTensor &tA, float fMin, float fMax) -> CTensor;
// >>>s_end(unary)

// <<<s_start(dot)
// --- dot product
/*---------------------------------------------------------
 * FN: fDot
 * DESC: dot product of two 1D tensors
 * PARMS: tA (first vector), tB (second vector)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto fDot(const CTensor &tA, const CTensor &tB) -> float;
// >>>s_end(dot)

// <<<s_start(index)
// --- index/slicing
/*---------------------------------------------------------
 * FN: SliceRow
 * DESC: extracts a single row from dim 0 as a view
 *       (does not copy data)
 * PARMS: tA (tensor), lIdx (row index)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto SliceRow(const CTensor &tA, int64_t lIdx) -> CTensor;

/*---------------------------------------------------------
 * FN: SliceRange
 * DESC: extracts rows [lStart, lEnd) from dim 0 as a view
 * PARMS: tA (tensor), lStart (start idx), lEnd (end idx)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto SliceRange(const CTensor &tA, int64_t lStart, int64_t lEnd) -> CTensor;

/*---------------------------------------------------------
 * FN: Gather
 * DESC: gathers rows from tA using indices in tIdx
 *       tIdx is 1D int32, output shape = (len(tIdx), ...)
 *       used for token embedding lookup
 * PARMS: tA (source), tIdx (index tensor, I32)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Gather(const CTensor &tA, const CTensor &tIdx) -> CTensor;
// >>>s_end(index)

// <<<s_start(construction)
// --- construction/manipulation
/*---------------------------------------------------------
 * FN: Concat
 * DESC: concatenates tensors along a dimension
 * PARMS: vtTensors (vector of tensor pointers), iDim (dim)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Concat(const std::vector<const CTensor *> &vtTensors, int iDim = 0) -> CTensor;

/*---------------------------------------------------------
 * FN: Repeat
 * DESC: repeats tensor along a dimension
 * PARMS: tA (tensor), iDim (dimension), iCount (repeats)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Repeat(const CTensor &tA, int iDim, int64_t iCount) -> CTensor;

/*---------------------------------------------------------
 * FN: Arange
 * DESC: creates a 1D tensor [fStart, fStart+1, ..., fEnd-1]
 * PARMS: fStart (start val), fEnd (end val), fStep (step)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Arange(float fStart, float fEnd, float fStep = 1.0f) -> CTensor;

/*---------------------------------------------------------
 * FN: TriMask
 * DESC: creates an (iSize x iSize) upper triangular mask
 *       with fFillVal above diagonal, 0 on and below
 *       used for causal attention masking
 * PARMS: iSize (matrix size), fFillVal (mask value)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto TriMask(int64_t iSize, float fFillVal = -1e9f) -> CTensor;

/*---------------------------------------------------------
 * FN: Where
 * DESC: element-wise conditional: out[i] = cond[i] ? tA[i] : tB[i]
 *       condition is true when value > 0
 * PARMS: tCond (condition), tA (true vals), tB (false vals)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Where(const CTensor &tCond, const CTensor &tA, const CTensor &tB) -> CTensor;
// >>>s_end(construction)

// <<<s_start(ipo)
// --- in place opers
/*---------------------------------------------------------
 * FN: CopyInto
 * DESC: copies tSrc data into tDst at row offset
 *       used for KV cache updates without allocation
 * PARMS: tDst (destination), tSrc (source), lRowOffset (row)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
void CopyInto(CTensor &tDst, const CTensor &tSrc, int64_t lRowOffset = 0);

/*---------------------------------------------------------
 * FN: FillInplace
 * DESC: fills entire tensor with a value in place
 * PARMS: tA (tensor to fill), fVal (value)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
void FillInplace(CTensor &tA, float fVal);
// >>>s_end(ipo)
} // namespace OP
} // namespace MT
