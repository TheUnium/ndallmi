// Created by Unium on 12.02.26

#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

namespace MT {

enum class EType { F32, F16, I32, I8 };

/*---------------------------------------------------------
 * FN: nGetTypeSize
 * DESC: returns byte size of a given dtype
 * PARMS: eType (dtype enum)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto nGetTypeSize(EType eType) -> size_t;

/*---------------------------------------------------------
 * FN: szGetTypeName
 * DESC: returns string name of dtype
 * PARMS: eType (dtype name)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto szGetTypeName(EType eType) -> const char *;

constexpr int mmDims = 8;

class CTensor {
private:
    // <<<s_start(internal)
    // --- internal helpers
    /*---------------------------------------------------------
     * FN: Allocate
     * DESC: allocs aligned memory for the tensor
     * PARMS: vlShape (dim size), eType (data type)
     * AUTH: unium (12.02.26)
     *-------------------------------------------------------*/
    void Allocate(const std::vector<int64_t> &vlShape, EType eType);

    /*---------------------------------------------------------
     * FN: ComputeStrides
     * DESC: computes row major strides from current strides
     * PARMS: none
     * AUTH: unium (12.02.26)
     *-------------------------------------------------------*/
    void ComputeStrides();
    // >>>s_end(internal)

    // <<<s_start(element)
    // --- element access
    /*---------------------------------------------------------
     * FN: lOffset
     * DESC: computes flat mem offset from multi dim indices
     * PARMS: lIndices (index per dim)
     * AUTH: unium (12.02.26)
     *-------------------------------------------------------*/
    auto lOffset(std::initializer_list<int64_t> lIndices) const -> int64_t;
    // >>>s_end(element)

public:
    // <<<v_start(metadata)
    int m_iNdim;
    int64_t m_lShape[mmDims];
    int64_t m_lStride[mmDims];
    EType m_eType;
    // >>>end(metadata)

    // <<<v_start(data)
    void *m_pData;
    size_t m_iDataSize;
    bool m_bOwnsData;
    // >>>end(data)

    // <<<s_start(constructors)
    // --- constructors/destructors
    /*---------------------------------------------------------
     * FN: CTensor (def)
     * DESC: constructs empty tensor with no data
     * PARMS: none
     * AUTH: unium (12.02.26)
     *-------------------------------------------------------*/
    CTensor();

    /*---------------------------------------------------------
     * FN: CTensor (initializer_list)
     * DESC: constructs a tensor with shape and dtype given
     * PARMS: lShape (dim sizes), eType (data type)
     * AUTH: unium (12.02.26)
     *-------------------------------------------------------*/
    CTensor(std::initializer_list<int64_t> lShape, EType eType = EType::F32);

    /*---------------------------------------------------------
     * FN: CTensor (vector)
     * DESC: constructs a tensor with shape vec and dtype given
     * PARMS: vlShape (dim sizes), EType (data type)
     * AUTH: unium (12.02.26)
     *-------------------------------------------------------*/
    CTensor(const std::vector<int64_t> &vlShape, EType eType = EType::F32);

    /*---------------------------------------------------------
     * FN: ~CTensor
     * DESC: destroys tensor and frees owned mem
     * PARMS: none
     * AUTH: unium (12.02.26)
     *-------------------------------------------------------*/
    ~CTensor();

    // no copy!! only move!!
    CTensor(const CTensor &) = delete;
    CTensor &operator=(const CTensor &) = delete;

    /*---------------------------------------------------------
     * FN: CTensor (move)
     * DESC: move constructs from another tensor and becomes
     *       romanian by stealing its data
     * PARMS: other (source tensor)
     * AUTH: unium (12.02.26)
     *-------------------------------------------------------*/
    CTensor(CTensor &&other) noexcept;

    /*---------------------------------------------------------
     * FN: operator= (move)
     * DESC: move assigns from another tensor and becomes
     *       romanian by stealing its data
     * PARMS: other (source tensor)
     * AUTH: unium (12.02.26)
     *-------------------------------------------------------*/
    CTensor &operator=(CTensor &&other) noexcept;

    /*---------------------------------------------------------
     * FN: Clone
     * DESC: deep copies the tensor, new tensor owns its data
     * PARMS: none
     * AUTH: unium (12.02.26)
     *-------------------------------------------------------*/
    auto Clone() const -> CTensor;
    // >>>s_end(destructor)

    // <<<s_start(factory_init_list)
    // --- factory methods (initializer_list)
    /*---------------------------------------------------------
     * FN: Zeros (initializer_list)
     * DESC: creates a tensor filled with zeros
     * PARMS: lShape (dims), eType (data type)
     * AUTH: unium (12.02.26 R: 20.02.26)
     *-------------------------------------------------------*/
    static auto Zeros(std::initializer_list<int64_t> lShape, EType eType = EType::F32) -> CTensor;

    /*---------------------------------------------------------
     * FN: Ones (initializer_list)
     * DESC: creates a tensor filled with ones
     * PARMS: lShape (dims), eType (data type)
     * AUTH: unium (12.02.26 R: 20.02.26)
     *-------------------------------------------------------*/
    static auto Ones(std::initializer_list<int64_t> lShape, EType eType = EType::F32) -> CTensor;

    /*---------------------------------------------------------
     * FN: Fill (initializer_list)
     * DESC: creates a tensor filled with a given value
     * PARMS: lShape (dims), fVal (fill value), eType
     *        (data type)
     * AUTH: unium (12.02.26 R: 20.02.26)
     *-------------------------------------------------------*/
    static auto Fill(std::initializer_list<int64_t> lShape, float fVal, EType eType = EType::F32) -> CTensor;

    /*---------------------------------------------------------
     * FN: Rand (initializer_list)
     * DESC: creates a tensor filled with random values in [0,1\
     * PARMS: lShape (dims), eType (data type)
     * AUTH: unium (12.02.26 R: 20.02.26)
     *-------------------------------------------------------*/
    static auto Rand(std::initializer_list<int64_t> lShape, EType eType = EType::F32) -> CTensor;
    // >>>s_end(factory_init_list)

    // <<<s_start(factory_vector)
    // ---- factory methods (vector) ----
    /*---------------------------------------------------------
     * FN: Zeros (vector)
     * DESC: creates a tensor filled with zeros
     * PARMS: vlShape (dims), eType (data type)
     * AUTH: unium (12.02.26 R: 20.02.26)
     *-------------------------------------------------------*/
    static auto Zeros(const std::vector<int64_t> &vlShape, EType eType = EType::F32) -> CTensor;

    /*---------------------------------------------------------
     * FN: Ones (vector)
     * DESC: creates a tensor filled with ones
     * PARMS: vlShape (dims), eType (data type)
     * AUTH: unium (12.02.26 R: 20.02.26)
     *-------------------------------------------------------*/
    static auto Ones(const std::vector<int64_t> &vlShape, EType eType = EType::F32) -> CTensor;

    /*---------------------------------------------------------
     * FN: Fill (vector)
     * DESC: creates a tensor filled with a given value
     * PARMS: vlShape (dim), fVal (fill value), eType
     *        (data type)
     * AUTH: unium (12.02.26 R: 20.02.26)
     *-------------------------------------------------------*/
    static auto Fill(const std::vector<int64_t> &vlShape, float fVal, EType eType = EType::F32) -> CTensor;

    /*---------------------------------------------------------
     * FN: Rand (vector)
     * DESC: creates a tensor filled with random values in [0,1)
     * PARMS: vlShape (dim), eType (data type)
     * AUTH: unium (12.02.26 R: 20.02.26)
     *-------------------------------------------------------*/
    static auto Rand(const std::vector<int64_t> &vlShape, EType eType = EType::F32) -> CTensor;
    // >>>s_end(factory_vector)

    // <<<s_start(element)
    // --- element access
    /*---------------------------------------------------------
     * FN: lNumel
     * DESC: returns total number of elements in the tensor
     * PARMS: none
     * AUTH: unium (12.02.26 R: 20.02.26)
     *-------------------------------------------------------*/
    auto lNumel() const -> int64_t;

    /*---------------------------------------------------------
     * FN: fAt
     * DESC: access element by multi dim index (mut)
     * PARMS: lIndices (index per dim)
     * AUTH: unium (12.02.26 R: 20.02.26)
     *-------------------------------------------------------*/
    auto fAt(std::initializer_list<int64_t> lIndices) -> float &;

    /*---------------------------------------------------------
     * FN: fAt (const)
     * DESC: access element by multi dim index (ro)
     * PARMS: lIndices (index per dim)
     * AUTH: unium (12.02.26 R: 20.02.26)
     *-------------------------------------------------------*/
    auto fAt(std::initializer_list<int64_t> lIndices) const -> float;

    /*---------------------------------------------------------
     * FN: fFlat
     * DESC: access element by flat index (mut)
     * PARMS: lIdx (flat index)
     * AUTH: unium (12.02.26 R: 20.02.26)
     *-------------------------------------------------------*/
    auto fFlat(int64_t lIdx) -> float &;

    /*---------------------------------------------------------
     * FN: fFlat (const)
     * DESC: access element by flat index (ro)
     * PARMS: lIdx (flat index)
     * AUTH: unium (12.02.26 R: 20.02.26)
     *-------------------------------------------------------*/
    auto fFlat(int64_t lIdx) const -> float;

    /*---------------------------------------------------------
     * FN: pfData
     * DESC: returns raw float pointer to tensor data (mut)
     * PARMS: none
     * AUTH: unium (12.02.26 R: 20.02.26)
     *-------------------------------------------------------*/
    auto pfData() -> float *;

    /*---------------------------------------------------------
     * FN: pfData (const)
     * DESC: returns raw float pointer to tensor data (ro)
     * PARMS: none
     * AUTH: unium (12.02.26 R: 20.02.26)
     *-------------------------------------------------------*/
    auto pfData() const -> const float *;
    // >>>s_end(element)

    // <<<s_start(shape_manip)
    // --- shape manipulation
    /*---------------------------------------------------------
     * FN: Reshape
     * DESC: returns a view with new shape, -1 infers one dim
     * PARMS: lNewShape (new dim sizes)
     * AUTH: unium (12.02.26 R: 20.02.26)
     *-------------------------------------------------------*/
    auto Reshape(std::initializer_list<int64_t> lNewShape) const -> CTensor;

    /*---------------------------------------------------------
     * FN: View
     * DESC: alias for Reshape on contiguous tensors
     * PARMS: lNewShape (new dim sizes)
     * AUTH: unium (12.02.26)
     *-------------------------------------------------------*/
    auto View(std::initializer_list<int64_t> lNewShape) const -> CTensor;

    /*---------------------------------------------------------
     * FN: Transpose
     * DESC: returns a view with two dims swapped
     * PARMS: iDim0 (first dim), iDim1 (second dim)
     * AUTH: unium (12.02.26)
     *-------------------------------------------------------*/
    auto Transpose(int iDim0, int iDim1) const -> CTensor;

    /*---------------------------------------------------------
     * FN: Contiguous
     * DESC: returns a contiguous copy if strides are non standard
     * PARMS: none
     * AUTH: unium (12.02.26)
     *-------------------------------------------------------*/
    auto Contiguous() const -> CTensor;

    /*---------------------------------------------------------
     * FN: bIsContiguous
     * DESC: checks if tensor memory layout is contiguous
     * PARMS: none
     * AUTH: unium (12.02.26)
     *-------------------------------------------------------*/
    auto bIsContiguous() const -> bool;
    // >>>s_end(shape_manip)

    // <<<s_start(utils)
    // --- printing utils
    /*---------------------------------------------------------
     * FN: Print
     * DESC: prints tensor data to stdout
     * PARMS: szName (optional label)
     * AUTH: unium (12.02.26)
     *-------------------------------------------------------*/
    void Print(const std::string &szName = "") const;

    /*---------------------------------------------------------
     * FN: PrintShape
     * DESC: prints tensor shape and dtype to stdout
     * PARMS: szName (optional label)
     * AUTH: unium (12.02.26)
     *-------------------------------------------------------*/
    void PrintShape(const std::string &szName = "") const;
    // >>>s_end(utils)
};
} // namespace MT
