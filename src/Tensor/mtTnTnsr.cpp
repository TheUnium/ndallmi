// Created by Unium on 20.02.26

#include "mtTnTnsr.hpp"
#include <algorithm>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace MT {

// <<<ignore
/*---------------------------------------------------------
 * FN: nGetTypeSize
 * DESC: returns byte size of a given dtype
 * PARMS: eType (dtype enum)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto nGetTypeSize(EType eType) -> size_t {
    switch (eType) {
    case EType::F32:
        return 4;
    case EType::F16:
        return 2;
    case EType::I32:
        return 4;
    case EType::I8:
        return 1;
    }
    return 0;
}

/*---------------------------------------------------------
 * FN: szGetTypeName
 * DESC: returns string name of a given dtype
 * PARMS: eType (dtype enum)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto szGetTypeName(EType eType) -> const char * {
    switch (eType) {
    case EType::F32:
        return "f32";
    case EType::F16:
        return "f16";
    case EType::I32:
        return "i32";
    case EType::I8:
        return "i8";
    }
    return "unknown";
}
// >>>ignore

// <<<s_start(constructors)
// --- constructors / destructions
/*---------------------------------------------------------
 * FN: CTensor (def)
 * DESC: constructs empty tensor with no data
 * PARMS: none
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
CTensor::CTensor() : m_iNdim(0), m_eType(EType::F32), m_pData(nullptr), m_iDataSize(0), m_bOwnsData(false) {
    std::memset(m_lShape, 0, sizeof(m_lShape));
    std::memset(m_lStride, 0, sizeof(m_lStride));
}

/*---------------------------------------------------------
 * FN: CTensor (initializer_list)
 * DESC: constructs a tensor with shape and dtype given
 * PARMS: lShape (dim sizes), eType (data type)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
CTensor::CTensor(std::initializer_list<int64_t> lShape, EType eType) { Allocate(std::vector<int64_t>(lShape), eType); }

/*---------------------------------------------------------
 * FN: CTensor (vector)
 * DESC: constructs a tensor with shape vec and dtype given
 * PARMS: vlShape (dim sizes), EType (data type)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
CTensor::CTensor(const std::vector<int64_t> &vlShape, EType eType) { Allocate(vlShape, eType); }

/*---------------------------------------------------------
 * FN: ~CTensor
 * DESC: destroys tensor and frees owned mem
 * PARMS: none
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
CTensor::~CTensor() {
    if (m_bOwnsData && m_pData) {
        std::free(m_pData);
        m_pData = nullptr;
    }
}

/*---------------------------------------------------------
 * FN: CTensor (move)
 * DESC: move constructs from another tensor and becomes
 *       romanian by stealing its data
 * PARMS: other (source tensor)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
CTensor::CTensor(CTensor &&other) noexcept
    : m_iNdim(other.m_iNdim), m_eType(other.m_eType), m_pData(other.m_pData), m_iDataSize(other.m_iDataSize),
      m_bOwnsData(other.m_bOwnsData) {
    std::memcpy(m_lShape, other.m_lShape, sizeof(m_lShape));
    std::memcpy(m_lStride, other.m_lStride, sizeof(m_lStride));
    other.m_pData = nullptr;
    other.m_bOwnsData = false;
    other.m_iNdim = 0;
}

/*---------------------------------------------------------
 * FN: operator= (move)
 * DESC: move assigns from another tensor and becomes
 *       romanian by stealing its data
 * PARMS: other (source tensor)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
CTensor &CTensor::operator=(CTensor &&other) noexcept {
    if (this != &other) {
        if (m_bOwnsData && m_pData) {
            std::free(m_pData);
        }
        m_iNdim = other.m_iNdim;
        m_eType = other.m_eType;
        m_pData = other.m_pData;
        m_iDataSize = other.m_iDataSize;
        m_bOwnsData = other.m_bOwnsData;
        std::memcpy(m_lShape, other.m_lShape, sizeof(m_lShape));
        std::memcpy(m_lStride, other.m_lStride, sizeof(m_lStride));
        other.m_pData = nullptr;
        other.m_bOwnsData = false;
        other.m_iNdim = 0;
    }
    return *this;
}

/*---------------------------------------------------------
 * FN: Clone
 * DESC: deep copies the tensor, new tensor owns its data
 * PARMS: none
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto CTensor::Clone() const -> CTensor {
    CTensor tOut;
    tOut.m_iNdim = m_iNdim;
    tOut.m_eType = m_eType;
    tOut.m_iDataSize = m_iDataSize;
    tOut.m_bOwnsData = true;
    std::memcpy(tOut.m_lShape, m_lShape, sizeof(m_lShape));
    std::memcpy(tOut.m_lStride, m_lStride, sizeof(m_lStride));

    tOut.m_pData = std::malloc(m_iDataSize);
    assert(tOut.m_pData && "[ct:clone] malloc failed");
    std::memcpy(tOut.m_pData, m_pData, m_iDataSize);
    return tOut;
}
// >>>s_end(constructors)

// <<<s_start(factory_init_list)
// --- factory methods (initializer_list)
/*---------------------------------------------------------
 * FN: Zeros (initializer_list)
 * DESC: creates a tensor filled with zeros
 * PARMS: lShape (dims), eType (data type)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto CTensor::Zeros(std::initializer_list<int64_t> lShape, EType eType) -> CTensor {
    return Zeros(std::vector<int64_t>(lShape), eType);
}

/*---------------------------------------------------------
 * FN: Ones (initializer_list)
 * DESC: creates a tensor filled with ones
 * PARMS: lShape (dims), eType (data type)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto CTensor::Ones(std::initializer_list<int64_t> lShape, EType eType) -> CTensor {
    return Ones(std::vector<int64_t>(lShape), eType);
}

/*---------------------------------------------------------
 * FN: Fill (initializer_list)
 * DESC: creates a tensor filled with a given value
 * PARMS: lShape (dims), fVal (fill value), eType
 *        (data type)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto CTensor::Fill(std::initializer_list<int64_t> lShape, float fVal, EType eType) -> CTensor {
    return Fill(std::vector<int64_t>(lShape), fVal, eType);
}

/*---------------------------------------------------------
 * FN: Rand (initializer_list)
 * DESC: creates a tensor filled with random values in [0,1\
 * PARMS: lShape (dims), eType (data type)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto CTensor::Rand(std::initializer_list<int64_t> lShape, EType eType) -> CTensor {
    return Rand(std::vector<int64_t>(lShape), eType);
}
// >>>s_end(factory_init_list)

// <<<s_start(factory_vector)
// ---- factory methods (vector) ----
/*---------------------------------------------------------
 * FN: Zeros (vector)
 * DESC: creates a tensor filled with zeros
 * PARMS: vlShape (dims), eType (data type)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto CTensor::Zeros(const std::vector<int64_t> &vlShape, EType eType) -> CTensor {
    CTensor t(vlShape, eType);
    std::memset(t.m_pData, 0, t.m_iDataSize);
    return t;
}

/*---------------------------------------------------------
 * FN: Ones (vector)
 * DESC: creates a tensor filled with ones
 * PARMS: vlShape (dims), eType (data type)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto CTensor::Ones(const std::vector<int64_t> &vlShape, EType eType) -> CTensor { return Fill(vlShape, 1.0f, eType); }

/*---------------------------------------------------------
 * FN: Fill (vector)
 * DESC: creates a tensor filled with a given value
 * PARMS: vlShape (dim), fVal (fill value), eType
 *        (data type)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto CTensor::Fill(const std::vector<int64_t> &vlShape, float fVal, EType eType) -> CTensor {
    CTensor t(vlShape, eType);
    assert(eType == EType::F32 && "[ct:fill:vector] fill only supports f32 for now");
    int64_t lN = t.lNumel();
    float *pfPtr = t.pfData();
    if (fVal == 0.0f) {
        std::memset(pfPtr, 0, lN * sizeof(float));
        return t;
    }

    pfPtr[0] = fVal;
    int64_t lFilled = 1;
    while (lFilled < lN) {
        int64_t lChunk = std::min(lFilled, lN - lFilled);
        std::memcpy(pfPtr + lFilled, pfPtr, lChunk * sizeof(float));
        lFilled += lChunk;
    }

    return t;
}

/*---------------------------------------------------------
 * FN: Rand (vector)
 * DESC: creates a tensor filled with random values in [0,1)
 * PARMS: vlShape (dim), eType (data type)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto CTensor::Rand(const std::vector<int64_t> &vlShape, EType eType) -> CTensor {
    CTensor t(vlShape, eType);
    assert(eType == EType::F32 && "[ct:rand:vector] rand only supports f32 for now");

    static std::mt19937 s_rng(42);
    int64_t lN = t.lNumel();
    float *pfPtr = t.pfData();

    // gen as uint32 and convert to [0,1] using bit magic
    // [ieee 754] force range [1, 2] and yoink 1.0
    auto *pU32 = reinterpret_cast<uint32_t *>(pfPtr);
    int64_t i = 0;

    for (; i + 3 < lN; i += 4) {
        uint32_t r0 = s_rng();
        uint32_t r1 = s_rng();
        uint32_t r2 = s_rng();
        uint32_t r3 = s_rng();
        pU32[i + 0] = (r0 >> 9) | 0x3F800000u;
        pU32[i + 1] = (r1 >> 9) | 0x3F800000u;
        pU32[i + 2] = (r2 >> 9) | 0x3F800000u;
        pU32[i + 3] = (r3 >> 9) | 0x3F800000u;
        pfPtr[i + 0] -= 1.0f;
        pfPtr[i + 1] -= 1.0f;
        pfPtr[i + 2] -= 1.0f;
        pfPtr[i + 3] -= 1.0f;
    }
    for (; i < lN; i++) {
        uint32_t r = s_rng();
        pU32[i] = (r >> 9) | 0x3F800000u;
        pfPtr[i] -= 1.0f;
    }

    return t;
}
// >>>s_end(factory_vector)

// <<<s_start(element)
// --- element access
/*---------------------------------------------------------
 * FN: lNumel
 * DESC: returns total number of elements in the tensor
 * PARMS: none
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto CTensor::lNumel() const -> int64_t {
    if (m_iNdim == 0)
        return 0;
    int64_t lN = 1;
    for (int i = 0; i < m_iNdim; i++) {
        lN *= m_lShape[i];
    }
    return lN;
}

/*---------------------------------------------------------
 * FN: lOffset
 * DESC: computes flat mem offset from multi dim indices
 * PARMS: lIndices (index per dim)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto CTensor::lOffset(std::initializer_list<int64_t> lIndices) const -> int64_t {
    assert((int)lIndices.size() == m_iNdim && "[ct:loffset] wrong number of indices womp womp");
    int64_t lOff = 0;
    int i = 0;
    for (auto lIdx : lIndices) {
        assert(lIdx >= 0 && lIdx < m_lShape[i] && "[ct:loffset] index out of bounds");
        lOff += lIdx * m_lStride[i];
        i++;
    }
    return lOff;
}

/*---------------------------------------------------------
 * FN: fAt
 * DESC: access element by multi dim index (mut)
 * PARMS: lIndices (index per dim)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto CTensor::fAt(std::initializer_list<int64_t> lIndices) -> float & {
    assert(m_eType == EType::F32 && "[ct:fat] fAt() is only for f32");
    return pfData()[lOffset(lIndices)];
}

/*---------------------------------------------------------
 * FN: fAt (const)
 * DESC: access element by multi dim index (ro)
 * PARMS: lIndices (index per dim)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto CTensor::fAt(std::initializer_list<int64_t> lIndices) const -> float {
    assert(m_eType == EType::F32 && "[ct:fat] fAt() is only for f32");
    return pfData()[lOffset(lIndices)];
}

/*---------------------------------------------------------
 * FN: fFlat
 * DESC: access element by flat index (mut)
 * PARMS: lIdx (flat index)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto CTensor::fFlat(int64_t lIdx) -> float & {
    assert(m_eType == EType::F32 && "[ct:fflat] fFlat() is only for f32");
    assert(lIdx >= 0 && lIdx < lNumel());
    return pfData()[lIdx];
}

/*---------------------------------------------------------
 * FN: fFlat (const)
 * DESC: access element by flat index (ro)
 * PARMS: lIdx (flat index)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto CTensor::fFlat(int64_t lIdx) const -> float {
    assert(m_eType == EType::F32 && "[ct:fflat] fFlat() is only for f32");
    assert(lIdx >= 0 && lIdx < lNumel());
    return pfData()[lIdx];
}

/*---------------------------------------------------------
 * FN: pfData
 * DESC: returns raw float pointer to tensor data (mut)
 * PARMS: none
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto CTensor::pfData() -> float * {
    assert(m_eType == EType::F32);
    return static_cast<float *>(m_pData);
}

/*---------------------------------------------------------
 * FN: pfData (const)
 * DESC: returns raw float pointer to tensor data (ro)
 * PARMS: none
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto CTensor::pfData() const -> const float * {
    assert(m_eType == EType::F32);
    return static_cast<const float *>(m_pData);
}
// >>>s_end(element)

// <<<s_start(shape_manip)
// --- shape manipulation
/*---------------------------------------------------------
 * FN: Reshape
 * DESC: returns a view with new shape, -1 infers one dim
 * PARMS: lNewShape (new dim sizes)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto CTensor::Reshape(std::initializer_list<int64_t> lNewShape) const -> CTensor {
    assert(bIsContiguous() && "[ct:reshape] rehsape requires a contiguous tensor!");

    std::vector<int64_t> vlNew(lNewShape);
    int64_t lNewNumel = 1;
    int iNegIdx = -1;
    for (int i = 0; i < (int)vlNew.size(); i++) {
        if (vlNew[i] == -1) {
            assert(iNegIdx == -1 && "[ct:reshape] only one -1 allowed in reshape");
            iNegIdx = i;
        } else {
            lNewNumel *= vlNew[i];
        }
    }

    if (iNegIdx >= 0) {
        vlNew[iNegIdx] = lNumel() / lNewNumel;
        lNewNumel *= vlNew[iNegIdx];
    }

    assert(lNewNumel == lNumel() && "[ct:reshape] element count mismatch");

    CTensor tOut;
    tOut.m_iNdim = (int)vlNew.size();
    tOut.m_eType = m_eType;
    tOut.m_pData = m_pData;
    tOut.m_iDataSize = m_iDataSize;
    tOut.m_bOwnsData = false;

    for (int i = 0; i < tOut.m_iNdim; i++) {
        tOut.m_lShape[i] = vlNew[i];
    }
    tOut.ComputeStrides();
    return tOut;
}

/*---------------------------------------------------------
 * FN: View
 * DESC: alias for Reshape on contiguous tensors
 * PARMS: lNewShape (new dim sizes)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto CTensor::View(std::initializer_list<int64_t> lNewShape) const -> CTensor { return Reshape(lNewShape); }

/*---------------------------------------------------------
 * FN: Transpose
 * DESC: returns a view with two dims swapped
 * PARMS: iDim0 (first dim), iDim1 (second dim)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto CTensor::Transpose(int iDim0, int iDim1) const -> CTensor {
    assert(iDim0 >= 0 && iDim0 < m_iNdim);
    assert(iDim1 >= 0 && iDim1 < m_iNdim);

    CTensor tOut;
    tOut.m_iNdim = m_iNdim;
    tOut.m_eType = m_eType;
    tOut.m_pData = m_pData;
    tOut.m_iDataSize = m_iDataSize;
    tOut.m_bOwnsData = false;

    std::memcpy(tOut.m_lShape, m_lShape, sizeof(m_lShape));
    std::memcpy(tOut.m_lStride, m_lStride, sizeof(m_lStride));

    std::swap(tOut.m_lShape[iDim0], tOut.m_lShape[iDim1]);
    std::swap(tOut.m_lStride[iDim0], tOut.m_lStride[iDim1]);

    return tOut;
}

/*---------------------------------------------------------
 * FN: bIsContiguous
 * DESC: checks if tensor memory layout is contiguous
 * PARMS: none
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto CTensor::bIsContiguous() const -> bool {
    if (m_iNdim == 0)
        return true;
    int64_t lExpected = 1;
    for (int i = m_iNdim - 1; i >= 0; i--) {
        if (m_lStride[i] != lExpected)
            return false;
        lExpected *= m_lShape[i];
    }
    return true;
}

/*---------------------------------------------------------
 * FN: Contiguous
 * DESC: returns a contiguous copy if strides are non standard
 * PARMS: none
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
auto CTensor::Contiguous() const -> CTensor {
    if (bIsContiguous()) {
        CTensor tOut;
        tOut.m_iNdim = m_iNdim;
        tOut.m_eType = m_eType;
        tOut.m_pData = m_pData;
        tOut.m_iDataSize = m_iDataSize;
        tOut.m_bOwnsData = false;
        std::memcpy(tOut.m_lShape, m_lShape, sizeof(m_lShape));
        std::memcpy(tOut.m_lStride, m_lStride, sizeof(m_lStride));
        return tOut;
    }

    assert(m_eType == EType::F32 && "[ct:contiguous] cont only supports f32 for now");

    CTensor tOut(std::vector<int64_t>(m_lShape, m_lShape + m_iNdim), m_eType);

    const float *pfSrc = static_cast<const float *>(m_pData);
    float *pfDst = tOut.pfData();

    // 2d transpose (literally 99.99% of cases)
    if (m_iNdim == 2) {
        int64_t lRows = m_lShape[0];
        int64_t lCols = m_lShape[1];
        int64_t lStride0 = m_lStride[0];
        int64_t lStride1 = m_lStride[1];

        // why onion whyyyy 😢😢😢😢😢😢 onion whyy 😢 why onion why make cpu cryy whyyyyy
        constexpr int64_t kBlock = 16;
        for (int64_t r = 0; r < lRows; r += kBlock) {
            int64_t rEnd = std::min(r + kBlock, lRows);
            for (int64_t c = 0; c < lCols; c += kBlock) {
                int64_t cEnd = std::min(c + kBlock, lCols);
                for (int64_t rr = r; rr < rEnd; rr++) {
                    int64_t lDstBase = rr * lCols;
                    int64_t lSrcBase = rr * lStride0;
                    for (int64_t cc = c; cc < cEnd; cc++) {
                        pfDst[lDstBase + cc] = pfSrc[lSrcBase + cc * lStride1];
                    }
                }
            }
        }
        return tOut;
    }

    int64_t lN = lNumel();
    int64_t lIdx[mmDims] = {};

    for (int64_t lFlat = 0; lFlat < lN; lFlat++) {
        int64_t lSrcOff = 0;
        for (int d = 0; d < m_iNdim; d++) {
            lSrcOff += lIdx[d] * m_lStride[d];
        }

        pfDst[lFlat] = pfSrc[lSrcOff];

        for (int d = m_iNdim - 1; d >= 0; d--) {
            if (++lIdx[d] < m_lShape[d])
                break;
            lIdx[d] = 0;
        }
    }

    return tOut;
}
// <<<s_end(shape_manip)

// <<<s_start(utils)
// --- printing utils
/*---------------------------------------------------------
 * FN: PrintShape
 * DESC: prints tensor shape and dtype to stdout
 * PARMS: szName (optional label)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
void CTensor::PrintShape(const std::string &szName) const {
    if (!szName.empty())
        std::cout << szName << ": ";
    std::cout << "CTensor(";
    for (int i = 0; i < m_iNdim; i++) {
        if (i > 0)
            std::cout << ", ";
        std::cout << m_lShape[i];
    }
    std::cout << ", " << szGetTypeName(m_eType) << ")" << std::endl;
}

/*---------------------------------------------------------
 * FN: Print
 * DESC: prints tensor data to stdout
 * PARMS: szName (optional label)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
void CTensor::Print(const std::string &szName) const {
    PrintShape(szName);

    if (m_eType != EType::F32) {
        std::cout << "  (printing only supported for f32)" << std::endl;
        return;
    }

    int64_t lN = lNumel();
    if (lN == 0) {
        std::cout << "  []" << std::endl;
        return;
    }

    int64_t lMaxShow = 20;
    bool bTruncated = lN > lMaxShow;

    std::cout << std::fixed << std::setprecision(4);

    if (m_iNdim == 1) {
        std::cout << "  [";
        for (int64_t i = 0; i < std::min(lN, lMaxShow); i++) {
            if (i > 0)
                std::cout << ", ";
            std::cout << fFlat(i);
        }
        if (bTruncated)
            std::cout << ", ...";
        std::cout << "]" << std::endl;
    } else if (m_iNdim == 2) {
        int64_t lRows = m_lShape[0];
        int64_t lCols = m_lShape[1];
        int64_t lMaxRows = std::min(lRows, (int64_t)8);
        int64_t lMaxCols = std::min(lCols, (int64_t)8);
        const float *pfPtr = pfData();

        for (int64_t r = 0; r < lMaxRows; r++) {
            std::cout << "  [";
            for (int64_t c = 0; c < lMaxCols; c++) {
                if (c > 0)
                    std::cout << ", ";
                int64_t lOff = r * m_lStride[0] + c * m_lStride[1];
                std::cout << pfPtr[lOff];
            }
            if (lCols > lMaxCols)
                std::cout << ", ...";
            std::cout << "]" << std::endl;
        }
        if (lRows > lMaxRows)
            std::cout << "  ..." << std::endl;
    } else {
        std::cout << "  [";
        for (int64_t i = 0; i < std::min(lN, lMaxShow); i++) {
            if (i > 0)
                std::cout << ", ";
            std::cout << fFlat(i);
        }
        if (bTruncated)
            std::cout << ", ...";
        std::cout << "]" << std::endl;
    }
}
// >>>s_end(utils)

// <<<s_start(internal)
// --- internal helpers
/*---------------------------------------------------------
 * FN: Allocate
 * DESC: allocs aligned memory for the tensor
 * PARMS: vlShape (dim size), eType (data type)
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
void CTensor::Allocate(const std::vector<int64_t> &vlShape, EType eType) {
    assert(vlShape.size() > 0 && vlShape.size() <= mmDims);

    m_iNdim = (int)vlShape.size();
    m_eType = eType;
    m_bOwnsData = true;

    std::memset(m_lShape, 0, sizeof(m_lShape));
    std::memset(m_lStride, 0, sizeof(m_lStride));

    for (int i = 0; i < m_iNdim; i++) {
        assert(vlShape[i] > 0 && "[ct:allocate] shape dimensions must be positive");
        m_lShape[i] = vlShape[i];
    }

    ComputeStrides();

    int64_t lN = lNumel();
    m_iDataSize = lN * nGetTypeSize(m_eType);

    m_pData = std::aligned_alloc(64, (m_iDataSize + 63) & ~63);
    assert(m_pData && "[ct:allocate] tensor allocation failed");
}

/*---------------------------------------------------------
 * FN: ComputeStrides
 * DESC: computes row major strides from current strides
 * PARMS: none
 * AUTH: unium (12.02.26)
 *-------------------------------------------------------*/
void CTensor::ComputeStrides() {
    if (m_iNdim == 0)
        return;
    m_lStride[m_iNdim - 1] = 1;
    for (int i = m_iNdim - 2; i >= 0; i--) {
        m_lStride[i] = m_lStride[i + 1] * m_lShape[i + 1];
    }
}
// >>>s_end(internal)
} // namespace MT
