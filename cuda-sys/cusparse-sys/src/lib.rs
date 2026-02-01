//! Raw FFI bindings to cuSPARSE.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use cuda_runtime_sys::cudaStream_t;
use libc::{c_char, c_int, c_void, size_t};

pub type cusparseStatus_t = c_int;
pub const CUSPARSE_STATUS_SUCCESS: cusparseStatus_t = 0;
pub const CUSPARSE_STATUS_NOT_INITIALIZED: cusparseStatus_t = 1;
pub const CUSPARSE_STATUS_ALLOC_FAILED: cusparseStatus_t = 2;
pub const CUSPARSE_STATUS_INVALID_VALUE: cusparseStatus_t = 3;
pub const CUSPARSE_STATUS_ARCH_MISMATCH: cusparseStatus_t = 4;
pub const CUSPARSE_STATUS_EXECUTION_FAILED: cusparseStatus_t = 6;
pub const CUSPARSE_STATUS_INTERNAL_ERROR: cusparseStatus_t = 7;
pub const CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: cusparseStatus_t = 8;
pub const CUSPARSE_STATUS_NOT_SUPPORTED: cusparseStatus_t = 9;
pub const CUSPARSE_STATUS_INSUFFICIENT_RESOURCES: cusparseStatus_t = 10;

#[repr(C)]
pub struct cusparseContext { _unused: [u8; 0] }
pub type cusparseHandle_t = *mut cusparseContext;

#[repr(C)]
pub struct cusparseMatDescr { _unused: [u8; 0] }
pub type cusparseMatDescr_t = *mut cusparseMatDescr;

#[repr(C)]
pub struct cusparseSpMatDescr { _unused: [u8; 0] }
pub type cusparseSpMatDescr_t = *mut cusparseSpMatDescr;

#[repr(C)]
pub struct cusparseDnVecDescr { _unused: [u8; 0] }
pub type cusparseDnVecDescr_t = *mut cusparseDnVecDescr;

#[repr(C)]
pub struct cusparseDnMatDescr { _unused: [u8; 0] }
pub type cusparseDnMatDescr_t = *mut cusparseDnMatDescr;

pub type cusparseIndexType_t = c_int;
pub const CUSPARSE_INDEX_16U: cusparseIndexType_t = 1;
pub const CUSPARSE_INDEX_32I: cusparseIndexType_t = 2;
pub const CUSPARSE_INDEX_64I: cusparseIndexType_t = 3;

pub type cusparseIndexBase_t = c_int;
pub const CUSPARSE_INDEX_BASE_ZERO: cusparseIndexBase_t = 0;
pub const CUSPARSE_INDEX_BASE_ONE: cusparseIndexBase_t = 1;

pub type cusparseOrder_t = c_int;
pub const CUSPARSE_ORDER_COL: cusparseOrder_t = 1;
pub const CUSPARSE_ORDER_ROW: cusparseOrder_t = 2;

pub type cusparseOperation_t = c_int;
pub const CUSPARSE_OPERATION_NON_TRANSPOSE: cusparseOperation_t = 0;
pub const CUSPARSE_OPERATION_TRANSPOSE: cusparseOperation_t = 1;
pub const CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE: cusparseOperation_t = 2;

pub type cudaDataType = c_int;
pub const CUDA_R_16F: cudaDataType = 2;
pub const CUDA_R_32F: cudaDataType = 0;
pub const CUDA_R_64F: cudaDataType = 1;
pub const CUDA_C_32F: cudaDataType = 4;
pub const CUDA_C_64F: cudaDataType = 5;

extern "C" {
    pub fn cusparseCreate(handle: *mut cusparseHandle_t) -> cusparseStatus_t;
    pub fn cusparseDestroy(handle: cusparseHandle_t) -> cusparseStatus_t;
    pub fn cusparseGetVersion(handle: cusparseHandle_t, version: *mut c_int) -> cusparseStatus_t;
    pub fn cusparseSetStream(handle: cusparseHandle_t, streamId: cudaStream_t) -> cusparseStatus_t;
    pub fn cusparseGetStream(handle: cusparseHandle_t, streamId: *mut cudaStream_t) -> cusparseStatus_t;
    pub fn cusparseGetErrorName(status: cusparseStatus_t) -> *const c_char;
    pub fn cusparseGetErrorString(status: cusparseStatus_t) -> *const c_char;

    // Sparse vector
    pub fn cusparseCreateDnVec(
        dnVecDescr: *mut cusparseDnVecDescr_t,
        size: i64,
        values: *mut c_void,
        valueType: cudaDataType,
    ) -> cusparseStatus_t;
    pub fn cusparseDestroyDnVec(dnVecDescr: cusparseDnVecDescr_t) -> cusparseStatus_t;

    // Sparse matrix CSR
    pub fn cusparseCreateCsr(
        spMatDescr: *mut cusparseSpMatDescr_t,
        rows: i64,
        cols: i64,
        nnz: i64,
        csrRowOffsets: *mut c_void,
        csrColInd: *mut c_void,
        csrValues: *mut c_void,
        csrRowOffsetsType: cusparseIndexType_t,
        csrColIndType: cusparseIndexType_t,
        idxBase: cusparseIndexBase_t,
        valueType: cudaDataType,
    ) -> cusparseStatus_t;
    pub fn cusparseDestroySpMat(spMatDescr: cusparseSpMatDescr_t) -> cusparseStatus_t;

    // Dense matrix
    pub fn cusparseCreateDnMat(
        dnMatDescr: *mut cusparseDnMatDescr_t,
        rows: i64,
        cols: i64,
        ld: i64,
        values: *mut c_void,
        valueType: cudaDataType,
        order: cusparseOrder_t,
    ) -> cusparseStatus_t;
    pub fn cusparseDestroyDnMat(dnMatDescr: cusparseDnMatDescr_t) -> cusparseStatus_t;

    // SpMV
    pub fn cusparseSpMV_bufferSize(
        handle: cusparseHandle_t,
        opA: cusparseOperation_t,
        alpha: *const c_void,
        matA: cusparseSpMatDescr_t,
        vecX: cusparseDnVecDescr_t,
        beta: *const c_void,
        vecY: cusparseDnVecDescr_t,
        computeType: cudaDataType,
        alg: c_int,
        bufferSize: *mut size_t,
    ) -> cusparseStatus_t;

    pub fn cusparseSpMV(
        handle: cusparseHandle_t,
        opA: cusparseOperation_t,
        alpha: *const c_void,
        matA: cusparseSpMatDescr_t,
        vecX: cusparseDnVecDescr_t,
        beta: *const c_void,
        vecY: cusparseDnVecDescr_t,
        computeType: cudaDataType,
        alg: c_int,
        externalBuffer: *mut c_void,
    ) -> cusparseStatus_t;

    // SpMM
    pub fn cusparseSpMM_bufferSize(
        handle: cusparseHandle_t,
        opA: cusparseOperation_t,
        opB: cusparseOperation_t,
        alpha: *const c_void,
        matA: cusparseSpMatDescr_t,
        matB: cusparseDnMatDescr_t,
        beta: *const c_void,
        matC: cusparseDnMatDescr_t,
        computeType: cudaDataType,
        alg: c_int,
        bufferSize: *mut size_t,
    ) -> cusparseStatus_t;

    pub fn cusparseSpMM(
        handle: cusparseHandle_t,
        opA: cusparseOperation_t,
        opB: cusparseOperation_t,
        alpha: *const c_void,
        matA: cusparseSpMatDescr_t,
        matB: cusparseDnMatDescr_t,
        beta: *const c_void,
        matC: cusparseDnMatDescr_t,
        computeType: cudaDataType,
        alg: c_int,
        externalBuffer: *mut c_void,
    ) -> cusparseStatus_t;
}
