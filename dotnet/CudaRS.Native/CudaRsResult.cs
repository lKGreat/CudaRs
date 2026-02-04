namespace CudaRS.Native;

public enum CudaRsResult : int
{
    Success = 0,
    ErrorInvalidValue = 1,
    ErrorOutOfMemory = 2,
    ErrorNotInitialized = 3,
    ErrorInvalidHandle = 4,
    ErrorNotSupported = 5,
    ErrorUnknown = 999,
}
