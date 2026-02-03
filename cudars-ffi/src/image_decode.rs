//! Unified image decode to GPU device memory.
//!
//! - JPEG: nvJPEG decode directly into a reusable device HWC (u8) buffer when available.
//! - JPEG fallback: Rust CPU decode (jpeg-decoder) into pinned host, then async H2D upload.
//! - PNG: Rust CPU decode (png crate) into pinned host, then async H2D upload.
//!
//! The decoder is handle-based and designed for pipelined throughput: each worker should
//! own its own decoder handle + CUDA stream.

use crate::runtime::{CudaRsStream, HandleManager, STREAMS};
use crate::CudaRsResult;
use libc::{c_int, c_uchar, c_void, size_t};
use std::io::Cursor;
use std::ptr;
use std::sync::{Arc, Mutex};

#[cfg(feature = "jpeg")]
use jpeg_decoder as jpegdec;
#[cfg(feature = "jpeg")]
use nvjpeg::{Backend, Handle as NvjpegHandle, JpegState};
#[cfg(feature = "jpeg")]
use nvjpeg_sys::{nvjpegDecode, nvjpegImage_t, NVJPEG_OUTPUT_RGBI};

pub type CudaRsImageDecoder = u64;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaRsImageFormat {
    Unknown = 0,
    Jpeg = 1,
    Png = 2,
}

struct ImageDecoder {
    device_id: i32,
    max_width: i32,
    max_height: i32,
    channels: i32,

    device_buffer: *mut u8,
    device_capacity: usize,

    pinned_host: *mut u8,
    pinned_capacity: usize,

    #[cfg(feature = "jpeg")]
    nvjpeg: Option<NvjpegHandle>,
    #[cfg(feature = "jpeg")]
    nvjpeg_state: Option<JpegState>,
}

unsafe impl Send for ImageDecoder {}

impl Drop for ImageDecoder {
    fn drop(&mut self) {
        unsafe {
            if !self.device_buffer.is_null() {
                cuda_runtime_sys::cudaFree(self.device_buffer as *mut c_void);
                self.device_buffer = ptr::null_mut();
                self.device_capacity = 0;
            }
            if !self.pinned_host.is_null() {
                cuda_runtime_sys::cudaFreeHost(self.pinned_host as *mut c_void);
                self.pinned_host = ptr::null_mut();
                self.pinned_capacity = 0;
            }
        }
    }
}

lazy_static::lazy_static! {
    static ref DECODERS: Mutex<HandleManager<Arc<Mutex<ImageDecoder>>>> = Mutex::new(HandleManager::new());
}

fn is_jpeg(data: &[u8]) -> bool {
    data.len() >= 2 && data[0] == 0xFF && data[1] == 0xD8
}

fn is_png(data: &[u8]) -> bool {
    const SIG: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    data.len() >= 8 && data[..8] == SIG
}

unsafe fn ensure_device_capacity(dec: &mut ImageDecoder, required: usize) -> Result<(), CudaRsResult> {
    if required <= dec.device_capacity && !dec.device_buffer.is_null() {
        return Ok(());
    }

    if !dec.device_buffer.is_null() {
        cuda_runtime_sys::cudaFree(dec.device_buffer as *mut c_void);
        dec.device_buffer = ptr::null_mut();
        dec.device_capacity = 0;
    }

    let mut out: *mut c_void = ptr::null_mut();
    if cuda_runtime_sys::cudaMalloc(&mut out, required) != cuda_runtime_sys::cudaSuccess {
        return Err(CudaRsResult::ErrorOutOfMemory);
    }

    dec.device_buffer = out as *mut u8;
    dec.device_capacity = required;
    Ok(())
}

unsafe fn ensure_pinned_capacity(dec: &mut ImageDecoder, required: usize) -> Result<(), CudaRsResult> {
    if required <= dec.pinned_capacity && !dec.pinned_host.is_null() {
        return Ok(());
    }

    if !dec.pinned_host.is_null() {
        cuda_runtime_sys::cudaFreeHost(dec.pinned_host as *mut c_void);
        dec.pinned_host = ptr::null_mut();
        dec.pinned_capacity = 0;
    }

    let mut out: *mut c_void = ptr::null_mut();
    if cuda_runtime_sys::cudaHostAlloc(&mut out, required, 0) != cuda_runtime_sys::cudaSuccess {
        return Err(CudaRsResult::ErrorOutOfMemory);
    }

    dec.pinned_host = out as *mut u8;
    dec.pinned_capacity = required;
    Ok(())
}

#[cfg(feature = "jpeg")]
fn try_create_nvjpeg() -> Option<(NvjpegHandle, JpegState)> {
    // Some installations behave differently depending on backend choice.
    let backends = [Backend::Hybrid, Backend::Default, Backend::GpuHybrid, Backend::Hardware];
    for backend in backends {
        if let Ok(h) = NvjpegHandle::with_backend(backend) {
            if let Ok(st) = JpegState::new(&h) {
                return Some((h, st));
            }
        }
    }
    None
}

#[cfg(feature = "jpeg")]
fn decode_jpeg_cpu_rgb(bytes: &[u8]) -> Result<(Vec<u8>, i32, i32), CudaRsResult> {
    let mut decoder = jpegdec::Decoder::new(Cursor::new(bytes));
    let pixels = decoder.decode().map_err(|_| CudaRsResult::ErrorInvalidValue)?;
    let info = decoder.info().ok_or(CudaRsResult::ErrorUnknown)?;

    let width = info.width as i32;
    let height = info.height as i32;

    match info.pixel_format {
        jpegdec::PixelFormat::RGB24 => Ok((pixels, width, height)),
        jpegdec::PixelFormat::L8 => {
            let mut rgb = Vec::with_capacity((width as usize) * (height as usize) * 3);
            for &g in pixels.iter() {
                rgb.push(g);
                rgb.push(g);
                rgb.push(g);
            }
            Ok((rgb, width, height))
        }
        _ => Err(CudaRsResult::ErrorNotSupported),
    }
}

#[no_mangle]
pub extern "C" fn cudars_image_decoder_create(
    out_handle: *mut CudaRsImageDecoder,
    max_width: c_int,
    max_height: c_int,
    channels: c_int,
) -> CudaRsResult {
    if out_handle.is_null() || max_width <= 0 || max_height <= 0 || channels <= 0 {
        return CudaRsResult::ErrorInvalidValue;
    }

    unsafe {
        let mut dev: i32 = 0;
        if cuda_runtime_sys::cudaGetDevice(&mut dev) != cuda_runtime_sys::cudaSuccess {
            return CudaRsResult::ErrorUnknown;
        }

        let max_bytes = (max_width * max_height * channels) as usize;

        let mut device_ptr: *mut c_void = ptr::null_mut();
        if cuda_runtime_sys::cudaMalloc(&mut device_ptr, max_bytes) != cuda_runtime_sys::cudaSuccess {
            return CudaRsResult::ErrorOutOfMemory;
        }

        let mut pinned_ptr: *mut c_void = ptr::null_mut();
        if cuda_runtime_sys::cudaHostAlloc(&mut pinned_ptr, max_bytes, 0) != cuda_runtime_sys::cudaSuccess {
            cuda_runtime_sys::cudaFree(device_ptr);
            return CudaRsResult::ErrorOutOfMemory;
        }

        #[cfg(feature = "jpeg")]
        let (nvjpeg, nvjpeg_state) = match try_create_nvjpeg() {
            Some((h, st)) => (Some(h), Some(st)),
            None => {
                // Keep decoder usable: JPEG will fall back to CPU decode (still in Rust).
                eprintln!("[cudars] nvJPEG init failed; falling back to CPU JPEG decode");
                (None, None)
            }
        };

        let dec = ImageDecoder {
            device_id: dev,
            max_width,
            max_height,
            channels,
            device_buffer: device_ptr as *mut u8,
            device_capacity: max_bytes,
            pinned_host: pinned_ptr as *mut u8,
            pinned_capacity: max_bytes,
            #[cfg(feature = "jpeg")]
            nvjpeg,
            #[cfg(feature = "jpeg")]
            nvjpeg_state,
        };

        let mut decoders = DECODERS.lock().unwrap();
        let id = decoders.insert(Arc::new(Mutex::new(dec)));
        *out_handle = id;
    }

    CudaRsResult::Success
}

#[no_mangle]
pub extern "C" fn cudars_image_decoder_destroy(handle: CudaRsImageDecoder) -> CudaRsResult {
    let mut decoders = DECODERS.lock().unwrap();
    match decoders.remove(handle) {
        Some(_) => CudaRsResult::Success,
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

#[no_mangle]
pub extern "C" fn cudars_image_decoder_decode_to_device(
    handle: CudaRsImageDecoder,
    data: *const c_uchar,
    len: size_t,
    stream: CudaRsStream,
    out_device_ptr: *mut *mut c_uchar,
    out_pitch_bytes: *mut c_int,
    out_width: *mut c_int,
    out_height: *mut c_int,
    out_format: *mut c_int,
) -> CudaRsResult {
    if data.is_null()
        || len == 0
        || out_device_ptr.is_null()
        || out_pitch_bytes.is_null()
        || out_width.is_null()
        || out_height.is_null()
        || out_format.is_null()
    {
        return CudaRsResult::ErrorInvalidValue;
    }

    let stream_raw = {
        let streams = STREAMS.lock().unwrap();
        let s = match streams.get(stream) {
            Some(s) => s,
            None => return CudaRsResult::ErrorInvalidHandle,
        };
        s.as_raw()
    };

    let bytes = unsafe { std::slice::from_raw_parts(data as *const u8, len as usize) };

    let dec = {
        let decoders = DECODERS.lock().unwrap();
        match decoders.get(handle) {
            Some(d) => Arc::clone(d),
            None => return CudaRsResult::ErrorInvalidHandle,
        }
    };
    let mut dec = dec.lock().unwrap();

    unsafe {
        cuda_runtime_sys::cudaSetDevice(dec.device_id);
    }

    // JPEG
    if is_jpeg(bytes) {
        #[cfg(not(feature = "jpeg"))]
        {
            let _ = stream_raw;
            return CudaRsResult::ErrorNotSupported;
        }

        #[cfg(feature = "jpeg")]
        {
            // Try nvJPEG if initialized.
            let nvjpeg_info = if let (Some(nv), Some(st)) = (dec.nvjpeg.as_ref(), dec.nvjpeg_state.as_ref()) {
                match nv.get_image_info(bytes) {
                    Ok(info) => Some((nv.as_raw(), st.as_raw(), info)),
                    Err(_) => None,
                }
            } else {
                None
            };

            if let Some((nv_raw, st_raw, info)) = nvjpeg_info {
                let w = info.width;
                let h = info.height;
                if w > 0
                    && h > 0
                    && w <= dec.max_width
                    && h <= dec.max_height
                    && dec.channels == 3
                {
                    let required = (w * h * dec.channels) as usize;
                    unsafe {
                        if let Err(e) = ensure_device_capacity(&mut *dec, required) {
                            return e;
                        }
                    }

                    let mut img: nvjpegImage_t = unsafe { std::mem::zeroed() };
                    img.channel[0] = dec.device_buffer;
                    img.pitch[0] = (w * dec.channels) as i32;

                    let status = unsafe {
                        nvjpegDecode(
                            nv_raw,
                            st_raw,
                            bytes.as_ptr(),
                            bytes.len(),
                            NVJPEG_OUTPUT_RGBI,
                            &mut img as *mut nvjpegImage_t,
                            stream_raw,
                        )
                    };

                    if status == 0 {
                        unsafe {
                            *out_device_ptr = dec.device_buffer as *mut c_uchar;
                            *out_pitch_bytes = (w * dec.channels) as c_int;
                            *out_width = w;
                            *out_height = h;
                            *out_format = CudaRsImageFormat::Jpeg as c_int;
                        }
                        return CudaRsResult::Success;
                    }
                }
            }

            // CPU fallback.
            if dec.channels != 3 {
                return CudaRsResult::ErrorNotSupported;
            }

            let (rgb, w, h) = match decode_jpeg_cpu_rgb(bytes) {
                Ok(v) => v,
                Err(e) => return e,
            };

            if w <= 0 || h <= 0 || w > dec.max_width || h > dec.max_height {
                return CudaRsResult::ErrorInvalidValue;
            }

            let required = (w * h * dec.channels) as usize;
            unsafe {
                if let Err(e) = ensure_pinned_capacity(&mut *dec, required) {
                    return e;
                }
                if let Err(e) = ensure_device_capacity(&mut *dec, required) {
                    return e;
                }

                ptr::copy_nonoverlapping(rgb.as_ptr(), dec.pinned_host, required);
                if cuda_runtime_sys::cudaMemcpyAsync(
                    dec.device_buffer as *mut c_void,
                    dec.pinned_host as *const c_void,
                    required,
                    cuda_runtime_sys::cudaMemcpyHostToDevice,
                    stream_raw,
                ) != cuda_runtime_sys::cudaSuccess
                {
                    return CudaRsResult::ErrorUnknown;
                }

                *out_device_ptr = dec.device_buffer as *mut c_uchar;
                *out_pitch_bytes = (w * dec.channels) as c_int;
                *out_width = w;
                *out_height = h;
                *out_format = CudaRsImageFormat::Jpeg as c_int;
            }

            return CudaRsResult::Success;
        }
    }

    // PNG
    if is_png(bytes) {
        let mut decoder = png::Decoder::new(Cursor::new(bytes));
        decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);

        let mut reader = match decoder.read_info() {
            Ok(r) => r,
            Err(_) => return CudaRsResult::ErrorInvalidValue,
        };

        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = match reader.next_frame(&mut buf) {
            Ok(i) => i,
            Err(_) => return CudaRsResult::ErrorInvalidValue,
        };

        let w = info.width as i32;
        let h = info.height as i32;
        if w <= 0 || h <= 0 {
            return CudaRsResult::ErrorInvalidValue;
        }
        if w > dec.max_width || h > dec.max_height {
            return CudaRsResult::ErrorInvalidValue;
        }
        if dec.channels != 3 {
            return CudaRsResult::ErrorNotSupported;
        }

        let required = (w * h * dec.channels) as usize;

        unsafe {
            if let Err(e) = ensure_device_capacity(&mut *dec, required) {
                return e;
            }
            if let Err(e) = ensure_pinned_capacity(&mut *dec, required) {
                return e;
            }

            let dst = std::slice::from_raw_parts_mut(dec.pinned_host, required);

            match info.color_type {
                png::ColorType::Rgb => {
                    dst.copy_from_slice(&buf[..required]);
                }
                png::ColorType::Rgba => {
                    for i in 0..(w as usize * h as usize) {
                        let s = i * 4;
                        let d = i * 3;
                        dst[d] = buf[s];
                        dst[d + 1] = buf[s + 1];
                        dst[d + 2] = buf[s + 2];
                    }
                }
                png::ColorType::Grayscale => {
                    for i in 0..(w as usize * h as usize) {
                        let g = buf[i];
                        let d = i * 3;
                        dst[d] = g;
                        dst[d + 1] = g;
                        dst[d + 2] = g;
                    }
                }
                png::ColorType::GrayscaleAlpha => {
                    for i in 0..(w as usize * h as usize) {
                        let s = i * 2;
                        let g = buf[s];
                        let d = i * 3;
                        dst[d] = g;
                        dst[d + 1] = g;
                        dst[d + 2] = g;
                    }
                }
                _ => return CudaRsResult::ErrorNotSupported,
            }

            if cuda_runtime_sys::cudaMemcpyAsync(
                dec.device_buffer as *mut c_void,
                dec.pinned_host as *const c_void,
                required,
                cuda_runtime_sys::cudaMemcpyHostToDevice,
                stream_raw,
            ) != cuda_runtime_sys::cudaSuccess
            {
                return CudaRsResult::ErrorUnknown;
            }

            *out_device_ptr = dec.device_buffer as *mut c_uchar;
            *out_pitch_bytes = (w * dec.channels) as c_int;
            *out_width = w;
            *out_height = h;
            *out_format = CudaRsImageFormat::Png as c_int;
        }

        return CudaRsResult::Success;
    }

    CudaRsResult::ErrorNotSupported
}
