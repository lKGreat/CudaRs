#![allow(dead_code)]

use std::io::Cursor;

use cudars_core::SdkErr;

use super::sdk_error::set_last_error;
use super::sdk_yolo_preprocess_meta::SdkYoloPreprocessMeta;

#[cfg(feature = "jpeg")]
use jpeg_decoder as jpegdec;

#[derive(Clone, Copy, Debug)]
pub enum InputLayout {
    Nchw,
    Nhwc,
}

pub fn decode_image_rgb(bytes: &[u8]) -> Result<(Vec<u8>, i32, i32), SdkErr> {
    if is_jpeg(bytes) {
        #[cfg(not(feature = "jpeg"))]
        {
            set_last_error("jpeg feature not enabled");
            return Err(SdkErr::Unsupported);
        }
        #[cfg(feature = "jpeg")]
        {
            return decode_jpeg_rgb(bytes);
        }
    }

    if is_png(bytes) {
        return decode_png_rgb(bytes);
    }

    set_last_error("unsupported image format");
    Err(SdkErr::Unsupported)
}

pub fn letterbox_u8_to_tensor(
    rgb: &[u8],
    input_width: i32,
    input_height: i32,
    target_w: i32,
    target_h: i32,
    channels: i32,
    layout: InputLayout,
) -> Result<(Vec<f32>, SdkYoloPreprocessMeta), SdkErr> {
    if input_width <= 0 || input_height <= 0 || target_w <= 0 || target_h <= 0 {
        set_last_error("invalid input dimensions");
        return Err(SdkErr::InvalidArg);
    }
    if channels != 3 {
        set_last_error("only 3-channel input supported");
        return Err(SdkErr::Unsupported);
    }

    let expected = (input_width as usize) * (input_height as usize) * (channels as usize);
    if rgb.len() < expected {
        set_last_error("input buffer too small");
        return Err(SdkErr::InvalidArg);
    }

    let scale = f32::min(target_w as f32 / input_width as f32, target_h as f32 / input_height as f32);
    let new_w = (input_width as f32 * scale).round() as i32;
    let new_h = (input_height as f32 * scale).round() as i32;
    if new_w <= 0 || new_h <= 0 {
        set_last_error("invalid resize result");
        return Err(SdkErr::InvalidArg);
    }
    let pad_x = (target_w - new_w) / 2;
    let pad_y = (target_h - new_h) / 2;

    let hw = (target_w as usize) * (target_h as usize);
    let mut output = vec![114.0f32 / 255.0f32; hw * channels as usize];

    for y in 0..target_h {
        for x in 0..target_w {
            let inside = x >= pad_x && x < pad_x + new_w && y >= pad_y && y < pad_y + new_h;
            if !inside {
                continue;
            }

            let src_x = ((x - pad_x) as f32 + 0.5f32) / scale - 0.5f32;
            let src_y = ((y - pad_y) as f32 + 0.5f32) / scale - 0.5f32;

            let mut x0 = src_x.floor() as i32;
            let mut y0 = src_y.floor() as i32;
            let mut x1 = x0 + 1;
            let mut y1 = y0 + 1;

            let mut fx = src_x - x0 as f32;
            let mut fy = src_y - y0 as f32;

            if x0 < 0 {
                x0 = 0;
                fx = 0.0;
            }
            if y0 < 0 {
                y0 = 0;
                fy = 0.0;
            }
            if x1 >= input_width {
                x1 = input_width - 1;
            }
            if y1 >= input_height {
                y1 = input_height - 1;
            }

            let idx00 = ((y0 * input_width + x0) as usize) * channels as usize;
            let idx01 = ((y0 * input_width + x1) as usize) * channels as usize;
            let idx10 = ((y1 * input_width + x0) as usize) * channels as usize;
            let idx11 = ((y1 * input_width + x1) as usize) * channels as usize;

            let c00_0 = rgb[idx00] as f32;
            let c00_1 = rgb[idx00 + 1] as f32;
            let c00_2 = rgb[idx00 + 2] as f32;

            let c01_0 = rgb[idx01] as f32;
            let c01_1 = rgb[idx01 + 1] as f32;
            let c01_2 = rgb[idx01 + 2] as f32;

            let c10_0 = rgb[idx10] as f32;
            let c10_1 = rgb[idx10 + 1] as f32;
            let c10_2 = rgb[idx10 + 2] as f32;

            let c11_0 = rgb[idx11] as f32;
            let c11_1 = rgb[idx11 + 1] as f32;
            let c11_2 = rgb[idx11 + 2] as f32;

            let w00 = (1.0f32 - fx) * (1.0f32 - fy);
            let w01 = fx * (1.0f32 - fy);
            let w10 = (1.0f32 - fx) * fy;
            let w11 = fx * fy;

            let r = (c00_0 * w00 + c01_0 * w01 + c10_0 * w10 + c11_0 * w11) / 255.0f32;
            let g = (c00_1 * w00 + c01_1 * w01 + c10_1 * w10 + c11_1 * w11) / 255.0f32;
            let b = (c00_2 * w00 + c01_2 * w01 + c10_2 * w10 + c11_2 * w11) / 255.0f32;

            let out_idx = (y as usize) * (target_w as usize) + (x as usize);
            match layout {
                InputLayout::Nchw => {
                    output[out_idx] = r;
                    output[out_idx + hw] = g;
                    output[out_idx + hw * 2] = b;
                }
                InputLayout::Nhwc => {
                    let base = out_idx * channels as usize;
                    output[base] = r;
                    output[base + 1] = g;
                    output[base + 2] = b;
                }
            }
        }
    }

    let meta = SdkYoloPreprocessMeta {
        scale,
        pad_x,
        pad_y,
        original_width: input_width,
        original_height: input_height,
    };

    Ok((output, meta))
}

fn is_jpeg(data: &[u8]) -> bool {
    data.len() >= 2 && data[0] == 0xFF && data[1] == 0xD8
}

fn is_png(data: &[u8]) -> bool {
    const SIG: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    data.len() >= 8 && data[..8] == SIG
}

#[cfg(feature = "jpeg")]
fn decode_jpeg_rgb(bytes: &[u8]) -> Result<(Vec<u8>, i32, i32), SdkErr> {
    let mut decoder = jpegdec::Decoder::new(Cursor::new(bytes));
    let pixels = decoder.decode().map_err(|_| {
        set_last_error("invalid jpeg");
        SdkErr::InvalidArg
    })?;
    let info = decoder.info().ok_or_else(|| {
        set_last_error("invalid jpeg info");
        SdkErr::InvalidArg
    })?;

    let width = info.width as i32;
    let height = info.height as i32;
    if width <= 0 || height <= 0 {
        set_last_error("invalid jpeg dimensions");
        return Err(SdkErr::InvalidArg);
    }

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
        _ => {
            set_last_error("unsupported jpeg pixel format");
            Err(SdkErr::Unsupported)
        }
    }
}

fn decode_png_rgb(bytes: &[u8]) -> Result<(Vec<u8>, i32, i32), SdkErr> {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);

    let mut reader = decoder.read_info().map_err(|_| {
        set_last_error("invalid png");
        SdkErr::InvalidArg
    })?;

    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|_| {
        set_last_error("invalid png");
        SdkErr::InvalidArg
    })?;

    let width = info.width as i32;
    let height = info.height as i32;
    if width <= 0 || height <= 0 {
        set_last_error("invalid png dimensions");
        return Err(SdkErr::InvalidArg);
    }

    let mut rgb = vec![0u8; (width as usize) * (height as usize) * 3];
    match info.color_type {
        png::ColorType::Rgb => {
            let required = rgb.len();
            rgb.copy_from_slice(&buf[..required]);
        }
        png::ColorType::Rgba => {
            for i in 0..(width as usize * height as usize) {
                let s = i * 4;
                let d = i * 3;
                rgb[d] = buf[s];
                rgb[d + 1] = buf[s + 1];
                rgb[d + 2] = buf[s + 2];
            }
        }
        png::ColorType::Grayscale => {
            for i in 0..(width as usize * height as usize) {
                let g = buf[i];
                let d = i * 3;
                rgb[d] = g;
                rgb[d + 1] = g;
                rgb[d + 2] = g;
            }
        }
        png::ColorType::GrayscaleAlpha => {
            for i in 0..(width as usize * height as usize) {
                let s = i * 2;
                let g = buf[s];
                let d = i * 3;
                rgb[d] = g;
                rgb[d + 1] = g;
                rgb[d + 2] = g;
            }
        }
        _ => {
            set_last_error("unsupported png color type");
            return Err(SdkErr::Unsupported);
        }
    }

    Ok((rgb, width, height))
}
