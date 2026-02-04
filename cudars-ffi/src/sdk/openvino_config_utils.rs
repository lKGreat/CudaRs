use crate::{CudaRsOvDevice};
use cudars_core::SdkErr;
use libc::c_int;
use serde_json::{Map, Value};

pub struct OpenVinoDeviceSpec {
    pub device: CudaRsOvDevice,
    pub device_index: c_int,
    pub device_name_override: Option<String>,
}

fn normalize_device_list(list: &str) -> String {
    list.split(',')
        .map(|v| v.trim().to_uppercase())
        .filter(|v| !v.is_empty())
        .collect::<Vec<_>>()
        .join(",")
}

pub fn parse_openvino_device(device: &str) -> Result<OpenVinoDeviceSpec, SdkErr> {
    let raw = device.trim();
    let d = raw.to_lowercase();

    if raw.is_empty() || d == "auto" {
        return Ok(OpenVinoDeviceSpec {
            device: CudaRsOvDevice::Auto,
            device_index: 0,
            device_name_override: None,
        });
    }
    if d == "cpu" {
        return Ok(OpenVinoDeviceSpec {
            device: CudaRsOvDevice::Cpu,
            device_index: 0,
            device_name_override: None,
        });
    }
    if d == "gpu" {
        return Ok(OpenVinoDeviceSpec {
            device: CudaRsOvDevice::Gpu,
            device_index: 0,
            device_name_override: None,
        });
    }
    if d.starts_with("gpu:") || d.starts_with("gpu.") {
        let idx = d[4..].parse::<c_int>().map_err(|_| SdkErr::InvalidArg)?;
        return Ok(OpenVinoDeviceSpec {
            device: CudaRsOvDevice::GpuIndex,
            device_index: idx,
            device_name_override: None,
        });
    }
    if d == "npu" {
        return Ok(OpenVinoDeviceSpec {
            device: CudaRsOvDevice::Npu,
            device_index: 0,
            device_name_override: None,
        });
    }
    if d == "nvidia" {
        return Ok(OpenVinoDeviceSpec {
            device: CudaRsOvDevice::Auto,
            device_index: 0,
            device_name_override: Some("NVIDIA".to_string()),
        });
    }
    if d.starts_with("auto:") {
        let list = normalize_device_list(&raw[5..]);
        return Ok(OpenVinoDeviceSpec {
            device: CudaRsOvDevice::Auto,
            device_index: 0,
            device_name_override: Some(format!("AUTO:{list}")),
        });
    }

    Ok(OpenVinoDeviceSpec {
        device: CudaRsOvDevice::Auto,
        device_index: 0,
        device_name_override: Some(raw.to_string()),
    })
}

fn map_has_key_ci(map: &Map<String, Value>, key: &str) -> bool {
    map.keys().any(|k| k.eq_ignore_ascii_case(key))
}

pub fn build_openvino_properties_json(
    base_json: &str,
    performance_mode: &str,
    num_requests: i32,
    cache_dir: &str,
    enable_mmap: Option<bool>,
) -> Result<Option<String>, SdkErr> {
    let mut map = if base_json.trim().is_empty() {
        Map::new()
    } else {
        let value: Value = serde_json::from_str(base_json).map_err(|_| SdkErr::InvalidArg)?;
        match value {
            Value::Object(obj) => obj,
            _ => return Err(SdkErr::InvalidArg),
        }
    };

    let perf = performance_mode.trim().to_lowercase();
    if !perf.is_empty() && !map_has_key_ci(&map, "PERFORMANCE_HINT") {
        let hint = if perf == "throughput" || perf == "tput" {
            "THROUGHPUT"
        } else if perf == "latency" {
            "LATENCY"
        } else {
            ""
        };
        if !hint.is_empty() {
            map.insert("PERFORMANCE_HINT".to_string(), Value::String(hint.to_string()));
        }
    }

    if num_requests > 0 && !map_has_key_ci(&map, "NUM_REQUESTS") && !map_has_key_ci(&map, "NUM_INFER_REQUESTS") {
        map.insert("NUM_REQUESTS".to_string(), Value::Number(serde_json::Number::from(num_requests)));
    }

    if !cache_dir.trim().is_empty() && !map_has_key_ci(&map, "CACHE_DIR") {
        map.insert("CACHE_DIR".to_string(), Value::String(cache_dir.trim().to_string()));
    }

    if let Some(enable) = enable_mmap {
        if !map_has_key_ci(&map, "ENABLE_MMAP") {
            map.insert("ENABLE_MMAP".to_string(), Value::String(if enable { "YES" } else { "NO" }.to_string()));
        }
    }

    if map.is_empty() {
        Ok(None)
    } else {
        Ok(Some(Value::Object(map).to_string()))
    }
}
