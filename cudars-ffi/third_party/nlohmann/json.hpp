#pragma once

// PaddleOCR includes "third_party/nlohmann/json.hpp". Provide a lightweight shim
// that forwards to the nlohmann/json header installed via vcpkg.
#include <nlohmann/json.hpp>
