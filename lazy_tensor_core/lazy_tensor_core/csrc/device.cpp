#include "lazy_tensor_core/csrc/device.h"

#include <c10/util/Optional.h>

#include "lazy_tensors/computation_client/computation_client.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/str_cat.h"
#include "lazy_tensors/str_split.h"

namespace torch_lazy_tensors {
namespace {

thread_local c10::optional<Device> g_current_device;

}  // namespace

Device::Device(const std::string& device_spec) {}

std::string Device::ToString() const {
  return lazy_tensors::StrCat("Default:", ordinal);
}

const Device* GetDefaultDevice() {
  static const Device* default_device = new Device("");
  return default_device;
}

Device GetCurrentDevice() {
  if (!g_current_device) {
    g_current_device = *GetDefaultDevice();
  }
  return *g_current_device;
}

Device SetCurrentDevice(const Device& device) {
  Device current = GetCurrentDevice();
  g_current_device = device;
  LTC_VLOG(2) << "New current device: " << device;
  return current;
}

}  // namespace torch_lazy_tensors
