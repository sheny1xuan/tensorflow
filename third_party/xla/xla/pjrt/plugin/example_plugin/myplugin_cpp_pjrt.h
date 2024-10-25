/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_PJRT_PLUGIN_EXAMPLE_PLUGIN_MYPLUGIN_CPP_PJRT_H_
#define XLA_PJRT_PLUGIN_EXAMPLE_PLUGIN_MYPLUGIN_CPP_PJRT_H_

#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"

namespace myplugin_pjrt {

class MypluginPjrtClient : public xla::PjRtClient {
 public:
  MypluginPjrtClient() = default;
  ~MypluginPjrtClient() override {};
  absl::string_view platform_name() const override;
  int process_index() const override;
  int device_count() const override { return 42; }
  int addressable_device_count() const override { return 43; }
  absl::Span<xla::PjRtDevice* const> devices() const override {
    return devices_;
  }
  absl::Span<xla::PjRtDevice* const> addressable_devices() const override {
    return devices_;
  }
  absl::Span<xla::PjRtMemorySpace* const> memory_spaces() const override {
    return memory_spaces_;
  }

  xla::PjRtPlatformId platform_id() const override;

  absl::string_view platform_version() const override {
    return "myplugin platform version";
  }

 private:
  std::vector<xla::PjRtDevice*> devices_;
  std::vector<xla::PjRtMemorySpace*> memory_spaces_;
};  // end class

}  // namespace myplugin_pjrt

#endif  // XLA_PJRT_PLUGIN_EXAMPLE_PLUGIN_MYPLUGIN_CPP_PJRT_H_
