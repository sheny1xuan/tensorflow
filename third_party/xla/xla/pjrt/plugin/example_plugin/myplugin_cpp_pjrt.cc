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

#include "xla/pjrt/plugin/example_plugin/myplugin_cpp_pjrt.h"

#include <cstdint>

#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "tsl/platform/fingerprint.h"

namespace myplugin_pjrt {

absl::string_view MypluginPjrtClient::platform_name() const {
  return "myplugin_pjrt_client";
}

int MypluginPjrtClient::process_index() const { return 0; }

xla::PjRtPlatformId MypluginPjrtClient::platform_id() const {
  constexpr char kMyBackendName[] = "my_plugin_backend";
  static const uint64_t kMyBackendId = tsl::Fingerprint64(kMyBackendName);
  return kMyBackendId;
}

}  // namespace myplugin_pjrt
