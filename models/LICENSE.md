# AI Model Licenses

This document details the individual licenses, origins, and terms of use for all pre-trained neural network models used by the **Open NPU Effects** project.

The source code of the Open NPU Effects project is distributed under the **MIT License** (see [`../LICENSE`](../LICENSE)). The neural network models bundled or downloaded by the project are subject to their respective open-source licenses, as described in detail below.

---

## Summary of Models and Licenses

| Model | Files | Purpose | Original Author / Project | License |
| :--- | :--- | :--- | :--- | :--- |
| **Intel PoCoNet** | `audio/noise-suppression-poconetlike-0001.*` | Real-time deep noise suppression for voice | [Intel Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) | [Apache 2.0](#1-intel-poconet-noise-suppression) |
| **YuNet Face Detector** | `video/face_detection_yunet_2023mar.onnx` | Fast face detection for Auto-Framing | [Shiqi Yu / OpenCV Zoo](https://github.com/opencv/opencv_zoo) | [Apache 2.0](#2-yunet-face-detector-opencv-zoo) |
| **MediaPipe Selfie Segmentation** | `video/selfie_segmentation_static.*` | Static person segmentation (144x256 FP16 NPU) | [Google LLC / MediaPipe](https://github.com/google/mediapipe) | [Apache 2.0](#3-mediapipe-selfie-segmentation-google-llc) |
| **MediaPipe Selfie Multiclass** | `video/selfie_multiclass.*` | Multiclass segmentation (skin, hair, clothes, body) | [Google LLC / MediaPipe](https://github.com/google/mediapipe) | [Apache 2.0](#4-mediapipe-selfie-multiclass-google-llc) |
| **MobileNetV3 LRASPP** | `video/chair_segmenter.*` | 21-class semantic segmentation (chair fallback) | [PyTorch / TorchVision Contributors](https://github.com/pytorch/vision) | [BSD 3-Clause](#5-mobilenetv3-lraspp-segmenter-pytorch--torchvision) |
| **YOLACT ResNet-50 FPN** | `video/chair_instance_segmenter.*` | Instance segmentation for precise chair retention | [Daniel Bolya et al. (UC Davis)](https://github.com/dbolya/yolact) | [MIT](#6-yolact-resnet-50-fpn-instance-segmenter-uc-davis--intel-omz) |
| **BiSeNet ResNet-18 Face Parser** | `video/face_parsing_bisenet.*` | Semantic face parsing & eyeglass preservation | [zllrunning](https://github.com/zllrunning/face-parsing.PyTorch) / [LiteRT Community](https://huggingface.co/litert-community/BiSeNet-Face-Parsing-LiteRT) | [MIT](#7-bisenet-resnet-18-face-parser-celebamask-hq) |
| **MODNet Portrait Matting** | `video/modnet_portrait_matting.*` | Optional high-fidelity person matting assist (fixes dark-clothing gaps in Selfie Multiclass) | [ZHKKKe (Zhanghan Ke) et al.](https://github.com/ZHKKKe/MODNet) / [Intel Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/modnet-photographic-portrait-matting) | [Apache 2.0](#8-modnet-portrait-matting-zhkkke--intel-omz) |

---

## 1. Intel PoCoNet Noise Suppression
- **Files:** `audio/noise-suppression-poconetlike-0001.xml`, `audio/noise-suppression-poconetlike-0001.bin`
- **Source:** [Intel Open Model Zoo](https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/noise-suppression-poconetlike-0001/FP16/)
- **Copyright:** Copyright (c) 2021-2024 Intel Corporation
- **License:** **Apache License, Version 2.0**

```text
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## 2. YuNet Face Detector (OpenCV Zoo)
- **File:** `video/face_detection_yunet_2023mar.onnx`
- **Source:** [OpenCV Zoo - Face Detection YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- **Authors:** Shiqi Yu (Shenzhen University) & OpenCV Team
- **Copyright:** Copyright (c) OpenCV Team and contributors
- **License:** **Apache License, Version 2.0**

```text
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## 3. MediaPipe Selfie Segmentation (Google LLC)
- **Files:** `video/selfie_segmentation_static.xml`, `video/selfie_segmentation_static.bin`
- **Source:** [Google MediaPipe](https://github.com/google/mediapipe) / [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
- **Copyright:** Copyright (c) 2020-2024 Google LLC
- **License:** **Apache License, Version 2.0**

```text
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## 4. MediaPipe Selfie Multiclass (Google LLC)
- **Files:** `video/selfie_multiclass.xml`, `video/selfie_multiclass.bin`
- **Source:** [Google MediaPipe](https://github.com/google/mediapipe) / [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
- **Copyright:** Copyright (c) 2021-2024 Google LLC
- **License:** **Apache License, Version 2.0**

```text
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## 5. MobileNetV3 LRASPP Segmenter (PyTorch / TorchVision)
- **Files:** `video/chair_segmenter.xml`, `video/chair_segmenter.bin`
- **Source:** [PyTorch TorchVision](https://github.com/pytorch/vision)
- **Copyright:** Copyright (c) Soumith Chintala 2016, PyTorch and TorchVision Contributors
- **License:** **BSD 3-Clause License**

```text
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## 6. YOLACT ResNet-50 FPN Instance Segmenter (UC Davis / Intel OMZ)
- **Files:** `video/chair_instance_segmenter.xml`, `video/chair_instance_segmenter.bin`
- **Source:** [dbolya/yolact](https://github.com/dbolya/yolact) / [Intel Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/yolact-resnet50-fpn-pytorch)
- **Authors:** Daniel Bolya, Chong Zhou, Fanyi Xiao, Yong Jae Lee (University of California, Davis)
- **Copyright:** Copyright (c) 2019 Daniel Bolya
- **License:** **MIT License**

```text
MIT License

Copyright (c) 2019 Daniel Bolya

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 7. BiSeNet ResNet-18 Face Parser (CelebAMask-HQ)
- **Files:** `video/face_parsing_bisenet.xml`, `video/face_parsing_bisenet.bin`
- **Source:** [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) / [LiteRT Community BiSeNet-Face-Parsing-LiteRT](https://huggingface.co/litert-community/BiSeNet-Face-Parsing-LiteRT)
- **Authors:** zllrunning (PyTorch implementation & CelebAMask-HQ training) / LiteRT Community (on-device conversion)
- **Copyright:** Copyright (c) 2019 zll
- **License:** **MIT License**

```text
MIT License

Copyright (c) 2019 zll

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 8. MODNet Portrait Matting (ZHKKKe / Intel OMZ)
- **Files:** `video/modnet_portrait_matting.xml`, `video/modnet_portrait_matting.bin`
- **Source:** [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet) (paper: https://arxiv.org/abs/2011.11961), checkpoint mirrored by Intel at [Open Model Zoo - modnet-photographic-portrait-matting](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/modnet-photographic-portrait-matting)
- **Copyright:** Copyright (c) 2020 Zhanghan Ke
- **License:** **Apache License, Version 2.0**
- **Conversion notes:** PyTorch checkpoint (`modnet_photographic_portrait_matting.ckpt`, MobileNetV2 backbone) exported to ONNX via the official `onnx/modnet_onnx.py` script from the source repo, then converted to OpenVINO IR. Preprocessing (BGR->RGB channel reversal, mean=127.5, scale=127.5) is baked into the IR graph via OpenVINO's PrePostProcessor, so the model accepts a raw `[1, 512, 512, 3]` uint8 BGR frame directly (no manual normalization needed in Python). Output is a `[1, 1, 512, 512]` float32 alpha matte in the `[0, 1]` range.

```text
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## Full Text of the Apache License, Version 2.0

```text
                              Apache License
                        Version 2.0, January 2004
                     http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

   "License" shall mean the terms and conditions for use, reproduction,
   and distribution as defined by Sections 1 through 9 of this document.

   "Licensor" shall mean the copyright owner or entity authorized by
   the copyright owner that is granting the License.

   "Legal Entity" shall mean the union of the acting entity and all
   other entities that control, are controlled by, or are under common
   control with that entity. For the purposes of this definition,
   "control" means (i) the power, direct or indirect, to cause the
   direction or management of such entity, whether by contract or
   otherwise, or (ii) ownership of fifty percent (50%) or more of the
   outstanding shares, or (iii) beneficial ownership of such entity.

   "You" (or "Your") shall mean an individual or Legal Entity
   exercising permissions granted by this License.

   "Source" form shall mean the preferred form for making modifications,
   including but not limited to software source code, documentation
   source, and configuration files.

   "Object" form shall mean any form resulting from mechanical
   transformation or translation of a Source form, including but
   not limited to compiled object code, generated documentation,
   and conversions to other media types.

   "Work" shall mean the work of authorship, whether in Source or
   Object form, made available under the License, as indicated by a
   copyright notice that is included in or attached to the work
   (an example is provided in the Appendix below).

   "Derivative Works" shall mean any work, whether in Source or Object
   form, that is based on (or derived from) the Work and for which the
   editorial revisions, annotations, elaborations, or other modifications
   represent, as a whole, an original work of authorship. For the purposes
   of this License, Derivative Works shall not include works that remain
   separable from, or merely link (or bind by name) to the interfaces of,
   the Work and Derivative Works thereof.

   "Contribution" shall mean any work of authorship, including
   the original version of the Work and any modifications or additions
   to that Work or Derivative Works thereof, that is intentionally
   submitted to Licensor for inclusion in the Work by the copyright owner
   or by an individual or Legal Entity authorized to submit on behalf of
   the copyright owner. For the purposes of this definition, "submitted"
   means any form of electronic, verbal, or written communication sent
   to the Licensor or its representatives, including but not limited to
   communication on electronic mailing lists, source code control systems,
   and issue tracking systems that are managed by, or on behalf of, the
   Licensor for the purpose of discussing and improving the Work, but
   excluding communication that is conspicuously marked or otherwise
   designated in writing by the copyright owner as "Not a Contribution."

   "Contributor" shall mean Licensor and any individual or Legal Entity
   on behalf of whom a Contribution has been received by Licensor and
   subsequently incorporated within the Work.

2. Grant of Copyright License. Subject to the terms and conditions of
   this License, each Contributor hereby grants to You a perpetual,
   worldwide, non-exclusive, no-charge, royalty-free, irrevocable
   copyright license to reproduce, prepare Derivative Works of,
   publicly display, publicly perform, sublicense, and distribute the
   Work and such Derivative Works in Source or Object form.

3. Grant of Patent License. Subject to the terms and conditions of
   this License, each Contributor hereby grants to You a perpetual,
   worldwide, non-exclusive, no-charge, royalty-free, irrevocable
   (except as stated in this section) patent license to make, have made,
   use, offer to sell, sell, import, and otherwise transfer the Work,
   where such license applies only to those patent claims licensable
   by such Contributor that are necessarily infringed by their
   Contribution(s) alone or by combination of their Contribution(s)
   with the Work to which such Contribution(s) was submitted. If You
   institute patent litigation against any entity (including a
   cross-claim or counterclaim in a lawsuit) alleging that the Work
   or a Contribution incorporated within the Work constitutes direct
   or contributory patent infringement, then any patent licenses
   granted to You under this License for that Work shall terminate
   as of the date such litigation is filed.

4. Redistribution. You may reproduce and distribute copies of the
   Work or Derivative Works thereof in any medium, with or without
   modifications, and in Source or Object form, provided that You
   meet the following conditions:

   (a) You must give any other recipients of the Work or
       Derivative Works a copy of this License; and

   (b) You must cause any modified files to carry prominent notices
       stating that You changed the files; and

   (c) You must retain, in the Source form of any Derivative Works
       that You distribute, all copyright, patent, trademark, and
       attribution notices from the Source form of the Work,
       excluding those notices that do not pertain to any part of
       the Derivative Works; and

   (d) If the Work includes a "NOTICE" text file as part of its
       distribution, then any Derivative Works that You distribute must
       include a readable copy of the attribution notices contained
       within such NOTICE file, excluding those notices that do not
       pertain to any part of the Derivative Works, in at least one
       of the following places: within a NOTICE text file distributed
       as part of the Derivative Works; within the Source form or
       documentation, if provided along with the Derivative Works; or,
       within a display generated by the Derivative Works, if and
       wherever such third-party notices normally appear. The contents
       of the NOTICE file are for informational purposes only and
       do not modify the License. You may add Your own attribution
       notices within Derivative Works that You distribute, alongside
       or as an addendum to the NOTICE text from the Work, provided
       that such additional attribution notices cannot be construed
       as modifying the License.

   You may add Your own copyright statement to Your modifications and
   may provide additional or different license terms and conditions
   for use, reproduction, or distribution of Your modifications, or
   for any such Derivative Works as a whole, provided Your use,
   reproduction, and distribution of the Work otherwise complies with
   the conditions stated in this License.

5. Submission of Contributions. Unless You explicitly state otherwise,
   any Contribution intentionally submitted for inclusion in the Work
   by You to the Licensor shall be under the terms and conditions of
   this License, without any additional terms or conditions.
   Notwithstanding the above, nothing herein shall supersede or modify
   the terms of any separate license agreement you may have executed
   with Licensor regarding such Contributions.

6. Trademarks. This License does not grant permission to use the trade
   names, trademarks, service marks, or product names of the Licensor,
   except as required for reasonable and customary use in describing the
   origin of the Work and reproducing the content of the NOTICE file.

7. Disclaimer of Warranty. Unless required by applicable law or
   agreed to in writing, Licensor provides the Work (and each
   Contributor provides its Contributions) on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
   implied, including, without limitation, any warranties or conditions
   of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
   PARTICULAR PURPOSE. You are solely responsible for determining the
   appropriateness of using or redistributing the Work and assume any
   risks associated with Your exercise of permissions under this License.

8. Limitation of Liability. In no event and under no legal theory,
   whether in tort (including negligence), contract, or otherwise,
   unless required by applicable law (such as deliberate and grossly
   negligent acts) or agreed to in writing, shall any Contributor be
   liable to You for damages, including any direct, indirect, special,
   incidental, or consequential damages of any character arising as a
   result of this License or out of the use or inability to use the
   Work (including but not limited to damages for loss of goodwill,
   work stoppage, computer failure or malfunction, or any and all
   other commercial damages or losses), even if such Contributor
   has been advised of the possibility of such damages.

9. Accepting Warranty or Additional Liability. While redistributing
   the Work or Derivative Works thereof, You may choose to offer,
   and charge a fee for, acceptance of support, warranty, indemnity,
   or other liability obligations and/or rights consistent with this
   License. However, in accepting such obligations, You may act only
   on Your own behalf and on Your sole responsibility, not on behalf
   of any other Contributor, and only if You agree to indemnify,
   defend, and hold each Contributor harmless for any liability
   incurred by, or claims asserted against, such Contributor by reason
   of your accepting any such warranty or additional liability.

END OF TERMS AND CONDITIONS
```
