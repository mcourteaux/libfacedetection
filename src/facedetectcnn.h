/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install, copy or
use the software.


                  License Agreement For libfacedetection
                     (3-clause BSD License)

Copyright (c) 2018-2021, Shiqi Yu, all rights reserved.
shiqi.yu@gmail.com

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#pragma once

#include <vector>

namespace libfacedetect {

extern void *(*aligned_alloc)(size_t size, size_t alignement);
extern void (*aligned_free)(void *ptr, size_t alignment);

struct FaceRect {
  float score;
  int x;
  int y;
  int w;
  int h;
  int lm[10];
};

#define API_FOR_ARCH                                                                            \
  int *facedetect_cnn(                                                                          \
    unsigned char result_buffer[0x9000], /* buffer memory for storing face detection results */ \
    unsigned char *rgb_image_data,       /* BGR order! */                                       \
    int width,                                                                                  \
    int height,                                                                                 \
    int step                                                                                    \
  );                                                                                            \
                                                                                                \
  std::vector<FaceRect> objectdetect_cnn(                                                       \
    const unsigned char *rgbImageData, /* BGR order! */                                         \
    int with,                                                                                   \
    int height,                                                                                 \
    int step                                                                                    \
  );

namespace avx2 {
API_FOR_ARCH;
}  // namespace avx2
namespace avx512 {
API_FOR_ARCH;
}  // namespace avx512
namespace neon {
API_FOR_ARCH;
}  // namespace neon
namespace scalar {
API_FOR_ARCH;
}  // namespace scalar

#undef API_FOR_ARCH

}  // namespace libfacedetect
