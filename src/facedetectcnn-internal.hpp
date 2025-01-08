#include <iostream>
#include <vector>
#include "facedetectcnn.h"
#include "facedetectcnn-namespace.hpp"
#include <cstring>


#ifndef MIN
#define MIN(a, b) ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#endif

namespace libfacedetect {

extern void* (*aligned_alloc)(size_t size, size_t alignement);
extern void (*aligned_free)(void *ptr, size_t alignment);

struct ConvInfoStruct {
  int channels;
  int num_filters;
  bool is_depthwise;
  bool is_pointwise;
  bool with_relu;
  float *pWeights;
  float *pBiases;
};

template <typename T>
class CDataBlob {
 public:
  int rows{0};
  int cols{0};
  int channels{0};     // in element
  int channelStep{0};  // in byte
  int cell_align{0};   // in byte
  T *data{nullptr};

 public:
  CDataBlob() = default;
  CDataBlob(int r, int c, int ch, int cell_align)
  {
    data = nullptr;
    create(r, c, ch, cell_align);
    // #warning "confirm later"
    setZero();
  }
  ~CDataBlob() { setNULL(); }

  CDataBlob(CDataBlob<T> &&other)
  {
    data = other.data;
    other.data = nullptr;
    rows = other.rows;
    cols = other.cols;
    channels = other.channels;
    channelStep = other.channelStep;
    cell_align = other.cell_align;
  }

  CDataBlob<T> &operator=(CDataBlob<T> &&other)
  {
    this->~CDataBlob();
    new (this) CDataBlob<T>(std::move(other));
    return *this;
  }

  void setNULL()
  {
    if (data) {
      aligned_free(data, cell_align);
    }
    rows = cols = channels = channelStep = cell_align = 0;
    data = nullptr;
  }

  void setZero()
  {
    if (data) {
      memset(data, 0, channelStep * rows * cols);
    }
  }

  inline bool isEmpty() const
  {
    return (rows <= 0 || cols <= 0 || channels == 0 || data == nullptr);
  }

  bool create(int r, int c, int ch, int cell_align)
  {
    setNULL();

    rows = r;
    cols = c;
    channels = ch;
    this->cell_align = cell_align;

    // alloc space for int8 array
    int remBytes = (sizeof(T) * channels) % cell_align;
    if (remBytes == 0) {
      this->channelStep = channels * sizeof(T);
    } else {
      this->channelStep = (channels * sizeof(T)) + cell_align - remBytes;
    }
    data = (T *)libfacedetect::aligned_alloc(size_t(rows) * cols * this->channelStep, cell_align);

    if (data == nullptr) {
      std::cerr << "Failed to alloc memeory for uint8 data blob: " << rows
                << "*" << cols << "*" << channels << std::endl;
      return false;
    }

    // memset(data, 0, width * height * channelStep);

    // the following code is faster than memset
    // but not only the padding bytes are set to zero.
    // BE CAREFUL!!!
    // #if defined(_OPENMP)
    // #pragma omp parallel for
    // #endif
    //  for (int r = 0; r < this->rows; r++)
    //  {
    //      for (int c = 0; c < this->cols; c++)
    //      {
    //          int pixel_end = this->channelStep / sizeof(T);
    //          T * pI = this->ptr(r, c);
    //          for (int ch = this->channels; ch < pixel_end; ch++)
    //              pI[ch] = 0;
    //      }
    //  }

    return true;
  }

  inline T *ptr(int r, int c)
  {
    if (r < 0 || r >= this->rows || c < 0 || c >= this->cols)
      return nullptr;

    return (this->data + (size_t(r) * this->cols + c) * this->channelStep / sizeof(T));
  }
  inline const T *ptr(int r, int c) const
  {
    if (r < 0 || r >= this->rows || c < 0 || c >= this->cols)
      return nullptr;

    return (this->data + (size_t(r) * this->cols + c) * this->channelStep / sizeof(T));
  }

  inline const T getElement(int r, int c, int ch) const
  {
    if (this->data) {
      if (r >= 0 && r < this->rows && c >= 0 && c < this->cols && ch >= 0 && ch < this->channels) {
        const T *p = this->ptr(r, c);
        return (p[ch]);
      }
    }

    return (T)(0);
  }

  friend std::ostream &operator<<(std::ostream &output, CDataBlob &dataBlob)
  {
    output << "DataBlob Size (channels, rows, cols) = ("
           << dataBlob.channels << ", " << dataBlob.rows << ", "
           << dataBlob.cols << ")" << std::endl;
    if (dataBlob.rows * dataBlob.cols * dataBlob.channels <= 16) {  // print the elements only when the total number is less than
                                                                    // 64
      for (int ch = 0; ch < dataBlob.channels; ch++) {
        output << "Channel " << ch << ": " << std::endl;

        for (int r = 0; r < dataBlob.rows; r++) {
          output << "(";
          for (int c = 0; c < dataBlob.cols; c++) {
            T *p = dataBlob.ptr(r, c);

            if (sizeof(T) < 4)
              output << (int)(p[ch]);
            else
              output << p[ch];

            if (c != dataBlob.cols - 1)
              output << ", ";
          }
          output << ")" << std::endl;
        }
      }
    } else {
      output << "(";
      int idx = 0;
      for (int r = 0; r < dataBlob.rows; ++r) {
        for (int c = 0; c < dataBlob.cols; ++c) {
          for (int ch = 0; ch < dataBlob.channels; ++ch) {
            output << dataBlob.getElement(r, c, ch) << ", ";
            ++idx;
            if (idx >= 16) {
              goto outloop;
            }
          }
        }
      }
    outloop:
      output << "..., "
             << dataBlob.getElement(dataBlob.rows - 1, dataBlob.cols - 1, dataBlob.channels - 1)
             << ")" << std::endl;
      float max_it = -500.f;
      float min_it = 500.f;
      for (int r = 0; r < dataBlob.rows; ++r) {
        for (int c = 0; c < dataBlob.cols; ++c) {
          for (int ch = 0; ch < dataBlob.channels; ++ch) {
            max_it = std::max(max_it, dataBlob.getElement(r, c, ch));
            min_it = std::min(min_it, dataBlob.getElement(r, c, ch));
          }
        }
      }
      output << "max_it: " << max_it << "    min_it: " << min_it
             << std::endl;
    }
    return output;
  }
};

template <typename T>
class Filters {
 public:
  int channels;
  int num_filters;
  bool is_depthwise;
  bool is_pointwise;
  bool with_relu;
  CDataBlob<T> weights;
  CDataBlob<T> biases;

  Filters()
  {
    channels = 0;
    num_filters = 0;
    is_depthwise = false;
    is_pointwise = false;
    with_relu = true;
  }

  void set(ConvInfoStruct &convinfo, int cell_align)
  {
    if (!std::is_same_v<T, float>) {
      std::cerr << "The data type must be float in this version." << std::endl;
      return;
    }
    // clang-format off
    if (!std::is_same_v<decltype(convinfo.pWeights), float *>
     || !std::is_same_v<decltype(convinfo.pBiases), float *>) {
      std::cerr << "The data type of the filter parameters must be float in this version." << std::endl;
      return;
    }
    // clang-format on

    this->channels = convinfo.channels;
    this->num_filters = convinfo.num_filters;
    this->is_depthwise = convinfo.is_depthwise;
    this->is_pointwise = convinfo.is_pointwise;
    this->with_relu = convinfo.with_relu;

    if (!this->is_depthwise && this->is_pointwise)  // 1x1 point wise
    {
      this->weights.create(1, num_filters, channels, cell_align);
    } else if (this->is_depthwise && !this->is_pointwise)  // 3x3 depth wise
    {
      this->weights.create(1, 9, channels, cell_align);
    } else {
      std::cerr << "Unsupported filter type. Only 1x1 point-wise and 3x3 "
                   "depth-wise are supported."
                << std::endl;
      return;
    }

    this->biases.create(1, 1, num_filters, cell_align);

    // the format of convinfo.pWeights/biases must meet the format in
    // this->weigths/biases
    for (int fidx = 0; fidx < this->weights.cols; fidx++) {
      memcpy(this->weights.ptr(0, fidx), convinfo.pWeights + channels * fidx, channels * sizeof(T));
    }
    memcpy(this->biases.ptr(0, 0), convinfo.pBiases, sizeof(T) * this->num_filters);
  }
};

namespace NAMESPACE_NAME {

CDataBlob<float> setDataFrom3x3S2P1to1x1S1P0FromImage(
  const unsigned char *inputData,
  int imgWidth,
  int imgHeight,
  int imgChannels,
  int imgWidthStep,
  int padDivisor = 32
);
CDataBlob<float> convolution(
  const CDataBlob<float> &inputData,
  const Filters<float> &filters,
  bool do_relu = true
);
CDataBlob<float> convolutionDP(
  const CDataBlob<float> &inputData,
  const Filters<float> &filtersP,
  const Filters<float> &filtersD,
  bool do_relu = true
);
CDataBlob<float> convolution4layerUnit(
  const CDataBlob<float> &inputData,
  const Filters<float> &filtersP1,
  const Filters<float> &filtersD1,
  const Filters<float> &filtersP2,
  const Filters<float> &filtersD2,
  bool do_relu = true
);
CDataBlob<float> maxpooling2x2S2(const CDataBlob<float> &inputData);

CDataBlob<float> elementAdd(
  const CDataBlob<float> &inputData1,
  const CDataBlob<float> &inputData2
);
CDataBlob<float> upsampleX2(const CDataBlob<float> &inputData);

CDataBlob<float> meshgrid(
  int feature_width,
  int feature_height,
  int stride,
  float offset = 0.0f
);

// TODO implement in SIMD
void bbox_decode(CDataBlob<float> &bbox_pred, const CDataBlob<float> &priors, int stride);
void kps_decode(CDataBlob<float> &bbox_pred, const CDataBlob<float> &priors, int stride);

template <typename T>
CDataBlob<T> blob2vector(const CDataBlob<T> &inputData);

template <typename T>
CDataBlob<T> concat3(
  const CDataBlob<T> &inputData1,
  const CDataBlob<T> &inputData2,
  const CDataBlob<T> &inputData3
);

// TODO implement in SIMD
void sigmoid(CDataBlob<float> &inputData);

std::vector<FaceRect> detection_output(
  const CDataBlob<float> &cls,
  const CDataBlob<float> &reg,
  const CDataBlob<float> &kps,
  const CDataBlob<float> &obj,
  float overlap_threshold,
  float confidence_threshold,
  int top_k,
  int keep_top_k
);

}  // namespace NAMESPACE_NAME

}  // namespace libfacedetect
