#if defined(_ENABLE_AVX512)
#define NAMESPACE_NAME avx512
#elif defined(_ENABLE_AVX2)
#define NAMESPACE_NAME avx2
#elif defined(_ENABLE_NEON)
#define NAMESPACE_NAME neon
#else
#define NAMESPACE_NAME scalar
#endif

namespace libfacedetect {
namespace NAMESPACE_NAME {

#if defined(_ENABLE_AVX512)
constexpr int CELL_ALIGN_BITS = 512;
#elif defined(_ENABLE_AVX2)
constexpr int CELL_ALIGN_BITS = 256;
#else
constexpr int CELL_ALIGN_BITS = 128;
#endif
constexpr int CELL_ALIGN = (CELL_ALIGN_BITS >> 3);

}  // namespace NAMESPACE_NAME
}  // namespace libfacedetect
