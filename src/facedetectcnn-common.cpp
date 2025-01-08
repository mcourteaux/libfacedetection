#include "facedetectcnn.h"
#include "facedetectcnn-internal.hpp"

#define NUM_CONV_LAYER 53

namespace libfacedetect {

extern ConvInfoStruct param_pConvInfo[NUM_CONV_LAYER];
Filters<float> g_pFilters[NUM_CONV_LAYER];

bool param_initialized = false;


void init_parameters(int cell_align)
{
  for (int i = 0; i < NUM_CONV_LAYER; i++) {
    g_pFilters[i].set(param_pConvInfo[i], cell_align);
  }
}

void *default_aligned_alloc(size_t size, size_t align)
{
  char *ptr, *ptr0;
  ptr0 = (char *)malloc((
    size_t
  )(size + align * ((size >= 4096) + 1L) + sizeof(char *)));

  if (!ptr0)
    return 0;

  // align the pointer
  ptr = (char *)(((size_t)(ptr0 + sizeof(char *) + 1) + align - 1) & ~(size_t)(align - 1));
  *(char **)(ptr - sizeof(char *)) = ptr0;

  return ptr;
}

void default_aligned_free(void *ptr, size_t align)
{
  // Pointer must be aligned by `align`
  if (ptr) {
    if (((size_t)ptr & (align - 1)) != 0)
      return;
    free(*((char **)ptr - 1));
  }
}

void* (*aligned_alloc)(size_t size, size_t alignement) = &default_aligned_alloc;
void (*aligned_free)(void *ptr, size_t alignment) = &default_aligned_free;

}  // namespace libfacedetect
