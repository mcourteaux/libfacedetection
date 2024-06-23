#include "facedetectcnn.h"

#define NUM_CONV_LAYER 53

extern ConvInfoStruct param_pConvInfo[NUM_CONV_LAYER];
Filters<float> g_pFilters[NUM_CONV_LAYER];

bool param_initialized = false;

void init_parameters() {
    for (int i = 0; i < NUM_CONV_LAYER; i++)
        g_pFilters[i] = param_pConvInfo[i];
}

void *myAlloc(size_t size) {
    char *ptr, *ptr0;
    ptr0 = (char *)malloc((
        size_t)(size + _MALLOC_ALIGN * ((size >= 4096) + 1L) + sizeof(char *)));

    if (!ptr0)
        return 0;

    // align the pointer
    ptr = (char *)(((size_t)(ptr0 + sizeof(char *) + 1) + _MALLOC_ALIGN - 1) &
                   ~(size_t)(_MALLOC_ALIGN - 1));
    *(char **)(ptr - sizeof(char *)) = ptr0;

    return ptr;
}

void myFree_(void *ptr) {
    // Pointer must be aligned by _MALLOC_ALIGN
    if (ptr) {
        if (((size_t)ptr & (_MALLOC_ALIGN - 1)) != 0)
            return;
        free(*((char **)ptr - 1));
    }
}

