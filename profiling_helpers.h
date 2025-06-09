#pragma once

#ifdef ENABLE_NVTX
  #include <nvtx3/nvToolsExt.h>

  // Renklerden oluşan palet (modu/katmanı ayırmak için)
  static const uint32_t NVTX_COLORS[] = {
    0xFF00FF00, // Yeşil
    0xFF0000FF, // Mavi
    0xFFFF0000, // Kırmızı
    0xFFFFFF00, // Sarı
    0xFF00FFFF, // Camgöbeği
    0xFFFF00FF, // Pembe
    0xFF888888, // Gri
    0xFF800000  // Kahverengi
  };
static int nvtx_color_index = 0;

// Gelişmiş push: mesaj, renk, payload, kategori ekler
#define NVTX_PUSH(name) do { \
nvtxEventAttributes_t attr = {}; \
attr.version       = NVTX_VERSION; \
attr.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
attr.colorType     = NVTX_COLOR_ARGB; \
attr.color         = NVTX_COLORS[nvtx_color_index++ % (sizeof(NVTX_COLORS)/sizeof(NVTX_COLORS[0]))]; \
attr.messageType   = NVTX_MESSAGE_TYPE_ASCII; \
attr.message.ascii = name; \
attr.category      = 0;  /* 0–63 arası kategori (ör: mode, kernel, memcopy) */ \
attr.payloadType   = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64; \
attr.payload.ullValue = (uint64_t)__LINE__;  /* payload olarak kaynak satırı */ \
nvtxRangePushEx(&attr); \
} while(0)

#define NVTX_POP() nvtxRangePop()

#else
#define NVTX_PUSH(name)
#define NVTX_POP()
#endif
