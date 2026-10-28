/*
 * wupz_abi.h — the one boundary between wupz-rs / upz-rs (Rust) and wupz-zig / upz-zig (Zig).
 *
 * FROZEN. Any change bumps WUPZ_ABI_VERSION and both sides in lockstep.
 * This header is a contract, not a convenience: if it churns, the seam is
 * broken and the "two toolchains, zero-cost" thesis is falsified.
 */
#ifndef WUPZ_ABI_H
#define WUPZ_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUPZ_ABI_VERSION 1u
#define WUPZ_GATE_0X62   0x62u
#define WUPZ_QUANT_QT55  0x55u

/*
 * 64-byte cache-line aligned Litany Wire Frame (`align(64)`).
 * Fits in a single CPU cache line (Apple Silicon M1-M4 / x86_64 AVX-512)
 * for zero-copy IPC and incremental pack-in-pass verification.
 */
#if defined(_MSC_VER)
#define WUPZ_ALIGN64 __declspec(align(64))
#elif defined(__GNUC__) || defined(__clang__)
#define WUPZ_ALIGN64 __attribute__((aligned(64)))
#else
#define WUPZ_ALIGN64
#endif

typedef struct WUPZ_ALIGN64 wupz_litany_wire_frame_s {
    uint8_t  frame32[32];    /* [0..32]  32B 0x62-gate witness frame */
    uint64_t bfp64_pos;      /* [32..40] BFP-64 +1 bitplane (fold & qt_mask) */
    uint64_t bfp64_neg;      /* [40..48] BFP-64 -1 bitplane (fold & ~qt_mask) */
    uint32_t ratio_p;        /* [48..52] GCD-reduced lossless ratio numerator p */
    uint32_t ratio_q;        /* [52..56] GCD-reduced lossless ratio denominator q */
    uint8_t  spqt_root[3];   /* [56..59] 1.585 bits/trit base-243 SPQT 15-trit root */
    uint8_t  qt_mask;        /* [59]     Quantization mask byte (0x55) */
    uint16_t lanes_active;   /* [60..62] Active lane ratio word (0x0101 = 1/1) */
    uint16_t wire_seal16;    /* [62..64] 16-bit XOR fold witness over [0..62] */
} wupz_litany_wire_frame_t;

/* wupz-rs: the ABI version the Rust core speaks. */
uint32_t wupz_abi_version(void);

/* wupz-zig: the ABI version the Zig kernels speak. Both must equal
 * WUPZ_ABI_VERSION — tests/abi_smoke.c asserts it at link time. */
uint32_t wupz_zig_abi_version(void);

/* wupz-zig: integrity fold. XOR of the little-endian u64 words, then the
 * trailing bytes. Allocation-free, order-stable, and its own witness: fold a
 * valid frame and you get zero if it was built as an XOR seal. */
uint64_t wupz_xor_fold(const uint8_t *ptr, size_t len);

/* wupz-zig: the number of differing bytes between two equal-length spans
 * (the delta size an incremental repack would touch). */
uint64_t wupz_byte_diff(const uint8_t *a, const uint8_t *b, size_t len);

/* wupz-zig: population count over a span (density hint). */
uint64_t wupz_popcount(const uint8_t *ptr, size_t len);

/* wupz-zig: seal a 64-byte cache-line-aligned Litany Wire frame at `gate` and `qt`.
 * Returns 1 on success, 0 on null pointer or invalid input. */
uint32_t wupz_seal_litany_wire_64b(
    wupz_litany_wire_frame_t *out_wire,
    const uint8_t *raw_ptr,
    size_t raw_len,
    const uint8_t *packed_ptr,
    size_t packed_len,
    uint8_t gate,
    uint8_t qt
);

/* wupz-zig: verify a 64-byte Litany Wire frame in constant shape.
 * Returns 1 if valid, 0 if tampered or null. */
uint32_t wupz_verify_litany_wire_64b(
    const wupz_litany_wire_frame_t *wire,
    const uint8_t *raw_ptr,
    size_t raw_len,
    const uint8_t *packed_ptr,
    size_t packed_len,
    uint8_t gate,
    uint8_t qt
);

/* DSE-resistant memory scrub helper for C/FFI callers (ZA-0003).
 * Uses volatile byte stores + compiler memory barrier so -O2/-O3 Dead Store
 * Elimination cannot strip stack/heap wipes prior to scope exit. */
static inline void wupz_zeroize(void *ptr, size_t len) {
    if (ptr == NULL || len == 0) {
        return;
    }
    volatile uint8_t *p = (volatile uint8_t *)ptr;
    for (size_t i = 0; i < len; ++i) {
        p[i] = 0u;
    }
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" : : "r"(ptr) : "memory");
#endif
}

#ifdef __cplusplus
}
#endif

#endif /* WUPZ_ABI_H */
