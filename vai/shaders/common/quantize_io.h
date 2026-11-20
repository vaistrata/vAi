// =============================================================================
// quantize_io.h - read a quantized [R, K]: a weight, or the activation
// quantize.comp makes for the integer kernels.
//
// One buffer holds every element's quant, row-major, then an fp16 scale per
// block of QUANTIZE_BLOCK elements along a row (vai/engine/quantize.h). A
// kernel reads it through two aliases: the quants as int words (Q8: four per
// word) and the scales as float16_t.
// =============================================================================
#ifndef VAI_COMMON_QUANTIZE_IO_H
#define VAI_COMMON_QUANTIZE_IO_H

// K elements a scale covers (vai::quantizeBlock).
#define QUANTIZE_BLOCK 32u

// The scale of element `e` of a Qint8 [R, K], as an fp16 index: past the R * K
// one-byte quants. K is whole blocks, so blocks follow element order.
#define SCALE_INDEX_Q8(R, K, e) ((R) * (K) / 2u + (e) / QUANTIZE_BLOCK)

// Read quant `e` from an int[] buffer of int8 quants as float.
#define READ_I8(buf, e) float(bitfieldExtract((buf)[(e) >> 2], int(8u * ((e) & 3u)), 8))

// The four int8 quants of word `w`, the lowest first, as vec4.
#define READ_4I8(w) vec4(bitfieldExtract((w), 0, 8), bitfieldExtract((w), 8, 8), bitfieldExtract((w), 16, 8), bitfieldExtract((w), 24, 8))

#endif // VAI_COMMON_QUANTIZE_IO_H
