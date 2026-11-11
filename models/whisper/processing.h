/*=============================================================================
  Written by   chaudharyshivansh9@gmail.com
  Brief        Whisper's audio frontend: 16 kHz samples -> normalized log-mel,
               matching HuggingFace WhisperFeatureExtractor.

    auto mel = whisper::logMel(samples.data() + start, count, cfg.numMels);
    melTensor.setFloats(mel);   // [numMels, 1, 3000]
=============================================================================*/

#ifndef WHISPER_PROCESSING_H
#define WHISPER_PROCESSING_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>
#include <vector>


namespace whisper {

inline constexpr uint32_t kSampleRate   = 16000;
inline constexpr uint32_t kFftSize      = 400;                  // 25 ms window
inline constexpr uint32_t kHopLength    = 160;                  // 10 ms hop
inline constexpr uint32_t kWindowFrames = 3000;                 // 30 s
inline constexpr uint32_t kWindowSamples = kWindowFrames * kHopLength;
inline constexpr uint32_t kFreqBins     = kFftSize / 2 + 1;


namespace detail {

// Slaney mel scale: linear below 1 kHz, logarithmic above
inline double hzToMel(double hz)
{
    const double logStep = std::log(6.4) / 27.0;
    return hz < 1000.0 ? 3.0 * hz / 200.0 : 15.0 + std::log(hz / 1000.0) / logStep;
}

inline double melToHz(double mel)
{
    const double logStep = std::log(6.4) / 27.0;
    return mel < 15.0 ? 200.0 * mel / 3.0 : 1000.0 * std::exp(logStep * (mel - 15.0));
}

// out[k] = sum_t in[t * stride] e^(-2πi kt/n), n = kFftSize / stride; w = kFftSize twiddles
inline void fft(const float* in, uint32_t n, uint32_t stride,
    const std::complex<float>* w, std::complex<float>* out)
{
    if (n % 2)   // odd length (25): direct DFT
    {
        for (uint32_t k = 0; k < n; ++k)
        {
            std::complex<float> sum = 0.0f;
            for (uint32_t t = 0; t < n; ++t)
                sum += in[t * stride] * w[(k * t % n) * stride];
            out[k] = sum;
        }
        return;
    }

    const uint32_t half = n / 2;
    fft(in,          half, stride * 2, w, out);
    fft(in + stride, half, stride * 2, w, out + half);
    for (uint32_t k = 0; k < half; ++k)
    {
        const std::complex<float> e = out[k];
        const std::complex<float> o = out[k + half] * w[k * stride];
        out[k] = e + o;
        out[k + half] = e - o;
    }
}

}   // namespace detail


// Slaney-normalized triangular filters [numMels, kFreqBins] over 0..8 kHz
inline std::vector<float> melFilters(uint32_t numMels)
{
    std::vector<double> edges(numMels + 2);
    const double melMax = detail::hzToMel(kSampleRate / 2.0);
    for (uint32_t i = 0; i < edges.size(); ++i)
        edges[i] = detail::melToHz(melMax * i / (numMels + 1));

    std::vector<float> filters(numMels * kFreqBins);
    for (uint32_t m = 0; m < numMels; ++m)
    {
        const double enorm = 2.0 / (edges[m + 2] - edges[m]);
        for (uint32_t k = 0; k < kFreqBins; ++k)
        {
            const double f    = (kSampleRate / 2.0) * k / (kFreqBins - 1);
            const double down = (f - edges[m]) / (edges[m + 1] - edges[m]);
            const double up   = (edges[m + 2] - f) / (edges[m + 2] - edges[m + 1]);
            filters[m * kFreqBins + k] = float(std::max(0.0, std::min(down, up)) * enorm);
        }
    }
    return filters;
}


// One 30 s window -> [numMels, kWindowFrames]; shorter input is zero-padded
inline std::vector<float> logMel(const float* samples, size_t count, uint32_t numMels)
{
    count = std::min<size_t>(count, kWindowSamples);

    // Reflect-pad by half a window on each side (center = True)
    const uint32_t pad = kFftSize / 2;
    std::vector<float> x(kWindowSamples + 2 * pad, 0.0f);
    std::copy_n(samples, count, x.begin() + pad);
    for (uint32_t i = 0; i < pad; ++i)
    {
        x[pad - 1 - i] = x[pad + 1 + i];
        x[pad + kWindowSamples + i] = x[pad + kWindowSamples - 2 - i];
    }

    std::vector<float> window(kFftSize);
    std::vector<std::complex<float>> twiddles(kFftSize);
    for (uint32_t n = 0; n < kFftSize; ++n)
    {
        const double a = 2.0 * std::numbers::pi * n / kFftSize;
        window[n]   = float(0.5 - 0.5 * std::cos(a));   // periodic Hann
        twiddles[n] = std::polar(1.0f, float(-a));
    }

    const std::vector<float> filters = melFilters(numMels);
    std::vector<float> mel(size_t(numMels) * kWindowFrames);
    std::vector<float> frame(kFftSize), power(kFreqBins);
    std::vector<std::complex<float>> spectrum(kFftSize);

    for (uint32_t f = 0; f < kWindowFrames; ++f)
    {
        const float* src = x.data() + size_t(f) * kHopLength;
        for (uint32_t n = 0; n < kFftSize; ++n)
            frame[n] = src[n] * window[n];

        detail::fft(frame.data(), kFftSize, 1, twiddles.data(), spectrum.data());
        for (uint32_t k = 0; k < kFreqBins; ++k)
            power[k] = std::norm(spectrum[k]);

        for (uint32_t m = 0; m < numMels; ++m)
        {
            const float* fl = filters.data() + size_t(m) * kFreqBins;
            float e = 0.0f;
            for (uint32_t k = 0; k < kFreqBins; ++k)
                e += fl[k] * power[k];
            mel[size_t(m) * kWindowFrames + f] = std::log10(std::max(e, 1e-10f));
        }
    }

    // Clamp to 8 decades below the peak, then scale to roughly [-1, 1]
    const float peak = *std::max_element(mel.begin(), mel.end());
    for (float& v : mel)
        v = (std::max(v, peak - 8.0f) + 4.0f) / 4.0f;
    return mel;
}

}   // namespace whisper

#endif // WHISPER_PROCESSING_H
