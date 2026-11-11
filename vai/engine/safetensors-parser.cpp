#include "safetensors-parser.h"
#include "vai/utils/tiny-json.h"
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>


namespace vai {

using namespace tinyjson;

struct TensorMeta
{
    std::string dtype;
    std::vector<uint32_t> shape;
    size_t dataOffset[2]; // start, end
};


struct FileCloser { void operator()(std::FILE* f) const { std::fclose(f); } };
using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

// A 64-bit seek. The C standard types fseek's offset as long, which is 32 bits on
// Windows, so a checkpoint past 2GB needs each platform's own name for it.
#if defined(_WIN32)
    #define VAI_SEEK(file, offset)  _fseeki64((file), (long long)(offset), SEEK_SET)
#else
    #define VAI_SEEK(file, offset)  fseeko((file), (off_t)(offset), SEEK_SET)
#endif


struct SafetensorsParserImpl
{
    std::unordered_map<std::string, TensorMeta> tensors;
    FilePtr file;
    std::string path;
    std::thread::id owner = std::this_thread::get_id();
    size_t startOfData = 0;

    // The file stays open and only its header is buffered. Each tensor is read
    // straight into the tensor that asked for it, so the checkpoint never sits in
    // memory a second time.
    //
    // stdio rather than ifstream: fread reads at 3.1GB/s here where ifstream::read
    // stalls at 1.2GB/s, whatever its buffering. Splitting a shard into hundreds of
    // seek+read calls costs nothing measurable on NVMe.
    SafetensorsParserImpl(const char* safetensorsFilePath)
    {
        path = safetensorsFilePath;
        file.reset(std::fopen(safetensorsFilePath, "rb"));
        if (!file)
            throw std::runtime_error("Could not open Safetensors file: " + std::string(safetensorsFilePath));
        std::setvbuf(file.get(), nullptr, _IONBF, 0);

        // 1. Read header length (first 8 bytes)
        uint64_t headerLength = 0;
        if (std::fread(&headerLength, 1, sizeof(headerLength), file.get()) != sizeof(headerLength))
            throw std::runtime_error("Failed to read header length.");

        // 2. Read JSON header
        std::vector<char> headerStr(headerLength + 1, '\0');
        if (std::fread(headerStr.data(), 1, headerLength, file.get()) != headerLength)
            throw std::runtime_error("Failed to read header string.");

        // 3. Parse JSON header
        parseHeader(headerStr.data(), headerLength);

        startOfData = sizeof(headerLength) + headerLength;
    }

    void readInto(const TensorMeta& meta, void* dst) const
    {
        readInto(meta, dst, 0, meta.dataOffset[1] - meta.dataOffset[0]);
    }

    void readInto(const TensorMeta& meta, void* dst, size_t byteOffset, size_t byteCount) const
    {
        const size_t bytes = meta.dataOffset[1] - meta.dataOffset[0];
        if (byteOffset + byteCount > bytes)
            throw std::runtime_error("Tensor read runs past the end of the tensor.");
        readAt(handle(), startOfData + meta.dataOffset[0] + byteOffset, dst, byteCount);
    }

    // The opening thread reads through the parser's own handle; any other thread
    // through one of its own, kept for that thread's lifetime, so reads from
    // several threads at once never share a file position.
    std::FILE* handle() const
    {
        if (std::this_thread::get_id() == owner)
            return file.get();

        thread_local std::unordered_map<std::string, FilePtr> handles;
        FilePtr& own = handles[path];
        if (!own)
        {
            own.reset(std::fopen(path.c_str(), "rb"));
            if (!own)
                throw std::runtime_error("Could not reopen Safetensors file: " + path);
            std::setvbuf(own.get(), nullptr, _IONBF, 0);
        }
        return own.get();
    }

    static void readAt(std::FILE* f, size_t offset, void* dst, size_t byteCount)
    {
        if (VAI_SEEK(f, offset) != 0)
            throw std::runtime_error("Failed to seek to tensor data.");
        if (std::fread(dst, 1, byteCount, f) != byteCount)
            throw std::runtime_error("Failed to read tensor data.");
    }


    void parseHeader(char* data, size_t len)
    {
        sj_Reader r = sj_reader(data, len);
        sj_Value root = sj_read(&r);

        if (root.type != SJ_OBJECT)
            throw std::runtime_error("Invalid Safetensors: root is not object");

        sj_Value key, val;
        while (sj_iter_object(&r, root, &key, &val))
        {
            std::string keyStr(key.start, key.end);
            if (keyStr == "__metadata__")
                continue;

            if (val.type != SJ_OBJECT)
                throw std::runtime_error("Invalid tensor entry: " + keyStr);

            TensorMeta meta;
            sj_Value k2, v2;
            while (sj_iter_object(&r, val, &k2, &v2))
            {
                std::string field(k2.start, k2.end);

                if (field == "dtype" && v2.type == SJ_STRING)
                    meta.dtype = std::string(v2.start, v2.end);
                else if (field == "shape" && v2.type == SJ_ARRAY)
                {
                    sj_Value dim;
                    while (sj_iter_array(&r, v2, &dim))
                        if (dim.type == SJ_NUMBER)
                            meta.shape.push_back(static_cast<uint32_t>(std::strtoul(dim.start, nullptr, 10)));
                }
                else if (field == "data_offsets" && v2.type == SJ_ARRAY)
                {
                    sj_Value off;
                    int idx = 0;
                    while (sj_iter_array(&r, v2, &off) && idx < 2)
                        if (off.type == SJ_NUMBER)
                            meta.dataOffset[idx++] = std::strtoull(off.start, nullptr, 10);
                }
            }
            tensors[keyStr] = std::move(meta);
        }

        if (r.error)
            throw std::runtime_error(std::string("JSON parse error: ") + r.error);
    }

    const TensorMeta& getMeta(std::string_view key) const
    {
        auto it = tensors.find(std::string(key));
        if (it == tensors.end())
            throw std::out_of_range("Tensor key not found: " + std::string(key));
        return it->second;
    }
};


SafetensorsParser::~SafetensorsParser() = default;
SafetensorsParser::SafetensorsParser(const char* safetensorsFilePath)
    : pImpl_(std::make_unique<SafetensorsParserImpl>(safetensorsFilePath))
{
}


std::vector<std::string> SafetensorsParser::getTensorNames() const
{
    std::vector<std::string> names;
    names.reserve(pImpl_->tensors.size());
    for (const auto& [key, _] : pImpl_->tensors)
        names.push_back(key);
    return names;
}


std::vector<uint32_t> SafetensorsParser::getShape(std::string_view key) const
{
    return pImpl_->getMeta(key).shape;
}


std::string SafetensorsParser::getDataType(std::string_view key) const
{
    return pImpl_->getMeta(key).dtype;
}


size_t SafetensorsParser::byteSize(std::string_view key) const
{
    const auto& meta = pImpl_->getMeta(key);
    const size_t bytes = meta.dataOffset[1] - meta.dataOffset[0];

    const size_t numElements = std::accumulate(meta.shape.begin(), meta.shape.end(), size_t(1), std::multiplies<size_t>());
    if (meta.dtype == "F32")
    {
        if (bytes != numElements * sizeof(float))
            throw std::runtime_error("Data size mismatch for F32 tensor.");
    }
    else if (meta.dtype == "F16" || meta.dtype == "BF16")
    {
        if (bytes != numElements * 2)
            throw std::runtime_error("Data size mismatch for F16/BF16 tensor.");
    }
    else if (meta.dtype == "Q4")   // vai's own (saveSafetensors): packed words, then fp16 scales
    {
        if (numElements % 32 != 0 || bytes != numElements / 2 + numElements / 32 * 2)
            throw std::runtime_error("Data size mismatch for Q4 tensor.");
    }
    else
        throw std::runtime_error("Unsupported dtype: " + meta.dtype);

    return bytes;
}


void SafetensorsParser::readInto(std::string_view key, void* dst) const
{
    pImpl_->readInto(pImpl_->getMeta(key), dst);
}


void SafetensorsParser::readInto(
    std::string_view key, void* dst, size_t byteOffset, size_t byteCount) const
{
    pImpl_->readInto(pImpl_->getMeta(key), dst, byteOffset, byteCount);
}


} // namespace vai
