#ifndef VAI_ENGINE_SAFETENSORS_PARSER_H
#define VAI_ENGINE_SAFETENSORS_PARSER_H

#include <vector>
#include <memory>
#include <string>
#include <string_view>
#include <cstdint>

namespace vai {

struct SafetensorsParserImpl;
class SafetensorsParser;

struct SafetensorRef
{
    const SafetensorsParser* parser;
    std::string key;

    std::vector<uint32_t> getShape() const;
    std::string getDataType() const;
    size_t byteSize() const;
    void readInto(void* dst) const;
    void readInto(void* dst, size_t byteOffset, size_t byteCount) const;
};


class SafetensorsParser
{
    std::unique_ptr<SafetensorsParserImpl> pImpl_;

public:
    SafetensorsParser(const char* safetensorsFilePath);
    ~SafetensorsParser();

    std::vector<std::string> getTensorNames() const;

    SafetensorRef operator[](std::string_view key) const
    {
        return { this, std::string(key) };
    }

    std::vector<uint32_t> getShape(std::string_view key) const;
    std::string getDataType(std::string_view key) const;

    // Bytes this tensor occupies in the file, as stored.
    size_t byteSize(std::string_view key) const;

    // Read this tensor straight from the file into dst, which must hold
    // byteSize(key) bytes. Nothing is buffered in between.
    void readInto(std::string_view key, void* dst) const;

    // ... or byteCount of it from byteOffset, for a caller that wants one run of
    // the leading axis and not the rest.
    void readInto(std::string_view key, void* dst, size_t byteOffset, size_t byteCount) const;
};


inline std::vector<uint32_t> SafetensorRef::getShape() const { return parser->getShape(key); }
inline std::string SafetensorRef::getDataType() const { return parser->getDataType(key); }
inline size_t SafetensorRef::byteSize() const { return parser->byteSize(key); }
inline void SafetensorRef::readInto(void* dst) const { parser->readInto(key, dst); }
inline void SafetensorRef::readInto(void* dst, size_t byteOffset, size_t byteCount) const
{
    parser->readInto(key, dst, byteOffset, byteCount);
}

} // namespace vai

#endif   // VAI_ENGINE_SAFETENSORS_PARSER_H
