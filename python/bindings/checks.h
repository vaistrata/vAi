#ifndef PYVAI_CHECKS_H
#define PYVAI_CHECKS_H

// Pre-checks shared across areas (node-only ones live in nodes/nodes.h).
#include "common.h"
#include <filesystem>
#include <string>

namespace pyvai {

// FileNotFoundError has no nb::*_error, so it goes through the C API.
inline void ensureFileExists(const char* who, const std::string& path)
{
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec)
    {
        PyErr_SetString(PyExc_FileNotFoundError,
            (std::string(who) + ": file not found: " + path).c_str());
        throw nb::python_error();
    }
}

inline void ensureDirExists(const char* who, const std::string& path)
{
    std::error_code ec;
    if (!std::filesystem::is_directory(path, ec) || ec)
    {
        PyErr_SetString(PyExc_FileNotFoundError,
            (std::string(who) + ": directory not found: " + path).c_str());
        throw nb::python_error();
    }
}

}   // namespace pyvai

#endif // PYVAI_CHECKS_H
