/*=============================================================================
  Written by   oneway@vaistrata.com
  Scoped timer - prints how long its scope took, in milliseconds, when it ends.

  The format string takes {} placeholders. With one per argument the output is
  "[formatted] => Nms"; with one more, the last {} is where N goes.
    TimeChecker t("Prefill {} tokens in {}ms", n);
=============================================================================*/

#ifndef VAI_UTILS_TIME_CHECKER_H
#define VAI_UTILS_TIME_CHECKER_H

#include <chrono>
#include <format>


namespace vai::utils {

class TimeChecker {
    std::string name;
    std::chrono::high_resolution_clock::time_point start;
    bool separateOutput = false;

    static constexpr std::string_view durationToken = "<DURATION>";

    static size_t countBraces(std::string_view fmt) {
        size_t count = 0;
        for (size_t i = 0; i + 1 < fmt.size(); ++i) {
            if (fmt[i] == '{' && fmt[i + 1] == '}') {
                ++count;
                ++i;
            }
        }
        return count;
    }

public:
    template <typename... Args>
    TimeChecker(std::string_view fmt, Args&&... args)
        : start(std::chrono::high_resolution_clock::now())
    {
        std::string fmtStr(fmt);
        size_t braceCount = countBraces(fmtStr);
        size_t argCount = sizeof...(Args);

        if (braceCount == argCount) {
            separateOutput = true;
            name = std::vformat(fmtStr, std::make_format_args(args...));
        }
        else {
            size_t pos = fmtStr.rfind("{}");
            if (pos != std::string::npos) {
                fmtStr.replace(pos, 2, durationToken);
            }
            name = std::vformat(fmtStr, std::make_format_args(args...));
        }
    }

    ~TimeChecker() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (separateOutput) {
            std::printf("[%s] => %lldms\n", name.c_str(), (long long)duration);
        }
        else {
            std::string output = name;
            size_t pos = output.find(durationToken);
            if (pos != std::string::npos) {
                output.replace(pos, durationToken.size(), std::to_string(duration));
            }
            std::printf("%s\n", output.c_str());
        }
        std::fflush(stdout);
    }
};


}   // namespace vai::utils

#endif   // VAI_UTILS_TIME_CHECKER_H
