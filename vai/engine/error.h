#ifndef VAI_ENGINE_ERROR_H
#define VAI_ENGINE_ERROR_H

#include <cstdlib>
#include <cstdio>


inline void vai_assert_impl(
    bool condition,
    const char* condition_str,
    const char* func,
    const char* file,
    int line
) {
    if (!condition) {
        fprintf(stderr,
            "[VAI_ASSERT FAILED] \"%s\" at %s in %s (line %d)\n",
            condition_str,
            func,
            file,
            line
        );
        std::abort();
    }
}

// The check must not be an argument to vai_assert_impl: that evaluates it in
// release too, which a graph-walking predicate cannot afford.
#ifndef NDEBUG
    #define VAI_ASSERT(expr) vai_assert_impl(static_cast<bool>(expr), #expr, __func__, __FILE__, __LINE__)
#else
    #define VAI_ASSERT(expr) ((void)0)
#endif


#endif   // VAI_ENGINE_ERROR_H
