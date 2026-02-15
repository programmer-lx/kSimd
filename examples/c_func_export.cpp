#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "c_func_export.cpp"
#include <kSimd/core/dispatch_this_file.hpp>
#include <kSimd/core/dispatch_core.hpp>

// non-named namespace: hide symbol
namespace
{
    // dynamic namespace : dispatch function
    namespace KSIMD_DYN_INSTRUCTION
    {
        template<typename T, bool tag>
        KSIMD_DYN_FUNC_ATTR void kernel_template(
            const T* KSIMD_RESTRICT src,
                  T* KSIMD_RESTRICT dst,
            const size_t            size
        ) noexcept
        {
            // do something...
            namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
            constexpr size_t lanes = ns::Lanes<T>;

            size_t i = 0;
            for (; i + lanes <= size; i += lanes)
            {
                ns::Batch<T> data = ns::loadu(src + i);
                ns::storeu(dst + i, data);
            }

            // tail
            if (const size_t tail = size - i; tail > 0)
            {
                ns::Batch<T> data = ns::loadu_partial(src + i, tail);
                ns::storeu_partial(dst + i, data, tail);
            }
        }
    } // namespace KSIMD_DYN_INSTRUCTION
} // namespace

#if KSIMD_ONCE

// non-named namespace: hide symbol
namespace
{
    // make function table
    template<typename T, bool tag>
    KSIMD_DYN_DISPATCH_FUNC(kernel_template, <T, tag>);
}

// export C functions

extern "C"
{
    KSIMD_DLL_EXPORT void kernel1(
        const float* KSIMD_RESTRICT src,
              float* KSIMD_RESTRICT dst,
        const uint64_t              size /* use fixed size data type */
    )
    {
        KSIMD_DYN_CALL(kernel_template, <float, false>)(src, dst, static_cast<size_t>(size));
    }

    KSIMD_DLL_EXPORT void kernel2(
        const float* KSIMD_RESTRICT src,
              float* KSIMD_RESTRICT dst,
        const uint64_t              size /* use fixed size data type */
    )
    {
        KSIMD_DYN_CALL(kernel_template, <float, true>)(src, dst, static_cast<size_t>(size));
    }
}

#endif
