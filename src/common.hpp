#ifndef ISLAY_COMMON_HPP
#define ISLAY_COMMON_HPP

// Compiler hints.
#if defined(_MSC_VER)
#define ISLAY_FORCEINLINE __forceinline
#else
#define ISLAY_FORCEINLINE inline __attribute__((always_inline))
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ISLAY_HOT __attribute__((hot))
#define ISLAY_FLATTEN __attribute__((flatten))
#else
#define ISLAY_HOT
#define ISLAY_FLATTEN
#endif

// Loop-only pragmas; use only after an interleaved A/B win.
#define ISLAY_PRAGMA(x) _Pragma(#x)
#if defined(__clang__)
#define ISLAY_UNROLL(n) ISLAY_PRAGMA(clang loop unroll_count(n))
#define ISLAY_NOUNROLL ISLAY_PRAGMA(clang loop unroll(disable))
#elif defined(__GNUC__)
#define ISLAY_UNROLL(n) ISLAY_PRAGMA(GCC unroll n)
#define ISLAY_NOUNROLL ISLAY_PRAGMA(GCC unroll 1)
#else
#define ISLAY_UNROLL(n)
#define ISLAY_NOUNROLL
#endif

// Never mark code that writes TT, counters, killers, or history as pure.
#if defined(__GNUC__) || defined(__clang__)
#define ISLAY_PURE __attribute__((pure))
#else
#define ISLAY_PURE
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ISLAY_PREFETCH(addr) __builtin_prefetch(addr)
#elif defined(_MSC_VER)
#include <intrin.h>
#define ISLAY_PREFETCH(addr) _mm_prefetch(reinterpret_cast<const char *>(addr), _MM_HINT_T0)
#else
#define ISLAY_PREFETCH(addr) ((void) 0)
#endif

// False assumptions are UB.
#if defined(__clang__)
#define ISLAY_ASSUME(expr) __builtin_assume(expr)
#elif defined(__GNUC__)
#define ISLAY_ASSUME(expr)                                                                                             \
  do {                                                                                                                 \
    if (!(expr))                                                                                                       \
      __builtin_unreachable();                                                                                         \
  } while (false)
#elif defined(_MSC_VER)
#define ISLAY_ASSUME(expr) __assume(expr)
#else
#define ISLAY_ASSUME(expr) ((void) 0)
#endif

#endif // ISLAY_COMMON_HPP
