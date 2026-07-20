/**
 * @file common.hpp
 * @brief Portable optimization attributes and hints.
 *
 * Thin wrappers so the hot paths can be annotated once and compile on Clang,
 * GCC and MSVC alike. Standard C++ attributes ([[likely]], [[nodiscard]]) are
 * used directly at call sites; the compiler-specific ones are funnelled through
 * the macros below.
 */
#ifndef ISLAY_COMMON_HPP
#define ISLAY_COMMON_HPP

// --- forced inlining --------------------------------------------------------
#if defined(_MSC_VER)
#define ISLAY_FORCEINLINE __forceinline
#else
#define ISLAY_FORCEINLINE inline __attribute__((always_inline))
#endif

// --- hot / flatten (encourage the optimizer on the recursion core) ----------
#if defined(__GNUC__) || defined(__clang__)
#define ISLAY_HOT __attribute__((hot))
#define ISLAY_FLATTEN __attribute__((flatten))
#else
#define ISLAY_HOT
#define ISLAY_FLATTEN
#endif

// --- loop unrolling ---------------------------------------------------------
// GCC and Clang share no spelling for this: Clang wants `#pragma clang loop
// unroll_count(n)`, GCC 8+ wants `#pragma GCC unroll n`. Neither is an
// attribute, so they must be funnelled through _Pragma. Place these on the
// `for` STATEMENT, never on a function -- there is no `unroll` function
// attribute in either compiler (one written that way is silently ignored).
//
// Use sparingly and only where measured: -O3 already unrolls small
// constant-trip-count loops, so these usually change nothing and can hurt
// (I-cache). Every use in this repo must survive an interleaved A/B.
//
// Currently used NOWHERE, on purpose. The one place tried -- the 7-iteration
// loop in Board::canonical() -- measured WORSE with ISLAY_UNROLL(7) (median
// 3576ms vs 3469ms, cached perft(13), interleaved A/B) and was removed. The
// macros stay because they are correct and portable (verified: clang emits
// `#pragma clang loop unroll_count(n)`, real GCC 16 emits `#pragma GCC unroll n`,
// both warning-clean) -- so a future MEASURED win can just use them.
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

// --- pure (result depends only on args + memory reads; NO side effects) -----
// An UNCHECKED promise: mark something that writes memory (a TT store, a node
// counter) and the compiler may CSE the call away, silently producing wrong
// results. Only ever put this on leaf functions that provably touch nothing.
//
// Currently used NOWHERE, on purpose. Measured on terminal_score/eval: exactly
// zero effect (681537 nodes / ~55ms either way, interleaved A/B at depth 13) --
// they are already inlined and the build uses LTO, so the compiler can see the
// bodies and `pure` tells it nothing it did not know. Removed rather than kept
// as decoration: it buys nothing and it is a footgun if copied onto anything
// that touches the TT, the node counter, killers or history.
#if defined(__GNUC__) || defined(__clang__)
#define ISLAY_PURE __attribute__((pure))
#else
#define ISLAY_PURE
#endif

// --- prefetch (start a load early; pure hint, never changes semantics) ------
#if defined(__GNUC__) || defined(__clang__)
#define ISLAY_PREFETCH(addr) __builtin_prefetch(addr)
#elif defined(_MSC_VER)
#include <intrin.h>
#define ISLAY_PREFETCH(addr) _mm_prefetch(reinterpret_cast<const char *>(addr), _MM_HINT_T0)
#else
#define ISLAY_PREFETCH(addr) ((void) 0)
#endif

// --- assume (nudge value-range reasoning; UB if the predicate is false) -----
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
