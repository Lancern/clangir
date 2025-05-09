//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// template <class Iter>
//   iterator insert(const_iterator position, Iter first, Iter last);

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <vector>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "test_allocator.h"
#include "test_iterators.h"
#include "min_allocator.h"
#include "asan_testing.h"

namespace adl {
struct S {};
void make_move_iterator(S*) {}
} // namespace adl

TEST_CONSTEXPR_CXX20 bool tests() {
  //
  // Tests for input_iterator
  // Vector may or may not reallocate during insertion -- test both cases.
  //
  { // Test insertion into a vector with less spare space available than the input range, triggering reallocation
    typedef std::vector<int> V;
    V v(100);
    int a[]     = {1, 2, 3, 4, 5};
    const int N = sizeof(a) / sizeof(a[0]);
    V::iterator i =
        v.insert(v.cbegin() + 10, cpp17_input_iterator<const int*>(a), cpp17_input_iterator<const int*>(a + N));
    assert(v.size() == 100 + N);
    assert(is_contiguous_container_asan_correct(v));
    assert(i == v.begin() + 10);
    int j;
    for (j = 0; j < 10; ++j)
      assert(v[j] == 0);
    for (std::size_t k = 0; k < N; ++j, ++k)
      assert(v[j] == a[k]);
    for (; j < 105; ++j)
      assert(v[j] == 0);
  }
  { // Test insertion into a vector with sufficient spare space where no reallocation happens
    typedef std::vector<int> V;
    V v(100);
    v.reserve(v.size() + 10);
    int a[]     = {1, 2, 3, 4, 5};
    const int N = sizeof(a) / sizeof(a[0]);
    V::iterator i =
        v.insert(v.cbegin() + 10, cpp17_input_iterator<const int*>(a), cpp17_input_iterator<const int*>(a + N));
    assert(v.size() == 100 + N);
    assert(is_contiguous_container_asan_correct(v));
    assert(i == v.begin() + 10);
    int j;
    for (j = 0; j < 10; ++j)
      assert(v[j] == 0);
    for (std::size_t k = 0; k < N; ++j, ++k)
      assert(v[j] == a[k]);
    for (; j < 105; ++j)
      assert(v[j] == 0);
  }

  //
  // Tests for forward_iterator
  //
  {
    typedef std::vector<int> V;
    V v(100);
    int a[]       = {1, 2, 3, 4, 5};
    const int N   = sizeof(a) / sizeof(a[0]);
    V::iterator i = v.insert(v.cbegin() + 10, forward_iterator<const int*>(a), forward_iterator<const int*>(a + N));
    assert(v.size() == 100 + N);
    assert(is_contiguous_container_asan_correct(v));
    assert(i == v.begin() + 10);
    int j;
    for (j = 0; j < 10; ++j)
      assert(v[j] == 0);
    for (std::size_t k = 0; k < N; ++j, ++k)
      assert(v[j] == a[k]);
    for (; j < 105; ++j)
      assert(v[j] == 0);
  }
  {
    typedef std::vector<int> V;
    V v(100);
    while (v.size() < v.capacity())
      v.push_back(0); // force reallocation
    std::size_t sz   = v.size();
    int a[]          = {1, 2, 3, 4, 5};
    const unsigned N = sizeof(a) / sizeof(a[0]);
    V::iterator i    = v.insert(v.cbegin() + 10, forward_iterator<const int*>(a), forward_iterator<const int*>(a + N));
    assert(v.size() == sz + N);
    assert(i == v.begin() + 10);
    std::size_t j;
    for (j = 0; j < 10; ++j)
      assert(v[j] == 0);
    for (std::size_t k = 0; k < N; ++j, ++k)
      assert(v[j] == a[k]);
    for (; j < v.size(); ++j)
      assert(v[j] == 0);
  }
  {
    typedef std::vector<int> V;
    V v(100);
    v.reserve(128); // force no reallocation
    std::size_t sz   = v.size();
    int a[]          = {1, 2, 3, 4, 5};
    const unsigned N = sizeof(a) / sizeof(a[0]);
    V::iterator i    = v.insert(v.cbegin() + 10, forward_iterator<const int*>(a), forward_iterator<const int*>(a + N));
    assert(v.size() == sz + N);
    assert(i == v.begin() + 10);
    std::size_t j;
    for (j = 0; j < 10; ++j)
      assert(v[j] == 0);
    for (std::size_t k = 0; k < N; ++j, ++k)
      assert(v[j] == a[k]);
    for (; j < v.size(); ++j)
      assert(v[j] == 0);
  }
  {
    typedef std::vector<int, limited_allocator<int, 308> > V;
    V v(100);
    int a[]     = {1, 2, 3, 4, 5};
    const int N = sizeof(a) / sizeof(a[0]);
    V::iterator i =
        v.insert(v.cbegin() + 10, cpp17_input_iterator<const int*>(a), cpp17_input_iterator<const int*>(a + N));
    assert(v.size() == 100 + N);
    assert(is_contiguous_container_asan_correct(v));
    assert(i == v.begin() + 10);
    int j;
    for (j = 0; j < 10; ++j)
      assert(v[j] == 0);
    for (std::size_t k = 0; k < N; ++j, ++k)
      assert(v[j] == a[k]);
    for (; j < 105; ++j)
      assert(v[j] == 0);
  }
  {
    typedef std::vector<int, limited_allocator<int, 300> > V;
    V v(100);
    int a[]       = {1, 2, 3, 4, 5};
    const int N   = sizeof(a) / sizeof(a[0]);
    V::iterator i = v.insert(v.cbegin() + 10, forward_iterator<const int*>(a), forward_iterator<const int*>(a + N));
    assert(v.size() == 100 + N);
    assert(is_contiguous_container_asan_correct(v));
    assert(i == v.begin() + 10);
    int j;
    for (j = 0; j < 10; ++j)
      assert(v[j] == 0);
    for (std::size_t k = 0; k < N; ++j, ++k)
      assert(v[j] == a[k]);
    for (; j < 105; ++j)
      assert(v[j] == 0);
  }
  { // Ensure that iterator-pair insert() doesn't use unexpected assignment.
    struct Wrapper {
      TEST_CONSTEXPR Wrapper(int n) : n_(n) {}

      int n_;

    private:
      void operator=(int);
    };

    int a[]                 = {1, 2, 3, 4, 5};
    const std::size_t count = sizeof(a) / sizeof(a[0]);
    std::vector<Wrapper> v;
    v.insert(v.end(), a, a + count);
    assert(v.size() == count);
    for (std::size_t i = 0; i != count; ++i)
      assert(v[i].n_ == a[i]);
  }
#if TEST_STD_VER >= 11
  {
    typedef std::vector<int, min_allocator<int> > V;
    V v(100);
    int a[]     = {1, 2, 3, 4, 5};
    const int N = sizeof(a) / sizeof(a[0]);
    V::iterator i =
        v.insert(v.cbegin() + 10, cpp17_input_iterator<const int*>(a), cpp17_input_iterator<const int*>(a + N));
    assert(v.size() == 100 + N);
    assert(is_contiguous_container_asan_correct(v));
    assert(i == v.begin() + 10);
    int j;
    for (j = 0; j < 10; ++j)
      assert(v[j] == 0);
    for (std::size_t k = 0; k < N; ++j, ++k)
      assert(v[j] == a[k]);
    for (; j < 105; ++j)
      assert(v[j] == 0);
  }
  {
    typedef std::vector<int, min_allocator<int> > V;
    V v(100);
    int a[]       = {1, 2, 3, 4, 5};
    const int N   = sizeof(a) / sizeof(a[0]);
    V::iterator i = v.insert(v.cbegin() + 10, forward_iterator<const int*>(a), forward_iterator<const int*>(a + N));
    assert(v.size() == 100 + N);
    assert(is_contiguous_container_asan_correct(v));
    assert(i == v.begin() + 10);
    int j;
    for (j = 0; j < 10; ++j)
      assert(v[j] == 0);
    for (std::size_t k = 0; k < N; ++j, ++k)
      assert(v[j] == a[k]);
    for (; j < 105; ++j)
      assert(v[j] == 0);
  }
#endif

  {
    std::vector<adl::S> s;
    s.insert(s.end(), cpp17_input_iterator<adl::S*>(nullptr), cpp17_input_iterator<adl::S*>(nullptr));
  }

  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER > 17
  static_assert(tests());
#endif
  return 0;
}
