/******************************************************************************/
/*       Copyright (C) 2014 Florent Hivert <Florent.Hivert@lri.fr>,           */
/*                                                                            */
/*  Distributed under the terms of the GNU General Public License (GPL)       */
/*                                                                            */
/*    This code is distributed in the hope that it will be useful,            */
/*    but WITHOUT ANY WARRANTY; without even the implied warranty of          */
/*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU       */
/*   General Public License for more details.                                 */
/*                                                                            */
/*  The full text of the GPL is available at:                                 */
/*                                                                            */
/*                  http://www.gnu.org/licenses/                              */
/******************************************************************************/

#ifndef HPCOMBI_PERM_GENERIC_HPP
#define HPCOMBI_PERM_GENERIC_HPP

#include "fonctions_gpu.cuh"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <stdlib.h> 
#include <ostream>
#include <iomanip>


namespace HPCombi {

template <size_t _Size, typename Expo = uint8_t> struct VectGeneric {

  static const constexpr size_t Size = _Size;
  std::array<Expo, Size> v;

  VectGeneric() = default;
  VectGeneric(std::initializer_list<Expo> il, Expo def = 0);
  VectGeneric(size_t plus, size_t mod);
  VectGeneric(Expo target);

  Expo operator[](uint64_t i) const { return v[i]; }
  Expo &operator[](uint64_t i) { return v[i]; }

  inline uint64_t first_diff(const VectGeneric &u, size_t bound = Size) const {
    for (uint64_t i = 0; i < bound; i++)
      if (v[i] != u[i])
        return i;
    return Size;
  }

  bool operator==(const VectGeneric &u) const { return first_diff(u) == Size; }
  bool operator!=(const VectGeneric &u) const { return first_diff(u) != Size; }

  bool operator<(const VectGeneric &u) const {
    uint64_t diff = first_diff(u);
    return (diff != Size) and v[diff] < u[diff];
  }

  char less_partial(const VectGeneric &u, int k) const {
    uint64_t diff = first_diff(u, k);
    return (diff == Size) ? 0 : char(v[diff]) - char(u[diff]);
  }

  VectGeneric permuted(const VectGeneric &u) const {
    VectGeneric res;
    for (uint64_t i = 0; i < Size; i++)
      res[i] = v[u[i]];
    return res;
  };
  
#ifdef USE_CUDA  
	VectGeneric permuted_gpu(const VectGeneric &u) const {

	  // Simple pointers are needed to cpy to GPU
	  const uint16_t* x = &v[0];
	  const uint16_t* y = &u.v[0];
	  VectGeneric res;
	  uint16_t* z = &res.v[0];
	  shufl_gpu<uint16_t>(x, y, z, Size);
	  return res;
	}
#endif  // USE_CUDA



  uint64_t first_non_zero(size_t bound = Size) const {
    for (uint64_t i = 0; i < bound; i++)
      if (v[i] != 0)
        return i;
    return Size;
  }
  uint64_t first_zero(size_t bound = Size) const {
    for (uint64_t i = 0; i < bound; i++)
      if (v[i] == 0)
        return i;
    return Size;
  }
  uint64_t last_non_zero(size_t bound = 16) const {
    for (int64_t i = bound - 1; i >= 0; i--)
      if (v[i] != 0)
        return i;
    return Size;
  }
  uint64_t last_zero(size_t bound = 16) const {
    for (int64_t i = bound - 1; i >= 0; i--)
      if (v[i] == 0)
        return i;
    return Size;
  }

  bool is_permutation(const size_t k = Size) const {
    auto temp = v;
    std::sort(temp.begin(), temp.end());
    for (uint64_t i = 0; i < Size; i++)
      if (temp[i] != i)
        return false;
    for (uint64_t i = k; i < Size; i++)
      if (v[i] != i)
        return false;
    return true;
  }
};

template <size_t Size, typename Expo>
std::ostream &operator<<(std::ostream &stream,
                         const VectGeneric<Size, Expo> &term) {
  stream << "[" << std::setw(2) << unsigned(term[0]);
  for (unsigned i = 1; i < Size; ++i)
    stream << "," << std::setw(2) << unsigned(term[i]);
  stream << "]";
  return stream;
}

template <size_t Size, typename Expo>
VectGeneric<Size, Expo>::VectGeneric(std::initializer_list<Expo> il, Expo def) {
  assert(il.size() <= Size);
  std::copy(il.begin(), il.end(), v.begin());
  for (uint64_t i = il.size(); i < Size; ++i)
    v[i] = def;
}

template <size_t Size, typename Expo>
VectGeneric<Size, Expo>::VectGeneric(size_t plus, size_t mod) {
  mod = (mod<=0) ? Size:mod;
  for (uint64_t i = 0; i < Size; ++i)
    v[i] = (i+plus)%mod;
  if(mod == -1){
	  std::random_shuffle(v.begin(), v.end());
  }
  else if(mod == -2){
	  for (uint64_t i = 0; i < Size; ++i)
	    v[i] = rand()%Size;
  }
}

template <size_t Size, typename Expo>
VectGeneric<Size, Expo>::VectGeneric(Expo target) {
  if(target >= Size || target < 0)
	target = 0;
  for (uint64_t i = 0; i < Size; ++i)
    v[i] = target;
}


// Definition since previously *only* declared
template <size_t _Size, typename Expo>
const constexpr size_t VectGeneric<_Size, Expo>::Size;

template <size_t _Size, typename Expo = uint8_t>
struct PermGeneric : public VectGeneric<_Size, Expo> {

  using vect = VectGeneric<_Size, Expo>;

  PermGeneric() = default;
  // PermGeneric() { for (uint64_t i=0; i < _Size; i++) this->v[i] = i; };
  PermGeneric(const vect u) : vect(u){};
  PermGeneric(std::initializer_list<Expo> il) {
    assert(il.size() <= vect::Size);
    std::copy(il.begin(), il.end(), this->v.begin());
    for (uint64_t i = il.size(); i < vect::Size; i++)
      this->v[i] = i;
  }

  PermGeneric operator*(const PermGeneric &p) const {
    return this->permuted(p);
  }
  static PermGeneric one() { return PermGeneric({}); }
  static PermGeneric elementary_transposition(uint64_t i) {
    assert(i < vect::Size);
    PermGeneric res{{}};
    res[i] = i + 1;
    res[i + 1] = i;
    return res;
  }
};

/*****************************************************************************/
/** Memory layout concepts check  ********************************************/
/*****************************************************************************/

static_assert(sizeof(VectGeneric<12>) == sizeof(PermGeneric<12>),
              "VectGeneric and PermGeneric have a different memory layout !");
static_assert(std::is_trivial<VectGeneric<12>>(),
              "VectGeneric is not a a trivial class !");
static_assert(std::is_trivial<PermGeneric<12>>(),
              "PermGeneric is not trivial !");

}  //  namespace HPCombi

namespace std {

template <size_t Size, typename Expo>
struct hash<HPCombi::VectGeneric<Size, Expo>> {
  size_t operator()(const HPCombi::VectGeneric<Size, Expo> &ar) const {
    size_t h = 0;
    for (size_t i = 0; i < HPCombi::VectGeneric<Size, Expo>::Size; i++)
      h = hash<Expo>()(ar[i]) + (h << 6) + (h << 16) - h;
    return h;
  }
};

}  // namespace std

#endif  // HPCOMBI_PERM_GENERIC_HPP
