//****************************************************************************//
//       Copyright (C) 2016 Florent Hivert <Florent.Hivert@lri.fr>,           //
//                                                                            //
//  Distributed under the terms of the GNU General Public License (GPL)       //
//                                                                            //
//    This code is distributed in the hope that it will be useful,            //
//    but WITHOUT ANY WARRANTY; without even the implied warranty of          //
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU       //
//   General Public License for more details.                                 //
//                                                                            //
//  The full text of the GPL is available at:                                 //
//                                                                            //
//                  http://www.gnu.org/licenses/                              //
//****************************************************************************//

#ifndef HPCOMBI_PERM_GENERIC_HPP
#define HPCOMBI_PERM_GENERIC_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <ostream>

namespace HPCombi {

template <size_t _Size, typename Expo = uint8_t> struct VectGeneric {

  static constexpr size_t Size() { return _Size; }
  std::array<Expo, _Size> v;

  VectGeneric() = default;
  VectGeneric(std::initializer_list<Expo> il, Expo def = 0);

  Expo operator[](uint64_t i) const { return v[i]; }
  Expo &operator[](uint64_t i) { return v[i]; }

  inline uint64_t first_diff(const VectGeneric &u, size_t bound = _Size) const {
    for (uint64_t i = 0; i < bound; i++)
      if (v[i] != u[i])
        return i;
    return _Size;
  }

  bool operator==(const VectGeneric &u) const { return first_diff(u) == _Size; }
  bool operator!=(const VectGeneric &u) const { return first_diff(u) != _Size; }

  bool operator<(const VectGeneric &u) const {
    uint64_t diff = first_diff(u);
    return (diff != _Size) and v[diff] < u[diff];
  }

  char less_partial(const VectGeneric &u, int k) const {
    uint64_t diff = first_diff(u, k);
    return (diff == _Size) ? 0
      : static_cast<char>(v[diff]) - static_cast<char>(u[diff]);
  }

  VectGeneric permuted(const VectGeneric &u) const {
    VectGeneric res;
    for (uint64_t i = 0; i < _Size; i++)
      res[i] = v[u[i]];
    return res;
  }

  uint64_t first_non_zero(size_t bound = _Size) const {
    for (uint64_t i = 0; i < bound; i++)
      if (v[i] != 0)
        return i;
    return _Size;
  }
  uint64_t first_zero(size_t bound = _Size) const {
    for (uint64_t i = 0; i < bound; i++)
      if (v[i] == 0)
        return i;
    return _Size;
  }
  uint64_t last_non_zero(size_t bound = 16) const {
    for (int64_t i = bound - 1; i >= 0; i--)
      if (v[i] != 0)
        return i;
    return _Size;
  }
  uint64_t last_zero(size_t bound = 16) const {
    for (int64_t i = bound - 1; i >= 0; i--)
      if (v[i] == 0)
        return i;
    return _Size;
  }

  bool is_permutation(const size_t k = _Size) const {
    auto temp = v;
    std::sort(temp.begin(), temp.end());
    for (uint64_t i = 0; i < _Size; i++)
      if (temp[i] != i)
        return false;
    for (uint64_t i = k; i < _Size; i++)
      if (v[i] != i)
        return false;
    return true;
  }
};

template <size_t _Size, typename Expo>
std::ostream &operator<<(std::ostream &stream,
                         const VectGeneric<_Size, Expo> &term) {
  stream << "[" << std::setw(2) << unsigned(term[0]);
  for (unsigned i = 1; i < _Size; ++i)
    stream << "," << std::setw(2) << unsigned(term[i]);
  stream << "]";
  return stream;
}

template <size_t _Size, typename Expo>
VectGeneric<_Size, Expo>::VectGeneric(std::initializer_list<Expo> il, Expo def) {
  assert(il.size() <= _Size);
  std::copy(il.begin(), il.end(), v.begin());
  for (uint64_t i = il.size(); i < _Size; ++i)
    v[i] = def;
}

template <size_t _Size, typename Expo = uint8_t>
struct PermGeneric : public VectGeneric<_Size, Expo> {

  using vect = VectGeneric<_Size, Expo>;

  PermGeneric() = default;
  // PermGeneric() { for (uint64_t i=0; i < _Size; i++) this->v[i] = i; }
  PermGeneric(const vect u) : vect(u) {}
  PermGeneric(std::initializer_list<Expo> il) {
    assert(il.size() <= _Size);
    std::copy(il.begin(), il.end(), this->v.begin());
    for (uint64_t i = il.size(); i < _Size; i++)
      this->v[i] = i;
  }

  PermGeneric operator*(const PermGeneric &p) const {
    return this->permuted(p);
  }
  static PermGeneric one() { return PermGeneric({}); }
  static PermGeneric elementary_transposition(uint64_t i) {
    assert(i < _Size);
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

template <size_t _Size, typename Expo>
struct hash<HPCombi::VectGeneric<_Size, Expo>> {
  size_t operator()(const HPCombi::VectGeneric<_Size, Expo> &ar) const {
    size_t h = 0;
    for (size_t i = 0; i < HPCombi::VectGeneric<_Size, Expo>::_Size; i++)
      h = hash<Expo>()(ar[i]) + (h << 6) + (h << 16) - h;
    return h;
  }
};

}  // namespace std

#endif  // HPCOMBI_PERM_GENERIC_HPP
