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

#include <cstdint>
#include <array>
#include <cassert>
#include <algorithm>
#include <ostream>


namespace HPCombi {

template < size_t _Size, typename Expo=uint8_t >
struct VectGeneric {

  static const constexpr size_t Size = _Size;
  std::array<Expo, Size> p;

  Expo operator[](uint64_t i) const { return p[i]; }
  Expo &operator[](uint64_t i) { return p[i]; }

  inline uint64_t first_diff(const VectGeneric &b, size_t bound = Size) const {
    for (uint64_t i=0; i<bound; i++) if (p[i] != b[i]) return i;
    return Size;
  }

  bool operator==(const VectGeneric &b) const {
    return first_diff(b) == Size;
  }
  bool operator!=(const VectGeneric &b) const {
    return first_diff(b) != Size;
  }

  bool operator < (const VectGeneric &b) const {
    uint64_t diff = first_diff(b);
    return (diff != Size) and p[diff] < b[diff];
  }

  char less_partial(const VectGeneric &b, int k) const {
    uint64_t diff = first_diff(b, k);
    return (diff == Size) ? 0 : char(p[diff]) - char(b[diff]);
  }

  VectGeneric permuted(const VectGeneric &other) const {
    VectGeneric res;
    for (uint64_t i=0; i<Size; i++) res.p[i] = p[other[i]];
    return res;
  };

  uint64_t first_non_zero(size_t bound=Size) const {
    for (uint64_t i=0; i<bound; i++) if (p[i] != 0) return i;
    return Size;
  }
  uint64_t first_zero(size_t  bound=Size) const {
    for (uint64_t i=0; i<bound; i++) if (p[i] == 0) return i;
    return Size;
  }
  uint64_t last_non_zero(size_t  bound=16) const {
    for (int64_t i=bound-1; i >= 0; i--) if (p[i] != 0) return i;
    return Size;
  }
  uint64_t last_zero(size_t  bound=16) const {
    for (int64_t i=bound-1; i >= 0; i--) if (p[i] == 0) return i;
    return Size;
  }

  bool is_permutation(const size_t k = Size) const {
    auto temp = this->p;
    std::sort(temp.begin(), temp.end());
    for (uint64_t i=0; i<Size; i++) if (temp[i] != i) return false;
    for (uint64_t i=k; i<Size; i++) if (p[i] != i) return false;
    return true;
  }
};


template < size_t Size, typename Expo >
std::ostream & operator<<(std::ostream & stream, const VectGeneric<Size,Expo> &term){
  stream << "[" << std::setw(2) << unsigned(term[0]);
  for (unsigned i=1; i < Size; ++i)
    stream << "," << std::setw(2) << unsigned(term[i]);
  stream << "]";
  return stream;
}


// Definition since previously *only* declared
template < size_t _Size, typename Expo >
const constexpr size_t VectGeneric<_Size,Expo>::Size;

template < size_t _Size, typename Expo=uint8_t >
struct PermGeneric : public VectGeneric<_Size, Expo> {

  using vect = VectGeneric<_Size, Expo>;

  PermGeneric() { for (uint64_t i=0; i < _Size; i++) this->p[i] = i; };
  PermGeneric(const vect v) : vect(v) {};
  PermGeneric(std::initializer_list<Expo> il) {
    assert (il.size() <= vect::Size);
    std::copy(il.begin(), il.end(), this->p.begin());
    for (uint64_t i = il.size(); i < vect::Size; i++) this->p[i] = i;
  }

  PermGeneric operator*(const PermGeneric&p) const { return this->permuted(p); }
  static PermGeneric one() { return {}; }
  static PermGeneric elementary_transposition(uint64_t i) {
    assert (i < vect::Size);
    PermGeneric res {}; res[i]=i+1; res[i+1]=i; return res; }

};

}   //  namespace HPCombi

namespace std {

  template<>
  template < size_t Size, typename Expo >
  struct hash< HPCombi::VectGeneric<Size,Expo> > {
    size_t operator () (const HPCombi::VectGeneric<Size,Expo> &ar) const {
      size_t h = 0;
      for (size_t i=0; i<HPCombi::VectGeneric<Size,Expo>::Size; i++)
	h = hash<Expo>()(ar[i]) + (h << 6) + (h << 16) - h;
      return h;
    }
  };

}   // namespace std


#endif   // HPCOMBI_PERM_GENERIC_HPP
