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

#ifndef HPCOMBI_VECT_GENERIC_HPP
#define HPCOMBI_VECT_GENERIC_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <ostream>

namespace HPCombi {



template <size_t Size, typename Expo = uint8_t>
std::array<Expo, Size> sorted_vect(std::array<Expo, Size> v) {
    std::sort(v.begin(), v.end());
    return v;
}

template <size_t Size, typename Expo = uint8_t>
uint64_t horiz_sum(const std::array<Expo, Size> &v) {
    Expo res = 0;
    for (uint64_t i = 0; i < Size; i++) res += v[i];
    return res;
}

template <size_t Size, typename Expo = uint8_t>
const std::array<Expo, Size> partial_sums(std::array<Expo, Size> v) {
    for (uint64_t i = 1; i < Size; i++) v[i] += v[i-1];
    return v;
}


/*
  
template <size_t Size, typename Expo = uint8_t>
  bool operator<(const std::array<Expo, Size> &u) const {
    uint64_t diff = first_diff(u);
    return (diff != Size) and v[diff] < u[diff];
  }

  char less_partial(const std::array<Expo, Size> &u, int k) const {
    uint64_t diff = first_diff(u, k);
    return (diff == Size) ? 0
      : static_cast<char>(v[diff]) - static_cast<char>(u[diff]);
  }


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
*/

template <size_t Size, typename Expo>
std::ostream &operator<<(std::ostream &stream,
                         const std::array<Expo, Size> &v) {
  stream << "[" << std::setw(2) << unsigned(v[0]);
  for (unsigned i = 1; i < Size; ++i)
    stream << "," << std::setw(2) << unsigned(v[i]);
  stream << "]";
  return stream;
}


template <size_t _Size, typename Expo = uint8_t>
struct VectGeneric {

    static const constexpr size_t Size = _Size;
    std::array<Expo, Size> v;

    VectGeneric() = default;
    VectGeneric(std::array<Expo, Size> _v) : v(_v) {};
    VectGeneric(std::initializer_list<Expo> il, Expo def = 0) {
        assert(il.size() <= Size);
        std::copy(il.begin(), il.end(), v.begin());
        std::fill(v.begin() + il.size(), v.end(), def);
    }

    Expo operator[](uint64_t i) const { return v[i]; }
    Expo &operator[](uint64_t i) { return v[i]; }

    uint64_t first_diff(const VectGeneric &u, size_t bound = Size) const {
        for (uint64_t i = 0; i < bound; i++)
            if (v[i] != u[i]) return i;
        return Size;
    }

    uint64_t last_diff(const VectGeneric &u, size_t bound = Size) const {
        while (bound != 0) {
            --bound;
            if (u[bound] != v[bound]) return bound;
        }
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

    void sort() { std::sort(v.begin(), v.end()); }

    bool is_sorted() const {
        for (uint64_t i = 1; i < Size; i++)
            if (v[i-1] < v[i]) return false;
        return true;
    }

    static VectGeneric random() {
        VectGeneric<Size, Expo> res = VectGeneric<Size, Expo>(0, 0);
        std::random_shuffle(res.begin(), res.end());
        return res;
    }

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

    uint64_t horiz_sum() {
        Expo res = 0;
        for (uint64_t i = 0; i < Size; i++) res += v[i];
        return res;
    }

    VectGeneric partial_sums() const {
        auto res = *this;
        for (uint64_t i = 1; i < Size; i++) res[i] += res[i-1];
        return res;
    }

};

}  // namespace std

#endif  // HPCOMBI_VECR_GENERIC_HPP
