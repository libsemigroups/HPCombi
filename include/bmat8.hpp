//
// libsemigroups - C++ library for semigroups and monoids
// Copyright (C) 2017 Finn Smith
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

// This file contains a declaration of fast boolean matrices up to dimension 8.

#ifndef HPCOMBI_BMAT8_HPP_INCLUDED
#define HPCOMBI_BMAT8_HPP_INCLUDED

#include <algorithm>  // for uniform_int_distribution, swap
#include <climits>    // for CHAR_BIT
#include <cstddef>    // for size_t
#include <cstdint>    // for uint64_t
#include <iostream>   // for operator<<, ostringstream
#include <random>     // for mt19937, random_device
#include <utility>    // for hash
#include <vector>      // for array
#include <array>      // for array

#ifndef HPCOMBI_ASSERT
#define HPCOMBI_ASSERT(x) assert(x)
#endif

namespace HPCombi {

//! Class for fast boolean matrices of dimension up to 8 x 8
//!
//! The methods for these small matrices over the boolean semiring
//! are more optimised than the generic methods for boolean matrices.
//! Note that all BMat8 are represented internally as an 8 x 8 matrix;
//! any entries not defined by the user are taken to be 0. This does
//! not affect the results of any calculations.
//!
//! BMat8 is a trivial class.
class BMat8 {
  public:
    //! A default constructor.
    //!
    //! This constructor gives no guarantees on what the matrix will contain.
    BMat8() = default;

    //! A constructor.
    //!
    //! This constructor initializes a BMat8 to have rows equal to the
    //! 8 chunks, of 8 bits each, of the binary representation of \p mat.
    inline explicit BMat8(uint64_t mat) : _data(mat) {}

    //! A constructor.
    //!
    //! This constructor initializes a matrix where the rows of the matrix
    //! are the vectors in \p mat.
    inline explicit BMat8(std::vector<std::vector<bool>> const &mat);

    //! A constructor.
    //!
    //! This is the copy constructor.
    BMat8(BMat8 const &) = default;

    //! A constructor.
    //!
    //! This is the move constructor.
    BMat8(BMat8 &&) = default;

    //! A constructor.
    //!
    //! This is the copy assignement constructor.
    BMat8 &operator=(BMat8 const &) = default;

    //! A constructor.
    //!
    //! This is the move assignment  constructor.
    BMat8 &operator=(BMat8 &&) = default;

    //! A default destructor.
    ~BMat8() = default;

    //! Returns \c true if \c this equals \p that.
    //!
    //! This method checks the mathematical equality of two BMat8 objects.
    bool operator==(BMat8 const &that) const { return _data == that._data; }

    //! Returns \c true if \c this does not equal \p that
    //!
    //! This method checks the mathematical inequality of two BMat8 objects.
    bool operator!=(BMat8 const &that) const { return _data != that._data; }

    //! Returns \c true if \c this is less than \p that.
    //!
    //! This method checks whether a BMat8 objects is less than another.
    //! We order by the results of to_int() for each matrix.
    bool operator<(BMat8 const &that) const { return _data < that._data; }

    //! Returns \c true if \c this is greater than \p that.
    //!
    //! This method checks whether a BMat8 objects is greater than another.
    //! We order by the results of to_int() for each matrix.
    bool operator>(BMat8 const &that) const { return _data > that._data; }

    //! Returns the entry in the (\p i, \p j)th position.
    //!
    //! This method returns the entry in the (\p i, \p j)th position.
    //! Note that since all matrices are internally represented as 8 x 8, it
    //! is possible to access entries that you might not believe exist.
    inline bool operator()(size_t i, size_t j) const;

    //! Sets the (\p i, \p j)th position to \p val.
    //!
    //! This method sets the (\p i, \p j)th entry of \c this to \p val.
    //! Uses the bit twiddle for setting bits found
    //! <a href=http://graphics.stanford.edu/~seander/bithacks>here</a>.
    inline void set(size_t i, size_t j, bool val);

    //! Returns the integer representation of \c this.
    //!
    //! Returns an unsigned integer obtained by interpreting an 8 x 8
    //! BMat8 as a sequence of 64 bits (reading rows left to right,
    //! from top to bottom) and then this sequence as an unsigned int.
    inline uint64_t to_int() const { return _data; }

    //! Returns the transpose of \c this
    //!
    //! Returns the standard matrix transpose of a BMat8.
    //! Uses the technique found in Knuth AoCP Vol. 4 Fasc. 1a, p. 15.
    inline BMat8 transpose() const;

    //! Returns the matrix product of \c this and \p that
    //!
    //! This method returns the standard matrix product (over the
    //! boolean semiring) of two BMat8 objects.
    inline BMat8 operator*(BMat8 const &that) const;

    inline BMat8 row_space_basis() const;
    inline BMat8 col_space_basis() const {
        return transpose().row_space_basis().transpose();
    }
    std::vector<uint8_t> rows() const;
    inline uint64_t row_space_size() const;
    inline uint64_t row_space_size_ref() const;

    //! Returns the identity BMat8
    //!
    //! This method returns the 8 x 8 BMat8 with 1s on the main diagonal.
    static inline BMat8 one() { return BMat8(0x8040201008040201); }

    //! Insertion operator
    //!
    //! This method allows BMat8 objects to be inserted into a ostream.
    inline std::ostream &operator<<(std::ostream &os);

    //! Returns a random BMat8
    //!
    //! This method returns a BMat8 chosen at random.
    inline static BMat8 random();

    //! Returns a random square BMat8 up to dimension \p dim.
    //!
    //! This method returns a BMat8 chosen at random, where only the
    //! top-left \p dim x \p dim entries may be non-zero.
    inline static BMat8 random(size_t dim);

    inline void swap(BMat8 &that) { std::swap(this->_data, that._data); }

    std::ostream & write(std::ostream &os) const;

#ifdef LIBSEMIGROUPS_DENSEHASHMAP
    // FIXME do this another way
    BMat8 empty_key() const { return BMat8(0xFF7FBFDFEFF7FBFE); }
#endif

  private:
    uint64_t _data;

    epu8 row_space_basis_internal() const;
};

}  // namespace HPCombi

#include "bmat8_impl.hpp"

namespace std {
template <> struct hash<HPCombi::BMat8> {
    size_t operator()(HPCombi::BMat8 const &bm) const {
        return hash<uint64_t>()(bm.to_int());
    }
};
}  // namespace std
#endif  // HPCOMBI_BMAT8_HPP_INCLUDED
