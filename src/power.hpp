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

#ifndef PERM16_POWER_HPP_INCLUDED
#define PERM16_POWER_HPP_INCLUDED

namespace HPCombi {

namespace power_helper {

  /** Forward declaration */
  // @brief Monoid structure used by default for type T by the pow function
  template <typename T> struct Monoid;

};

/** A generic compile time squaring function
 *
 *  @param x      the number to square
 *  @return       x squared
 *
 *  To use for a specific type the user should pass a Monoid structure (see
 *  below) as second parameter to the template. Alternatively a default monoid
 *  structure can be defined for a given type by overloading the struct
 *  power_helper::Monoid<T>
 *
 *  A Monoid structure is required to define two static members
 *  - one : the unit of the monoid
 *  - T prod(T, T) : the product of two elements in the monoid
 *
 *  Note: unfortunately boost::math::power is not enought general to handle
 *  monoids whose unit is not constructed as T(1).
 */
template<typename T, typename M = power_helper::Monoid<T> >
constexpr T square(const T x) { return M::prod(x, x); }

/** @brief A generic compile time exponentiation function
 *
 *  @tparam exp    the power
 *  @arg x      the number to exponentiate
 *  @return       x to the power exp
 *
 *  @details Raise x to the exponent exp where exp is known at compile
 *  time. We use the classica binary algorithm, but the recursion is unfolded
 *  and optimized at compile time giving an assembly code which is just a
 *  sequence of multiplication.
 *
 *  To use for a specific type the user should pass a Monoid structure (see
 *  below) as third parameter to the template. Alternatively a default monoid
 *  structure can be defined for a given type by overloading the struct
 *  power_helper::Monoid<T>
 *
 *  A Monoid structure is required to define two static members
 *  - one : the unit of the monoid
 *  - T prod(T, T) : the product of two elements in the monoid
 *
 *  Note: unfortunately boost::math::power is not enought general to handle
 *  monoids whose unit is not constructed as T(1).
 */
template<unsigned exp, typename T, typename M = power_helper::Monoid<T> >
constexpr T pow(const T x) {
  return
    (exp == 0) ? M::one :
    (exp % 2 == 0) ? square<T, M>(pow<unsigned(exp/2), T, M>(x)) :
    M::prod(x, square<T, M>(pow<unsigned(exp/2), T, M>(x)));
}



namespace power_helper {

/** Default class for numeric multiplicative monoids
 */
template <typename T> struct Monoid {
  /** @brief the one of the type T */
  static constexpr T one = 1;
  /** @brief the product of two element in the type T */
  static constexpr T prod(T a, T b) { return a * b; }
};

};  // namespace power_helper

};  // namespace HPCombi

#endif  // PERM16_POWER_HPP_INCLUDED
