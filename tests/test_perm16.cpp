/******************************************************************************/
/*       Copyright (C) 2017 Florent Hivert <Florent.Hivert@lri.fr>,           */
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

#define BOOST_TEST_MODULE Perm16Tests

#include "perm16.hpp"
#include <boost/test/unit_test.hpp>

using HPCombi::Perm16;
using HPCombi::Vect16;

BOOST_AUTO_TEST_CASE(Vect16TestEq) { BOOST_CHECK_EQUAL(Vect16(), Vect16()); }

BOOST_AUTO_TEST_CASE(Perm16TestEq) {
  BOOST_CHECK_EQUAL(Perm16::one() * Perm16::one(), Perm16::one());
}
