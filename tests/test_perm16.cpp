#define BOOST_TEST_MODULE Perm16Tests

#include "perm16.hpp"
#include <boost/test/unit_test.hpp>

using HPCombi::Vect16;
using HPCombi::Perm16;

BOOST_AUTO_TEST_CASE(Vect16TestEq)
{
  BOOST_CHECK_EQUAL(Vect16(), Vect16());
}

BOOST_AUTO_TEST_CASE(Perm16TestEq)
{
  BOOST_CHECK_EQUAL(Perm16::one() * Perm16::one(), Perm16::one());
}
