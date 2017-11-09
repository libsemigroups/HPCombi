#define BOOST_TEST_MODULE TotoTests

#include "perm16.hpp"
#include <boost/test/unit_test.hpp>

using namespace IVMPG;

BOOST_AUTO_TEST_CASE(TestFonct)
{
  BOOST_CHECK_EQUAL(Perm16::one() * Perm16::one(), Perm16::one());
}
