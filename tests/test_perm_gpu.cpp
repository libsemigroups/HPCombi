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

#define BOOST_TEST_MODULE PermgpuTests

#include "perm16.hpp"
#include "perm_generic.hpp"
#include <boost/test/unit_test.hpp>


using HPCombi::epu8;
using HPCombi::Vect16;
using HPCombi::PTransf16;
using HPCombi::Transf16;
using HPCombi::Perm16;
using HPCombi::VectGeneric;


struct Fix {
  Fix() : zero(Vect16({}, 0)),
          P01(Vect16({0, 1}, 0)),
          P10(Vect16({1, 0}, 0)),
          P11(Vect16({1, 1}, 0)),
          P1(Vect16({}, 1)),
          RandT({3, 1, 0, 14, 15, 13, 5, 10, 2, 11, 6, 12, 7, 4, 8, 9}),
          PPa({1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
          PPb({1, 2, 3, 6, 0, 5, 4, 7, 8, 9, 10, 11, 12, 15, 14, 13}),
          RandPerm(RandT),
          Tlist({zero, P01, P10, P11, P1, RandT, epu8(PPa), epu8(PPb)}),
          
		  //~ id_16(VectGeneric<16, uint8_t>(0, 0)),
		  //~ plus_one_16(VectGeneric<16, uint8_t>(1, 0)),
		  //~ mod_8_16(VectGeneric<16, uint8_t>(0, 16)),
		  //~ prem_16(VectGeneric<16, uint8_t>(7, 13)),
          
          randShuf(VectGeneric<1024, uint16_t>(0, -1)),
          rand(VectGeneric<1024, uint16_t>(0, -2)),
          zeros(VectGeneric<1024, uint16_t>(0)),
          
		  id_64(VectGeneric<64, uint16_t>(0, 0)),
		  plus_one_64(VectGeneric<64, uint16_t>(1, 0)),
		  mod_32_64(VectGeneric<64, uint16_t>(0, 32)),
		  prem_64(VectGeneric<64, uint16_t>(13, 61)),
		  
		  id_1024(VectGeneric<1024, uint16_t>(0, 0)),
		  plus_one_1024(VectGeneric<1024, uint16_t>(1, 0)),
		  mod_32_1024(VectGeneric<1024, uint16_t>(0, 32)),
		  prem_1024(VectGeneric<1024, uint16_t>(13, 131)),
		  
		  id_2048(VectGeneric<2048, uint16_t>(0, 0)),
		  plus_one_2048(VectGeneric<2048, uint16_t>(1, 0)),
		  mod_32_2048(VectGeneric<2048, uint16_t>(0, 32)),
		  prem_2048(VectGeneric<2048, uint16_t>(13, 131)),
		  
		  id_8192(VectGeneric<8192, uint16_t>(0, 0)),
		  plus_one_8192(VectGeneric<8192, uint16_t>(1, 0)),
		  mod_32_8192(VectGeneric<8192, uint16_t>(0, 32)),
		  prem_8192(VectGeneric<8192, uint16_t>(13, 131)),
		  
		  id_32768(VectGeneric<32768, uint16_t>(0, 0)),
		  plus_one_32768(VectGeneric<32768, uint16_t>(1, 0)),
		  mod_32_32768(VectGeneric<32768, uint16_t>(0, 32)),
		  prem_32768(VectGeneric<32768, uint16_t>(13, 131)),
		  
		  id_131072(VectGeneric<131072, uint32_t>(0, 0)),
		  plus_one_131072(VectGeneric<131072, uint32_t>(1, 0)),
		  mod_32_131072(VectGeneric<131072, uint32_t>(0, 32)),
		  prem_131072(VectGeneric<131072, uint32_t>(13, 131)),
		  
		  id_701(VectGeneric<701, uint16_t>(0, 0)),
		  plus_one_701(VectGeneric<701, uint16_t>(1, 0)),
		  mod_32_701(VectGeneric<701, uint16_t>(0, 32)),
		  prem_701(VectGeneric<701, uint16_t>(13, 131))
		  
          {BOOST_TEST_MESSAGE("setup fixture");}
		  
  ~Fix() { BOOST_TEST_MESSAGE("teardown fixture"); }

  const Transf16 zero, P01, P10, P11, P1, RandT;
  const Perm16 PPa, PPb, RandPerm;
  const std::vector<Transf16> Tlist;
  VectGeneric<16, uint8_t> id_16, plus_one_16, mod_8_16, prem_16, zerogen;
  VectGeneric<64, uint16_t> id_64, plus_one_64, mod_32_64, prem_64;
  VectGeneric<1024, uint16_t> id_1024, plus_one_1024, mod_32_1024, prem_1024;
  VectGeneric<2048, uint16_t> id_2048, plus_one_2048, mod_32_2048, prem_2048;
  VectGeneric<8192, uint16_t> id_8192, plus_one_8192, mod_32_8192, prem_8192;
  VectGeneric<32768, uint16_t> id_32768, plus_one_32768, mod_32_32768, prem_32768;
  VectGeneric<131072, uint32_t> id_131072, plus_one_131072, mod_32_131072, prem_131072;
  VectGeneric<701, uint16_t> id_701, plus_one_701, mod_32_701, prem_701;
  VectGeneric<1024, uint16_t> randShuf, rand, zeros;


};
//____________________________________________________________________________//


//____________________________________________________________________________//

BOOST_AUTO_TEST_SUITE(fonctions_gpu_test16)
//____________________________________________________________________________//

BOOST_FIXTURE_TEST_CASE(GPU16, Fix) {
  BOOST_CHECK_EQUAL(Fix::PPa.permuted(Fix::RandT), Fix::PPa.permuted_gpu(Fix::RandT));
  BOOST_CHECK_EQUAL(Fix::PPb.permuted(Fix::RandT), Fix::PPb.permuted_gpu(Fix::RandT));
  BOOST_CHECK_EQUAL(Fix::RandPerm.permuted(Fix::P11), Fix::RandPerm.permuted_gpu(Fix::P11));
  BOOST_CHECK_EQUAL(Fix::P01.permuted(Fix::P10), Fix::P01.permuted_gpu(Fix::P10));
  BOOST_CHECK_EQUAL(Fix::PPa.permuted(Fix::P1), Fix::PPa.permuted_gpu(Fix::P1));

}

BOOST_AUTO_TEST_SUITE_END()
//____________________________________________________________________________//
//____________________________________________________________________________//

//____________________________________________________________________________//

BOOST_AUTO_TEST_SUITE(fonctions_gpu_test_64)
//____________________________________________________________________________//

BOOST_FIXTURE_TEST_CASE(GPU_64, Fix) {
  BOOST_CHECK_EQUAL(Fix::id_64.permuted(Fix::prem_64), Fix::id_64.permuted_gpu(Fix::prem_64));
  BOOST_CHECK_EQUAL(Fix::plus_one_64.permuted(Fix::prem_64), Fix::plus_one_64.permuted_gpu(Fix::prem_64));
  BOOST_CHECK_EQUAL(Fix::plus_one_64.permuted(Fix::mod_32_64), Fix::plus_one_64.permuted_gpu(Fix::mod_32_64));
  BOOST_CHECK_EQUAL(Fix::prem_64.permuted(Fix::mod_32_64), Fix::prem_64.permuted_gpu(Fix::mod_32_64));

}

BOOST_AUTO_TEST_SUITE_END()
//____________________________________________________________________________//
//____________________________________________________________________________//

//____________________________________________________________________________//

BOOST_AUTO_TEST_SUITE(fonctions_gpu_test_1024)
//____________________________________________________________________________//

BOOST_FIXTURE_TEST_CASE(GPU_1024, Fix) {
  BOOST_CHECK_EQUAL(Fix::id_1024.permuted(Fix::prem_1024), Fix::id_1024.permuted_gpu(Fix::prem_1024));
  BOOST_CHECK_EQUAL(Fix::plus_one_1024.permuted(Fix::prem_1024), Fix::plus_one_1024.permuted_gpu(Fix::prem_1024));
  BOOST_CHECK_EQUAL(Fix::plus_one_1024.permuted(Fix::mod_32_1024), Fix::plus_one_1024.permuted_gpu(Fix::mod_32_1024));
  BOOST_CHECK_EQUAL(Fix::prem_1024.permuted(Fix::mod_32_1024), Fix::prem_1024.permuted_gpu(Fix::mod_32_1024));

}

BOOST_AUTO_TEST_SUITE_END()
//____________________________________________________________________________//
//____________________________________________________________________________//

BOOST_AUTO_TEST_SUITE(fonctions_gpu_test_2048)
//____________________________________________________________________________//

BOOST_FIXTURE_TEST_CASE(GPU_2048, Fix) {
  BOOST_CHECK_EQUAL(Fix::id_2048.permuted(Fix::prem_2048), Fix::id_2048.permuted_gpu(Fix::prem_2048));
  BOOST_CHECK_EQUAL(Fix::plus_one_2048.permuted(Fix::prem_2048), Fix::plus_one_2048.permuted_gpu(Fix::prem_2048));
  BOOST_CHECK_EQUAL(Fix::plus_one_2048.permuted(Fix::mod_32_2048), Fix::plus_one_2048.permuted_gpu(Fix::mod_32_2048));
  BOOST_CHECK_EQUAL(Fix::prem_2048.permuted(Fix::mod_32_2048), Fix::prem_2048.permuted_gpu(Fix::mod_32_2048));

}

BOOST_AUTO_TEST_SUITE_END()
//____________________________________________________________________________//
//____________________________________________________________________________//

BOOST_AUTO_TEST_SUITE(fonctions_gpu_test_8192)
//____________________________________________________________________________//

BOOST_FIXTURE_TEST_CASE(GPU_8192, Fix) {
  BOOST_CHECK_EQUAL(Fix::id_8192.permuted(Fix::prem_8192), Fix::id_8192.permuted_gpu(Fix::prem_8192));
  BOOST_CHECK_EQUAL(Fix::plus_one_8192.permuted(Fix::prem_8192), Fix::plus_one_8192.permuted_gpu(Fix::prem_8192));
  BOOST_CHECK_EQUAL(Fix::plus_one_8192.permuted(Fix::mod_32_8192), Fix::plus_one_8192.permuted_gpu(Fix::mod_32_8192));
  BOOST_CHECK_EQUAL(Fix::prem_8192.permuted(Fix::mod_32_8192), Fix::prem_8192.permuted_gpu(Fix::mod_32_8192));

}

BOOST_AUTO_TEST_SUITE_END()
//____________________________________________________________________________//
//____________________________________________________________________________//

BOOST_AUTO_TEST_SUITE(fonctions_gpu_test_32768)
//____________________________________________________________________________//

BOOST_FIXTURE_TEST_CASE(GPU_32768, Fix) {
  BOOST_CHECK_EQUAL(Fix::id_32768.permuted(Fix::prem_32768), Fix::id_32768.permuted_gpu(Fix::prem_32768));
  BOOST_CHECK_EQUAL(Fix::plus_one_32768.permuted(Fix::prem_32768), Fix::plus_one_32768.permuted_gpu(Fix::prem_32768));
  BOOST_CHECK_EQUAL(Fix::plus_one_32768.permuted(Fix::mod_32_32768), Fix::plus_one_32768.permuted_gpu(Fix::mod_32_32768));
  BOOST_CHECK_EQUAL(Fix::prem_32768.permuted(Fix::mod_32_32768), Fix::prem_32768.permuted_gpu(Fix::mod_32_32768));

}

BOOST_AUTO_TEST_SUITE_END()
//____________________________________________________________________________//
//____________________________________________________________________________//

BOOST_AUTO_TEST_SUITE(fonctions_gpu_test_131072)
//____________________________________________________________________________//

BOOST_FIXTURE_TEST_CASE(GPU_131072, Fix) {
  BOOST_CHECK_EQUAL(Fix::id_131072.permuted(Fix::prem_131072), Fix::id_131072.permuted_gpu(Fix::prem_131072));
  BOOST_CHECK_EQUAL(Fix::plus_one_131072.permuted(Fix::prem_131072), Fix::plus_one_131072.permuted_gpu(Fix::prem_131072));
  BOOST_CHECK_EQUAL(Fix::plus_one_131072.permuted(Fix::mod_32_131072), Fix::plus_one_131072.permuted_gpu(Fix::mod_32_131072));
  BOOST_CHECK_EQUAL(Fix::prem_131072.permuted(Fix::mod_32_131072), Fix::prem_131072.permuted_gpu(Fix::mod_32_131072));

}

BOOST_AUTO_TEST_SUITE_END()
//____________________________________________________________________________//
//____________________________________________________________________________//

//____________________________________________________________________________//

BOOST_AUTO_TEST_SUITE(fonctions_gpu_test_701)
//____________________________________________________________________________//

BOOST_FIXTURE_TEST_CASE(GPU_701, Fix) {
  BOOST_CHECK_EQUAL(Fix::id_701.permuted(Fix::prem_701), Fix::id_701.permuted_gpu(Fix::prem_701));
  BOOST_CHECK_EQUAL(Fix::plus_one_701.permuted(Fix::prem_701), Fix::plus_one_701.permuted_gpu(Fix::prem_701));
  BOOST_CHECK_EQUAL(Fix::plus_one_701.permuted(Fix::mod_32_701), Fix::plus_one_701.permuted_gpu(Fix::mod_32_701));
  BOOST_CHECK_EQUAL(Fix::prem_701.permuted(Fix::mod_32_701), Fix::prem_701.permuted_gpu(Fix::mod_32_701));

}

BOOST_AUTO_TEST_SUITE_END()
//____________________________________________________________________________//
//____________________________________________________________________________//
//____________________________________________________________________________//

BOOST_AUTO_TEST_SUITE(fonctions_gpu_test_Rand)
//____________________________________________________________________________//

BOOST_FIXTURE_TEST_CASE(GPU_Rand, Fix) {
  BOOST_CHECK_EQUAL(Fix::id_1024.permuted(Fix::id_1024), Fix::id_1024.permuted_gpu(Fix::id_1024));
  BOOST_CHECK_EQUAL(Fix::randShuf.permuted(Fix::randShuf), Fix::randShuf.permuted_gpu(Fix::randShuf));
  BOOST_CHECK_EQUAL(Fix::rand.permuted(Fix::rand), Fix::rand.permuted_gpu(Fix::rand));
  BOOST_CHECK_EQUAL(Fix::zeros.permuted(Fix::zeros), Fix::zeros.permuted_gpu(Fix::zeros));

}

BOOST_AUTO_TEST_SUITE_END()
//____________________________________________________________________________//
//____________________________________________________________________________//





