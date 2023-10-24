//****************************************************************************//
//       Copyright (C) 2023 James D. Mitchell <jdm3@st-andrews.ac.uk>         //
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

#include <iostream>

#include <catch2/catch_test_case_info.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

struct HPCombiListener : Catch::EventListenerBase {
    using EventListenerBase::EventListenerBase;  // inherit constructor

    void testCaseStarting(Catch::TestCaseInfo const &testInfo) override {
        std::cout << testInfo.tagsAsString() << " " << testInfo.name
                  << std::endl;
    }
    void testCaseEnded(Catch::TestCaseStats const &testInfo) override {}
    void sectionStarting(Catch::SectionInfo const &sectionStats) override {}
    void sectionEnded(Catch::SectionStats const &sectionStats) override {}
    void testCasePartialStarting(Catch::TestCaseInfo const &testInfo,
                                 uint64_t partNumber) override {}
    void testCasePartialEnded(Catch::TestCaseStats const &testCaseStats,
                              uint64_t partNumber) override {}
};

CATCH_REGISTER_LISTENER(HPCombiListener)
