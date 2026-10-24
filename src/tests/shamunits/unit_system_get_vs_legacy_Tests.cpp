// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file unit_system_get_vs_legacy_Tests.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Temporary: UnitSystem's Unit-value interface must match the legacy enum-based getters
 *
 * Every unit x prefix x power in several systems, runtime and template paths, f64 and f32.
 * Delete together with legacy_units (Phase D).
 */

#include "shambase/aliases_float.hpp"
#include "shamtest/shamtest.hpp"
#include "shamunits/Names.hpp"
#include "shamunits/UnitSystem.hpp"
#include "shamunits/prefix_table.hpp"
#include "shamunits/unit_table.hpp"
#include <cmath>
#include <string>

namespace {
    using namespace shamunits;

    template<class T>
    void require_rel(const std::string &label, T got, T expected, T tol) {
        if (!std::isnormal(expected) || !std::isnormal(got)) {
            return; // out of range for T in this system: nothing to compare
        }
        REQUIRE_FLOAT_EQUAL_NAMED(label, got / expected, T(1), tol);
    }

    /// Runtime path: every unit, alone and with every prefix, powers -3..3
    template<class T>
    void compare_runtime(UnitSystem<T> &s, const std::string &sys, T tol) {
        for (const NamedUnit &e : units::registry) {
            std::string uname(e.long_name);
            auto legacy_u = legacy_units::unit_from_name(uname);
            for (const NamedUnit &p : prefix::registry) {
                std::string pname(p.long_name);
                auto legacy_p = unit_prefix_from_name(pname);
                for (int pw = -3; pw <= 3; pw++) {
                    std::string label = sys + " " + pname + " " + uname + "^" + std::to_string(pw);
                    require_rel(
                        label + " get",
                        s.get(p.unit * e.unit, pw),
                        s.runtime_get(legacy_p, legacy_u, pw),
                        tol);
                    require_rel(
                        label + " to",
                        s.to(p.unit * e.unit, pw),
                        s.runtime_to(legacy_p, legacy_u, pw),
                        tol);
                }
            }
        }
    }

} // namespace

NEW_TEST(Unittest, "shamunits/unit_system_get_vs_legacy", 1) {
    UnitSystem<f64> si{}, astro{3.15576e13, 1.495978707e11, 1.98847e30}, cgs{1, 1e-2, 1e-3};
    UnitSystem<f64> primes{2, 3, 5, 7, 11, 13, 17};
    compare_runtime<f64>(si, "f64 si", 1e-12);
    compare_runtime<f64>(astro, "f64 astro", 1e-12);
    compare_runtime<f64>(cgs, "f64 cgs", 1e-12);
    compare_runtime<f64>(primes, "f64 primes", 1e-12);

    // f32: the legacy getters lose precision in the astro system (BUG 6), so compare elsewhere
    UnitSystem<f32> si32{}, cgs32{1, 1e-2f, 1e-3f}, primes32{2, 3, 5, 7, 11, 13, 17};
    compare_runtime<f32>(si32, "f32 si", 1e-5f);
    compare_runtime<f32>(cgs32, "f32 cgs", 1e-5f);
    compare_runtime<f32>(primes32, "f32 primes", 1e-5f);

    // f32 astro: compare against the f64 result of the new path instead
    UnitSystem<f32> astro32{3.15576e13f, 1.495978707e11f, 1.98847e30f};
    for (const NamedUnit &e : units::registry) {
        for (const NamedUnit &p : prefix::registry) {
            for (int pw = -3; pw <= 3; pw++) {
                Unit<> u = upow(p.unit * e.unit, pw);
                require_rel(
                    "f32 astro vs f64 " + std::string(p.long_name) + " " + std::string(e.long_name)
                        + "^" + std::to_string(pw),
                    astro32.get(u),
                    f32(astro.get(u)),
                    1e-5f);
            }
        }
    }

    // Template path for the units the C++ call sites use: identical values, not just close
    for (UnitSystem<f64> *s : {&si, &astro, &cgs, &primes}) {
        REQUIRE_EQUAL(s->to<units::kilogram>(), s->to<legacy_units::kilogram>());
        REQUIRE_EQUAL(s->to<units::metre>(), s->to<legacy_units::metre>());
        REQUIRE_EQUAL(s->to<units::second>(), s->to<legacy_units::second>());
        REQUIRE_EQUAL(s->get<units::hertz>(), s->get<legacy_units::Hertz>());
        REQUIRE_EQUAL(
            s->get<units::astronomical_unit>(), s->get<legacy_units::astronomical_unit>());
        REQUIRE_EQUAL(s->get<units::year>(), s->get<legacy_units::years>());
    }
}
