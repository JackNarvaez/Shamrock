// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file unit_tables_vs_legacy_Tests.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Temporary: the new unit and prefix tables must match the legacy enum-based ones
 *
 * Catches transcription errors in unit_table.hpp / prefix_table.hpp: same names, and the same
 * value for every unit and prefix in several unit systems. Delete together with legacy_units.
 */

#include "shambase/aliases_float.hpp"
#include "shamtest/shamtest.hpp"
#include "shamunits/Names.hpp"
#include "shamunits/UnitSystem.hpp"
#include "shamunits/details/utils.hpp"
#include "shamunits/prefix_table.hpp"
#include "shamunits/unit_table.hpp"
#include <cmath>
#include <string>

namespace {
    using namespace shamunits;
    using details::find_named;
    using details::ipow;

    /// Value of `u` in system `s`, the way the new UnitSystem::get(Unit) will compute it
    f64 new_value(const UnitSystem<f64> &s, const Unit<> &u) {
        return u.si_factor * ipow(s.s, u.dim.second, s.s_inv) * ipow(s.m, u.dim.metre, s.m_inv)
               * ipow(s.kg, u.dim.kilogram, s.kg_inv) * ipow(s.A, u.dim.ampere, s.A_inv)
               * ipow(s.K, u.dim.kelvin, s.K_inv) * ipow(s.mol, u.dim.mole, s.mol_inv)
               * ipow(s.cd, u.dim.candela, s.cd_inv);
    }

    void require_rel(const std::string &label, f64 got, f64 expected) {
        REQUIRE_FLOAT_EQUAL_NAMED(label, got / expected, 1., 1e-13);
    }

} // namespace

NEW_TEST(Unittest, "shamunits/unit_tables_vs_legacy", 1) {

    // Same names, both directions
    for (const auto &[name, legacy] : legacy_units::map_name_to_unit) {
        REQUIRE_NAMED(
            "unit name in new table: " + name, find_named(units::registry, name) != nullptr);
    }
    for (const NamedUnit &e : units::registry) {
        REQUIRE_NAMED(
            "unit name in legacy table: " + std::string(e.long_name),
            legacy_units::map_name_to_unit.count(std::string(e.long_name)) == 1
                && legacy_units::map_name_to_unit.count(std::string(e.short_name)) == 1);
    }
    for (const auto &[name, legacy] : map_name_to_unit_prefix) {
        REQUIRE_NAMED(
            "prefix name in new table: " + name, find_named(prefix::registry, name) != nullptr);
    }
    REQUIRE_EQUAL(map_name_to_unit_prefix.size(), 2 * prefix::registry.size());

    // Same values: every unit, with and without every prefix, in several systems
    UnitSystem<f64> systems[] = {
        UnitSystem<f64>{},
        UnitSystem<f64>{3.15576e13, 1.495978707e11, 1.98847e30},
        UnitSystem<f64>{1, 1e-2, 1e-3},
        UnitSystem<f64>{2, 3, 5, 7, 11, 13, 17},
    };
    for (UnitSystem<f64> &s : systems) {
        for (const NamedUnit &e : units::registry) {
            std::string uname(e.long_name);
            auto legacy_u = legacy_units::unit_from_name(uname);

            require_rel("unit " + uname, new_value(s, e.unit), s.runtime_get(None, legacy_u, 1));

            for (const NamedUnit &p : prefix::registry) {
                std::string pname(p.long_name);
                require_rel(
                    "prefix " + pname + " " + uname,
                    new_value(s, p.unit * e.unit),
                    s.runtime_get(unit_prefix_from_name(pname), legacy_u, 1));
            }
        }
    }
}
