#pragma once

#include "tdse/parameters.hpp"

#include <iosfwd>
#include <string>

namespace tdse {

/** Parse a Quantum ESPRESSO-style namelist file into `p` (defaults already set). */
void parse_namelist_file(const std::string &path, Parameters &p);

/** Apply one `key = value` inside namelist `section` (names are case-insensitive). */
void apply_namelist_assignment(Parameters &p,
                               const std::string &section,
                               const std::string &key,
                               const std::string &value,
                               const std::string &origin);

/** Cross-check dimensions, propagator/basis, and fill prefix-derived file names. */
void finalize_parameters(Parameters &p);

/** Write a commented template namelist to `os`. */
void write_input_template(std::ostream &os);

const char *trap_name(TrapKind t);
const char *basis_name(const Parameters &p);

} // namespace tdse
