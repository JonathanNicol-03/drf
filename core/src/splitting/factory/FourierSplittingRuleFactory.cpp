/*-------------------------------------------------------------------------------
 This file is part of distributional random forest (drf).
 
 drf is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 drf is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with drf. If not, see <http://www.gnu.org/licenses/>.
#-------------------------------------------------------------------------------*/

#include "splitting/factory/FourierSplittingRuleFactory.h"
#include "splitting/FourierSplittingRule.h"

namespace drf {

std::unique_ptr<SplittingRule> FourierSplittingRuleFactory::create(const Data& data,
                                                                      const TreeOptions& options) const {
    (void) data;
    return std::unique_ptr<SplittingRule>(
        new FourierSplittingRule(options.get_alpha(),
                                         options.get_min_obs(),
                                         options.get_max_weight_ratio()));
}

} // namespace drf
