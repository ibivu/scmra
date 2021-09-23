"""Utilitiy functions to interact with cplex.Cplex objects."""

import cplex


###############################################################################
#
# functions
#

def set_rhs_linear(cpx, newval):
    """Set rhs of linear constraints to newval.

    Parameters
    ----------
    cpx : cplex.Cplex object

    newval : float
    """
    for constr in cpx.linear_constraints.get_names():
        if constr.endswith('lb'):  # Reset lower bounds
            cpx.linear_constraints.set_rhs(constr, -newval)
        elif constr.endswith('ub'):  # Reset upper bounds
            cpx.linear_constraints.set_rhs(constr, newval)
        else:
            print("""WARNING: Constraint doesn't end with lb or ub
                   neither upper nor lower bound""")


def set_indicator_status(cpx, indicator, status):
    """Set indicator active or inactive.

    This can be used to force interactions to be present or absent in a network
    reconstruction.

    Parameters
    ----------
    cpx : cplex.Cplex instance

    indicator : string.
        Name of indicator.

    status : {0, 1}
        0 to force indicator to be inactive, 1 active
    """
    assert indicator in cpx.variables.get_names(), indicator + " is not a \
    variable"
    if status == 1:
        assert cpx.variables.get_upper_bounds(indicator) == 1, indicator + \
         " is already set inactive, cannot be set to active"
        cpx.variables.set_lower_bounds(indicator, 1)
        cpx.objective.set_linear(indicator, 0.)
    elif status == 0:
        assert cpx.variables.get_lower_bounds(indicator) == 0, indicator + \
           " is already set active, cannot be set to inactive"
        cpx.variables.set_upper_bounds(indicator, 0)
        cpx.objective.set_linear(indicator, 0.)
    else:
        raise ValueError("status should be either 1 or 0.")


# def set_rhs_indicator(cpx, newval):
#     """Set rhs of all indicator constraints to newval.

#     This can be used to update the tolerance for r.
#     """
#     # Check if number of indicator constraints doesn't change

#     # First, get list of indicator variables
#     inames = []
#     for var in cpx.variables.get_names():
#         if cpx.variables.get_types(var) == "B":
#             inames.append(var)
#     # Remove all indicator constraints
#     nind_old = cpx.indicator_constraints.get_num()
#     cpx.indicator_constraints.delete()
#     for iij in inames:
#         rij = 'r' + iij[1:]
#         nij = 'ind' + iij[1:]
#         constr = cplex.SparsePair(ind=[rij], val=[1.])
#         cpx.indicator_constraints.add(indvar=iij, complemented=1,
#                                       rhs=newval, sense='L',
#                                       lin_expr=constr,
#                                       name=nij + "_ub")
#         cpx.indicator_constraints.add(indvar=iij, complemented=1,
#                                       rhs=-newval, sense='G',
#                                       lin_expr=constr,
#                                       name=nij + "_lb")
#     nind_new = cpx.indicator_constraints.get_num()
#     if not nind_new == nind_old:
#         print("WARNING: number of indicator constraints changed.")
#         print("from" + str(nind_old) + "to" + str(nind_new))


def set_vars_positive(cpx, var_names):
    """"Restrict variables to be positive."""
    lbounds = []
    for name in var_names:
        assert name in cpx.variables.get_names(), name + ' not a variable'
        lbounds.append((name, 0.))
    cpx.variables.set_lower_bounds(lbounds)


def set_vars_negative(cpx, var_names):
    """"Restrict variables to be negative."""
    ubounds = []
    for name in var_names:
        assert name in cpx.variables.get_names(), name + ' not a variable'
        ubounds.append((name, 0.))
    cpx.variables.set_upper_bounds(ubounds)
