"""Classes and functions to generate cplex.Cplex for sc-MRA problem."""

import cplex
import numpy as np
import pandas as pd
import sympy
import itertools
from scmra.cplexutils import set_indicator_status
from scmra.scdata import ScData

###############################################################################
#
# Classes


class ScMraProblem(ScData):
    """Class to generate sc-MRA problem and associated cplex.Cplex object.

    Inherits from ScData.

    Parameters
    ---------
    scdata : scdata.ScData object

    bounds : float > 0, optional (default=cplex.infinity)
        Set the bounds on all continious variables in the models. This is
        overwritten if more specific (r_bounds, s_bounds, p_bounds) bounds are
        set.

    r_bounds : float > 0, optional
        Sets/overwrites the bounds on local response coefficient

    eta : float in [0, 1], optional (default = 0.0)
        Weight of interactions vs errors in objective functions. Higher eta
         favors sparser solutions.

    alpha: "automatic" or float in [0, 1], optional (default = "automatic").
        This parameter sets the weight relative weight of residuals from 
        complete and incomplete nodes in the objective functions.

    prior_network : List of 2-tuples, optional
        If provided, possible interactions are restricted to this set

    mode: None, simple or comparative.
        Used to set if the reconstruction should be comparative or simple. If
        set to comparative, a differnet network for each group will be 
        reconstructed with an L0 penalty on differences between the groups. 

    Attributes
    ----------
    nodes : list of strings
        Names of the nodes in the model

    rglob : pandas.DataFrame
        Dataframe containing measurements of node activity (i.e. phospho-
        proteins). Rows correspond to nodes. Columns correspond to cells.
        Entries are the scaled diviations from the bulk mean.

    rtot:
        Dataframe containing measurements of total node abundance (i.e. total
         protein). Rows correspond to nodes. Columns correspond to cells.
        Entries are the scaled diviations from the bulk mean. Should have the
        same columns and index as rglob.

    scp : cplex.Cplex object
        Contains the ScMra problem formulation.
    """

    def __init__(
            self, scdata,
            bounds=cplex.infinity,
            r_bounds=None,
            s_bounds=None, p_bounds=None,
            ec_bounds=None, ei_bounds=None,
            eta=0.0,
            alpha='automatic',
            prior_network=None,
            mode=None):
        """Initialization from ScData object."""
        # From input
        super().__init__(scdata.rglob, scdata.rtot,
                         scdata.cell_annot, scdata.tx_annot)

        # Set alpha value
        assert 0. <= eta < 1, "eta should be in [0,1)"
        self._eta = eta
        if alpha == 'automatic':
            self._alpha = estimate_alpha(scdata)
        else:
            assert 0 < alpha <= 1, "alpha is invalid, should be 'automatic' or \
                in (0,1]"
            self._alpha = alpha

        # Check prior network
        if prior_network is not None:
            for n, m in prior_network:
                assert (n in scdata.nodes),\
                    "invalid node " + n + " in prior_network"
                assert (m in scdata.nodes), \
                    "invalid node " + n + " in prior_network"
        self._prior_network = prior_network

        self._r_bounds = r_bounds if r_bounds is not None else bounds
        self._s_bounds = s_bounds if s_bounds is not None else bounds
        self._p_bounds = p_bounds if p_bounds is not None else bounds
        self._ec_bounds = ec_bounds if ec_bounds is not None else bounds
        self._ei_bounds = ei_bounds if ei_bounds is not None else bounds
        # self._estimate_sr = estimate_sr
        # Construct helper objects

        self._rloc_vars = [
            "_".join(["r", node_i, node_j]) for node_i, node_j in
            itertools.product(self.nodes, self.nodes) if
            (node_i != node_j)
        ]

        self._indicators = _generate_indicators(
            self.nodes, self._prior_network)

        self._s_vars = ["s_"+node for node in self.nodes_complete]
        
        if self.tx_annot is not None:
            self._p_vars = [
                "_".join(["p", tx, node]) for tx, node in self.tx_annot.items()
                if node is not None]
        
        self._residuals_complete = [
            '_'.join(['e', node, cell]) for node, cell in
            itertools.product(self.nodes_complete, self.cells)
        ]
        self._residuals_incomplete = [
            '_'.join(['e', node, cell]) for node, cell in
            itertools.product(self.nodes_incomplete, self.cells)
        ]
        self.add_cpx()

    @property
    def indicators(self):
        "return list of indicators for edge presence."
        return self._indicators

    @property
    def stot_symbols(self):
        """return sympy.Matrix with s matrix. Elements are sympy.Symbols"""
        return sympy.Matrix(np.diag(sympy.symbols(self._s_vars)))

    @property
    def residuals_symbols(self):
        """return sympy.Matrix with residuals matrix. Elements are
        sympy.Symbols"""
        mat = sympy.Matrix(
            sympy.symbols(self._residuals_complete +
                          self._residuals_incomplete)
        ).reshape(len(self.nodes), len(self._cells))
        return mat

    @property
    def residuals_complete_symbols(self):
        """return sympy.Matrix with residuals matrix. Elements are
        sympy.Symbols"""
        mat = sympy.Matrix(
            sympy.symbols(self._residuals_complete)
        ).reshape(len(self.nodes_complete), len(self._cells))
        return mat

    @property
    def residuals_incomplete_symbols(self):
        """return sympy.Matrix with residuals matrix. Elements are
        sympy.Symbols"""
        mat = sympy.Matrix(
            sympy.symbols(self._residuals_incomplete)
        ).reshape(len(self.nodes_incomplete), len(self._cells))
        return mat

    # @property
    # def matrixform_complete(self):
    #     """Return sympy.Matrix with the MRA equations for complete nodes: \
    #          rloc.R + s.R^tot"""
    #     rloc = sympy.Matrix(generate_rloc_symbols(
    #         self.nodes_complete, self.nodes, self._prior_network)
    #     )
    #     stot = sympy.Matrix(np.diag(sympy.symbols(self._s_vars)))
    #     mat = (np.dot(rloc, self.rglob) + np.dot(stot, self.rtot))
    #     return sympy.Matrix(mat)

    # @property
    # def matrixform_incomplete(self):
    #     """Return sympy.Matrix with the MRA equations for incomplete nodes: \
    #          rloc.R + sr"""
    #     rloc = sympy.Matrix(generate_rloc_symbols(
    #         self.nodes_incomplete, self.nodes, self._prior_network)
    #     )
    #     # if self._estimate_sr:
    #     #     sr = sympy.Matrix([sympy.Symbol(var) for var in self._sr_vars]).reshape(
    #     #         len(self.nodes_incomplete), len(self._cells)
    #     #     )
    #     #     mat = np.dot(rloc, self.rglob) + 1. * sr
    #     # else:
    #     mat = np.dot(rloc, self.rglob)
    #     return sympy.Matrix(mat)

    def _cpx_add_variables(self):
        """Add variables to cplex.Cplex object"""
        # Add local response coefficients as variables
        n_rloc = len(self._rloc_vars)
        self.cpx.variables.add(
            names=self._rloc_vars,
            types=[self.cpx.variables.type.continuous] * n_rloc,
            lb=[-1*self._r_bounds] * n_rloc,
            ub=[self._r_bounds] * n_rloc
        )

        # Add diagonal element of r matrix
        self.cpx.variables.add(
            names=['r_i_i'],
            types=[self.cpx.variables.type.continuous],
            lb=[-1.], ub=[-1.]
        )

        # Add sensitivities of complete nodes as variables
        n_s = len(self._s_vars)
        self.cpx.variables.add(
            names=self._s_vars,
            types=[self.cpx.variables.type.continuous] * n_s,
            lb=[0.] * n_s,  # protein, phospho-protein assumed to be positively
            ub=[self._s_bounds] * n_s
        )

        # Add perturbation strengths as variables
        if self.tx_annot is not None:
            n_p = len(self._p_vars)
            self.cpx.variables.add(
                names=self._p_vars,
                types=[self.cpx.variables.type.continuous] * n_p,
                lb=[-1*self._p_bounds] * n_p,
                ub=[self._p_bounds] * n_p
            )

        # Add residuals of complete nodes as variables
        n_ec = len(self._residuals_complete)
        self.cpx.variables.add(
            names=self._residuals_complete,
            types=[self.cpx.variables.type.continuous] * n_ec,
            lb=[-1*self._ec_bounds] * n_ec,
            ub=[self._ec_bounds] * n_ec
        )
        # Add residuals of incomplete nodes as variables
        n_ei = len(self._residuals_incomplete)
        self.cpx.variables.add(
            names=self._residuals_incomplete,
            types=[self.cpx.variables.type.continuous] * n_ei,
            lb=[-1*self._ei_bounds] * n_ei,
            ub=[self._ei_bounds] * n_ei
        )

        # Add indicator constraint on presence/absence of edges as variables
        n_i = len(self._indicators)
        self.cpx.variables.add(
            names=self._indicators,
            types=[self.cpx.variables.type.binary] * n_i,
            lb=[0] * n_i,
            ub=[1] * n_i,
            obj=[self._eta] * n_i
        )

    def _cpx_mra_eqns(self):
        lin_exprs = []
        # Iterate over all MRA equations

        # The MRA equations for complete nodes (protein + phospho measured)
        # Temp helper objects to prevent recreating these too often
        tmp_rloc_complete = generate_rloc_symbols(
            self.nodes_complete, self.nodes, self._prior_network)
        tmp_stot = self.stot_symbols
        if self.tx_annot is not None:
            tmp_pert_complete = generate_pert_symbols(
                self.nodes_complete, self.cells, 
                self.cell_annot_inv, self.tx_annot
            )
        tmp_res_complete = self.residuals_complete_symbols

        # Iterate over all complete node and each cell
        for node_idx in range(len(self.nodes_complete)):
            for cell_idx in range(len(self.cells)):

                # The contribution of other nodes
                var = [str(r) for r in tmp_rloc_complete[node_idx, :]]
                coef = list(np.array(self.rglob)[:, cell_idx])

                # Remove absent interactions
                indices = [i for i, x in enumerate(var) if x == '0']
                var = [i for j, i in enumerate(var) if j not in indices]
                coef = [i for j, i in enumerate(coef) if j not in indices]

                # Contribution of deviations in total protein
                var.append(str(tmp_stot[node_idx, node_idx]))
                coef.append(np.array(self.rtot)[node_idx, cell_idx])

                # Contribution of perturbation (if perturbations were done)
                if self.tx_annot is not None and\
                        tmp_pert_complete[node_idx, cell_idx] != 0:
                    pvar = tmp_pert_complete[node_idx, cell_idx]
                    assert pvar in self._p_vars
                    var.append(pvar)
                    coef.append(1.)

                # The residual (which appears in the objective function)
                var.append(str(tmp_res_complete[node_idx, cell_idx]))
                coef.append(1.)

                lin_exprs.append([var, coef])  # Add constraint to linexps

        # The MRA equations for incomplete nodes (only phospho measured)
        # Temp helper objects
        tmp_rloc_incomplete = generate_rloc_symbols(
            self.nodes_incomplete, self.nodes, self._prior_network)
        if self.tx_annot is not None:
            tmp_pert_incomplete = generate_pert_symbols(
                self.nodes_incomplete, self.cells, 
                self.cell_annot_inv, self.tx_annot
            )
        tmp_res_incomplete = self.residuals_incomplete_symbols
        # Iterate over all incomplete nodes in each cell
        for node_idx in range(len(self.nodes_incomplete)):
            for cell_idx in range(len(self.cells)):

                # The contribution of other nodes
                var = [str(r) for r in tmp_rloc_incomplete[node_idx, :]]
                coef = list(np.array(self.rglob)[:, cell_idx])

                # Remove absent interactions
                indices = [i for i, x in enumerate(var) if x == '0']
                var = [i for j, i in enumerate(var) if j not in indices]
                coef = [i for j, i in enumerate(coef) if j not in indices]

                # Contribution of perturbation if annotations are present
                if self.tx_annot is not None and\
                        tmp_pert_incomplete[node_idx, cell_idx] != 0:
                    pvar = tmp_pert_incomplete[node_idx, cell_idx]
                    assert pvar in self._p_vars
                    var.append(pvar)
                    coef.append(1.)

                # The residual (which appears in the objective function)
                var.append(str(tmp_res_incomplete[node_idx, cell_idx]))
                coef.append(1.)

                lin_exprs.append([var, coef])  # Add constraint to linexps

        # Add all MRA equations to the Cplex object
        self.cpx.linear_constraints.add(
            lin_expr=lin_exprs,
            senses=["E"] * len(lin_exprs),
            rhs=[0.] * len(lin_exprs)
        )

    def add_cpx(self):
        """Create cplex MIQP problem."""
        # Initialize the Cplex object.
        self.cpx = cplex.Cplex()

        self.cpx.set_problem_type(cplex.Cplex.problem_type.MIQP)
        self.cpx.objective.set_sense(self.cpx.objective.sense.minimize)

        self._cpx_add_variables()
        self._cpx_mra_eqns()

        # ---------------------------------------------------------------------
        # Add quadratic part to objective function
        for ec in self._residuals_complete:
            self.cpx.objective.set_quadratic_coefficients(ec, ec, 1.-self._eta)

        for ei in self._residuals_incomplete:
            self.cpx.objective.set_quadratic_coefficients(
                ei, ei, self._alpha * (1.-self._eta))

        # -------------------------------------------------------------------
        # Construct indicator constraints for presence of edge
        for rvar in self._rloc_vars:
            # rvar is expected to have form r_nodei_nodej
            r_elements = rvar.split('_')
            assert len(r_elements) == 3
            assert r_elements[0] == 'r'
            assert r_elements[1] in self.nodes and r_elements[2] in self.nodes
            ivar = '_'.join(["I", r_elements[1], r_elements[2]])
            assert ivar in self._indicators
            name = '_'.join(["Ind", r_elements[1], r_elements[2]])
            constr = cplex.SparsePair(ind=[rvar], val=[1.])
            self.cpx.indicator_constraints.add(
                indvar=ivar,
                complemented=1,
                rhs=0., sense='E',
                lin_expr=constr,
                name=name
            )


class ScCnrProblem(ScData):
    """Class to generate sc-CNR problem and associated cplex.Cplex object."""

    def __init__(
            self, scdata,
            eta=0.0, theta=0.0,
            alpha='automatic',
            bounds=cplex.infinity,
            r_bounds=None,
            dev_bounds=None,
            s_bounds=None, p_bounds=None,
            ec_bounds=None, ei_bounds=None,
            prior_network=None):
        """Initialization from ScData object."""
        # From input
        super().__init__(scdata.rglob, scdata.rtot,
                         scdata.cell_annot, scdata.tx_annot, scdata.group_annot)

        # Set hyper parameters value
        assert 0. <= eta < 1, "eta should be in [0,1)"
        self._eta = eta
        self._theta = theta
        if alpha == 'automatic':
            self._alpha = estimate_alpha(scdata)
        else:
            assert 0 < alpha <= 1, \
            """alpha should be `automatic` or in (0, 1]"""
            self._alpha = alpha

        # Check prior network
        if prior_network is not None:
            for n, m in prior_network:
                assert (n in scdata.nodes),\
                    "invalid node " + n + " in prior_network"
                assert (m in scdata.nodes), \
                    "invalid node " + n + " in prior_network"
        self._prior_network = prior_network

        # Bounds
        self._r_bounds = r_bounds if r_bounds is not None else bounds
        self._dev_bounds = dev_bounds if dev_bounds is not None else bounds
        self._s_bounds = s_bounds if s_bounds is not None else bounds
        self._p_bounds = p_bounds if p_bounds is not None else bounds
        self._ec_bounds = ec_bounds if ec_bounds is not None else bounds
        self._ei_bounds = ei_bounds if ei_bounds is not None else bounds
        # self._estimate_sr = estimate_sr
        # Construct helper objects

        # Variables related to local response matrix
        if prior_network is None:
            self._rloc_vars = [
                "_".join(["r", group, node_i, node_j]) for group, node_i, node_j in
                itertools.product(self.groups, self.nodes, self.nodes) if
                (node_i != node_j)
            ]
        self._dev_vars = ["dev_" + r for r in self._rloc_vars]

        self._indicators = _generate_indicators(
            self.nodes, self._prior_network)
        self._dev_indicators = _generate_indicators(self.nodes,
                                                    self._prior_network,
                                                    base='IDev')
        
        # Variables related to s matrix (sensitivity of p-prot to tot-prot)
        self._s_vars = ["_".join(["s", group, node]) for group, node in
                        itertools.product(self.groups, self.nodes_complete)]
        self._sdev_vars = ["dev_" + s for s in self._s_vars]
        self._sdev_indicators = ["ISDev_" +
                                 node for node in self.nodes_complete]

        # Variables related to perturbations 
        if self.tx_annot is not None:
            self._p_vars = []
            for group in self.groups:
                for tx, node in self.tx_annot.items():
                    if node is not None:
                        self._p_vars.append("_".join(["p", group, tx, node]))
            self._pdev_vars = ["dev_" + p for p in self._p_vars]
            self._pdev_indicators = [
                "_".join(['IPDev', tx, node]) for 
                tx, node in self.tx_annot.items() if 
                node is not None]            
        
        # Residuals
        self._residuals_complete = [
            '_'.join(['e', node, cell]) for node, cell in
            itertools.product(self.nodes_complete, self.cells)
        ]
        self._residuals_incomplete = [
            '_'.join(['e', node, cell]) for node, cell in
            itertools.product(self.nodes_incomplete, self.cells)
        ]
        self.add_cpx()

    @property
    def indicators(self):
        "return list of indicators for edge presence."
        return self._indicators

    @property
    def stot_symbols(self):
        """return dict of sympy.Matrixes with s matrix. Elements are sympy.Symbols"""
        stot_dict = {}
        for group in self.groups:
            svg = ["_".join(["s", group, node])
                   for node in self.nodes_complete]
            stot_dict[group] = sympy.Matrix(np.diag(sympy.symbols(svg)))
        return stot_dict

    @property
    def residuals_complete_symbols(self):
        """return dict of sympy.Matrix with residuals matrix. Elements are
        sympy.Symbols"""
        mat = sympy.Matrix(
            sympy.symbols(self._residuals_complete)
        ).reshape(len(self.nodes_complete), len(self._cells))
        return mat

    @property
    def residuals_incomplete_symbols(self):
        """return sympy.Matrix with residuals matrix. Elements are
        sympy.Symbols"""
        mat = sympy.Matrix(
            sympy.symbols(self._residuals_incomplete)
        ).reshape(len(self.nodes_incomplete), len(self._cells))
        return mat

    def _cpx_add_variables(self):
        """Add variables to cplex.Cplex object"""
        # Add local response coefficients as variables
        n_rloc = len(self._rloc_vars)
        self.cpx.variables.add(
            names=self._rloc_vars,
            types=[self.cpx.variables.type.continuous] * n_rloc,
            lb=[-1*self._r_bounds] * n_rloc,
            ub=[self._r_bounds] * n_rloc
        )

        # Add diagonal element of r matrix
        self.cpx.variables.add(
            names=['r_i_i'],
            types=[self.cpx.variables.type.continuous],
            lb=[-1.], ub=[-1.]
        )

        # Add sensitivities of complete nodes as variables
        n_s = len(self._s_vars)
        self.cpx.variables.add(
            names=self._s_vars,
            types=[self.cpx.variables.type.continuous] * n_s,
            lb=[0.] * n_s,  # protein, phospho-protein assumed to be positively
            ub=[self._s_bounds] * n_s
        )

        # Add direct drug effect  as variables
        if self.tx_annot is not None:
            n_p = len(self._p_vars)
            self.cpx.variables.add(
                names=self._p_vars,
                types=[self.cpx.variables.type.continuous] * n_p,
                lb=[-1*self._p_bounds] * n_p,
                ub=[self._p_bounds] * n_p
            )

        # Add residuals of complete nodes as variables
        n_ec = len(self._residuals_complete)
        self.cpx.variables.add(
            names=self._residuals_complete,
            types=[self.cpx.variables.type.continuous] * n_ec,
            lb=[-1*self._ec_bounds] * n_ec,
            ub=[self._ec_bounds] * n_ec
        )
        # Add residuals of incomplete nodes as variables
        n_ei = len(self._residuals_incomplete)
        self.cpx.variables.add(
            names=self._residuals_incomplete,
            types=[self.cpx.variables.type.continuous] * n_ei,
            lb=[-1*self._ei_bounds] * n_ei,
            ub=[self._ei_bounds] * n_ei
        )

        # Add indicator constraint on presence/absence of edges as variables
        n_i = len(self._indicators)
        self.cpx.variables.add(
            names=self._indicators,
            types=[self.cpx.variables.type.binary] * n_i,
            lb=[0] * n_i,
            ub=[1] * n_i,
            obj=[self._eta] * n_i
        )

        # Add the deviations from mean as variables to cpx object
        n_dev_vars = len(self._dev_vars) + len(self._sdev_vars) 
        dev_names = self._dev_vars + self._sdev_vars
        if self.tx_annot is not None:
            n_dev_vars += len(self._pdev_vars)
            dev_names += self._pdev_vars
        self.cpx.variables.add(
            names= dev_names,
            types=[self.cpx.variables.type.continuous] * n_dev_vars,
            lb=[-self._dev_bounds] * n_dev_vars,
            ub=[self._dev_bounds] * n_dev_vars
        )

        # Add indicator constraint on differences between groups.
        n_idev = len(self._dev_indicators) + len(self._sdev_indicators)
        idev_names = self._dev_indicators + self._sdev_indicators
        if self.tx_annot is not None:
            n_idev += len(self._pdev_indicators)
            idev_names += self._pdev_indicators
        self.cpx.variables.add(
            names= idev_names,
            types=[self.cpx.variables.type.binary] * n_idev,
            lb=[0] * n_idev,
            ub=[1] * n_idev,
            obj=[self._theta] * n_idev
        )
        
    def _cpx_add_mra_eqns(self):
        lin_exprs = []

        # The MRA equations for complete nodes (protein + phospho measured)
        # Temp helper objects to prevent recreating these too often
        tmp_rloc_complete = {
            group: generate_rloc_symbols(
                self.nodes_complete, self.nodes,
                self._prior_network, prefix=group
            )
            for group in self.groups
        }
        tmp_stot = self.stot_symbols
        if self.tx_annot is not None:
            tmp_pert_complete = {
                group: generate_pert_symbols(
                self.nodes_complete, self.cells, 
                self.cell_annot_inv, self.tx_annot, prefix=group
            ) for group in self.groups
        }
        tmp_res_complete = self.residuals_complete_symbols

        # Iterate over all complete node and each cell
        for node_idx in range(len(self.nodes_complete)):
            for cell_idx in range(len(self.cells)):

                # Get the group this cell belongs to
                group = self.group_annot_inv[self.cells[cell_idx]]

                # The contribution of other nodes
                var = [str(r) for r in tmp_rloc_complete[group][node_idx, :]]
                coef = list(np.array(self.rglob)[:, cell_idx])

                # Remove absent interactions
                indices = [i for i, x in enumerate(var) if x == '0']
                var = [i for j, i in enumerate(var) if j not in indices]
                coef = [i for j, i in enumerate(coef) if j not in indices]

                # Contribution of deviations in total protein
                var.append(str(tmp_stot[group][node_idx, node_idx]))
                coef.append(np.array(self.rtot)[node_idx, cell_idx])

                # Contribution of perturbation (if perturbations were done)
                if self.tx_annot is not None and\
                    tmp_pert_complete[group][node_idx, cell_idx] != 0:
                    pvar = tmp_pert_complete[group][node_idx, cell_idx]
                    assert pvar in self._p_vars
                    var.append(pvar)
                    coef.append(1.)

                # The residual (which appears in the objective function)
                var.append(str(tmp_res_complete[node_idx, cell_idx]))
                coef.append(1.)

                lin_exprs.append([var, coef])  # Add constraint to linexps

        # The MRA equations for incomplete nodes (only phospho measured)
        # Temp helper objects
        tmp_rloc_incomplete = {
            group: generate_rloc_symbols(
                self.nodes_incomplete, self.nodes,
                self._prior_network, group)
            for group in self.groups
        }
        if self.tx_annot is not None:
            tmp_pert_incomplete = {
                group: generate_pert_symbols(
                    self.nodes_incomplete, self.cells,
                    self.cell_annot_inv, self.tx_annot, prefix=group
                ) for group in self.groups
            }
        tmp_res_incomplete = self.residuals_incomplete_symbols

        # Iterate over all incomplete nodes in each cell
        for node_idx in range(len(self.nodes_incomplete)):
            for cell_idx in range(len(self.cells)):

                # Get the group this cell belongs to
                group = self.group_annot_inv[self.cells[cell_idx]]

                # The contribution of other nodes
                var = [str(r) for r in tmp_rloc_incomplete[group][node_idx, :]]
                coef = list(np.array(self.rglob)[:, cell_idx])

                # Remove absent interactions
                indices = [i for i, x in enumerate(var) if x == '0']
                var = [i for j, i in enumerate(var) if j not in indices]
                coef = [i for j, i in enumerate(coef) if j not in indices]

                # Contribution of perturbation if annotations are present
                if self.tx_annot is not None and\
                    tmp_pert_incomplete[group][node_idx, cell_idx] != 0:
                    pvar = tmp_pert_incomplete[group][node_idx, cell_idx]
                    assert pvar in self._p_vars
                    var.append(pvar)
                    coef.append(1.)

                # The residual (which appears in the objective function)
                var.append(str(tmp_res_incomplete[node_idx, cell_idx]))
                coef.append(1.)

                lin_exprs.append([var, coef])  # Add constraint to linexps

        # Add all MRA equations to the Cplex object
        self.cpx.linear_constraints.add(
            lin_expr=lin_exprs,
            senses=["E"] * len(lin_exprs),
            rhs=[0.] * len(lin_exprs)
        )

    def _cpx_add_cnr_constraints(self):

        n_groups = len(self.groups)
        dev_constr = []
        dev_constr_names = []
        dev_indicator_constr = []
        dev_indicator_constr_names = []

        for rvar in self._rloc_vars:
            base, group, nd1, nd2 = rvar.split("_")
            devvar = "dev_" + rvar
            ivar = '_'.join(["IDev", nd1, nd2])
            assert base == "r"
            assert group in self.groups
            assert nd1 in self.nodes and nd2 in self.nodes
            assert devvar in self._dev_vars
            assert ivar in self._dev_indicators

            var = [devvar, rvar]
            # NOTE: must be floats
            coef = [-1., (n_groups - 1.) / n_groups]

            # Iterate over all groups
            other_groups = set(self.groups)
            other_groups.remove(group)
            # Add all other groups
            for other in other_groups:
                newvar = "_".join([base, other, nd1, nd2])
                assert newvar in self._rloc_vars
                var.append(newvar)
                coef.append(-1. / n_groups)
            dev_constr.append([var, coef])
            dev_constr_names.append("diffeq_" + rvar)

            # Add the indicator constraint
            # IDev_nd1_nd2 = 0 => dev_r_group_nd1_nd2 = 0
            name = '_'.join(["IndDev", group, nd1, nd2])
            constr = cplex.SparsePair(ind=[devvar], val=[1.])
            self.cpx.indicator_constraints.add(indvar=ivar, complemented=1,
                                               rhs=0., sense='E',
                                               lin_expr=constr,
                                               name=name)

        for svar in self._s_vars:
            base, group, node = svar.split("_")
            devvar = "dev_" + svar
            ivar = '_'.join(["ISDev", node])
            assert base == "s"
            assert group in self.groups
            assert node in self.nodes
            assert devvar in self._sdev_vars
            assert ivar in self._sdev_indicators

            var = [devvar, svar]
            # NOTE: must be floats
            coef = [-1., (n_groups - 1.) / n_groups]

            # Iterate over all groups
            other_groups = set(self.groups)
            other_groups.remove(group)
            # Add all other cell lines
            for other in other_groups:
                newvar = "_".join([base, other, node])
                assert newvar in self._s_vars
                var.append(newvar)
                coef.append(-1. / n_groups)
            dev_constr.append([var, coef])
            dev_constr_names.append("diffeq_" + svar)

            # Add the indicator constraint
            # ISDev_node = 0 => dev_s_group_node = 0
            name = '_'.join(["IndSDev", group, node])
            constr = cplex.SparsePair(ind=[devvar], val=[1.])
            self.cpx.indicator_constraints.add(indvar=ivar, complemented=1,
                                               rhs=0., sense='E',
                                               lin_expr=constr,
                                               name=name)
            
            if self.tx_annot is not None:
                for pvar in self._p_vars:
                    base, group, tx, node = pvar.split("_")
                    devvar = "dev_" + pvar
                    ivar = "_".join(["IPDev", tx, node])
                    assert base == "p"
                    assert group in self.groups
                    assert node in self.nodes
                    assert devvar in self._pdev_vars
                    assert ivar in self._pdev_indicators

                    var = [devvar, pvar]
                    # NOTE: must be floats
                    coef = [-1., (n_groups - 1.) / n_groups]

                    # Iterate over all groups
                    other_groups = set(self.groups)
                    other_groups.remove(group)
                    # Add all other groups
                    for other in other_groups:
                        newvar = "_".join([base, other, tx, node])
                        assert newvar in self._p_vars
                        var.append(newvar)
                        coef.append(-1. / n_groups)
                    dev_constr.append([var, coef])
                    dev_constr_names.append("diffeq_" + pvar)

                    # Add the indicator constraint
                    # IPDev_tx_node = 0 => dev_p_group_tx_node = 0
                    name = '_'.join(["IndPDev", group, tx, node])
                    constr = cplex.SparsePair(ind=[devvar], val=[1.])
                    self.cpx.indicator_constraints.add(indvar=ivar, complemented=1, 
                                                    rhs=0., sense='E',
                                                    lin_expr=constr,
                                                    name=name)

        # Add the constraints for deviations from mean
        self.cpx.linear_constraints.add(lin_expr=dev_constr,
                                        senses=["E"] * len(dev_constr),
                                        rhs=[0.] * len(dev_constr),
                                        names=dev_constr_names)

    def add_cpx(self):
        """Create cplex MIQP problem."""
        # Initialize the Cplex object.
        self.cpx = cplex.Cplex()

        self.cpx.set_problem_type(cplex.Cplex.problem_type.MIQP)
        self.cpx.objective.set_sense(self.cpx.objective.sense.minimize)

        self._cpx_add_variables()
        self._cpx_add_mra_eqns()
        self._cpx_add_cnr_constraints()

        # ---------------------------------------------------------------------
        # Add quadratic part to objective function
        for ec in self._residuals_complete:
            self.cpx.objective.set_quadratic_coefficients(ec, ec, 1.-self._eta)

        for ei in self._residuals_incomplete:
            self.cpx.objective.set_quadratic_coefficients(ei, ei,
                                                          self._alpha * (1.-self._eta))

        # -------------------------------------------------------------------
        # Construct indicator constraints for presence of edge
        for rvar in self._rloc_vars:
            # rvar is expected to have form r_group_nodei_nodej
            rbase, grp, nd1, nd2 = rvar.split('_')
            # assert len(r_elements) == 4
            assert rbase == 'r'
            assert grp in self.groups
            assert nd1 in self.nodes and nd2 in self.nodes
            ivar = '_'.join(["I", nd1, nd2])
            assert ivar in self._indicators
            name = '_'.join(["Ind", nd1, nd2])
            constr = cplex.SparsePair(ind=[rvar], val=[1.])
            self.cpx.indicator_constraints.add(
                indvar=ivar,
                complemented=1,
                rhs=0., sense='E',
                lin_expr=constr,
                name=name
            )


###############################################################################
#
# Helper functions

def _generate_indicators(nodes, prior_network, base='I'):
    """Return list with indicator names (optional: based on prior network)."""
    ind_lst = []
    if prior_network:
        for edge in prior_network:
            ind_lst.append('_'.join([base, edge[0], edge[1]]))
    else:
        ind_lst = ["_".join([base, node_i, node_j]) for node_i, node_j in
                   itertools.product(nodes, nodes) if node_i != node_j]
    return ind_lst


def generate_rloc_symbols(rows, columns, prior_network, prefix=None):
    """"Generate matrix containing local response coefficients as symbols.

    Parameters
    ----------
    rows: list of nodes. This can be all nodes to get complete rloc, or
    (in)complete nodes to get submatrix

    nodes : tuple of strings

    prior_network : list of 2-tuples, optional

    Returns
    -------
    MxN np.array with dtype=sympy.Symbol with as ij-th elements r_nodei_nodej
    and on the diagonal: r_i_i
    """
    ncn = len(rows)  # rows of rloc (i.e. "downstream" nodes)
    nn = len(columns)  # columns of rloc (i.e. upstream and thus all nodes)
    mat = np.zeros((ncn, nn), dtype=sympy.Symbol)
    BASE = "r_"
    if prefix:
        assert isinstance(prefix, str)
        BASE = BASE + prefix + "_"

    for i in range(ncn):
        for j in range(nn):
            edge = (rows[i], columns[j])
            if rows[i] == columns[j]:
                mat[i][j] = sympy.Symbol('r_i_i')
            elif (prior_network is None) or (edge in prior_network):
                mat[i][j] = sympy.Symbol(BASE + rows[i] + '_' + columns[j])
    return mat


def generate_pert_symbols(nodes, cells, cell_annot_inv, tx_annot, prefix=None):
    """"Generate matrix containing local response coefficients as symbols.

    Parameters
    ----------
    rows: list of nodes. This can be all nodes to get complete p-matrix, or
    (in)complete nodes to get submatrix

    columns : list of cells

    cell_annot: dict mapping cells to treatments

    tx_annot: dict mapping treatments to perturbed nodes

    Returns
    -------
    MxN np.array with dtype=sympy.Symbol with as ij-th elements p_pertk_nodei
    """
    mat = np.zeros((len(nodes), len(cells)), dtype=sympy.Symbol)
    BASE = "p_"
    if prefix:
        assert isinstance(prefix, str)
        BASE = BASE + prefix + "_"
    for cell in cells:
        # Get perturbed nodes:
        tx = cell_annot_inv[cell]
        perturbed_node = tx_annot[tx]
        if perturbed_node in nodes:
            node_idx = nodes.index(perturbed_node)
            cell_idx = cells.index(cell)
            mat[node_idx][cell_idx] = BASE + tx + "_" + perturbed_node

    return mat


def estimate_alpha(scdata):
    """Returns mean of upper and lower bound estimate for best alpha value"""

    # If there are no incomplete nodes, there is no need to set alpha
    if len(scdata.nodes_incomplete) == 0:
        return 1.

    print("Estimating best value for alpha")
    # Get upper bound for alpha
    ub = estimate_alpha_upperbound(scdata)
    print("Estimated upper bound for alpha: " + str(ub))

    # Get lower bound for alpha
    lb = estimate_alpha_lowerbound(scdata)
    print("Estimated lower bound for alpha: " + str(lb))

    # Return average of both
    alpha = (ub + lb)/2
    print("Estimated alpha: " + str(alpha))
    return alpha


def estimate_alpha_upperbound(scd):
    """Estimate upper bound on alpha"""
    # Build problem without edges
    p = ScMraProblem(scd, alpha=1., prior_network=[])
    p.cpx.solve()
    # Get residuals of complete and incomplete nodes
    ec = p.cpx.solution.get_values(p._residuals_complete)
    ei = p.cpx.solution.get_values(p._residuals_incomplete)
    ub = np.square(ec).mean()/np.square(ei).mean()
    return ub


def estimate_alpha_lowerbound(scd):
    """Estimate lower bound on alpha"""
    # Build problem without edge-penalties

    p = ScMraProblem(scd, alpha=1, eta=0)
    p.cpx.solve()
    # Get residuals of complete and incomplete nodes
    ec = p.cpx.solution.get_values(p._residuals_complete)
    ei = p.cpx.solution.get_values(p._residuals_incomplete)
    lb = np.square(ec).mean()/np.square(ei).mean()
    return lb


def set_interactions_status(scproblem, interaction_list, status):
    """Force interaction to be absent/persent.

    Parameters:
    -----------
    scproblem: an ScMra or ScCnr object

    interaction_list: list of tuples
        tuples should have form (node_i, node_j): with n_j --> n_i

    status: {1, 0}
        1. Interaction is present, 0 absent
    """
    indicator_lst = ['_'.join(['I', n_i, n_j]) for n_i, n_j in interaction_list]

    for indicator in indicator_lst:
        assert indicator in scproblem.indicators, indicator + "not in list."
        # print("setting indicator " + indicator + " to " + str(status))
        set_indicator_status(scproblem.cpx, indicator, status)