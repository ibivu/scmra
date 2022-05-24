"""Classes and functions to store and analyze ScMra optimization results."""
import numpy as np
import pandas as pd
import re
from scmra.scdata import ScData
import networkx as nx
import cplex
import numpy as np
import sympy
import itertools
from scmra.scdata import ScData

###############################################################################
#
# Classes


class ScMraResult(ScData):
    """Class to hold information from a sc-MRA network reconstruction.

    Inherits from ScData.

    Parameters
    ---------
    scproblem :

    Attributes
    ----------
    """

    def __init__(self, scproblem):
        """Initialization from ScMraProblem object."""
        # From input
        super().__init__(scproblem.rglob, scproblem.rtot)
        self.vardict = _get_vardict_from_cpx(scproblem.cpx, solidx=0)
        self.imap = _get_imap_from_scp(scproblem)
        self.rloc = _get_rloc_from_scp(scproblem)
        self.stot = _get_stot_from_scp(scproblem, self.vardict)
        if scproblem.tx_annot is not None:
            self.pertstrength = {var: self.vardict[var] for var in
                                 scproblem._p_vars}
        self.residuals_complete = _get_residuals_from_scp(
            scproblem, self.vardict, 'complete')
        self.residuals_incomplete = _get_residuals_from_scp(
            scproblem, self.vardict, 'incomplete')
        self.hyperparameters = {
            "alpha": scproblem._alpha,
            "eta": scproblem._eta}
        self.annotation = _get_tx_annot_from_scp(scproblem)

        #'new' rloc_i (dict with key=treatment and values=rloc values) of inhibition which is inhibition coef = 1-(rloc_i/rloc)
        self.rloc_i = _get_rloc_i_from_scp(scproblem)

    @property
    def n_edges(self):
        "number of edges in solution"
        return self.imap.sum().sum()

    @property
    def mssr(self):
        """Mean sum of squares of residuals."""
        mean_ssr = 0
        n_res = np.size(self.residuals_complete) + \
            np.size(self.residuals_incomplete)
        mean_ssr += np.sum(np.array(np.square(self.residuals_complete)))/n_res
        mean_ssr += np.sum(np.array(np.square(self.residuals_incomplete)))/n_res
        return mean_ssr

    @property
    def mssr_complete(self):
        """Mean sum of squares of residuals of complete nodes."""
        mean_ssr = 0
        n_res = np.size(self.residuals_complete)
        mean_ssr += np.sum(np.array(np.square(self.residuals_complete)))/n_res
        return mean_ssr

    @property
    def mssr_incomplete(self):
        """Mean sum of squares of residuals of complete nodes."""
        mean_ssr = 0
        n_res = np.size(self.residuals_incomplete)
        mean_ssr += np.sum(np.array(np.square(self.residuals_incomplete)))/n_res
        return mean_ssr

    @property
    def graph(self):
        """Generate graph from complete solution.

        This graph includes perturbations as nodes, and edges from
        perturbations to the affected nodes.

        input:
        * sol: A ScCnrResult object.

        output: An networkx DiGraph object
        """
        g = nx.DiGraph()
        # Go over all variables from the solution.
        for var, val in self.vardict.items():
            # select the local response coefficients
            if var.startswith("r_") and var != "r_i_i":
                base, n1, n2 = var.split("_")
                from_n = n2
                to_n = n1
                # Only add edge if in network
                if self.vardict['_'.join(['I', to_n, from_n])]:
                    # Color according to sign of perturbation.
                    if val > 0:
                        col = 'green'
                        sign = 'positive'
                    else:
                        col = 'red'
                        sign = 'negative'
                    g.add_edge(from_n, to_n, weight=val, color=col,
                               penwidth=abs(val))
        return g

        


class ScCnrResult(ScData):
    """Class to hold information from a sc-MRA network reconstruction.

    Inherits from ScData.

    Parameters
    ---------
    scproblem :

    Attributes
    ----------
    """

    def __init__(self, scproblem):
        """Initialization from ScCnrProblem object."""
        # From input
        super().__init__(
            scproblem.rglob, scproblem.rtot,
            group_annot=scproblem.group_annot
        )
        self.vardict = _get_vardict_from_cpx(scproblem.cpx, solidx=0)
        self.imap = _get_imap_from_scp(scproblem)
        self.rloc = {
            group: _get_rloc_from_scp(scproblem) for group in scproblem.groups
        }
        self.stot = {
            group: _get_stot_from_scp(scproblem, self.vardict, group)
            for group in scproblem.groups
        }
        self.residuals_complete = _get_residuals_from_scp(
            scproblem, self.vardict, 'complete')
        self.residuals_incomplete = _get_residuals_from_scp(
            scproblem, self.vardict, 'incomplete')
        self.hyperparameters = {
            "alpha": scproblem._alpha,
            "eta": scproblem._eta,
            "theta": scproblem._theta}

    @property
    def allowed_deviations(self):
        """Return indicator variables relating to difference between lines."""
        return {var: self.vardict[var] for var, val in self.vardict.items()
                if var.startswith(('IDev_', 'ISDev_', 'IPDev_'))}

    @property
    def deviations_overview(self):
        """Summarize non-zero deviations.

        Returns
        -------
        pd.DataFrame
        """
        indicators = [key for key, val in
                      self.allowed_deviations.items() if val == 1]
        df = pd.DataFrame(columns=list(self.groups) + ['mean'],
                          index=indicators)
        names = []

        for i in indicators:
            info = i.split('_')
            assert info[0] in ['IDev', 'ISDev', 'IPDev'], str(
                i) + ' has unexpected form'
            if info[0] == 'IDev':
                vars_lst = ['_'.join(['r', group] + info[1:]) for group in
                            self.groups]
                vars_vals = [self.vardict[v] for v in vars_lst]
                vars_vals.append(np.mean(vars_vals))
                df.loc[i] = vars_vals
                names += [('_'.join(['r'] + info[1:]))]
            elif info[0] == 'ISDev':
                vars_lst = ['_'.join(['s', group] + info[1:]) for
                            group in self.groups]
                vars_vals = [self.vardict[v] for v in vars_lst]
                vars_vals.append(np.mean(vars_vals))
                df.loc[i] = vars_vals
                names.append('_'.join(['s'] + info[1:]))
            elif info[0] == 'IPDev':
                vars_lst = ['_'.join(['p', group] + info[1:]) for
                            group in self.groups]
                vars_vals = [self.vardict[v] for v in vars_lst]
                vars_vals.append(np.mean(vars_vals))
                df.loc[i] = vars_vals
                names.append('_'.join(['p'] + info[1:]))
        df.index = names
        return df.sort_index()
        
    @property
    def n_edges(self):
        "number of edges in solution"
        return self.imap.sum().sum()

    @property
    def n_deviations(self):
        "number of edges that differ between groups"
        return sum(self.allowed_deviations.values())
    
    @property
    def mssr(self):
        """Mean sum of squares of residuals."""
        mean_ssr = 0
        n_res = np.size(self.residuals_complete) + \
            np.size(self.residuals_incomplete)
        mean_ssr += np.sum(np.array(np.square(self.residuals_complete)))/n_res
        mean_ssr += np.sum(np.array(np.square(self.residuals_incomplete)))/n_res
        return mean_ssr

##############################################################################
#
# Helper functions (private)


# For extracting information from cplex object ------------------------------
def _get_vardict_from_cpx(cpx, solidx):
    """Return dict with var: value as entries, for selected solution."""
    var_names = cpx.variables.get_names()
    var_vals = cpx.solution.pool.get_values(solidx, var_names)
    return dict(zip(var_names, var_vals))


def _get_imap_from_scp(scp, solidx=0):
    """
    Returns: interaction map, square np.array containing 0s and 1s.

    0: no interaction
    1: interaction
    """
    assert scp.cpx.solution.is_primal_feasible()
    assert solidx < scp.cpx.solution.pool.get_num()

    allvars = scp.cpx.variables.get_names()
    vars_lst = [var for var in allvars if var.startswith('I_')]
    vars_vals = scp.cpx.solution.pool.get_values(solidx, vars_lst)
    vars_dict = dict(zip(vars_lst, vars_vals))

    mat = np.array([0] * len(scp.nodes) **
                   2).reshape(len(scp.nodes), len(scp.nodes))
    reg_sp = re.compile(
        '(' + '|'.join(scp.nodes) + ')_(' + '|'.join(scp.nodes) + ')$')
    for key, value in vars_dict.items():
        # Extract nodes from key
        sps = re.findall(reg_sp, key)
        assert len(sps) == 1, key + " has unexpected form"
        sps = sps[0]
        assert len(sps) == 2, key + " has unexpected form"
        i = scp.nodes.index(sps[0])
        j = scp.nodes.index(sps[1])
        assert '_'.join(['I', sps[0], sps[1]]) in vars_lst
        mat[i][j] = value

    df = pd.DataFrame(mat)
    df.index = scp.nodes
    df.columns = scp.nodes

    return df


def _get_rloc_from_scp(scp, prefix=None, solidx=0):
    """Construct local response matrix from cplex solution.

    Parameter
    ---------
    scp :  ScMraProblem object

    nodes : Tuple of nodes in the reconstructed network

    prefix : str. (optional)
        Used if rloc is part of cell line panel

    solidx : int. (optional)
        Used to select solution from solution pool

    Returns
    -------
    Square pandas.DataFrame
    """

    base = "r"
    if prefix:
        assert isinstance(prefix, str)
        base = "_".join([base, prefix])
    assert scp.cpx.solution.is_primal_feasible()
    assert solidx < scp.cpx.solution.pool.get_num()

    allvars = scp.cpx.variables.get_names()
    vars_lst = [var for var in allvars if var.startswith(base + '_')]
    vars_vals = scp.cpx.solution.pool.get_values(solidx, vars_lst)
    vars_dict = dict(zip(vars_lst, vars_vals))

    if 'r_i_i' in vars_dict.keys():
        del vars_dict['r_i_i']

    mat = -1. * np.identity(len(scp.nodes))

    reg_sp = re.compile(
        '(' + '|'.join(scp.nodes) + ')_(' + '|'.join(scp.nodes) + ')$')
    for key, value in vars_dict.items():
        sps = re.findall(reg_sp, key)
        assert len(sps) == 1, key + " has unexpected form"
        sps = sps[0]
        assert len(sps) == 2, key + " has unexpected form"
        mat[scp.nodes.index(sps[0])][scp.nodes.index(sps[1])] = value

    df = pd.DataFrame(mat)
    df.index = scp.nodes
    df.columns = scp.nodes

    return df


def _get_rloc_i_from_scp(scp, prefix=None, solidx=0):

    pertubRlocDict = {}
    if( (scp.tx_annot is not None) and (scp._modelPertRloc or scp._modelPertNode)):
        for tx,node in scp.tx_annot.items():

            if(tx == 'ctr'): continue
            base = "INH_" + str(tx)
            if prefix:
                assert isinstance(prefix, str)
                base = "_".join([base, prefix])
            assert scp.cpx.solution.is_primal_feasible()
            assert solidx < scp.cpx.solution.pool.get_num()

            allvars = scp.cpx.variables.get_names()
            vars_lst = [var for var in allvars if var.startswith(base + '_')]

            vars_vals = scp.cpx.solution.pool.get_values(solidx, vars_lst)
            vars_dict = dict(zip(vars_lst, vars_vals))

            if 'r_i_i' in vars_dict.keys():
                del vars_dict['r_i_i']

            mat = -1. * np.identity(len(scp.nodes))

            reg_sp = re.compile(
                '(' + '|'.join(scp.nodes) + ')_(' + '|'.join(scp.nodes) + ')$')
            for key, value in vars_dict.items():
                sps = re.findall(reg_sp, key)
                assert len(sps) == 1, key + " has unexpected form"
                sps = sps[0]
                assert len(sps) == 2, key + " has unexpected form"
                mat[scp.nodes.index(sps[0])][scp.nodes.index(sps[1])] = value

            df = pd.DataFrame(mat)
            df.index = scp.nodes
            df.columns = scp.nodes

            pertubRlocDict[tx] = df
    return(pertubRlocDict)

def _get_residuals_from_scp(scp, vardict, which_nodes='all', 
                            prefix=None, solidx=0):
    """Return data-frame containing residuals of nodes with both phospho and
    total protein measured."""
    assert scp.cpx.solution.is_primal_feasible()
    assert solidx < scp.cpx.solution.pool.get_num()
    assert which_nodes in ['complete', 'incomplete', 'all']
    if which_nodes == 'all':
        nodes = scp.nodes
        residuals = np.array(scp.residuals_symbols)
    elif which_nodes == 'complete':
        nodes = scp.nodes_complete
        residuals = np.array(scp.residuals_complete_symbols)
    else:
        nodes = scp.nodes_incomplete
        residuals = np.array(scp.residuals_incomplete_symbols)

    for res in np.nditer(residuals, flags=["refs_ok", "zerosize_ok"],
                         op_flags=["readwrite"]):
        res[...] = vardict[str(res)]

    df = pd.DataFrame(
        residuals,
        index=nodes, columns=scp.cells
    )
    return df


def _get_stot_from_scp(scp, vardict, group=None, solidx=0):
    assert scp.cpx.solution.is_primal_feasible()
    assert solidx < scp.cpx.solution.pool.get_num()
    if group:
        stot = np.array(scp.stot_symbols[group])
    else:
        stot = np.array(scp.stot_symbols)

    if(stot.size != 0):
        for s in np.nditer(stot, flags=["refs_ok"],  op_flags=["readwrite"]):
            if str(s) in scp._s_vars:
                s[...] = vardict[str(s)]

    df = pd.DataFrame(stot,
                      index=scp.nodes_complete, columns=scp.nodes_complete
                      )
    return df

def _get_tx_annot_from_scp(scp):
    return scp.tx_annot
