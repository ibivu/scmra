"""Classes to hold input data and problem description for single cell MRA."""


class ScData:
    """" Class to contain data from a perturbation experiment.

    Parameters
    ----------
    rglob : pandas.DataFrame
        Dataframe containing measurements of node activity (i.e. phospho-
        proteins). Rows correspond to nodes. Columns correspond to cells.
        Entries are the scaled diviations from the bulk mean.

    rtot: pandas.DataFrame
        Dataframe containing measurements of total node abundance (i.e. total
         protein). Rows correspond to nodes. Columns correspond to cells.
        Entries are the scaled diviations from the bulk mean. Should have the
        same columns and index as rglob.

    cell_annot: dict (optional), only used dataset contains treatments/
        perturbation experiments. Keys are treatments, values are cells. If 
        cell_annot is specified, tx_annot should also be specified.

    tx_annot: dict (optional), only used dataset contains treatments/perturbation
        experiments. Keys are treatments, values are the affected nodes. For 
        control treatments, use None.

    group_annot: dict (optional), used when cells in the experiment originate
        from different groups (e.g. cell lines or cell states), and the
        networks of these groups should be compared. Keys are cells, values are
        group.
    """

    def __init__(self, rglob, rtot,
                 cell_annot=None,
                 tx_annot=None,
                 group_annot=None):
        """Initialization."""
        self._rglob = rglob.copy(deep=True)
        self._rtot = rtot.copy(deep=True)
        # Check if cells match
        assert tuple(self._rglob.columns) == tuple(self._rtot.columns)
        self._cells = tuple(self._rglob.columns)
        assert "_" not in ''.join(self.cells), "_ are not allowed in cell\
            identifiers." 
        # Check if nodes match
        assert set(self._rtot.index).issubset(set(self._rglob.index))
        # Get nodes from columns
        self._nodes_complete = list(self._rtot.index)
        self._nodes_incomplete = [node for node in self._rglob.index
                                  if node not in tuple(self._rtot.index)]
        assert(set(self.nodes) == set(self._rglob.index))
        assert "_" not in ''.join(self.nodes), "_ are not allowed in node names"
        # Reorder rows of rglob
        if list(self._rglob.index) != self.nodes:
            print("Beware: Rows of perturbation data are reordered:\nold:\t" +
                  str(list(self._rglob.index)) + "\nnew:\t" + str(self.nodes))
            self._rglob = self._rglob.loc[self.nodes]

        # Add annotations of cells and treatment
        # Check consistency of cell and treatment annotation
        if cell_annot is not None:
            all_annotated_cells = [item for sublist in cell_annot.values() for
                                   item in sublist]
            assert len(all_annotated_cells) == len(set(all_annotated_cells))
            assert set(all_annotated_cells) == set(self.cells), "Some cells\
                lack treatment annotation"
            assert tx_annot is not None, "If cell_annot is specified, tx_annot\
                should also be specified."
            assert set(cell_annot.keys()) == set(tx_annot.keys()),\
                "Some treatments lack target annotation"
            assert set(tx_annot.values()).issubset(set(self.nodes+[None])),\
                "Invalid treatment target annotation."
            assert "_" not in "".join(cell_annot.keys()), "_ not allowed in \
                treatment names."
        self._cell_annot = cell_annot
        self._tx_annot = tx_annot

        if group_annot is not None:
            for var, val in group_annot.items():
                assert "_" not in var, var + " is illegal group name"
            self._group_annot = group_annot

            assert set(rglob.columns) == set(self.group_annot_inv.keys()), \
                "Some cells lack group annotation."

    @property
    def cells(self):
        return self._cells

    @property
    def nodes(self):
        """Get nodes, list of str indicating the measured epitopes."""
        return self._nodes_complete + self._nodes_incomplete

    @property
    def nodes_complete(self):
        """Get nodes for which both phosphosite and total protein is measured.
        Returns tuple of str indicating the measured epitopes."""
        return self._nodes_complete

    @property
    def nodes_incomplete(self):
        """Get nodes for which both phosphosite and total protein is measured.
        Returns list of str indicating the measured epitopes."""
        return self._nodes_incomplete

    @property
    def cells(self):
        """Get cells, list of str indicating the cell ids."""
        return self._cells

    @property
    def rglob(self):
        """Get dict of global response matrix (i.e. node activity measurements)
        ."""
        return self._rglob

    @property
    def rtot(self):
        """Get dict of global response matrix (i.e. node activity measurements)
        ."""
        return self._rtot

    @property
    def cell_annot(self):
        "Dict mapping treatments to cells. Keys are treatments"
        return self._cell_annot

    @property
    def cell_annot_inv(self):
        """Dict mapping cells to treatments. Inverse of cell_annot. Keys are
        treatments."""
        cell_annot_inv = dict()
        for tx, cells in self.cell_annot.items():
            for cell in cells:
                cell_annot_inv[cell] = tx
        return cell_annot_inv

    @property
    def tx_annot(self):
        "Dict mapping treatments to perturbed nodes"
        return self._tx_annot

    @property
    def treatments(self):
        return list(self._tx_annot.keys())

    @property
    def group_annot(self):
        "Dict mapping groups to cells. Keys are groups, values are list of cells"
        return self._group_annot

    @property
    def group_annot_inv(self): 
        "Dict mapping groups to cells. Dict with cells as keys, groups as vals."
        gai = dict()
        for group, cells in self.group_annot.items():
            for c in cells:
                gai[c] = group
        return gai

    @property
    def groups(self):
        """Get treatment/cell state groups. Tuple of strings."""
        return list(self.group_annot.keys())
