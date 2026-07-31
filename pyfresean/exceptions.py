class PyFRESEANWarning(UserWarning):
    """Recoverable issue during PyFRESEAN analysis or transformations."""


class MissingBoxWarning(PyFRESEANWarning):
    """Trajectory frame has no periodic box dimensions."""


class MissingBeadMassWarning(PyFRESEANWarning):
    """CG bead masses were restored from the coarse-grain mapping."""


class TopologyFormatWarning(PyFRESEANWarning):
    """Topology file format may not preserve bead masses on reload."""
