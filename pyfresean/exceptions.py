class pyFRESEANWarning(UserWarning):
    """Recoverable issue during pyFRESEAN analysis or transformations."""


class MissingBoxWarning(pyFRESEANWarning):
    """Trajectory frame has no periodic box dimensions."""


class MissingVelocityWarning(pyFRESEANWarning):
    """Trajectory frame has no velocities."""


class MissingBeadMassWarning(pyFRESEANWarning):
    """CG bead masses were restored from the coarse-grain mapping."""


class TopologyFormatWarning(pyFRESEANWarning):
    """Topology file format may not preserve bead masses on reload."""


class PlumedAllAtomWarning(pyFRESEANWarning):
    """A PLUMED DIRECTION file was given coarse-grained mode vectors."""
