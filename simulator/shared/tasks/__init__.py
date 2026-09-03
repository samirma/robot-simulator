"""Tasks an engine can stage into whatever scene it has compiled.

A task here is two things bolted together: the *staging* -- the objects and cameras it
needs, injected into an engine's `MjSpec` before compile -- and the *arbiter*, which
measures the world afterwards and answers whether the task is done. Both halves are
engine-neutral, so the same task runs in a MolmoSpaces house and a RoboCasa kitchen and
a client cannot tell which one it is connected to.
"""
