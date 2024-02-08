from src.diffeomorphisms.vanilla_manifold import Vanilla_into_Manifold
from src.diffeomorphisms.simple_diffeomorphisms.stereographic_sphere import StereographicSphereChart
from src.manifolds.sphere import Sphere

class Vanilla_into_Sphere(Vanilla_into_Manifold):
    def __init__(self, d, offset, orthogonal):
        super().__init__(Sphere(d), offset=offset, orthogonal=orthogonal, chart=StereographicSphereChart(d))