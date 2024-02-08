from src.diffeomorphisms.vanilla_manifold import Vanilla_into_Manifold
from src.diffeomorphisms.simple_diffeomorphisms.stereographic_hyperboloid import StereographicHyperboloidChart
from src.manifolds.hyperboloid import Hyperboloid

class Vanilla_into_Hyperboloid(Vanilla_into_Manifold):
    def __init__(self, d, offset, orthogonal):
        super().__init__(Hyperboloid(d), offset=offset, orthogonal=orthogonal, chart=StereographicHyperboloidChart(d))
