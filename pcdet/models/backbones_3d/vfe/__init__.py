from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE, Radar7PillarVFE
from .dynamic_mean_vfe import DynamicMeanVFE
from .dynamic_pillar_vfe import DynamicPillarVFE, DynamicPillarVFESimple2D
from .dynamic_voxel_vfe import DynamicVoxelVFE
from .image_vfe import ImageVFE
from .vfe_template import VFETemplate

__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'ImageVFE': ImageVFE,
    'DynMeanVFE': DynamicMeanVFE,
    'DynPillarVFE': DynamicPillarVFE,
    'DynamicPillarVFESimple2D': DynamicPillarVFESimple2D,
    'DynamicVoxelVFE': DynamicVoxelVFE,
    'Radar7PillarVFE': Radar7PillarVFE,
    'Radar8PillarVFE': Radar7PillarVFE,
}