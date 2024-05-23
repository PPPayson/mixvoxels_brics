from .llff_video import LLFFVideoDataset, SSDDataset
from .brics import BRICSDataset
from .render_circle import RenderCircleDataset
from .render_org import RenderOrgDataset
dataset_dict = {'ssd': SSDDataset, 'llffvideo':LLFFVideoDataset, 'brics':BRICSDataset, 'circle':RenderCircleDataset, 'org':RenderOrgDataset }
