from .cbam import CBAM
from .da_att import Self_Att
from .ema import EMA
from .bam import BAM


attn_types = {
            'cbam': CBAM,
            'self': Self_Att,
            'ema': EMA,
            'bam': BAM
        }