"""LongMemEval-MC10 pipeline — reuses locomo_mc10's 10-choice letter-match reward."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tasks.locomo_mc10.pipeline_mc10 import MC10Pipeline as _MC10Pipeline


class LMEMC10Pipeline(_MC10Pipeline):
    """Identical behaviour; named separately for wandb / checkpoint clarity."""
    pass
