from .early_fusion_gru_models import EarlyFusionCnnRNN
from .early_fusion_tsfm_models import EarlyFusionCnnTransformer
from .early_fusion_tsfm_models_modified import EarlyFusionCnnTransformer_Modified

REGISTERED_MODELS = {
    "EarlyFusionCnnTransformer": EarlyFusionCnnTransformer,
    "EarlyFusionCnnRNN": EarlyFusionCnnRNN,
    "EarlyFusionCnnTransformer_Modified": EarlyFusionCnnTransformer_Modified,
}
