from .maxpooling_mil import MaxPoolingMIL
from .meanpooling_mil import MeanPoolingMIL
from .mil_rnn import MILRNN, MIX_RNN
from .abmil import GatedAttention, ABMIL, MixABMIL, IRL_ABMIL, Stable_ABMIL
from .clam import CLAMSB, CLAMMB
from .Supvision import Supvision
#from .transmil import stableTransMIL, TransMIL, MixTransMILTrainer


model_fn = {
    'max_pooling_mil': MaxPoolingMIL, 'mean_pooling_mil': MeanPoolingMIL,
    'mil_rnn': MILRNN,
    'ab_mil_gate': GatedAttention,
    'clam_sb': CLAMSB, 'clam_mb': CLAMMB,
    'origin_abmil': ABMIL,
    'mix_abmil': IRL_ABMIL,
    'mix_rnn': MIX_RNN,
    'supvision': Supvision,
    'Stable_ABMIL': Stable_ABMIL
    #'stable_transmil': stableTransMIL,
    #'transmil': TransMIL,
    #'mixtransmil': MixTransMILTrainer
    }
    