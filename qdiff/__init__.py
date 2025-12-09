from qdiff.block_recon import block_reconstruction
from qdiff.layer_recon import layer_reconstruction, layer_reconstruction_modiff
from qdiff.quant_block import BaseQuantBlock
from qdiff.quant_layer import QuantModule
from qdiff.quant_model import QuantModel

# INT4 versions (following MoDiff paper - 3-4 bit activation quantization)
from qdiff.quant_layer_int4 import (
    UniformAffineQuantizerINT4,
    QuantModuleINT4,
    convert_to_int4_module,
)
from qdiff.quant_model_int4 import (
    QuantModelINT4,
    recon_model_int4,
    recon_model_modiff_int4,
)
