from qdiff.block_recon import block_reconstruction
from qdiff.layer_recon import layer_reconstruction, layer_reconstruction_modiff
from qdiff.quant_block import BaseQuantBlock
from qdiff.quant_layer import QuantModule
from qdiff.quant_model import QuantModel

# INT8 versions (recommended for production - faster inference!)
from qdiff.block_recon_int8 import block_reconstruction_int8
from qdiff.layer_recon_int8 import layer_reconstruction_int8, layer_reconstruction_modiff_int8
from qdiff.quant_block_int8 import BaseQuantBlockINT8
from qdiff.quant_layer_int8 import QuantModuleINT8
from qdiff.quant_model_int8 import QuantModelINT8

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
