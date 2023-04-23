
from src.bert_model import BertModel
from src.bert_model import BertConfig
import mindspore.common.dtype as mstype
bert_net_cfg_base = BertConfig(
    seq_length=128,
    vocab_size=21128,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    use_relative_positions=False,
    dtype=mstype.float32,
    compute_type=mstype.float16
)
bert_net_cfg_nezha = BertConfig(
    seq_length=128,
    vocab_size=21128,
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    use_relative_positions=True,
    dtype=mstype.float32,
    compute_type=mstype.float16
)
def create_network(name, *args, **kwargs):
    '''
    Create bert network for base and nezha.
    '''
    if name == 'bert_base':
        if "seq_length" in kwargs:
            bert_net_cfg_base.seq_length = kwargs["seq_length"]
        is_training = kwargs.get("is_training", False)
        return BertModel(bert_net_cfg_base, is_training, *args)
    if name == 'bert_nezha':
        if "seq_length" in kwargs:
            bert_net_cfg_nezha.seq_length = kwargs["seq_length"]
        is_training = kwargs.get("is_training", False)
        return BertModel(bert_net_cfg_nezha, is_training, *args)
    raise NotImplementedError(f"{name} is not implemented in the repo")
