"""Public interface to the Core GPT model"""

import tensorflow as tf
import gpt_core_v2
import codec

def top_k_logits(logits, k):
    """top k logits"""
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )

def get_encoder(model_name, models_dir):
    """extend the codec encoder"""
    return codec.get_encoder(model_name, models_dir)

def default_hparams():
    """default hparams"""
    return gpt_core_v2.default_hparams()

def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch, _ = logits.shape.as_list()
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf.stack([
        tf.range(0, batch),
        # number of indices to include
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
    ], axis=-1)
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,
    )

@tf.function
def submit_query(
    hparams=gpt_core_v2.default_hparams(),
    length=50,
    model_name='124M',
    model_dir='../models',
    start_token=None,
    batch_size=None,
    context="Hello, how are you today?",
    temperature=1,
    top_k=0,
    top_p=1
):
    """submit a query to the model"""

    codec_instance = codec.get_encoder(model_name,model_dir)

    context = codec_instance.encode(context) # [15496, 11, 703, 389, 345, 1909, 30]
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    def determine_length(provided_length=50, max_length=50):
        if provided_length is None:
            return max_length
        if provided_length > max_length:
            print(f"Warning: Provided length ({provided_length}) exceeds maximum length ({max_length}). Using maximum length.")
            return max_length
        return provided_length

    length = determine_length(provided_length=length, max_length=hparams.n_ctx)

    def step(hparams, tokens, past=None):
        lm_output = gpt_core_v2.model(hparams=hparams, input_tokens=tokens, past=past, reuse=True)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(gpt_core_v2.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        def body(past, prev, output):
            next_outputs = step(hparams, prev, past=past)
            logits = next_outputs['logits'][:, -1, :] / tf.compat.v1.to_float(temperature)
            logits = top_k_logits(logits, k=top_k)
            logits = top_p_logits(logits, p=top_p)
            samples = tf.compat.v1.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            return [
                next_outputs['presents'] if past is None else tf.concat([past, next_outputs['presents']], axis=-2),
                samples,
                tf.concat([output, samples], axis=1)
            ]

        past, prev, output = body(None, context, context)

        def cond(*args):
            return True

        _, _, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length - 1,
            loop_vars=[
                past,
                prev,
                output
            ],
            shape_invariants=[
                tf.TensorShape(gpt_core_v2.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )

        # Decode the generated output
        text_completion = codec_instance.decode(tokens[0])

        return text_completion
"""
class GPT2Model(tf.keras.Model):
    def __init__(self, hparams):
        super(GPT2Model, self).__init__()
        # Initialize your model layers here based on hparams
        # For example:
        self.embed = tf.keras.layers.Embedding(hparams.vocab_size, hparams.n_embd)
        self.transformer_blocks = [
            TransformerBlock(hparams) for _ in range(hparams.n_layer)
        ]
        self.fc = tf.keras.layers.Dense(hparams.vocab_size)

    def call(self, inputs):
        x = self.embed(inputs)
        for block in self.transformer_blocks:
            x = block(x)
        return self.fc(x)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, hparams):
        super(TransformerBlock, self).__init__()
        # Initialize transformer block layers
        # (attention, feed forward, layer norm, etc.)

    def call(self, inputs):
        # Implement the forward pass of the transformer block
        pass

def create_model(hparams):
    return GPT2Model(hparams)
"""
