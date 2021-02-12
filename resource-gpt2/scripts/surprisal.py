import tensorflow as tf

import model

# maybe needs to be None?
BATCH_SIZE = 1

def get_per_subword_surprisal(*, corpus, hparams, encoder):
    start_token = encoder.encoder['<|endoftext|>']
    context = tf.fill([BATCH_SIZE, 1], start_token)

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=BATCH_SIZE))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('get_per_word_surprisal'):
        # word is a list of integers (encoded chunks)
        def body(corpus, i, past, prev, surprisals):
            # chunk should be a scalar here
            chunk = corpus[i]
            next_outputs = step(hparams, prev, past=past)
            # dimension is batch_size x vocab_size
            logits = next_outputs['logits'][:, -1, :]
            softmax = tf.nn.softmax(logits)
            surp = tf.math.scalar_mul(-1, tf.math.log(softmax[0, chunk]))
            # TODO assuming here that batch size is 1.
            # find a better solultion
            return [
                corpus,
                tf.add(i, 1),
                next_outputs['presents'] if past is None else tf.concat([past, next_outputs['presents']], axis=-2),
                tf.reshape(chunk, [1, 1]),
                tf.concat([surprisals, tf.reshape(surp, [1, 1])], axis=1),
            ]

        corpus = tf.constant(corpus)
        i = tf.constant(0)
        corpus, i, past, prev, surprisals = body(corpus, i, None, context, tf.constant([[]]))

        corpus_length = corpus.shape[0].value

        def cond(corpus, i, past, prev, surprisals):
            return tf.less(i, corpus_length)

        _, _, _, _, surprisals = tf.while_loop(
            cond=cond, body=body,
            loop_vars=[corpus, i, past, prev, surprisals],
            shape_invariants=[
                corpus.get_shape(),
                i.get_shape(),
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=BATCH_SIZE)),
                tf.TensorShape([BATCH_SIZE, None]),
                tf.TensorShape([BATCH_SIZE, None]),
            ],
            back_prop=False,
        )

        return surprisals
