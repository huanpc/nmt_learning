# NMT step-by-step learning

Final project for VietAI course (ML + DL)

## Documents

- [BLEU score](https://machinelearningmastery.com/calculate-bleu-score-for-text-python/)
- [Visualize Attention Matrix](https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb)
- Attention
  - [Paper](https://arxiv.org/abs/1409.0473)
  - [Explained 1 - VN](https://viblo.asia/p/machine-learning-attention-attention-attention-eW65GPJYKDO)

## Reports

#### Vi -> En

- Standard params with beam
    ```
    !python -m nmt.nmt.nmt \
    --attention=scaled_luong \
    --src=vi --tgt=en \
    --vocab_prefix=nmt_data/vocab \
    --train_prefix=nmt_data/train \
    --dev_prefix=nmt_data/tst2012 \
    --test_prefix=nmt_data/tst2013 \
    --out_dir=gdrive/My\ Drive/nmt_attention_model \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --infer_mode=beam_search\
    --beam_width=10\
    --num_layers=2 \
    --num_units=512 \
    --dropout=0.2 \
    --metrics=bleu \
    --encoder_type=bi\
    --decay_scheme=luong234
    ```

    *Res* (standard.ipynb)

    ```
    Best bleu, step 11000 lr 0.0625 step-time 0.00s wps 0.00K ppl 0.00 gN 0.00 dev ppl 9.88, dev bleu 21.7, test ppl 8.68, test bleu 24.4, Mon Jan 21 05:13:22 2019
    ```

- Adam optimizer with learning_rate=0.001 and beam
    
    ```
    --attention=scaled_luong \
    --src=vi --tgt=en \
    --vocab_prefix=nmt_data/vocab  \
    --train_prefix=nmt_data/train \
    --dev_prefix=nmt_data/tst2012  \
    --test_prefix=nmt_data/tst2013 \
    --out_dir=gdrive/My\ Drive/model_temp \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --infer_mode=beam_search\
    --beam_width=10\
    --num_layers=2 \
    --num_units=512 \
    --dropout=0.2 \
    --metrics=bleu \
    --encoder_type=bi\
    --decay_scheme=luong234
    ```

    *Res* (adam.ipynb)
    ```
    # Best bleu, step 4000 lr 6.25e-05 step-time 0.00s wps 0.00K ppl 0.00 gN 0.00 dev ppl 10.66, dev bleu 20.4, test ppl 9.55, test bleu 22.9, Tue Jan 22 02:56:35 2019
    ```

- Standard params with greedy
    
    *Res*
    ```
    # Best bleu, step 12000 lr 0.125 step-time 0.46s wps 11.90K ppl 4.83 gN 5.92 dev ppl 9.82, dev bleu 20.7, test ppl 8.36, test bleu 23.7, Tue Jan 22 06:36:15 2019
    Time: Tue Jan 22 04:46:04 2019 -> Tue Jan 22 06:35:35 2019
    ```

#### En -> Vi

- Standard params with greedy
  

- Standard params with beam
    
    ```
    --attention=scaled_luong \
    --src=en --tgt=vi \
    --vocab_prefix=nmt_data/vocab  \
    --train_prefix=nmt_data/train \
    --dev_prefix=nmt_data/tst2012  \
    --test_prefix=nmt_data/tst2013 \
    --out_dir=gdrive/My\ Drive/model_temp \
    --hparams_path=nmt/standard_hparams/iwslt15_beam.json
    ```
    
    *Res*
    ```
    # Best bleu, step 11000 lr 0.125 step-time 0.42s wps 13.27K ppl 4.36 gN 6.37 dev ppl 11.83, dev bleu 24.1, test ppl 10.92, test bleu 26.4, Tue Jan 22 09:40:25 2019
    Time: Tue Jan 22 07:54:47 2019 -> Tue Jan 22 09:39:20 2019
    ```

- Standard params with beam + adam 
    
    ```
    --attention=scaled_luong \
    --src=en --tgt=vi \
    --vocab_prefix=nmt_data/vocab  \
    --train_prefix=nmt_data/train \
    --dev_prefix=nmt_data/tst2012  \
    --test_prefix=nmt_data/tst2013 \
    --out_dir=gdrive/My\ Drive/model_temp \
    --hparams_path=nmt/standard_hparams/iwslt15_beam.json
    ```
    
    *Res*
    ```
    # Best bleu, step 11000 lr 0.125 step-time 0.42s wps 13.27K ppl 4.36 gN 6.37 dev ppl 11.83, dev bleu 24.1, test ppl 10.92, test bleu 26.4, Tue Jan 22 09:40:25 2019
    ```