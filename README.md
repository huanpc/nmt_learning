# NMT step-by-step learning

Final project for VietAI course (ML + DL)

## Documents

- [BLEU score](https://machinelearningmastery.com/calculate-bleu-score-for-text-python/)
- [Visualize Attention Matrix](https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb)
- Attention
  - [Paper](https://arxiv.org/abs/1409.0473)
  - [Explained 1 - VN](https://viblo.asia/p/machine-learning-attention-attention-attention-eW65GPJYKDO)

## Reports

- 24k_epoch, infer_mode=beam, num_layers=4, num_unit=128 (3.ipynb)

    ``` 
        --attention=scaled_luong \
            --src=vi --tgt=en \
            --vocab_prefix=nmt_data/vocab  \
            --train_prefix=nmt_data/train \
            --dev_prefix=nmt_data/tst2012  \
            --test_prefix=nmt_data/tst2013 \
            --out_dir=nmt_attention_model \
            --num_train_steps=24000 \
            --steps_per_stats=100 \
            --infer_mode=beam_search\
            --beam_width=10\
            --num_layers=4 \
            --num_units=128 \
            --dropout=0.2 \
            --metrics=bleu 
    ```

    result: 
    ```
    Best bleu, step 24000 lr 1 step-time 0.48s wps 11.42K ppl 14.60 gN 5.06 dev ppl 14.69, dev bleu 16.5, test ppl 13.43, test bleu 19.1, Fri Jan 18 16:25:06 2019
    ```


- 12k_epoch, 512 unit,  num_layer=2, steps = 24k (4.ipynb)
  ```
  --attention=scaled_luong \
    --src=vi --tgt=en \
    --vocab_prefix=nmt_data/vocab  \
    --train_prefix=nmt_data/train \
    --dev_prefix=nmt_data/tst2012  \
    --test_prefix=nmt_data/tst2013 \
    --out_dir=nmt_attention_model \
    --num_train_steps=24000 \
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

  - Standard params by tut: iwslt15.json
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