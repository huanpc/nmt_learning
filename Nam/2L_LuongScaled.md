# 2L LuongScaled

## HPARAMS

    --attention:scaled_luong \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_units=128 --unit_type=lstm \
    --dropout=0.2 --num_layers=2 \

    rest is default

## RESULT

```log
# Fri Jan 18 11:53:54 > 13:40:35
    step-time 0.26s wps 21.86K ppl 14.77 gN 5.49
    dev ppl 13.85, dev bleu 16.2
    test ppl 12.86, test bleu 17.6
```
