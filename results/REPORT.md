# Results (FLEURS hi_in test, 418 utts)

WER/CER in %, 95% bootstrap CI over utterances; `indic` normaliser.

## Model x condition (WER)

| model | clean | white_10dB | white_0dB | babble_10dB | babble_0dB | reverb_0.6s | telephony |
|---|---|---|---|---|---|---|---|
| lora_aug | 25.0 | 39.0 | 63.5 | 33.5 | 87.6 | 39.3 | 28.1 |
| lora_clean | 25.8 | 42.2 | 71.6 | 35.4 | 94.1 | 41.6 | 29.0 |
| whisper-base | 127.2 | – | – | – | – | – | – |
| whisper-small | 66.6 | 84.6 | 104.7 | 81.1 | 125.1 | 80.4 | 66.4 |
| whisper-tiny | 140.3 | – | – | – | – | – | – |

## Clean-set detail

| model | WER | CER | off-script | halluc. | trunc. |
|---|---|---|---|---|---|
| lora_aug | 25.0 [23.7, 26.4] | 10.0 | 0.0% | 0.0% | 0.0% |
| lora_clean | 25.8 [24.1, 27.8] | 10.3 | 0.0% | 0.5% | 0.0% |
| whisper-base | 127.2 [119.4, 135.0] | 110.5 | 79.2% | 9.6% | 5.3% |
| whisper-small | 66.6 [62.6, 71.1] | 34.0 | 0.0% | 4.8% | 2.9% |
| whisper-tiny | 140.3 [128.7, 153.3] | 123.4 | 95.5% | 8.9% | 5.0% |

## Normaliser sensitivity (clean)

| model | raw | whisper_basic | indic | indic_lenient |
|---|---|---|---|---|
| lora_aug | 29.9 | 16.9 | 25.0 | 24.0 |
| lora_clean | 30.8 | 17.0 | 25.8 | 24.9 |
| whisper-base | 122.8 | 109.3 | 127.2 | 127.2 |
| whisper-small | 71.4 | 48.8 | 66.6 | 65.6 |
| whisper-tiny | 138.0 | 117.4 | 140.3 | 140.3 |

## Paired bootstrap (ΔWER = B − A, p = P(B not better))

| A → B | condition | ΔWER | p |
|---|---|---|---|
| whisper-small → lora_clean | clean | -40.7 | 0.000 |
| whisper-small → lora_clean | white_10dB | -42.4 | 0.000 |
| whisper-small → lora_clean | white_0dB | -33.0 | 0.000 |
| whisper-small → lora_clean | babble_10dB | -45.7 | 0.000 |
| whisper-small → lora_clean | babble_0dB | -31.0 | 0.000 |
| whisper-small → lora_clean | reverb_0.6s | -38.8 | 0.000 |
| whisper-small → lora_clean | telephony | -37.4 | 0.000 |
| whisper-small → lora_aug | clean | -41.6 | 0.000 |
| whisper-small → lora_aug | white_10dB | -45.6 | 0.000 |
| whisper-small → lora_aug | white_0dB | -41.2 | 0.000 |
| whisper-small → lora_aug | babble_10dB | -47.6 | 0.000 |
| whisper-small → lora_aug | babble_0dB | -37.4 | 0.000 |
| whisper-small → lora_aug | reverb_0.6s | -41.1 | 0.000 |
| whisper-small → lora_aug | telephony | -38.3 | 0.000 |
| lora_clean → lora_aug | clean | -0.9 | 0.056 |
| lora_clean → lora_aug | white_10dB | -3.2 | 0.000 |
| lora_clean → lora_aug | white_0dB | -8.2 | 0.000 |
| lora_clean → lora_aug | babble_10dB | -1.9 | 0.014 |
| lora_clean → lora_aug | babble_0dB | -6.5 | 0.002 |
| lora_clean → lora_aug | reverb_0.6s | -2.3 | 0.000 |
| lora_clean → lora_aug | telephony | -0.9 | 0.056 |
