# Baseline In NMT for English--Icelandic

The MIT License (MIT)

Copyright (c) 2020 Miðeind ehf

## Disclaimer

This is based on the GreynirT2T pipeline for NMT, it is no loner maintained since it's core Tensor2Tensor has been de facto abandoned for PyTorch. See the repoository [GreynirSeq](https://github.com/mideind/GreynirSeq) for future development on NMT for Icelandic.

## Intro

For serving with docker, store models under `conf/models/`. The baseline models available on CLARIN trained on a cleaned up and filtered variation of ParIce along with backtranslation data. One model for each direction English--Icelandic and Icelandic--English.

## Serve models over http

### Simple setup

Having installed docker and docker-compose, run

```bash
docker-compose up
```

and you will be able to query a translation server for translations from Icelandic to English and vice versa using e.g. curl in the following manner:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"model": "transformer-bt", "target": "en", "source": "is", "pgs": ["Hvað er klukkan?"]}' localhost:5005/translate.api
```

witch returns

```
[{"batch_prediction_key":[0],"outputs":"What time is it?","scores":-0.642857373}]
```

### In production

To serve this in production a proper webserver such as nginx should be set up as a proxy and https used since sensitive data might be translated.
