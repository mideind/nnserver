#!/usr/bin/env python3
"""
    Reynir: Natural language processing for Icelandic

    Neural Network Query Client

    Copyright (C) 2018 Miðeind ehf.

       This program is free software: you can redistribute it and/or modify
       it under the terms of the GNU General Public License as published by
       the Free Software Foundation, either version 3 of the License, or
       (at your option) any later version.
       This program is distributed in the hope that it will be useful,
       but WITHOUT ANY WARRANTY; without even the implied warranty of
       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
       GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see http://www.gnu.org/licenses/.


    This module implements a server that provides an Icelandic text interface to
    a Tensorflow model server running a text-to-parse-tree neural network.

    Example usage:
    python nnserver.py -lh 0.0.0.0 -lp 8080 -mh localhost -mp 9001

    To curl a running nnserver try:
    curl --header "Content-Type: application/json" \
        --request POST \
        http://localhost:8080/translate.api \
        --data '{"pgs":["Hvernig komstu þangað?"],"signature_name":"serving_default"}'

    To test a running model server directly, try:
    curl --header "Content-Type: application/json" \
        --request POST \
        http://localhost:8080/v1/models/transformer:predict \
        --data '{"instances":[{"input":{"b64":"CiAKHgoGaW5wdXRzEhQaEgoQpweFF0oOBsI+EclIBP4cAQ=="}}],"signature_name":"serving_default"}'

    To test a running model server directly and receive gRPC error feedback (more informative than RESTful):
    cd tf/lib/python3.5/site-packages
    python tensor2tensor/serving/query.py \
        --server 'localhost:8081' \
        --servable_name transformer \
        --data_dir ~/t2t_data \
        --t2t_usr_dir ~/t2t_usr \
        --problem translate_enis16k_rev \
        --inputs_once "Kominn."
"""

import base64
import json
import os
import requests
import itertools

from tensor2tensor.data_generators import text_encoder
from tensorflow.core.example import feature_pb2
from tensorflow.core.example import example_pb2

from flask import Flask, jsonify, request

from nnserver import _ENIS_VOCAB, _ONMT_EN_VOCAB, _ONMT_IS_VOCAB
from nnserver.composite_encoder import CompositeTokenEncoder

from subword_nmt import apply_bpe

EOS_ID = text_encoder.EOS_ID
PAD_ID = text_encoder.PAD_ID

app = Flask(__name__)


MODEL_NAMES = {
    "transformer": {
        "is-en": "translate_enis16k_v4_rev-avg-ckpt-2.10M",
        "en-is": "translate_enis16k_v4.baseline-avg-ckpt-2.25M"
    },
    "bilstm": {
        "en-is": "translate_enis16k_v4.onmt-bilstm",
        "is-en": "translate_enis16k_v4.onmt-bilstm_rev"
    }
}

class _SubwordNmtEncoder:
    """Wrap subword-nmt's BPE encoder with Tensor2tensors api"""

    def __init__(self, path):
        with open(path, "r") as fp:
            self._bpe = apply_bpe.BPE(fp)

    def encode(self, text):
        return self._bpe.process_line(text)

    def decode(self, flat_text):
        res = flat_text.replace("@@ ", "")
        if len(res) > 1 and flat_text[-2] == "@@":
            return res[:-2]
        return res

    def decode_list(self, flat_text):
        return flat_text

class NnServer:
    """ Client that mimics the HTTP RESTful interface of
        a tensorflow model server, but accepts plain text. """

    _tfms_version = "v1"
    _model_name = "transformer"
    _verb = "predict"
    src_enc = None
    tgt_enc = None

    @classmethod
    def request(cls, pgs, tgt_pgs=None, model_name=None):
        """ Send serialized request to remote model server """

        if model_name is None:
            model_name = cls._model_name

        ms_host = os.environ.get("MS_HOST", app.config.get("out_host"))
        ms_port = os.environ.get("MS_PORT", app.config.get("out_port"))

        url = "http://{host}:{port}/{version}/models/{model}:{verb}".format(
            port=ms_port,
            host=ms_host,
            version=cls._tfms_version,
            model=model_name,
            verb=cls._verb,
        )
        payload = cls.package_data(pgs, tgt_pgs)
        headers = {"content-type": "application/json"}

        app.logger.debug(payload)

        resp = requests.post(url, json=payload)
        resp.raise_for_status()

        obj = json.loads(resp.text)
        results = cls.extract_results(
            obj, pgs, tgt_pgs=tgt_pgs, src_enc=cls.src_enc, tgt_enc=cls.tgt_enc
        )
        return results

    @classmethod
    def extract_results(
        cls, resp_json_obj, pgs, tgt_pgs=None, src_enc=None, tgt_enc=None
    ):
        src_enc = src_enc or cls.src_enc
        tgt_enc = tgt_enc or cls.tgt_enc

        def process_response_instance(instance, src_enc=src_enc, tgt_enc=tgt_enc):
            scores = instance["scores"]
            output_ids = instance["outputs"]

            app.logger.debug("scores: " + str(scores))
            app.logger.debug("output_ids: " + str(output_ids))

            # Strip padding and eos token
            length = len(output_ids)
            pad_start = output_ids.index(PAD_ID) if PAD_ID in output_ids else length
            eos_start = output_ids.index(EOS_ID) if EOS_ID in output_ids else length
            sent_end = min(pad_start, eos_start)
            outputs = tgt_enc.decode(output_ids[:sent_end])

            app.logger.debug(
                "tokenized and depadded: "
                + str(tgt_enc.decode_list(output_ids[:sent_end]))
            )
            app.logger.info(outputs)

            instance["outputs"] = outputs
            return instance

        predictions = resp_json_obj["predictions"]
        results = [
            process_response_instance(inst) for (inst, sent) in zip(predictions, pgs)
        ]
        return results

    @classmethod
    def package_data(cls, pgs, tgt_pgs=None):

        def serialize_to_instance(
            src_segment, src_enc=None, tgt_segment=None, tgt_enc=None
        ):
            """ Encodes a single sentence into the format expected by the RESTful
                interface of tensorflow_model_server running an exported tensor2tensor
                transformer translation model
            """

            src_enc = src_enc or cls.src_enc
            tgt_enc = tgt_enc or cls.tgt_enc

            input_ids = src_enc.encode(src_segment) + [EOS_ID]
            app.logger.info("input_segment: " + src_segment)
            app.logger.debug("input_subtokens: " + str(src_enc.decode_list(input_ids)))
            app.logger.debug("input_ids: " + str(input_ids))

            int64_list = feature_pb2.Int64List(value=input_ids)
            feature = feature_pb2.Feature(int64_list=int64_list)
            feature_map = {"inputs": feature}

            if tgt_segment is not None:
                tgt_ids = tgt_enc.encode(tgt_segment) + [EOS_ID]
                app.logger.info("target_segment: " + tgt_segment)
                app.logger.debug("target_subtokens: " + str(tgt_enc.decode_list(tgt_ids)))
                app.logger.debug("target_ids: " + str(tgt_ids))
                tgt_int64_list = feature_pb2.Int64List(value=tgt_ids)
                tgt_feature = feature_pb2.Feature(int64_list=tgt_int64_list)
                feature_map["targets"] = tgt_feature

            features = feature_pb2.Features(feature=feature_map)
            example = example_pb2.Example(features=features)

            b64_example = base64.b64encode(example.SerializeToString()).decode()
            return {"input": {"b64": b64_example}}

        tgt_pgs = tgt_pgs or itertools.repeat(None)
        instances = [
            serialize_to_instance(segment, tgt_segment=tgt_segment)
            for (segment, tgt_segment) in zip(pgs, tgt_pgs)
        ]
        payload = {"signature_name": "serving_default", "instances": instances}
        return payload



class ParsingServer(NnServer):
    """ Client that accepts plain text Icelandic
        and returns a flattened parse tree according
        to the Reynir schema """

    src_enc = text_encoder.SubwordTextEncoder(_ENIS_VOCAB)
    tgt_enc = CompositeTokenEncoder()
    _model_name = "parse"


class TranslateServer(NnServer):
    """ Client that accepts plain text Icelandic
        and returns an English translation of the text """

    src_enc = text_encoder.SubwordTextEncoder(_ENIS_VOCAB)
    tgt_enc = src_enc
    _model_name = "translate_v2"


class TranslationScoringServer(NnServer):
    """ Client that accepts source and target text and returns
        subword-wise estimate of translation probabilities"""

    src_enc = text_encoder.SubwordTextEncoder(_ENIS_VOCAB)
    tgt_enc = src_enc
    _model_name = "translate_enis16k_v3-scorer"

    @classmethod
    def extract_results(
        cls, resp_json_obj, pgs, tgt_pgs=None, src_enc=None, tgt_enc=None
    ):
        src_enc = src_enc or cls.src_enc
        tgt_enc = src_enc or cls.tgt_enc

        def process_response_instance(instance, src_enc=None, tgt_enc=None):
            # Strip padding and eos token
            output_ids = instance["outputs"]
            length = len(output_ids)
            pad_start = output_ids.index(PAD_ID) if PAD_ID in output_ids else length
            eos_start = output_ids.index(EOS_ID) if EOS_ID in output_ids else length
            sent_end = min(pad_start, eos_start)
            sent_end_with_eos = sent_end + 1

            log_probs = instance["scores"]
            log_probs = log_probs[:sent_end_with_eos]

            alpha = 0.7
            penalty = ((len(log_probs) + 1) / 6) ** alpha
            score = sum(log_probs) / penalty

            app.logger.debug("log_probs: " + str(log_probs))
            app.logger.debug("scores: " + str(score))

            return instance

        predictions = resp_json_obj["predictions"]
        results = [
            process_response_instance(inst) for (inst, sent) in zip(predictions, pgs)
        ]
        return results


class OpenNMTTranslationServer(NnServer):
    """ Same as TranslateServer, except uses subword-nmt as the encoder
        along with using the OpenNMT model api"""

    @classmethod
    def package_data(cls, pgs, tgt_pgs=None):
        batch = [
            cls.src_enc.encode(segment).split() for segment in pgs
        ]
        batch_width = max(len(item) for item in batch)

        # this might not be necessary
        # lengths = []
        # padded_batch = []
        # for item in batch:
        #     length = len(item)
        #     padding = [""] * (batch_width - length)
        #     lengths.append(length)
        #     item.extend(padding)
        #     padded_batch.append(item)

        instances = [
            {"tokens":item, "length":len(item)} for item in batch
        ]

        payload = {"signature_name": "serving_default", "instances": instances}
        return payload

    @classmethod
    def extract_results(
        cls, resp_json_obj, pgs, tgt_pgs = None, src_enc = None, tgt_enc = None
    ):
        tgt_enc = tgt_enc or cls.tgt_enc

        def process_response_instance(instance):
            log_probs = instance["log_probs"]
            tokens = instance["tokens"]

            app.logger.debug("log_probs: " + str(log_probs))
            app.logger.debug("tokens: " + str(tokens))

            lengths = instance["length"]
            # strip eos token
            outputs = []
            for i in range(len(tokens)):
                length = lengths[i]
                outputs.append(tgt_enc.decode(" ".join(tokens[i][:length])))

            app.logger.info(outputs)

            instance = {
                "outputs": "\n\n".join(outputs),
                "scores": log_probs,
            }
            return instance

        predictions = resp_json_obj["predictions"]
        results = [
            process_response_instance(inst) for (inst, sent) in zip(predictions, pgs)
        ]
        return results


class OpenNMTTranslationServerEnIs(OpenNMTTranslationServer):
    """ Same as TranslateServer, except uses subword-nmt as the encoder
        along with using the OpenNMT model api"""

    src_enc = _SubwordNmtEncoder(_ONMT_EN_VOCAB)
    tgt_enc = _SubwordNmtEncoder(_ONMT_IS_VOCAB)
    _model_name = "translate_enis16k_v4.onmt-bilstm"


class OpenNMTTranslationServerIsEn(OpenNMTTranslationServer):
    """Reverse direction of OpenNMTTranslationServerEnIs"""

    src_enc = _SubwordNmtEncoder(_ONMT_EN_VOCAB)
    tgt_enc = _SubwordNmtEncoder(_ONMT_IS_VOCAB)
    _model_name = "translate_enis16k_v4.onmt-bilstm_rev"


@app.route("/parse.api", methods=["POST"])
def parse_api():
    try:
        req_body = request.data.decode("utf-8")
        obj = json.loads(req_body)
        # TODO: validate form?
        pgs = obj["pgs"]
        model_response = ParsingServer.request(pgs)
        resp = jsonify(model_response)
    except Exception as error:
        resp = jsonify(valid=False, reason="Invalid request")
        app.logger.exception(error)
    resp.headers["Content-Type"] = "application/json; charset=utf-8"
    return resp


@app.route("/translate.api", methods=["POST"])
def translate_api():
    try:
        req_body = request.data.decode("utf-8")
        obj = json.loads(req_body)
        pgs = obj["pgs"]
        model_name = MODEL_NAMES[obj["model"]][
            "{}-{}".format(obj["source"], obj["target"])
        ]
        if "lstm" in obj["model"]:
            onmt_server = OpenNMTTranslationServerIsEn
            if "en" in obj["source"]:
                onmt_server = OpenNMTTranslationServerEnIs
            model_response = onmt_server.request(pgs, model_name=model_name)
        else:
            model_response = TranslateServer.request(pgs, model_name=model_name)
        resp = jsonify(model_response)
    except Exception as error:
        resp = jsonify(valid=False, reason="Invalid request")
        app.logger.exception(error)
    resp.headers["Content-Type"] = "application/json; charset=utf-8"
    return resp


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Middleware server that provides a textual interface to tensorflow model server"
        )
    )
    parser.add_argument(
        "-lh",
        "--listen_host",
        dest="IN_HOST",
        default="0.0.0.0",
        required=False,
        type=str,
        help="Hostname to listen on",
    )
    parser.add_argument(
        "-lp",
        "--listen_port",
        dest="IN_PORT",
        default="8080",
        required=False,
        type=str,
        help="Port to listen on",
    )
    parser.add_argument(
        "-mh",
        "--model_host",
        dest="OUT_HOST",
        default="localhost",
        required=False,
        type=str,
        help="Hostname of model server",
    )
    parser.add_argument(
        "-mp",
        "--model_port",
        dest="OUT_PORT",
        default="9000",
        required=False,
        type=str,
        help="Port of model server",
    )
    parser.add_argument(
        "--debug",
        dest="DEBUG",
        default=False,
        action="store_true",
        required=False,
        help="Emit debug information messages",
    )
    parser.add_argument(
        "--only",
        dest="ONLY",
        default=False,
        required=False,
        type=str,
        choices=["parse", "translate"],
        help="Only use one model (otherwise both).",
    )
    args = parser.parse_args()
    app.config["out_host"] = args.OUT_HOST
    app.config["out_port"] = args.OUT_PORT
    app.run(threaded=True, debug=args.DEBUG, host=args.IN_HOST, port=args.IN_PORT)
