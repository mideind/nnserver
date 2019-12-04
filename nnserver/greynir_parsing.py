
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf

from nnserver import composite_encoder
from nnserver import (
  _ENIS_VOCAB,
  _PARSING_VOCAB
)

EOS = text_encoder.EOS_ID


def tabbed_generator_samples(source_path):
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    for line in source_file:
      if line and "\t" in line:
        parts = line.split("\t", 1)
        source, target = parts[0].strip(), parts[1].strip()
        yield {
          "inputs": source,
          "targets": target,
        }


class TranslateProblemV2(translate.TranslateProblem):

  def __init__(self, was_reversed=False, was_copy=False):
    super(TranslateProblemV2, self).__init__(was_reversed, was_copy)

  def generate_encoded_samples(self, data_dir, tmp_dir, train):
    vocabs = self.feature_encoders(data_dir)
    source_vocab = vocabs["inputs"]
    target_vocab = vocabs["targets"]

    eos_list = [EOS]
    for sample in self.generate_samples(data_dir, tmp_dir, train):
      yield {
        "inputs": source_vocab.encode(sample["inputs"]) + eos_list,
        "targets": target_vocab.encode(sample["targets"]) + eos_list,
      }


@registry.register_problem
class ParsingIcelandic16kV5(TranslateProblemV2):

  @property
  def source_vocab_size(self):
    return 16384  # 16384

  @property
  def targeted_vocab_size(self):
    return 165  # 16384

  def feature_encoders(self, data_dir):
    enis_vocab = text_encoder.SubwordTextEncoder(_ENIS_VOCAB)
    parse_token_vocab = composite_encoder.CompositeTokenEncoder(_PARSING_VOCAB)
    return {
        "inputs": enis_vocab,
        "targets": parse_token_vocab
    }

  def generate_samples(self, data_dir, tmp_dir, train):
    tag = "train" if train else "dev"
    parse_source_fname = "parsing_%s.pairs" % tag
    parse_source_path = os.path.join("/data", parse_source_fname)
    return tabbed_generator_samples(parse_source_path)
