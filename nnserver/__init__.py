"""
Reynir: Natural language processing for Icelandic

Copyright (C) 2019 Miðeind ehf.

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
"""

import os
import pkg_resources


NNSERVER_RESOURCES = os.getenv("NNSERVER_RESOURCES", "/nnserver/nnserver/resources")
NNSERVER_ENIS_VOCAB = os.getenv("NNSERVER_ENIS_VOCAB", "vocab.translate_enis16k.16384.subwords")
NNSERVER_OPENNMT_IS_VOCAB = os.getenv("NNSERVER_ENIS_VOCAB", "vocab.translate_enis16k_v4.is.nmt-bpe")
NNSERVER_OPENNMT_EN_VOCAB = os.getenv("NNSERVER_ENIS_VOCAB", "vocab.translate_enis16k_v4.en.nmt-bpe")

try:
    _RESOURCES = pkg_resources.resource_filename(__package__, "resources")
except (KeyError, TypeError, ValueError) as e:
    _RESOURCES = NNSERVER_RESOURCES
_ENIS_VOCAB = os.path.join(_RESOURCES, NNSERVER_ENIS_VOCAB)
_ONMT_EN_VOCAB = os.path.join(_RESOURCES, NNSERVER_OPENNMT_EN_VOCAB)
_ONMT_IS_VOCAB = os.path.join(_RESOURCES, NNSERVER_OPENNMT_IS_VOCAB)
