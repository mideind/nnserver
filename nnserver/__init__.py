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


try:
    _RESOURCES = pkg_resources.resource_filename(__package__, "resources")
except (KeyError, TypeError, ValueError) as e:
    _RESOURCES = "/nnserver/nnserver/resources"
_ENIS_VOCAB = os.path.join(_RESOURCES, "vocab.translate_enis16k.16384.subwords")
_PARSING_VOCAB = os.path.join(_RESOURCES, "parsing_tokens_180729.txt")
