#!/usr/bin/env/python
# coding=utf-8

"""
    Reynir: Natural language processing for Icelandic

    Neural Network Parsing Encoder

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


    This module implements a composite subword encoder for tokens on
    the output side of the text-to-parse-tree model, i.e. grammar
    nonterminals, terminals and their variants.

"""

from tensor2tensor.data_generators import text_encoder

UNK = "<UNK>"
EOS_ID = text_encoder.EOS_ID


class GRAMMAR_ITEMS:
    nonterminals = {
        "ADJP",
        ###
        "ADVP",
        ####
        "ADVP",
        "ADVP-DATE",
        "ADVP-DATE-ABS",
        "ADVP-DATE-REL",
        "ADVP-DIR",
        "ADVP-DUR-ABS",
        "ADVP-DUR-REL",
        "ADVP-DUR-TIME",
        "ADVP-LOC",
        "ADVP-PCL",
        "ADVP-TIMESTAMP",
        "ADVP-TIMESTAMP-ABS",
        "ADVP-TIMESTAMP-REL",
        "ADVP-TMP-SET",
        ####
        "FOREIGN",
        ###
        "P",
        "PP",
        "PP-LOC",
        "PP-DIR",
        ####
        "C",
        "CP-ADV-ACK",
        "CP-ADV-CAUSE",
        "CP-ADV-CMP",
        "CP-ADV-COND",
        "CP-ADV-CONS",
        "CP-ADV-PURP",
        "CP-ADV-TEMP",
        "CP-EXPLAIN",
        "CP-QUE",
        "CP-QUOTE",
        "CP-REL",
        "CP-SOURCE",
        "CP-THT",
        ####
        "IP",
        "IP-INF",
        ####
        "NP",
        "NP-ADDR",
        "NP-AGE",
        "NP-ADP",
        "NP-COMPANY",
        "NP-DAT",
        "NP-ES",
        "NP-IOBJ",
        "NP-MEASURE",
        "NP-OBJ",
        "NP-PERSON",
        "NP-POSS",
        "NP-PRD",
        "NP-SUBJ",
        "NP-SOURCE",
        "NP-TITLE",
        ####
        "S",
        "S-COND",
        "S-CONS",
        "S-EXPLAIN",
        "S-HEADING",
        "S-MAIN",
        "S-PREFIX",
        "S-QUE",
        "S-QUOTE",
        ####
        "S0",
        "S0-X",
        ###
        "TO",
        ####
        "VP",
        "VP-AUX",
    }
    categories = {
        "abfn",
        "abbrev",
        "ao",
        "ártal",
        "dags",
        "dagsföst",
        "dagsafs",
        "entity",
        "eo",
        "fn",
        "foreign",
        "fs",
        "fyrirtæki",
        "gata",
        "gr",
        "grm",
        "lén",
        "lo",
        "mælieining",
        "myllumerki",
        "nhm",
        "no",
        "person",
        "p",
        "pfn",
        "prósenta",
        "raðnr",
        "sérnafn",
        "sequence",
        "so",
        "st",
        "stt",
        "tala",
        "talameðbókstaf",
        "tímapunktur",
        "tímapunkturafs",
        "tímapunkturfast",
        "tími",
        "to",
        "töl",
        "tölvupóstfang",
        "uh",
        "x",
    }

    class VARIANTS:
        article = {"gr"}
        abbrev = {"abbrev"}
        cases = {"nf", "þf", "þgf", "ef"}
        genders = {"kk", "kvk", "hk"}
        numbers = {"et", "ft"}
        persons = {"p1", "p2", "p3"}
        tense = {"þt", "nt"}
        degree = {"fst", "mst", "est", "esb", "evb"}
        strength = {"sb", "vb"}
        voice = {"mm", "gm"}
        mood = {"fh", "vh", "nh", "bh", "lhnt", "lhþt"}
        supine = {"sagnb"}
        clitic = {"sn"}
        impersonal = {"es", "subj", "none", "op"}
        lo_obj = {"sþf", "sþgf", "sef"}
        all_subvariants = (
            *article,
            *abbrev,
            *cases,
            *genders,
            *numbers,
            *persons,
            *tense,
            *degree,
            *strength,
            *voice,
            *mood,
            *supine,
            *clitic,
            *impersonal,
            *lo_obj,
        )


class CompositeTokenEncoder:
    """Composite token encoder, behaves similarly to a wordpiece or subword encoder
    except uses a handcrafted vocabulary of grammatical items for Icelandic.
    Encoder has the same outward behaviour as Tensor2tensors SubwordTextEncoder.

    Subtokens should follow a simple right recursive rule with a couple of exceptions,
    namely case-control of verbs."""

    def __init__(self, reorder=True):
        self._num_reserved_ids = len(text_encoder.RESERVED_TOKENS)
        self._reorder = reorder
        self._preprocess_word = lambda x: x

        nonterminals = list(GRAMMAR_ITEMS.nonterminals) + [
            "/" + i for i in GRAMMAR_ITEMS.nonterminals
        ]
        head_tokens = list(GRAMMAR_ITEMS.categories) + ["so_0", "so_1", "so_2"]
        tail_tokens = list(GRAMMAR_ITEMS.VARIANTS.all_subvariants)

        self._all_tokens = (
            text_encoder.RESERVED_TOKENS
            + nonterminals
            + head_tokens
            + tail_tokens
            + [UNK]
        )

        self._tok_id_to_tok_str = dict(enumerate(self._all_tokens))
        offset = len(text_encoder.RESERVED_TOKENS)
        self._ftok_to_tok_id = dict(
            (tok, idx + offset) for (idx, tok) in enumerate(nonterminals)
        )
        offset += len(self._ftok_to_tok_id)
        self._htok_to_tok_id = dict(
            (tok, idx + offset) for (idx, tok) in enumerate(head_tokens)
        )
        offset += len(self._htok_to_tok_id)
        self._ttok_to_tok_id = dict(
            (tok, idx + offset) for (idx, tok) in enumerate(tail_tokens)
        )
        offset += len(self._ttok_to_tok_id)
        self._oov_id = offset

        self._ftok_ids = set(self._ftok_to_tok_id.values())
        self._htok_ids = set(self._htok_to_tok_id.values())
        self._ttok_ids = set(self._ttok_to_tok_id.values())

    def encode(self, string):
        result = self._tokens_to_subtoken_ids(list(string.split(" ")))
        return result

    def _tokens_to_subtoken_ids(self, words):
        result = []
        for word in words:
            result.extend(self._token_to_subtoken_ids(word))
        return result

    def _token_to_subtoken_ids(self, token):
        token = self._preprocess_word(token)
        if token in self._ftok_to_tok_id:
            return [self._ftok_to_tok_id[token]]
        elif token in self._htok_to_tok_id:
            return [self._htok_to_tok_id[token]]
        if "_" not in token:
            return [self._oov_id]
        subtokens = token.split("_")
        head, t1 = subtokens[:2]

        tail_sort_start = 1
        tail_start = 1
        canonical = []
        if head == "so" and subtokens[tail_start] in {"0", "1", "2"}:
            tail_start += 1
            tail_sort_start += 1 + int(t1)
            canonical.append(head + "_" + t1)
            head = head + "_" + t1
            canonical.extend(subtokens[tail_start:tail_sort_start])
        else:
            canonical.append(head)

        canonical_sort_start = len(canonical)
        parts = subtokens[tail_sort_start:]
        if "op" in parts and "es" in parts:
            parts.remove("op")
            parts.remove("es")
            canonical.extend(("op", "es"))
        elif "op" in parts and "subj" in parts:
            parts.remove("op")
            parts.remove("subj")
            canonical.extend(("op", "subj"))
        elif "op" in parts:
            parts.remove("op")
            canonical.append("op")

        if "lh" in parts and "nt" in parts:
            parts.remove("lh")
            parts.remove("nt")
            parts.append("lhnt")
            pass

        if self._reorder:
            parts = sorted(parts)
        canonical.extend(sorted(parts))

        if head not in self._htok_to_tok_id or not all(
            t in self._ttok_to_tok_id for t in canonical[1:]
        ):
            return [self._oov_id]

        head = [self._htok_to_tok_id[head]]
        tail = [self._ttok_to_tok_id[t] for t in canonical[1:]]
        ids = head + tail
        return ids

    def decode(self, ids):
        result = []
        current = []
        tok_id = None
        for tok_id in ids:
            if 0 <= tok_id < self._num_reserved_ids:
                result.append(current)
                current = [text_encoder.RESERVED_TOKENS[tok_id]]
                continue

            subtok = self._tok_id_to_tok_str[tok_id]
            if tok_id in self._ftok_ids or tok_id in self._htok_ids:
                # is nonterminal or head part of terminal, start new composite_token
                result.append(current)
                current = [subtok]
            elif tok_id in self._ttok_ids:
                current.append(subtok)
            else:
                current.append(UNK)

        if tok_id is not None:
            result.append(current)
        result = ["_".join(subtokens) for subtokens in result if subtokens]
        return " ".join(result)

    def decode_list(self, ids):
        result = []
        for tok_id in ids:
            if tok_id in self._ttok_ids:
                result.append(
                    "_" + self._tok_id_to_tok_str[tok_id]
                )  # to show that it should merge to preceding token
            elif tok_id in self._all_tokens:
                result.append(self._tok_id_to_tok_str[tok_id])
            else:
                result.append(UNK)
        return result

    def _tokens_to_subtoken_ids(self, tokens):
        result = []
        for token in tokens:
            result.extend(self._token_to_subtoken_ids(token))
        return result

    @property
    def vocab_size(self):
        return (
            self._num_reserved_ids
            + len(self._ftok_to_tok_id)
            + len(self._htok_to_tok_id)
            + len(self._ttok_to_tok_id)
        )

<<<<<<< Updated upstream
=======

def test_roundtrip():
    sample = (
        "P S-MAIN IP NP-SUBJ pfn_et_nf_p3 no_ft_ef_kvk to_nf_ft_kvk"
        " so_subj_op_þf_et_mm_vh_þt so_1_þf_subj_op_þf so_2_þf_þgf_gm_nh"
        " gata_kk_þf to_ef_ft_kk to_et_kvk_þf so_2_þf_ef_ft_gm_nt_p3_vh"
        " so_1_ef_et_fh_gm_p2_þt /NP-SUBJ /IP /S-MAIN /P"
    )
    default_encoder = CompositeTokenEncoder()
    subtoken_ids = default_encoder.encode(sample)
    decoded_sample = default_encoder.decode(subtoken_ids)
    assert sample == decoded_sample, "Encoding roundtrip does not match"
>>>>>>> Stashed changes
