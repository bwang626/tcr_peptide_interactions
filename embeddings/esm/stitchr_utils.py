"""
Utilities for stitching full-length TCR coding sequences via stitchr.

Reference data (IMGT FASTA, J/C motifs, codon table) is loaded once per
(chain, species) pair and reused across calls, so bulk processing is efficient.

Failure modes and retry strategy
---------------------------------
1. C-terminus not found / string index error
   Cause: CDR3 is missing the terminal junction residue (almost always 'F').
   Fix:   Append 'F' to the CDR3 and retry.

2. N-terminus not found
   Cause: CDR3 doesn't start with the conserved Cys, OR the V-gene junction
          can't be detected (e.g. TRBV17 edge case).
   Fix:   Retry with skip_n_checks=True, which lets stitchr anchor the CDR3
          at the last conserved Cys in the V sequence regardless of CDR3 start.

3. Leader sequence not found (e.g. TRBV8-1)
   Cause: Gene name not in stitchr's IMGT LEADER data (deprecated/renamed).
   Fix:   Retry with no_leader=True (omit signal peptide from output).

4. Chain mismatch / bad gene name
   Cause: TRA gene paired with TRB gene, or non-IMGT names.
   Fix:   Return None — unfixable data errors.
"""

import warnings
from typing import Optional

from Stitchr import stitchr as st
from Stitchr import stitchrfunctions as fxn

_gene_types = list(fxn.regions.values())

# Module-level caches keyed by species or (chain, species)
_motif_cache: dict = {}   # species -> (j_res, low_conf_js, c_res, codon_dict)
_imgt_cache: dict = {}    # (chain, species) -> (imgt_dat, functionality, partial)

_C_TERM_ERRORS = frozenset([
    "Unable to locate C terminus of CDR3 in J gene correctly.",
    "string index out of range",
    "CDR3 seemingly deleted",
])
_N_TERM_ERROR = "Unable to locate N terminus of CDR3 in V gene correctly."
_LEADER_ERROR = "LEADER sequence region has not been found"


def _load_motifs(species: str):
    if species not in _motif_cache:
        j_res, low_conf_js = fxn.get_j_motifs(species)
        c_res = fxn.get_c_motifs(species)
        codon_dict = fxn.get_optimal_codons("", species)
        _motif_cache[species] = (j_res, low_conf_js, c_res, codon_dict)
    return _motif_cache[species]


def _load_imgt(chain: str, species: str):
    key = (chain, species)
    if key not in _imgt_cache:
        imgt_dat, functionality, partial = fxn.get_ref_data(chain, _gene_types, species)
        _imgt_cache[key] = (imgt_dat, functionality, partial)
    return _imgt_cache[key]


def _try_stitch(specific_args, imgt_dat, functionality, partial,
                codon_dict, c_res, j_res, low_conf_js):
    """Single stitch attempt; returns result dict or raises."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return st.stitch(
            specific_args, imgt_dat, functionality, partial,
            codon_dict, specific_args["j_warning_threshold"],
            {}, c_res, j_res, low_conf_js,
        )


def _expected_terminal(j_gene: str, j_res: dict) -> str:
    """
    Return the expected CDR3 C-terminal residue for this J gene.

    j_res keys are full allele names (e.g. 'TRBJ2-7*01'). If the caller omits
    the allele, we check the *01 allele. Falls back to 'F' (correct for >95%
    of TRB J genes).
    """
    lookup = j_gene if "*" in j_gene else j_gene + "*01"
    return j_res.get(lookup, "F")


def _stitch_with_fallbacks(specific_args, imgt_dat, functionality, partial,
                           codon_dict, c_res, j_res, low_conf_js):
    """
    Try stitching with up to 8 flag combinations. Returns stitch_dict or None.

    Pre-processing applied before the grid:
      • expected_term  — J-gene-specific terminal residue from j_res (not 'F')
      • base_cdr3      — CDR3 with 'C' prepended if the leading Cys is absent;
                         this makes skip_n_checks more reliable by giving stitchr
                         a valid anchor even when the CDR3 lacks the conserved Cys

    Flags varied in the grid:
      extra_term    — append expected_term (repairs missing C-terminal residue)
      skip_n        — skip_n_checks=True (repairs N-term mismatch / V-junction issue)
      force_no_ldr  — no_leader=True (repairs V gene absent from IMGT LEADER data)

    Note: the caller's no_leader value is always respected; force_no_ldr only
    escalates to True, never overrides a True→False.
    """
    raw_cdr3 = specific_args["cdr3"]
    base_no_leader = specific_args["no_leader"]  # preserve caller's intent

    # Resolve the correct terminal residue for this specific J gene
    expected_term = _expected_terminal(specific_args["j"], j_res)

    # Prepend conserved Cys if missing so skip_n_checks has a valid anchor
    base_cdr3 = raw_cdr3 if raw_cdr3.startswith("C") else "C" + raw_cdr3

    for extra_term, skip_n, force_no_ldr in [
        (False, False, False),  # 1. as-is (with C prepended if needed)
        (True,  False, False),  # 2. append terminal  (missing C-terminal residue)
        (False, True,  False),  # 3. skip_n            (N-term mismatch)
        (True,  True,  False),  # 4. append terminal + skip_n  (both N and C issues)
        (False, False, True),   # 5. no_leader          (LEADER gene missing from IMGT)
        (False, True,  True),   # 6. no_leader + skip_n
        (True,  False, True),   # 7. no_leader + append terminal
        (True,  True,  True),   # 8. all fixes combined
    ]:
        args = dict(specific_args)
        args["cdr3"] = base_cdr3 + (expected_term if extra_term else "")
        args["skip_n_checks"] = skip_n
        args["no_leader"] = base_no_leader or force_no_ldr
        try:
            return _try_stitch(args, imgt_dat, functionality, partial,
                               codon_dict, c_res, j_res, low_conf_js)
        except Exception:
            continue

    return None


def _build_raw_args(v_gene, j_gene, cdr3, species, no_leader):
    return {
        "v": v_gene,
        "j": j_gene,
        "cdr3": cdr3,
        "c": "",
        "l": "",
        "name": "",
        "5_prime_seq": "",
        "3_prime_seq": "",
        "mode": "NT",
        "species": species,
        "skip_c_checks": False,
        "skip_n_checks": False,
        "no_leader": no_leader,
        "seamless": False,
        "codon_usage_path": "",
        "j_warning_threshold": 3,
        "extra_genes": False,
        "preferred_alleles_path": "",
        "suppress_warnings": True,
        "aa": "",
    }


def get_full_sequence(
    v_gene: str,
    j_gene: str,
    cdr3: str,
    species: str = "HUMAN",
    no_leader: bool = False,
) -> Optional[str]:
    """
    Stitch a full-length TCR coding nucleotide sequence from V, J, and CDR3.

    CDR3 should include the conserved terminal residues (leading C, trailing
    F/W) but missing terminals are recovered automatically where possible.
    Gene names should be IMGT format; allele suffix (*01) is optional.

    Returns the nucleotide sequence string, or None if stitching fails after
    all fallback attempts.
    """
    species = species.upper()
    try:
        specific_args, chain = fxn.sort_input(_build_raw_args(v_gene, j_gene, cdr3, species, no_leader))
    except Exception:
        return None

    j_res, low_conf_js, c_res, codon_dict = _load_motifs(species)
    imgt_dat, functionality, partial = _load_imgt(chain, species)

    result = _stitch_with_fallbacks(specific_args, imgt_dat, functionality, partial,
                                    codon_dict, c_res, j_res, low_conf_js)
    return result["stitched_nt"] if result is not None else None


def get_full_aa_sequence(
    v_gene: str,
    j_gene: str,
    cdr3: str,
    species: str = "HUMAN",
    no_leader: bool = False,
) -> Optional[str]:
    """
    Same as get_full_sequence but returns the translated amino acid sequence.
    Handles the translation offset introduced by any 5' padding.
    """
    species = species.upper()
    try:
        specific_args, chain = fxn.sort_input(_build_raw_args(v_gene, j_gene, cdr3, species, no_leader))
    except Exception:
        return None

    j_res, low_conf_js, c_res, codon_dict = _load_motifs(species)
    imgt_dat, functionality, partial = _load_imgt(chain, species)

    result = _stitch_with_fallbacks(specific_args, imgt_dat, functionality, partial,
                                    codon_dict, c_res, j_res, low_conf_js)
    if result is None:
        return None
    return fxn.translate_nt("N" * result["translation_offset"] + result["stitched_nt"])
