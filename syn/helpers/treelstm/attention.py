class AttentionWeights:
    MAX_WEIGHT = 1.0
    CORE_WORD = 1.0 * MAX_WEIGHT
    MODIFIER_WORD = 0.75 * MAX_WEIGHT
    FUNCTION_WORD = 0.5 * MAX_WEIGHT
    NON_WORD = 0.25 * MAX_WEIGHT
    NON_INFORMATIVE_WORD = 0.0


def get_attention_weight(tag: str) -> float:
    weight = {
        'core_word': AttentionWeights.CORE_WORD,
        'modifier_word': AttentionWeights.MODIFIER_WORD,
        'function_word': AttentionWeights.FUNCTION_WORD,
        'non_word': AttentionWeights.NON_WORD,
        'non_informative_word': AttentionWeights.NON_INFORMATIVE_WORD
    }
    return weight[tag]


# https://www.ling.upenn.edu/hist-corpora/annotation/labels.htm
# http://www.surdeanu.info/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html
def get_part_of_speech_weight(tag: str):
    pos_weight = {
        #  Clause Level
        "ROOT": "non_informative_word",
        "S": "non_informative_word",
        "SBAR": "non_informative_word",
        "SBARQ": "non_informative_word",
        "SINV": "non_informative_word",
        "SQ": "non_informative_word",

        # Phrase Leve
        "ADJP": "modifier_word",
        "ADVP": "modifier_word",
        "CONJP": "function_word",
        "FRAG": "core_word",
        "INTJ": "modifier_word",
        "LST": "core_word",
        "NAC": "core_word",
        "NP": "core_word",
        "NX": "core_word",
        "PP": "function_word",
        "PRN": "core_word",
        "PRT": "function_word",
        "QP": "core_word",
        "RRC": "core_word",
        "UCP": "core_word",
        "VP": "core_word",
        "WHADJP": "modifier_word",
        "WHADVP": "modifier_word",
        "WHNP": "core_word",
        "WHPP": "function_word",
        "X": "core_word",

        # Word level
        "CC": "function_word",
        "CD": "non_word",
        "DT": "function_word",
        "EX": "function_word",
        "FW": "core_word",
        "IN": "function_word",
        "JJ": "modifier_word",
        "JJR": "modifier_word",
        "JJS": "modifier_word",
        "LS": "core_word",
        "MD": "modifier_word",
        "NN": "core_word",
        "NNS": "core_word",
        "NNP": "core_word",
        "NNPS": "core_word",
        "PDT": "function_word",
        "POS": "core_word",
        "PRP": "function_word",
        "PRP$": "function_word",
        "RB": "modifier_word",
        "RBR": "modifier_word",
        "RBS": "modifier_word",
        "RP": "function_word",
        "SYM": "non_word",
        "TO": "function_word",
        "UH": "modifier_word",
        "VB": "core_word",
        "VBD": "core_word",
        "VBG": "core_word",
        "VBN": "core_word",
        "VBP": "core_word",
        "VBZ": "core_word",
        "WDT": "core_word",
        "WP": "core_word",
        "WP$": "core_word",
        "WRB": "core_word",

        # Form/function discrepancies
        "-ADV": "non_informative_word",
        "-NOM": "non_informative_word",

        # Grammatical role
        "-DTV": "non_informative_word",
        "-LGS": "non_informative_word",
        "-PRD": "non_informative_word",
        "PUT": "non_informative_word",
        "-SBJ": "non_informative_word",
        "-TPC": "non_informative_word",
        "-VOC": "non_informative_word",

        # Adverbials
        "-BNF": "non_informative_word",
        "-DIR": "non_informative_word",
        "-EXT": "non_informative_word",
        "-LOC": "non_informative_word",
        "-MNR": "non_informative_word",
        "-PRP": "non_informative_word",
        "-TMP": "non_informative_word",

        # Miscellaneous
        "-CLR": "non_informative_word",
        "-CLF": "non_informative_word",
        "-HLN": "non_informative_word",
        "-TTL": "non_informative_word",

        # TreeBinarizer
        "GW": "non_informative_word",
        "ADD": "non_informative_word",
        "NFP": "non_informative_word",
        "AFX": "non_informative_word",
        "HYPH": "non_informative_word",

        # Nonstructural labels
        "META": "non_informative_word",

        # Stanford
        ".": "non_word",
        "$": "non_word",
        "NML": "core_word",
        "NP-TMP": "core_word"
    }
    # KeyError: '@NP'
    # if (value.equals(v.getTag()) || value.equals("@".concat(v.getTag()))) {
    # KeyError: 'ROOT'
    # KeyError: 'NML'
    return get_attention_weight(tag=pos_weight[tag.replace('@', '')])
