DEFAULT_MODEL = "CohereLabs/tiny-aya-global"
DEFAULT_DATASET = "1024m/LID"
DEFAULT_DATASET_FILE = "Data_Hackathon/LID-500.parquet"

LANG_TO_ISO = {
    "amharic": "amh",
    "arabic": "ara",
    "bangla": "ben",
    "banyumasan": "jav",
    "basque": "eus",
    "bulgarian": "bul",
    "burmese": "mya",
    "catalan": "cat",
    "chinese": "zho",
    "croatian": "hrv",
    "czech": "ces",
    "danish": "dan",
    "dutch": "nld",
    "estonian": "est",
    "finnish": "fin",
    "french": "fra",
    "galician": "glg",
    "german": "deu",
    "greek": "ell",
    "gujarati": "guj",
    "hausa": "hau",
    "hebrew": "heb",
    "hindi": "hin",
    "hungarian": "hun",
    "igbo": "ibo",
    "indonesian": "ind",
    "irish": "gle",
    "italian": "ita",
    "japanese": "jpn",
    "khmer": "khm",
    "korean": "kor",
    "lao": "lao",
    "latvian": "lav",
    "lithuanian": "lit",
    "malagasy": "mlg",
    "malay": "msa",
    "maltese": "mlt",
    "marathi": "mar",
    "nepali": "nep",
    "norwegian": "nor",
    "persian": "fas",
    "polish": "pol",
    "portuguese": "por",
    "punjabi": "pan",
    "romanian": "ron",
    "russian": "rus",
    "serbian": "srp",
    "shona": "sna",
    "simple english": "eng",
    "slovak": "slk",
    "slovenian": "slv",
    "spanish": "spa",
    "swahili": "swh",
    "swedish": "swe",
    "tagalog": "tgl",
    "tamil": "tam",
    "telugu": "tel",
    "thai": "tha",
    "turkish": "tur",
    "ukrainian": "ukr",
    "urdu": "urd",
    "vietnamese": "vie",
    "welsh": "cym",
    "wolof": "wol",
    "xhosa": "xho",
    "yoruba": "yor",
    "zulu": "zul",
}

VALID_OPTIONS = sorted(LANG_TO_ISO.values())

TASK_DESCRIPTION = (
    "The task is language identification, Read the given input text "
    "and respond with the detected language's 3 letter code (ISO-693-3)"
)

LANG_CODES_BLOCK = "\n".join(
    f"{k} : {v}" for k, v in sorted(LANG_TO_ISO.items())
)
