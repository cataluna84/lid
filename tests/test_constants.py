from lid.constants import LANG_CODES_BLOCK, LANG_TO_ISO, VALID_OPTIONS


def test_lang_to_iso_count():
    assert len(LANG_TO_ISO) == 67


def test_valid_options_sorted():
    assert sorted(VALID_OPTIONS) == VALID_OPTIONS


def test_all_iso_codes_three_chars():
    for code in VALID_OPTIONS:
        assert len(code) == 3, f"ISO code '{code}' is not 3 characters"


def test_lang_codes_block_not_empty():
    assert len(LANG_CODES_BLOCK) > 0
    assert "amharic : amh" in LANG_CODES_BLOCK
