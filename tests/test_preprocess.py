from sentiment.preprocess import clean_text


def test_removes_html_tags_and_entities():
    assert clean_text("Great<br /><br />movie &amp; cast") == "great movie cast"


def test_removes_urls():
    assert clean_text("see https://example.com now") == "see now"


def test_keeps_negations_and_emphasis():
    assert clean_text("I don't like it!!!") == "i don't like it!!"


def test_lowercases_and_collapses_spaces():
    assert clean_text("  SO    Boring  ") == "so boring"


def test_shortens_repeated_letters():
    assert clean_text("sooooo good") == "soo good"
