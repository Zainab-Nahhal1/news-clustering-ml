from source.main import preprocess_text

def test_preprocess_text():
    text = "Breaking News!!! AI and Machine-Learning are Evolving."
    processed = preprocess_text(text)
    assert isinstance(processed, str)
    assert 'breaking' in processed
