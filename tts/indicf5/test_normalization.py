import re

def normalize_text_extended(text):
    # Existing basic cleanup (from indicf5-openai.py)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 1. Acronyms: All Caps (2 or more letters) -> Add dots
    # e.g. FAQ -> F.A.Q., PH -> P.H.
    def add_dots_to_acronyms(match):
        word = match.group(0)
        return ".".join(list(word)) + "."
    
    # Pattern: discrete words, all caps, length >= 2
    text = re.sub(r'\b[A-Z]{2,}\b', add_dots_to_acronyms, text)
    
    # 2. Short consonant words (1-3 letters, no vowels aeiou/y) -> Add dots
    # e.g. ml -> m.l.
    def add_dots_to_short_consonants(match):
        word = match.group(0)
        return ".".join(list(word)) + "."
    
    # Regex: word boundary, 1-3 lowercase consonants (no aeiouy)
    # consonants: bcdfghjklmnpqrstvwxz
    consonants = "bcdfghjklmnpqrstvwxz"
    pattern = r'\b[' + consonants + r']{1,3}\b'
    text = re.sub(pattern, add_dots_to_short_consonants, text)
    
    return text

def test():
    test_cases = [
        ("FAQ PH AM PM", "F.A.Q. P.H. A.M. P.M."),
        ("hello world", "hello world"),
        ("ml", "m.l."),
        ("kg", "k.g."),
        ("cm", "c.m."),
        ("dr", "d.r."),
        ("mr", "m.r."),
        ("cat", "cat"),
        ("by", "by"),
        ("try", "try"),
        ("Assamese bhaxa Bharatiya bhaxar ekhon.", "Assamese bhaxa Bharatiya bhaxar ekhon."),
        ("I AM HERE", "I A.M. H.E.R.E."),
    ]
    
    for input_text, expected in test_cases:
        result = normalize_text_extended(input_text)
        print(f"Input:    '{input_text}'\nOutput:   '{result}'\nExpected: '{expected}'\nMatch:    {result == expected}\n")

if __name__ == "__main__":
    test()
