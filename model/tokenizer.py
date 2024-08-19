class UTF8Tokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.special_tokens = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2}
        self.char_to_id = {chr(i): i + len(self.special_tokens) for i in range(128)}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}

    def encode(self, text):
        return [self.special_tokens['<BOS>']] + [self.char_to_id.get(c, c.encode('utf-8')[0]) for c in text] + [self.special_tokens['<EOS>']]

    def decode(self, tokens):
        return ''.join([self.id_to_char.get(t, chr(t)) for t in tokens if t not in self.special_tokens.values()])