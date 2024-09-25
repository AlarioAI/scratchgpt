from scratchgpt.tokenizer.char_tokenizer import CharTokenizer, Utf8Tokenizer


test_data = [
    "Привет, как дела? 😊", 
    "Я люблю читать книги.", 
    "Москва - это красивый город.", 
    "안녕하세요 👋", 
    "나는 당신을 만나서 행복해요 😊", 
    "서울은 아름다운 도시입니다.", 
    "Ciao, come stai? 😊", 
    "Amo leggere libri.", 
    "Roma è una città bellissima.", 
    "Hello, how are you? 👋", 
    "I love to read books.", 
    "New York City is a bustling metropolis 🗽️"
]

def test_basic_char_tokenizer():
    test_corpus = "".join(test_data)
    tokenizer = CharTokenizer(test_corpus)

    for test_sample in test_data:
        encoded = tokenizer.encode(test_sample)
        decoded = tokenizer.decode(encoded)

        assert test_sample == decoded, "Oh no, this thing failed"


def test_utf8_tokenizer():
    tokenizer = Utf8Tokenizer()

    for test_sample in test_data:
        encoded = tokenizer.encode(test_sample)
        decoded = tokenizer.decode(encoded)

        assert test_sample == decoded, "Oh no, Utf8Tokenizer failed"

    print(f"{tokenizer.vocab_size=}")
