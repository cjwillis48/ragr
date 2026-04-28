from app.services.wikipedia import is_wikipedia_url, parse_wikipedia_url


class TestParseWikipediaUrl:
    def test_english_article(self):
        assert parse_wikipedia_url("https://en.wikipedia.org/wiki/Hades") == ("en", "Hades")

    def test_french_article(self):
        assert parse_wikipedia_url("https://fr.wikipedia.org/wiki/Paris") == ("fr", "Paris")

    def test_article_with_parentheses(self):
        assert parse_wikipedia_url("https://en.wikipedia.org/wiki/Hades_(video_game)") == ("en", "Hades_(video_game)")

    def test_url_encoded_title(self):
        result = parse_wikipedia_url("https://en.wikipedia.org/wiki/%E4%B8%AD%E5%9B%BD")
        assert result == ("en", "中国")

    def test_article_with_subpage(self):
        assert parse_wikipedia_url("https://en.wikipedia.org/wiki/Greek_underworld") == ("en", "Greek_underworld")

    def test_non_wiki_path(self):
        assert parse_wikipedia_url("https://en.wikipedia.org/w/index.php?title=Hades") is None

    def test_non_wikipedia_domain(self):
        assert parse_wikipedia_url("https://example.com/wiki/Hades") is None

    def test_http_scheme(self):
        assert parse_wikipedia_url("http://en.wikipedia.org/wiki/Hades") == ("en", "Hades")

    def test_main_page(self):
        assert parse_wikipedia_url("https://en.wikipedia.org/wiki/Main_Page") == ("en", "Main_Page")

    def test_no_path(self):
        assert parse_wikipedia_url("https://en.wikipedia.org/") is None

    def test_language_with_variant(self):
        assert parse_wikipedia_url("https://zh-min-nan.wikipedia.org/wiki/Test") == ("zh-min-nan", "Test")


class TestIsWikipediaUrl:
    def test_wikipedia_url(self):
        assert is_wikipedia_url("https://en.wikipedia.org/wiki/Hades") is True

    def test_non_wikipedia_url(self):
        assert is_wikipedia_url("https://example.com/wiki/Hades") is False

    def test_wikipedia_non_article(self):
        assert is_wikipedia_url("https://en.wikipedia.org/w/index.php") is False
