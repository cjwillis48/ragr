from app.services.html import parse_html, strip_html


class TestStripHtml:
    def test_inline_elements_do_not_break_a_sentence(self):
        """The regression that shredded every HTML source into fragments."""
        html = '<body><p>RAGr costs $10 per <a href="/x">month</a> for the starter plan.</p></body>'
        assert strip_html(html) == "RAGr costs $10 per month for the starter plan."

    def test_nested_inline_markup_stays_inline(self):
        html = "<body><p>The god <em>Hades</em> ruled the <strong>underworld</strong> alone.</p></body>"
        assert strip_html(html) == "The god *Hades* ruled the **underworld** alone."

    def test_headings_become_atx_markdown(self):
        html = "<body><h1>Hades</h1><h2>Name</h2><p>Text.</p></body>"
        out = strip_html(html)
        assert "# Hades" in out
        assert "## Name" in out

    def test_list_items_become_bullets(self):
        html = "<body><ul><li>Cypress</li><li>Narcissus</li></ul></body>"
        out = strip_html(html)
        assert "* Cypress" in out
        assert "* Narcissus" in out

    def test_paragraphs_separated_by_blank_line(self):
        html = "<body><p>First para.</p><p>Second para.</p></body>"
        assert strip_html(html) == "First para.\n\nSecond para."

    def test_boilerplate_tags_removed(self):
        html = (
            "<body><nav>Home About</nav><header>Site</header>"
            "<p>Real content.</p>"
            "<aside>Sidebar</aside><footer>Copyright</footer></body>"
        )
        out = strip_html(html)
        assert "Real content." in out
        for junk in ("Home About", "Site", "Sidebar", "Copyright"):
            assert junk not in out

    def test_navigation_containers_removed_by_class(self):
        html = (
            '<body><div class="navbox">Greek deities Zeus Hera Poseidon</div>'
            '<table class="infobox"><tr><td>Abode</td><td>Underworld</td></tr></table>'
            "<p>Hades is the god of the dead.</p></body>"
        )
        out = strip_html(html)
        assert "Hades is the god of the dead." in out
        assert "Greek deities" not in out
        assert "Abode" not in out

    def test_role_attributes_removed(self):
        html = '<body><div role="navigation">Menu</div><p>Body text.</p></body>'
        out = strip_html(html)
        assert "Body text." in out
        assert "Menu" not in out

    def test_excess_blank_lines_collapsed(self):
        html = "<body><p>One.</p><div></div><div></div><p>Two.</p></body>"
        assert "\n\n\n" not in strip_html(html)

    def test_empty_and_malformed_input(self):
        assert strip_html("") == ""
        assert strip_html("<html>") == ""
        assert "text" in strip_html("<body><p>text</p>")

    def test_image_alt_text_does_not_leak_markup(self):
        html = '<body><p>See <img src="/a.png" alt="a picture"> here.</p></body>'
        out = strip_html(html)
        assert "](" not in out
        assert "/a.png" not in out


class TestParseHtml:
    def test_returns_markdown_text_and_links(self):
        html = (
            '<body><h2>Title</h2><p>Body with a <a href="/page">link</a> inline.</p>'
            '<a href="/other">Other</a></body>'
        )
        text, links = parse_html(html, "https://example.com/", "example.com", None)
        assert "## Title" in text
        assert "Body with a link inline." in text
        assert "https://example.com/page" in links
        assert "https://example.com/other" in links

    def test_offsite_links_excluded(self):
        html = '<body><a href="https://other.com/x">x</a><a href="/y">y</a></body>'
        _, links = parse_html(html, "https://example.com/", "example.com", None)
        assert links == ["https://example.com/y"]

    def test_prefix_filter_applied(self):
        html = '<body><a href="/docs/a">a</a><a href="/blog/b">b</a></body>'
        _, links = parse_html(html, "https://example.com/", "example.com", "/docs")
        assert links == ["https://example.com/docs/a"]

    def test_non_http_schemes_excluded(self):
        html = '<body><a href="mailto:x@example.com">mail</a><a href="/ok">ok</a></body>'
        _, links = parse_html(html, "https://example.com/", "example.com", None)
        assert links == ["https://example.com/ok"]

    def test_links_harvested_from_boilerplate_before_stripping(self):
        """Nav links still drive the crawl even though nav text is dropped."""
        html = '<body><nav><a href="/hidden">Hidden</a></nav><p>Body.</p></body>'
        text, links = parse_html(html, "https://example.com/", "example.com", None)
        assert "https://example.com/hidden" in links
        assert "Hidden" not in text
