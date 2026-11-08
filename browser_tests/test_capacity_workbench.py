"""Browser coverage for the packaged LP and capacity-planning workbench."""

import json
import socket
from threading import Thread
from time import monotonic, sleep

from playwright.sync_api import expect, sync_playwright
import pytest
import uvicorn

from simplex_codes._api import app


@pytest.fixture
def url():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        address = f"http://127.0.0.1:{listener.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning", ws="none"))
        thread = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = monotonic() + 10
            while not server.started:
                assert thread.is_alive() and monotonic() < deadline, "Workbench did not start"
                sleep(0.05)
            yield address
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            assert not thread.is_alive()


def test_upload_compare_and_download_scenarios(url, tmp_path):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 390, "height": 844})
        assert page.goto(url).ok

        page.get_by_role("button", name="Modeli çöz").click()
        expect(page.get_by_role("status").filter(has_text="Optimal çözüm")).to_be_visible()
        expect(page.locator("#steps tr")).to_have_count(3)

        upload = page.get_by_label("Üretim verisi (CSV)")
        upload.set_input_files({
            "name": "capacity.csv", "mimeType": "text/csv",
            "buffer": "product,profit,Depo,İşçilik\nMasa,8,1,9\nSandalye,5,1,5\n".encode(),
        })
        expect(page.get_by_text("capacity.csv · 2 ürün")).to_be_visible()
        for label, value in (
            ("Depo temel kapasite", "6"), ("Depo alternatif kapasite", "6"),
            ("İşçilik temel kapasite", "45"), ("İşçilik alternatif kapasite", "54"),
        ):
            page.get_by_label(label).fill(value)
        page.get_by_role("button", name="Senaryoları karşılaştır").click()
        expect(page.get_by_text("Kâr farkı: 6,75")).to_be_visible()
        expect(page.get_by_role("table", name="Temel senaryo kaynak kullanımı")).to_contain_text("Aktif")
        with page.expect_download() as event:
            page.get_by_role("button", name="Sonucu JSON indir").click()
        exported = tmp_path / event.value.suggested_filename
        event.value.save_as(exported)
        assert json.loads(exported.read_text())["profit_delta"] == pytest.approx(6.75)
        assert page.evaluate("document.documentElement.scrollWidth === innerWidth")
        browser.close()
