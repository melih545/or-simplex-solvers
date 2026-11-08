"use strict";

(() => {
  const get = (id) => document.getElementById(id);
  const format = new Intl.NumberFormat("tr-TR", { maximumSignificantDigits: 8 }).format;
  const form = get("scenario-form");
  const error = get("scenario-error");
  const compare = get("compare");
  const download = get("export-comparison");
  let csv = "";
  let capacityInputs = [];
  let report = null;
  let revision = 0;

  function invalidate() {
    revision += 1;
    report = null;
    error.hidden = true;
    download.disabled = true;
    get("scenario-result").hidden = true;
  }

  function showError(message) {
    error.textContent = message;
    error.hidden = false;
  }

  async function post(path, payload) {
    const response = await fetch("/api/v1/scenarios/" + path, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "İstek başarısız oldu.");
    return result;
  }

  function capacity(name, label, value) {
    const input = document.createElement("input");
    input.type = "number";
    input.min = "0";
    input.step = "any";
    input.required = true;
    input.value = value ?? "";
    input.setAttribute("aria-label", name + " " + label);
    input.addEventListener("input", () => {
      invalidate();
      compare.disabled = false;
    });
    return input;
  }

  async function load(source, label, sample = false) {
    invalidate();
    const version = revision;
    compare.disabled = true;
    capacityInputs = [];
    get("scenario-capacities").hidden = true;
    get("scenario-source").textContent = "CSV okunuyor…";
    try {
      const content = await source;
      if (version !== revision) return;
      const model = await post("import", { csv: content });
      if (version !== revision) return;
      csv = content;
      get("scenario-source").textContent = label + " · " + model.products.length + " ürün";
      const body = get("capacity-rows");
      body.replaceChildren();
      const examples = { Depo: [6, 6], İşçilik: [45, 54] };
      capacityInputs = model.resources.map((name) => {
        const defaults = sample ? examples[name] : [];
        const baseline = capacity(name, "temel kapasite", defaults[0]);
        const alternative = capacity(name, "alternatif kapasite", defaults[1]);
        const row = body.insertRow();
        const title = document.createElement("th");
        title.scope = "row";
        title.textContent = name;
        row.append(title);
        row.insertCell().append(baseline);
        row.insertCell().append(alternative);
        return { name, baseline, alternative };
      });
      get("scenario-capacities").hidden = false;
      compare.disabled = false;
    } catch (failure) {
      if (version === revision) {
        get("scenario-source").textContent = "";
        showError(failure.message);
      }
    }
  }

  function element(tag, content) {
    const node = document.createElement(tag);
    node.textContent = content;
    return node;
  }

  function showScenario(name, data) {
    const panel = document.createElement("article");
    panel.className = "scenario-report";
    panel.append(element("h3", name));
    if (data.status !== "optimal") {
      const statuses = {
        unbounded: "Kâr sınırsız.", infeasible: "Uygun çözüm yok.",
        numerical_error: "Sonuç sayısal olarak doğrulanamadı.", iteration_limit: "Pivot sınırına ulaşıldı.",
      };
      panel.append(element("p", statuses[data.status] || data.error || data.status));
      return panel;
    }
    panel.append(element("p", "Kâr: " + format(data.profit)));
    panel.append(element("h4", "Üretim miktarları"));
    for (const [product, quantity] of Object.entries(data.products)) {
      panel.append(element("p", product + ": " + format(quantity)));
    }
    panel.append(element("h4", "Kaynak kullanımı"));
    const wrapper = document.createElement("div");
    wrapper.className = "table-scroll";
    const table = document.createElement("table");
    table.setAttribute("aria-label", name + " kaynak kullanımı");
    const header = table.createTHead().insertRow();
    for (const label of ["Kaynak", "Kullanım", "Kapasite", "Kalan", "Durum"]) {
      header.append(element("th", label));
    }
    const body = table.createTBody();
    for (const resource of data.resources) {
      const row = body.insertRow();
      for (const value of [resource.name, format(resource.used), format(resource.limit), format(resource.slack)]) {
        row.insertCell().textContent = value;
      }
      const status = row.insertCell();
      status.textContent = resource.active ? "Aktif" : "Boşluk var";
      if (resource.active) status.className = "active-resource";
    }
    wrapper.append(table);
    panel.append(wrapper);
    return panel;
  }

  get("scenario-example").addEventListener("click", () => {
    get("production-csv").value = "";
    const source = fetch("/capacity_example.csv").then((response) => {
      if (!response.ok) throw new Error("Örnek CSV yüklenemedi.");
      return response.text();
    });
    load(source, "Üretim planı örneği", true);
  });
  get("production-csv").addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) {
      const source = file.size > 32_768
        ? Promise.reject(new Error("CSV dosyası 32 KB sınırını aşıyor."))
        : file.text();
      load(source, file.name);
    }
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (compare.disabled) return;
    invalidate();
    const version = revision;
    compare.disabled = true;
    const limits = (side) => Object.fromEntries(capacityInputs.map(({ name, [side]: input }) => [name, input.valueAsNumber]));
    const data = { csv, baseline: limits("baseline"), alternative: limits("alternative") };
    try {
      const result = await post("compare", data);
      if (version !== revision) return;
      report = result;
      get("profit-delta").textContent = "profit_delta" in result
        ? "Kâr farkı: " + format(result.profit_delta)
        : "Kâr farkı hesaplanamadı.";
      get("scenario-reports").replaceChildren(
        showScenario("Temel senaryo", result.baseline), showScenario("Alternatif senaryo", result.alternative),
      );
      get("scenario-result").hidden = false;
      download.disabled = false;
    } catch (failure) {
      if (version === revision) showError(failure.message);
    } finally {
      if (version === revision) compare.disabled = false;
    }
  });
  download.addEventListener("click", () => {
    if (!report) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "capacity-comparison.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
})();
