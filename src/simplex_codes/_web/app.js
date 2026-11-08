"use strict";

const byId = (id) => document.getElementById(id);
const displayNumber = new Intl.NumberFormat("tr-TR", { maximumSignificantDigits: 8 });
const format = (value) => displayNumber.format(value);
const form = byId("model");
let variables = [];
let constraints = [];
let revision = 0;

function cell(row, value, header = false) {
  const element = document.createElement(header ? "th" : "td");
  if (header) element.scope = "row";
  if (value instanceof Node) element.append(value);
  else element.textContent = value;
  row.append(element);
}

function markEdited() {
  revision += 1;
  byId("error").hidden = true;
  byId("status").textContent = byId("report").hidden ? "Çözüme hazır" : "Model değişti";
}

function numberInput(label, value, update, required = true) {
  const input = document.createElement("input");
  input.type = "number";
  input.step = "any";
  input.required = required;
  input.value = value;
  if (!required) input.placeholder = "Serbest";
  input.setAttribute("aria-label", label);
  input.addEventListener("input", () => {
    update(input.value);
    markEdited();
  });
  return input;
}

function removeButton(label, remove, disabled = false) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "remove";
  button.textContent = "Sil";
  button.disabled = disabled;
  button.setAttribute("aria-label", label);
  button.addEventListener("click", () => {
    remove();
    renderModel();
    markEdited();
    byId(label.startsWith("Kısıt") ? "add-constraint" : "add-variable").focus();
  });
  return button;
}

function renderModel() {
  const objective = byId("variables");
  objective.replaceChildren();
  const header = document.createElement("tr");
  for (const [index, variable] of variables.entries()) {
    const name = "x" + (index + 1);
    const row = document.createElement("tr");
    cell(row, name, true);
    cell(row, numberInput(name + " amaç katsayısı", variable.cost, (value) => { variable.cost = value; }));
    cell(row, numberInput(name + " alt sınırı", variable.lower, (value) => { variable.lower = value; }, false));
    cell(row, numberInput(name + " üst sınırı", variable.upper, (value) => { variable.upper = value; }, false));
    cell(row, removeButton(name + " değişkenini sil", () => {
      variables.splice(index, 1);
      constraints.forEach((constraint) => constraint.coefficients.splice(index, 1));
    }, variables.length === 1));
    objective.append(row);
  }

  for (const label of ["Satır", ...variables.map((_, index) => "x" + (index + 1)), "İlişki", "Sağ taraf", ""]) {
    const heading = document.createElement("th");
    heading.scope = "col";
    heading.textContent = label;
    if (!label) heading.setAttribute("aria-label", "İşlem");
    header.append(heading);
  }
  byId("constraint-head").replaceChildren(header);

  const rows = byId("constraints");
  rows.replaceChildren();
  for (const [index, constraint] of constraints.entries()) {
    const label = "Kısıt " + (index + 1);
    const row = document.createElement("tr");
    cell(row, index + 1, true);
    for (const [column, value] of constraint.coefficients.entries()) {
      cell(row, numberInput(label + " x" + (column + 1) + " katsayısı", value, (updated) => {
        constraint.coefficients[column] = updated;
      }));
    }
    const relation = document.createElement("select");
    relation.setAttribute("aria-label", label + " ilişkisi");
    relation.add(new Option("≤", "<="));
    relation.add(new Option("≥", ">="));
    relation.add(new Option("=", "="));
    relation.value = constraint.relation;
    relation.addEventListener("change", () => {
      constraint.relation = relation.value;
      markEdited();
    });
    cell(row, relation);
    cell(row, numberInput(label + " sağ tarafı", constraint.rhs, (value) => { constraint.rhs = value; }));
    cell(row, removeButton(label + " satırını sil", () => constraints.splice(index, 1)));
    rows.append(row);
  }
  byId("add-variable").disabled = variables.length >= 12;
  byId("add-constraint").disabled = constraints.length >= 30;
}

function loadExample() {
  variables = [8, 5].map((cost) => ({ cost, lower: "0", upper: "" }));
  constraints = [
    { coefficients: [1, 1], relation: "<=", rhs: 6 },
    { coefficients: [9, 5], relation: "<=", rhs: 45 },
  ];
  byId("sense").value = "max";
  byId("report").hidden = true;
  byId("placeholder").hidden = false;
  renderModel();
  markEdited();
}

function showPairs(id, pairs) {
  byId(id).replaceChildren(...pairs.map(([label, value]) => {
    const wrapper = document.createElement("div");
    const name = document.createElement("dt");
    const number = document.createElement("dd");
    name.textContent = label;
    number.textContent = format(value);
    wrapper.append(name, number);
    return wrapper;
  }));
}

function showReport(result) {
  showPairs("metrics", [["Amaç değeri", result.objective_value], ["Pivot sayısı", result.iterations]]);
  showPairs("values", result.solution.map((value, index) => ["x" + (index + 1), value]));
  const chart = byId("chart");
  chart.hidden = !result.chart;
  if (result.chart) chart.src = "data:image/png;base64," + result.chart;
  else chart.removeAttribute("src");
  byId("chart-note").textContent = result.chart
    ? "Renkli alan uygun bölgeyi, numaralar pivot sırasını gösterir. Faz 1 adımları bu alanın dışında kalabilir."
    : result.chart_message
      ? "Grafiği göstermek için viz eklentisini kur."
      : "Grafik iki değişkenli modellerde gösterilir. Tüm adımlar aşağıdaki tabloda yer alır.";
  const steps = document.createDocumentFragment();
  const columnName = (column) => column === null ? "—" : "Sütun " + (column + 1);
  for (const step of result.steps) {
    const row = document.createElement("tr");
    cell(row, step.iteration === 0 ? "0 · Başlangıç" : step.iteration, true);
    for (const value of [step.phase, columnName(step.entering), columnName(step.leaving),
      "(" + step.solution.map(format).join("; ") + ")", format(step.objective_value)]) {
      cell(row, value);
    }
    steps.append(row);
  }
  byId("steps").replaceChildren(steps);
  byId("placeholder").hidden = true;
  byId("report").hidden = false;
  byId("status").textContent = "Optimal çözüm";
}

byId("example").addEventListener("click", loadExample);
byId("sense").addEventListener("change", markEdited);
byId("add-variable").addEventListener("click", () => {
  variables.push({ cost: "0", lower: "0", upper: "" });
  constraints.forEach((constraint) => constraint.coefficients.push("0"));
  renderModel();
  markEdited();
  byId("variables").lastElementChild.querySelector("input").focus();
});
byId("add-constraint").addEventListener("click", () => {
  constraints.push({ coefficients: variables.map(() => "0"), relation: "<=", rhs: "0" });
  renderModel();
  markEdited();
  byId("constraints").lastElementChild.querySelector("input").focus();
});
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const solveButton = byId("solve");
  if (solveButton.disabled) return;
  const version = revision;
  const endpoint = (value) => value === "" ? null : Number(value);
  solveButton.disabled = true;
  byId("status").textContent = "Çözülüyor…";
  byId("error").hidden = true;
  byId("report").hidden = true;
  byId("placeholder").hidden = true;
  try {
    const model = {
      c: variables.map((variable) => Number(variable.cost)),
      A: constraints.map((constraint) => constraint.coefficients.map(Number)),
      b: constraints.map((constraint) => Number(constraint.rhs)),
      constraint_senses: constraints.map((constraint) => constraint.relation),
      bounds: variables.map((variable) => [endpoint(variable.lower), endpoint(variable.upper)]),
      sense: byId("sense").value,
    };
    const invalidBound = model.bounds.findIndex(([lo, hi]) => lo !== null && hi !== null && lo > hi);
    if (invalidBound >= 0) throw new Error("x" + (invalidBound + 1) + " için alt sınır üst sınırı aşamaz.");

    const response = await fetch("/api/v1/solve?trace=true&chart=true", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(model),
    });
    const result = await response.json();
    if (result.status !== "optimal") {
      const messages = {
        infeasible: "Bu kısıtları birlikte sağlayan bir çözüm yok.",
        unbounded: "Amaç değeri sınırsız. Kısıtları ve değişken sınırlarını kontrol et.",
        iteration_limit: "Çözücü pivot sınırına ulaştı.",
        numerical_error: "Sonuç sayısal olarak doğrulanamadı. Modelin katsayı ölçeklerini kontrol et.",
      };
      throw new Error(messages[result.status] || result.error || "Çözüm isteği başarısız oldu.");
    }
    if (revision === version) showReport(result);
  } catch (error) {
    if (revision === version) {
      byId("status").textContent = "Çözüm bulunamadı";
      byId("error").textContent = error.message;
      byId("error").hidden = false;
    }
  } finally {
    solveButton.disabled = false;
  }
});

loadExample();
