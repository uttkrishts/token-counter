const state = {
  providers: {},
  files: [],
};

const els = {
  activeEncoding: document.querySelector("#activeEncoding"),
  breakdown: document.querySelector("#breakdown"),
  choice: document.querySelector("#choice"),
  choiceLabel: document.querySelector("#choiceLabel"),
  choiceType: document.querySelector("#choiceType"),
  choiceTypeLabel: document.querySelector("#choiceTypeLabel"),
  clearPrompt: document.querySelector("#clearPrompt"),
  combinedTokens: document.querySelector("#combinedTokens"),
  countButton: document.querySelector("#countButton"),
  customLabel: document.querySelector("#customLabel"),
  customChoice: document.querySelector("#customChoice"),
  dropzone: document.querySelector("#dropzone"),
  estimateList: document.querySelector("#estimateList"),
  estimatePanel: document.querySelector("#estimatePanel"),
  fileInput: document.querySelector("#fileInput"),
  fileList: document.querySelector("#fileList"),
  fileTokens: document.querySelector("#fileTokens"),
  mediaDetail: document.querySelector("#mediaDetail"),
  ocrMode: document.querySelector("#ocrMode"),
  prompt: document.querySelector("#prompt"),
  promptTokens: document.querySelector("#promptTokens"),
  provider: document.querySelector("#provider"),
  status: document.querySelector("#status"),
};

function formatNumber(value) {
  return new Intl.NumberFormat().format(value || 0);
}

function setStatus(message, isError = false) {
  els.status.textContent = message;
  els.status.classList.toggle("error", isError);
}

function option(label, value) {
  const item = document.createElement("option");
  item.textContent = label;
  item.value = value;
  return item;
}

function replaceOptions(select, items) {
  select.replaceChildren(...items.map((item) => option(item.label, item.value)));
}

function selectedProvider() {
  return els.provider.value;
}

function providerData(name = selectedProvider()) {
  return state.providers[name] || {};
}

function setCustomState({ disabled, label, placeholder }) {
  els.customChoice.disabled = disabled;
  els.customChoice.value = disabled ? "" : els.customChoice.value;
  els.customLabel.textContent = label;
  els.customChoice.placeholder = placeholder;
}

function populateModeOptions() {
  const provider = selectedProvider();

  if (provider === "openai") {
    els.choiceType.disabled = false;
    els.choiceTypeLabel.textContent = "Token source";
    replaceOptions(els.choiceType, [
      { label: "Model", value: "model" },
      { label: "Encoding", value: "encoding" },
    ]);
    setCustomState({
      disabled: false,
      label: "Custom model or encoding",
      placeholder: "Optional, e.g. gpt-4o-mini",
    });
  } else if (provider === "gemini") {
    els.choiceType.disabled = true;
    els.choiceTypeLabel.textContent = "Token source";
    replaceOptions(els.choiceType, [{ label: "Model", value: "model" }]);
    setCustomState({
      disabled: false,
      label: "Custom Gemini model",
      placeholder: "Optional, e.g. gemini-2.5-flash",
    });
  } else {
    els.choiceType.disabled = true;
    els.choiceTypeLabel.textContent = "Token source";
    replaceOptions(els.choiceType, [{ label: "Estimate", value: "estimate" }]);
    setCustomState({
      disabled: true,
      label: "Custom model",
      placeholder: "Claude estimates do not use model names",
    });
  }

  els.choice.replaceChildren();
  populateChoices();
}

function populateChoices() {
  const provider = selectedProvider();
  const data = providerData(provider);
  els.choice.replaceChildren();

  if (!data.available) {
    els.choiceLabel.textContent = "Unavailable";
    els.choice.append(option("Install dependencies first", ""));
    setStatus(`${provider} support unavailable. Run: ${data.install}`, true);
    return;
  }

  if (provider === "openai" && els.choiceType.value === "encoding") {
    els.choiceLabel.textContent = "Encoding";
    for (const name of data.encodings) {
      els.choice.append(option(name, name));
    }
    return;
  }

  if (provider === "openai") {
    els.choiceLabel.textContent = "Model";
    for (const model of data.models) {
      els.choice.append(option(`${model.name} (${model.encoding})`, model.name));
    }

    if (data.modelPrefixes.length) {
      const group = document.createElement("optgroup");
      group.label = "Versioned model prefixes";
      for (const item of data.modelPrefixes) {
        group.append(option(`${item.prefix}* (${item.encoding})`, item.prefix));
      }
      els.choice.append(group);
    }
    return;
  }

  if (provider === "gemini") {
    els.choiceLabel.textContent = "Gemini model";
    for (const model of data.models) {
      els.choice.append(option(model.name, model.name));
    }
    return;
  }

  els.choiceLabel.textContent = "Estimate profile";
  for (const estimate of data.estimates) {
    els.choice.append(option(estimate.label, estimate.name));
  }
}

async function loadModels() {
  const response = await fetch("/api/models");
  const data = await response.json();

  state.providers = data.providers;
  populateModeOptions();
  const openaiCount = state.providers.openai?.models?.length || 0;
  const geminiCount = state.providers.gemini?.models?.length || 0;
  setStatus(`Loaded ${openaiCount} OpenAI models, ${geminiCount} Gemini local model options, and Claude estimate profiles.`);
}

function currentChoice() {
  if (selectedProvider() === "claude") {
    return els.choice.value;
  }

  const custom = els.customChoice.value.trim();
  return custom || els.choice.value;
}

function renderFiles() {
  els.fileList.replaceChildren();

  if (!state.files.length) {
    const empty = document.createElement("li");
    empty.innerHTML = "<span>No files selected.</span>";
    els.fileList.append(empty);
    return;
  }

  state.files.forEach((file, index) => {
    const li = document.createElement("li");

    const meta = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = file.name;
    const detail = document.createElement("span");
    const size = file.size
      ? `${formatNumber(file.size)} bytes`
      : `${formatNumber(file.content?.length || 0)} characters`;
    detail.textContent = `${(file.kind || "text").toUpperCase()} - ${size}`;
    meta.append(name, detail);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => {
      state.files.splice(index, 1);
      renderFiles();
    });

    li.append(meta, remove);
    els.fileList.append(li);
  });
}

function fileKind(file) {
  const name = file.name.toLowerCase();
  if (file.type === "application/pdf" || name.endsWith(".pdf")) {
    return "pdf";
  }
  if (file.type.startsWith("image/") || /\.(png|jpe?g|webp|gif)$/.test(name)) {
    return "image";
  }
  return "text";
}

function readAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      const result = String(reader.result || "");
      resolve(result.includes(",") ? result.split(",").pop() : result);
    });
    reader.addEventListener("error", () => reject(reader.error));
    reader.readAsDataURL(file);
  });
}

async function addFiles(files) {
  for (const file of [...files]) {
    const kind = fileKind(file);
    if (kind === "text") {
      const content = await file.text();
      state.files.push({
        name: file.name,
        type: file.type || "text/plain",
        size: file.size,
        kind,
        content,
      });
    } else {
      const data = await readAsBase64(file);
      state.files.push({
        name: file.name,
        type: file.type,
        size: file.size,
        kind,
        data,
      });
    }
  }
  renderFiles();
}

function renderBreakdown(data) {
  const rows = [
    {
      name: "Prompt",
      type: "Text",
      chars: data.prompt.chars,
      bytes: data.prompt.bytes,
      textTokens: data.prompt.tokens,
      mediaTokens: 0,
      tokens: data.prompt.tokens,
    },
    ...data.files,
  ];

  if (!rows.length) {
    els.breakdown.innerHTML = '<tr><td colspan="6" class="empty">No content counted.</td></tr>';
    return;
  }

  els.breakdown.replaceChildren(
    ...rows.map((row) => {
      const tr = document.createElement("tr");
      for (const value of [
        row.name,
        row.type || "Text",
        row.textTokens ?? 0,
        row.mediaTokens ?? 0,
        row.bytes,
        row.tokens,
      ]) {
        const td = document.createElement("td");
        td.textContent = typeof value === "number" ? formatNumber(value) : value;
        tr.append(td);
      }
      return tr;
    }),
  );
}

function renderEstimates(estimates) {
  els.estimatePanel.classList.toggle("hidden", !estimates.length);
  els.estimateList.replaceChildren();

  for (const item of estimates) {
    const card = document.createElement("div");
    card.className = "estimate-item";

    const label = document.createElement("span");
    label.textContent = item.label;
    const tokens = document.createElement("strong");
    tokens.textContent = formatNumber(item.tokens);
    const method = document.createElement("small");
    method.textContent = item.method;

    card.append(label, tokens, method);
    els.estimateList.append(card);
  }
}

async function countTokens() {
  const choice = currentChoice();
  if (!choice) {
    setStatus("Choose a tokenizer first.", true);
    return;
  }

  els.countButton.disabled = true;
  setStatus("Counting tokens...");

  try {
    const response = await fetch("/api/count", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: selectedProvider(),
        choice,
        choiceType: els.choiceType.value,
        mediaDetail: els.mediaDetail.value,
        ocrMode: els.ocrMode.value,
        prompt: els.prompt.value,
        files: state.files,
      }),
    });
    const data = await response.json();
    if (!data.ok) {
      throw new Error(data.error || "Token count failed.");
    }

    els.promptTokens.textContent = formatNumber(data.prompt.tokens);
    els.fileTokens.textContent = formatNumber(data.fileTokens);
    els.combinedTokens.textContent = formatNumber(data.combinedTokens);
    const quality = data.exact ? "Exact" : "Estimate";
    const media = data.mediaTokens ? `, media ${formatNumber(data.mediaTokens)}` : "";
    els.activeEncoding.textContent = `${quality}: ${data.method} (${data.choice}${media})`;
    renderBreakdown(data);
    renderEstimates(data.estimates || []);
    setStatus("Token count updated.");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    els.countButton.disabled = false;
  }
}

els.provider.addEventListener("change", populateModeOptions);
els.choiceType.addEventListener("change", populateChoices);
els.countButton.addEventListener("click", countTokens);
els.clearPrompt.addEventListener("click", () => {
  els.prompt.value = "";
});
els.fileInput.addEventListener("change", (event) => {
  addFiles(event.target.files);
  event.target.value = "";
});

for (const eventName of ["dragenter", "dragover"]) {
  els.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropzone.classList.add("active");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  els.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropzone.classList.remove("active");
  });
}

els.dropzone.addEventListener("drop", (event) => {
  addFiles(event.dataTransfer.files);
});

renderFiles();
loadModels().catch((error) => setStatus(error.message, true));
