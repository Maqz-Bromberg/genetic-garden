/* genetic-garden studio -- interactive front end for the Python renderer.
 *
 * All drawing happens server-side; this file only collects parameters,
 * posts them, and drops the returned SVG into the page. Keeping it that
 * way means what you see here is exactly what the build pipeline emits.
 */
"use strict";

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls) => { const n = document.createElement(tag); if (cls) n.className = cls; return n; };

let META = null;
let TAB = "strip";

const state = {
  id: "apollo-rewrite",
  kingdom: "tree",
  genome: null,      // working copy; null means "derive from id"
  n: 8,
  full: 21,
  size: 340,
  detail: "full",
  fit: "mature",
  gamma: 0.55,
  ground: true,
  background: "#f4f1e8",
  variantSalt: "v0",
  gardenSeed: 3,
  gardenCount: 26,
  gardenHorizon: 0.2,
  gardenHeight: 500,
  divSalt: 0,
  divScale: 0.07,
};

/* ------------------------------------------------------------------ net */

async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function requestBody(include) {
  return {
    id: state.id,
    kingdom: state.kingdom,
    genome: state.genome,
    contributions: state.n,
    full: state.full,
    size: state.size,
    detail: state.detail,
    fit: state.fit,
    gamma: state.gamma,
    ground: state.ground,
    background: state.background,
    include,
    variant_salt: state.variantSalt,
    variant_count: 8,
    div_salt: state.divSalt,
    div_scale: state.divScale,
  };
}

/* --------------------------------------------------------------- render */

let pending = null;

async function refresh(include) {
  const want = include || ["main", TAB];
  try {
    const data = await post("/api/render", requestBody(want.filter(Boolean)));
    // Adopt the genome the server expressed, so the sliders track reality
    // after a reseed, a kingdom switch or a reset.
    state.genome = data.genome;
    state.kingdom = data.kingdom;
    paintKingdoms();
    paintGrowth(data);
    if (data.main) paintMain(data.main);
    if (data.strip) paintStrip(data.strip);
    if (data.icons) paintIcons(data.icons);
    if (data.variants) paintVariants(data.variants);
    if (data.divergence) paintDivergence(data.divergence);
    paintGenes(data.traits);
  } catch (err) {
    $("#stats").innerHTML = `<span class="err">${err.message}</span>`;
  }
}

function schedule(include, delay) {
  clearTimeout(pending);
  pending = setTimeout(() => refresh(include), delay === undefined ? 90 : delay);
}

/* --------------------------------------------------------------- paint */

function paintMain(main) {
  $("#plant").innerHTML = main.svg;
  $("#stats").innerHTML = [
    ["size", `${(main.bytes / 1024).toFixed(1)} KB`],
    ["branches", main.branches],
    ["leaves", main.leaves],
    ["masses", main.masses],
    ["blooms", main.blooms],
  ].map(([k, v]) => `${k} <b>${v}</b>`).join("");
}

function paintGrowth(d) {
  $("#growth-readout").innerHTML = [
    ["stage", d.stage],
    ["maturity", d.maturity.toFixed(2)],
    ["depth", d.depth.toFixed(2)],
    ["leaf", d.leafiness.toFixed(2)],
    ["bloom", d.bloom.toFixed(2)],
    ["wood", d.woodiness.toFixed(2)],
  ].map(([k, v]) => `${k} <b>${v}</b>`).join("");
}

function figure(svg, label, caption) {
  const f = el("figure");
  f.innerHTML = svg;
  if (label !== undefined) {
    const c = el("figcaption");
    c.innerHTML = caption ? `${label}<br>${caption}` : label;
    f.appendChild(c);
  }
  return f;
}

function fill(node, children) {
  node.replaceChildren(...children);
}

function paintStrip(strip) {
  fill($("#strip"), strip.map((s) => figure(s.svg, `n=${s.n}`, s.stage)));
}

function paintIcons(icons) {
  fill($("#icons"), icons.map((entry) => {
    const f = el("figure");
    const set = el("div", "iconset");
    entry.sizes.forEach((s) => {
      const box = el("span");
      box.innerHTML = s.svg;
      set.appendChild(box);
    });
    f.appendChild(set);
    const c = el("figcaption");
    c.textContent = `n=${entry.n}`;
    f.appendChild(c);
    return f;
  }));
}

function paintVariants(variants) {
  fill($("#variants"), variants.map((v) => {
    const f = figure(v.svg, v.id);
    f.style.cursor = "pointer";
    f.title = "Load this genome into the editor";
    f.onclick = () => {
      state.genome = v.genome;
      state.id = v.id;
      $("#pid").value = v.id;
      refresh();
    };
    return f;
  }));
}

function paintDivergence(rows) {
  fill($("#divergence"), rows.map((row) => {
    const r = el("div", "divrow");
    const tag = el("span", "tag");
    tag.textContent = `n=${row.n}`;
    r.appendChild(tag);
    row.svgs.forEach((svg) => {
      const box = el("span");
      box.innerHTML = svg;
      r.appendChild(box);
    });
    return r;
  }));
}

function paintKingdoms() {
  const host = $("#kingdoms");
  if (host.childElementCount !== META.kingdoms.length) {
    fill(host, META.kingdoms.map((k) => {
      const b = el("button");
      b.textContent = k;
      b.dataset.kingdom = k;
      b.onclick = () => {
        // Keep the genes, change the kingdom: the same chromosome expressed
        // as a tree and as a flower shows the family resemblance.
        state.kingdom = k;
        refresh();
      };
      return b;
    }));
  }
  [...host.children].forEach((b) => {
    b.classList.toggle("on", b.dataset.kingdom === state.kingdom);
  });
}

/* ---------------------------------------------------------- gene editor */

let geneInputs = {};

function buildGenes() {
  const host = $("#genes");
  geneInputs = {};
  const groups = Object.entries(META.groups).map(([name, genes], i) => {
    const d = el("details");
    if (i === 0) d.open = true;
    const s = el("summary");
    s.textContent = name;
    d.appendChild(s);
    genes.forEach((g) => {
      const row = el("div", "gene");
      const label = el("span");
      label.textContent = g;
      label.title = g;
      const input = el("input");
      input.type = "range";
      input.min = 0; input.max = 1; input.step = 0.002;
      input.oninput = () => {
        if (!state.genome) return;
        state.genome.genes[g] = parseFloat(input.value);
        schedule(["main", TAB], 60);
      };
      const out = el("em");
      row.append(label, input, out);
      d.appendChild(row);
      geneInputs[g] = { input, out };
    });
    return d;
  });
  fill(host, groups);
}

function paintGenes(traits) {
  if (!state.genome) return;
  for (const [g, refs] of Object.entries(geneInputs)) {
    const v = state.genome.genes[g];
    if (document.activeElement !== refs.input) refs.input.value = v;
    const t = traits ? traits[g] : undefined;
    refs.out.textContent = t === undefined ? v.toFixed(2)
      : (Math.abs(t) >= 100 ? t.toFixed(0) : t.toFixed(2));
    refs.out.title = `gene ${v.toFixed(3)} → trait ${t}`;
  }
}

/* -------------------------------------------------------------- wiring */

function bindRange(id, key, fmt, include, delay) {
  const input = $("#" + id);
  const out = $("#" + id + "-out");
  const sync = () => {
    const v = parseFloat(input.value);
    state[key] = v;
    if (out) out.textContent = fmt ? fmt(v) : v;
  };
  input.addEventListener("input", () => { sync(); schedule(include, delay); });
  sync();
}

function bindSelect(id, key) {
  const node = $("#" + id);
  node.value = state[key];
  node.addEventListener("change", () => { state[key] = node.value; refresh(); });
}

async function boot() {
  META = await (await fetch("/api/meta")).json();
  state.full = META.full;
  buildGenes();

  bindRange("n", "n", (v) => v, ["main", "strip", "icons", "variants"], 60);
  bindRange("full", "full", (v) => v, undefined, 150);
  bindRange("size", "size", (v) => v, ["main"], 60);
  bindRange("gamma", "gamma", (v) => v.toFixed(2), undefined, 60);
  bindSelect("detail", "detail");
  bindSelect("fit", "fit");

  $("#pid").addEventListener("input", (e) => {
    state.id = e.target.value.trim() || "apollo";
    state.genome = null;            // re-derive from the id
    schedule(undefined, 220);
  });

  $("#ground").addEventListener("change", (e) => {
    state.ground = e.target.checked; refresh();
  });
  const bgSync = () => {
    state.background = $("#bg-on").checked ? $("#bg").value : null;
    refresh();
  };
  $("#bg-on").addEventListener("change", bgSync);
  $("#bg").addEventListener("input", () => {
    if ($("#bg-on").checked) state.background = $("#bg").value;
    schedule(undefined, 120);
  });
  $("#bg").addEventListener("change", bgSync);

  $("#reseed").onclick = () => {
    if (state.genome) state.genome.seed = Math.floor(Math.random() * 2 ** 40);
    refresh();
  };
  $("#randomize").onclick = () => {
    if (!state.genome) return;
    for (const g of META.genes) state.genome.genes[g] = Math.random();
    state.genome.seed = Math.floor(Math.random() * 2 ** 40);
    refresh();
  };
  $("#reset").onclick = () => { state.genome = null; refresh(); };

  const mut = $("#mut"), mutOut = $("#mut-out");
  mut.addEventListener("input", () => mutOut.textContent = parseFloat(mut.value).toFixed(2));
  $("#mutate").onclick = () => {
    if (!state.genome) return;
    const rate = parseFloat(mut.value);
    for (const g of META.genes) {
      if (Math.random() < rate) {
        const delta = (Math.random() + Math.random() + Math.random() - 1.5) * 0.14;
        state.genome.genes[g] = Math.min(1, Math.max(0, state.genome.genes[g] + delta));
      }
    }
    refresh();
  };

  $("#copy").onclick = async () => {
    await navigator.clipboard.writeText(JSON.stringify(state.genome, null, 2));
    $("#copy").textContent = "copied";
    setTimeout(() => $("#copy").textContent = "copy JSON", 1200);
  };

  $("#download").onclick = () => {
    const svg = $("#plant").innerHTML;
    const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
    const a = el("a");
    a.href = url;
    a.download = `${state.id}-${state.kingdom}-${state.n}.svg`;
    a.click();
    URL.revokeObjectURL(url);
  };

  $("#tabs").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-tab]");
    if (!btn) return;
    TAB = btn.dataset.tab;
    [...$("#tabs").children].forEach((b) => b.classList.toggle("on", b === btn));
    document.querySelectorAll(".tabpane").forEach((p) => {
      p.classList.toggle("hidden", p.dataset.pane !== TAB);
    });
    if (TAB === "breed") doBreed();
    else if (TAB === "garden") doGarden();
    else refresh(["main", TAB]);
  });

  $("#reroll-variants").onclick = () => {
    state.variantSalt = "v" + Math.floor(Math.random() * 1e6);
    refresh(["variants"]);
  };
  $("#reroll-div").onclick = () => {
    state.divSalt = Math.floor(Math.random() * 1e6);
    refresh(["divergence"]);
  };
  const divScale = $("#div-scale");
  divScale.addEventListener("input", () => {
    state.divScale = parseFloat(divScale.value);
    $("#div-scale-out").textContent = state.divScale.toFixed(2);
    schedule(["divergence"], 140);
  });

  const gardenRange = (id, key, fmt) => {
    const input = $("#" + id), out = $("#" + id + "-out");
    input.addEventListener("input", () => {
      state[key] = parseFloat(input.value);
      out.textContent = fmt ? fmt(state[key]) : state[key];
      clearTimeout(pending);
      pending = setTimeout(doGarden, 260);
    });
  };
  gardenRange("g-count", "gardenCount");
  gardenRange("g-horizon", "gardenHorizon", (v) => v.toFixed(2));
  gardenRange("g-height", "gardenHeight");
  $("#reroll-garden").onclick = () => {
    state.gardenSeed = Math.floor(Math.random() * 1e6);
    doGarden();
  };

  const bmut = $("#bmut");
  bmut.addEventListener("input", () => {
    $("#bmut-out").textContent = parseFloat(bmut.value).toFixed(2);
  });
  $("#dobreed").onclick = doBreed;

  await refresh();
}

async function doGarden() {
  const host = $("#garden");
  host.style.opacity = "0.45";
  try {
    // Width follows the panel, so the plane fills whatever space there is.
    const width = Math.max(320, Math.round(host.clientWidth || 900));
    const data = await post("/api/garden", {
      count: state.gardenCount,
      seed: state.gardenSeed,
      width,
      height: state.gardenHeight,
      horizon: state.gardenHorizon,
      full: state.full,
    });
    host.innerHTML = data.svg;
    const kinds = {};
    data.plants.forEach((p) => { kinds[p.kingdom] = (kinds[p.kingdom] || 0) + 1; });
    $("#garden-stats").innerHTML = [
      ["planted", `${data.planted}/${data.requested}`],
      ["trees", kinds.tree || 0],
      ["bushes", kinds.bush || 0],
      ["flowers", kinds.flower || 0],
      ["seed", data.seed],
      ["size", `${(data.bytes / 1024).toFixed(0)} KB`],
    ].map(([k, v]) => `${k} <b>${v}</b>`).join("");
  } catch (err) {
    host.innerHTML = `<span class="err">${err.message}</span>`;
  } finally {
    host.style.opacity = "1";
  }
}

async function doBreed() {
  try {
    const data = await post("/api/breed", {
      a: $("#pa").value.trim() || "apollo",
      b: $("#pb").value.trim() || "borealis",
      kingdom: state.kingdom,
      mutate: parseFloat($("#bmut").value),
      seed: Math.floor(Math.random() * 1e6),
      contributions: state.n,
      full: state.full,
      size: 170,
      background: state.background,
    });
    const host = $("#breed");
    const kids = [];
    kids.push(figure(data.parents[0].svg, data.parents[0].id));
    const plus = el("span", "plus"); plus.textContent = "×";
    kids.push(plus);
    kids.push(figure(data.parents[1].svg, data.parents[1].id));
    const eq = el("span", "plus"); eq.textContent = "→";
    kids.push(eq);
    const child = figure(data.child.svg, "child");
    child.style.cursor = "pointer";
    child.title = "Load this genome into the editor";
    child.onclick = () => { state.genome = data.child.genome; refresh(); };
    kids.push(child);
    fill(host, kids);
  } catch (err) {
    $("#breed").innerHTML = `<span class="err">${err.message}</span>`;
  }
}

boot();
