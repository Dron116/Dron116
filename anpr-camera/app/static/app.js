const live = document.getElementById("live");
const historyEl = document.getElementById("history");
const searchEl = document.getElementById("search");
const sourceLine = document.getElementById("source-line");
const modeChip = document.getElementById("mode-chip");
const countChip = document.getElementById("count-chip");
const clock = document.getElementById("clock");

function pad(value) {
  return String(value).padStart(2, "0");
}

function formatTime(ts) {
  const date = new Date(ts * 1000);
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

function splitPlate(plate) {
  const compact = (plate || "").replace(/\s/g, "");
  if (compact.length < 8) {
    return { letter: "—", num: "---", tail: "--", reg: "--" };
  }
  return {
    letter: compact[0],
    num: compact.slice(1, 4),
    tail: compact.slice(4, 6),
    reg: compact.slice(6),
  };
}

function renderPlate(plate, meta) {
  const parts = splitPlate(plate);
  document.getElementById("p-letter").textContent = parts.letter;
  document.getElementById("p-num").textContent = parts.num;
  document.getElementById("p-tail").textContent = parts.tail;
  document.getElementById("p-reg").textContent = parts.reg || "--";
  document.getElementById("last-meta").textContent = meta || "Ожидание проезда";
}

function renderHistory(items) {
  historyEl.innerHTML = items
    .map(
      (item) => `
      <li>
        <span>${item.display}</span>
        <time>${formatTime(item.created_at)}</time>
      </li>`
    )
    .join("");
}

async function refresh() {
  const status = await fetch("/api/status").then((r) => r.json());
  const camera = status.camera || {};
  const stats = status.stats || {};
  const mode = camera.mode === "live" ? "live" : "demo";
  modeChip.textContent = mode === "live" ? "Камера" : "Демо";
  modeChip.className = `chip ${mode}`;
  countChip.textContent = `${stats.unique || 0} уник. / ${stats.total || 0} всего`;
  sourceLine.textContent = camera.connected
    ? `Источник: ${camera.source || camera.camera_host}`
    : `Камера ${camera.camera_host} недоступна, включён демо-поток`;
  document.getElementById("cam-host").textContent = camera.camera_host || "—";
  document.getElementById("cam-mode").textContent = camera.mode || "—";
  document.getElementById("cam-source").textContent = camera.source || "—";
  document.getElementById("cam-error").textContent = camera.last_error || "нет";
  document.getElementById("res-label").textContent = camera.width
    ? `${camera.width}×${camera.height}`
    : "нет кадра";

  if (status.latest && status.latest[0]) {
    const last = status.latest[0];
    renderPlate(last.plate, `уверенность ${Math.round(last.confidence * 100)}%`);
  } else if (stats.last) {
    renderPlate(stats.last.plate, `последняя фиксация ${formatTime(stats.last.created_at)}`);
  }

  const query = searchEl.value.trim();
  const detections = await fetch(`/api/detections?limit=40&q=${encodeURIComponent(query)}`).then((r) =>
    r.json()
  );
  renderHistory(detections.items || []);
}

document.getElementById("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = document.getElementById("photo").files[0];
  if (!file) return;
  const body = new FormData();
  body.append("file", file);
  const result = await fetch("/api/recognize", { method: "POST", body }).then((r) => r.json());
  if (result.items && result.items[0]) {
    renderPlate(result.items[0].plate, "распознано из файла");
  }
  refresh();
});

document.getElementById("clear-btn").addEventListener("click", async () => {
  await fetch("/api/detections", { method: "DELETE" });
  renderPlate("", "Журнал очищен");
  refresh();
});

searchEl.addEventListener("input", () => {
  refresh();
});

setInterval(() => {
  clock.textContent = new Date().toLocaleTimeString("ru-RU");
}, 1000);

setInterval(refresh, 2000);
refresh();

try {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.addEventListener("message", () => refresh());
} catch (err) {
  console.warn(err);
}

live.addEventListener("error", () => {
  sourceLine.textContent = "Нет видеопотока, перезагрузите страницу";
});
