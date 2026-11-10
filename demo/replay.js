"use strict";

const get = (id) => document.getElementById(id);
const trackSelect = get("track");
const scrub = get("scrub");
const play = get("play");
const plot = get("plot");
const image = get("scene-image");
const colors = { lru: "#9db0c9", fixed: "#ffbd73", short_only: "#64dfc2" };
const buttons = Array.from(get("policies").querySelectorAll("button"));
const plotLeft = 42;
const plotRight = 12;
let policy = "short_only";
let frame = 0;
let timer = null;

function currentTrack() {
  return REPLAY_DATA[trackSelect.value];
}

function chart(track) {
  const context = plot.getContext("2d");
  const width = plot.clientWidth;
  const height = plot.clientHeight;
  const scale = window.devicePixelRatio || 1;
  plot.width = Math.round(width * scale);
  plot.height = Math.round(height * scale);
  context.scale(scale, scale);
  const bottom = height - 21;
  const right = width - plotRight;
  const last = track.cameras.length - 1;
  const limit = Math.max(40, Math.ceil(Math.max(...Object.values(track.policies).flatMap(
    (values) => values.frames.slice(1).map((sample) => sample.wall))) / 10) * 10);
  const x = (index) => plotLeft + (right - plotLeft) * index / last;
  const y = (ms) => bottom - (bottom - 8) * ms / limit;
  const line = (x1, y1, x2, y2) => {
    context.beginPath();
    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
    context.stroke();
  };
  context.font = "12px Segoe UI, sans-serif";
  context.fillStyle = "#a9b9cc";
  context.strokeStyle = "#34475b";
  for (const tick of [0, limit / 2, limit]) {
    line(plotLeft, y(tick), right, y(tick));
    context.fillText(String(tick), 4, y(tick) + 4);
  }
  context.fillText("ms", 4, height - 3);
  context.fillText("0", plotLeft, height - 3);
  context.fillText(String(last), right - context.measureText(String(last)).width, height - 3);
  context.strokeStyle = "#b87863";
  context.setLineDash([5, 4]);
  line(plotLeft, y(33.3), right, y(33.3));
  context.setLineDash([]);
  const plotted = [...Object.keys(track.policies).filter((name) => name !== policy), policy];
  for (const name of plotted) {
    context.strokeStyle = colors[name];
    context.globalAlpha = name === policy ? 1 : 0.47;
    context.lineWidth = name === policy ? 2.2 : 1.15;
    context.beginPath();
    track.policies[name].frames.forEach((sample, index) => {
      if (index === 0) return;
      if (index === 1) context.moveTo(x(index), y(sample.wall));
      else context.lineTo(x(index), y(sample.wall));
    });
    context.stroke();
  }
  context.globalAlpha = 1;
  context.lineWidth = 1;
  context.strokeStyle = "#eef4fc";
  line(x(frame), 8, x(frame), bottom);
  if (frame > 0) {
    context.fillStyle = colors[policy];
    context.beginPath();
    context.arc(x(frame), y(track.policies[policy].frames[frame].wall), 4, 0, Math.PI * 2);
    context.fill();
  }
}

function render() {
  const track = currentTrack();
  const values = track.policies[policy];
  const sample = values.frames[frame];
  const path = "media/" + trackSelect.value + "/" + String(frame).padStart(4, "0") + ".png";
  if (image.getAttribute("src") !== path) image.src = path;
  scrub.value = frame;
  get("frame-label").textContent = "第 " + frame + " 帧 / " + (track.cameras.length - 1);
  get("camera-label").textContent = track.cameras[frame];
  get("wall").textContent = sample.wall.toFixed(2) + " ms";
  get("p99").textContent = values.p99.toFixed(2) + " ms";
  for (const key of ["demand", "prefetch", "resident"]) {
    get(key).textContent = sample[key].toFixed(2) + " MiB";
  }
  for (const key of ["transfer", "wasted"]) {
    get(key).textContent = values[key].toFixed(2) + " MiB";
  }
  get("slow").textContent = values.slow + " / " + values.samples;
  get("resident-bar").style.width = 100 * sample.resident / track.budget + "%";
  buttons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.policy === policy)));
  chart(track);
}

function seek(index) {
  frame = Math.max(0, Math.min(index, currentTrack().cameras.length - 1));
  render();
}

function pause() {
  clearInterval(timer);
  timer = null;
  play.textContent = "播放";
}

function resetTrack() {
  pause();
  scrub.max = currentTrack().cameras.length - 1;
  const jumps = trackSelect.value === "capture_order" ? [15, 183, 194] : [30, 60, 90];
  get("jumps").replaceChildren(...jumps.map((index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "第 " + index + " 帧";
    button.addEventListener("click", () => { pause(); seek(index); });
    return button;
  }));
  seek(0);
}

play.addEventListener("click", () => {
  if (timer !== null) {
    pause();
    return;
  }
  if (frame === Number(scrub.max)) seek(0);
  play.textContent = "暂停";
  timer = setInterval(() => {
    seek(frame + 1);
    if (frame === Number(scrub.max)) pause();
  }, 85);
});
scrub.addEventListener("input", () => { pause(); seek(Number(scrub.value)); });
plot.addEventListener("click", (event) => {
  pause();
  seek(Math.round((event.offsetX - plotLeft) / (plot.clientWidth - plotLeft - plotRight) * Number(scrub.max)));
});
trackSelect.addEventListener("change", resetTrack);
buttons.forEach((button) => button.addEventListener("click", () => { policy = button.dataset.policy; render(); }));
window.addEventListener("resize", () => chart(currentTrack()));
resetTrack();
