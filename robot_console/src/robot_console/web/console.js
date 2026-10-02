// Robot console page: rosbridge transport, typed discovery/validation (the same algorithm and
// the same packaged profiles as robot_console/discovery.py), live cameras and bounded controls
// whose changes are sent at once. Plain browser JavaScript, no external dependencies.
"use strict";
(() => {
  const RC = (window.RC = { events: [] });
  const $ = (id) => document.getElementById(id);
  const log = (e) => { RC.events.push(Object.assign({ t: Date.now() }, e)); };

  // ------------------------------------------------------------------ dialect
  function normType(t) {
    const p = String(t || "").trim().split("/");
    if (p.length === 3 && ["msg", "srv", "action"].includes(p[1])) return p[0] + "/" + p[2];
    return String(t || "").trim();
  }
  function typeDialect(t) {
    const p = String(t || "").split("/");
    if (p.length === 3 && ["msg", "srv", "action"].includes(p[1])) return "ros2";
    if (p.length === 2 && p[0] && p[1]) return "ros1";
    return null;
  }
  let INFRA = null;
  function isInfra(name, kind) {
    if (INFRA.topics.includes(name) || INFRA.prefixes.some((p) => name.startsWith(p))) return true;
    if (name === "/rosapi" || name === "/rosbridge_websocket") return true;
    if (kind === "service" && INFRA.node_service_suffixes.some((s) => name.endsWith(s))) return true;
    return false;
  }
  function ros1ActionTopics(name, type) {
    let [pkg, base] = normType(type).split("/");
    if (base.endsWith("Action")) base = base.slice(0, -6);
    return {
      [name + "/goal"]: `${pkg}/${base}ActionGoal`, [name + "/cancel"]: "actionlib_msgs/GoalID",
      [name + "/status"]: "actionlib_msgs/GoalStatusArray",
      [name + "/feedback"]: `${pkg}/${base}ActionFeedback`, [name + "/result"]: `${pkg}/${base}ActionResult`,
    };
  }

  // ------------------------------------------------------------------ transport
  class RosConn {
    constructor(url) {
      this.url = url; this.ws = null; this.ids = 0; this.pending = new Map(); this.subs = new Map();
      this.advertised = new Map(); this.userClosed = false; this.closed = false; this.onlost = null;
      this.handlers = new Map();
    }
    connect(timeoutMs = 5000) {
      return new Promise((resolve, reject) => {
        let done = false;
        const timer = setTimeout(() => { if (!done) { done = true; reject(new Error("timeout")); try { this.ws.close(); } catch (e) {} } }, timeoutMs);
        try { this.ws = new WebSocket(this.url); } catch (e) { clearTimeout(timer); reject(e); return; }
        this.ws.onopen = () => { if (!done) { done = true; clearTimeout(timer); resolve(this); } };
        this.ws.onerror = () => { if (!done) { done = true; clearTimeout(timer); reject(new Error("cannot connect to " + this.url)); } };
        this.ws.onclose = () => this._closed();
        this.ws.onmessage = (ev) => this._message(ev.data);
      });
    }
    get open() { return this.ws && this.ws.readyState === WebSocket.OPEN && !this.closed; }
    _closed() {
      if (this.closed) return;
      this.closed = true;
      for (const [, p] of this.pending) p.reject(new Error("connection lost"));
      this.pending.clear();
      if (!this.userClosed && this.onlost) this.onlost();
    }
    _message(data) {
      let m; try { m = JSON.parse(data); } catch (e) { return; }
      if (m.op === "publish") { const cbs = this.subs.get(m.topic); if (cbs) for (const [, cb] of cbs) { try { cb(m.msg || {}); } catch (e) { console.error(e); } } }
      else if (m.op === "service_response") {
        const p = this.pending.get(String(m.id)); if (!p) return; this.pending.delete(String(m.id));
        if (m.result === false) p.reject(new Error(typeof m.values === "string" ? m.values : JSON.stringify(m.values)));
        else p.resolve(m.values || {});
      } else if (this.handlers.has(m.op + ":" + m.id)) { this.handlers.get(m.op + ":" + m.id)(m); }
    }
    send(msg) {
      if (!this.open) throw new Error("not connected");
      this.ws.send(JSON.stringify(msg));
    }
    call(service, args = {}, timeoutMs = 5000) {
      const id = "call:" + (++this.ids);
      return new Promise((resolve, reject) => {
        const t = setTimeout(() => { if (this.pending.delete(id)) reject(new Error(`${service} did not answer within ${timeoutMs / 1000} s`)); }, timeoutMs);
        this.pending.set(id, { resolve: (v) => { clearTimeout(t); resolve(v); }, reject: (e) => { clearTimeout(t); reject(e); } });
        try { this.send({ op: "call_service", id, service, args }); } catch (e) { this.pending.delete(id); clearTimeout(t); reject(e); }
      });
    }
    subscribe(topic, type, cb, throttle = 0) {
      const id = "sub:" + (++this.ids);
      if (!this.subs.has(topic)) this.subs.set(topic, new Map());
      this.subs.get(topic).set(id, cb);
      const msg = { op: "subscribe", id, topic, throttle_rate: throttle, queue_length: 1 };
      if (type) msg.type = type;
      this.send(msg);
      return id;
    }
    unsubscribe(id) {
      for (const [topic, m] of this.subs) if (m.delete(id)) { try { this.send({ op: "unsubscribe", id, topic }); } catch (e) {} }
    }
    advertise(topic, type) {
      if (this.advertised.get(topic) === type) return;
      this.send({ op: "advertise", id: "adv:" + (++this.ids), topic, type });
      this.advertised.set(topic, type);
    }
    publish(topic, msg, type) { if (type) this.advertise(topic, type); this.send({ op: "publish", topic, msg }); }
    close() {
      this.userClosed = true;
      try { for (const [topic, m] of this.subs) for (const [id] of m) this.send({ op: "unsubscribe", id, topic }); } catch (e) {}
      try { for (const [topic] of this.advertised) this.send({ op: "unadvertise", topic }); } catch (e) {}
      try { this.ws.close(); } catch (e) {}
      this.closed = true;
    }
  }

  // ------------------------------------------------------------------ graph
  async function fetchGraph(conn) {
    const tr = await conn.call("/rosapi/topics");
    const topics = {}; (tr.topics || []).forEach((n, i) => { topics[n] = (tr.types || [])[i] || ""; });
    const services_list = (await conn.call("/rosapi/services")).services || [];
    let dialect = null;
    for (const t of Object.values(topics)) { dialect = typeDialect(t); if (dialect === "ros2") break; }
    if (services_list.includes("/rosapi/get_ros_version")) {
      try { const v = await conn.call("/rosapi/get_ros_version"); dialect = Number(v.version) === 2 ? "ros2" : "ros1"; } catch (e) {}
    }
    const services = {};
    await Promise.all(services_list.filter((s) => !isInfra(s, "service")).map(async (s) => {
      try { services[s] = String((await conn.call("/rosapi/service_type", { service: s })).type || ""); } catch (e) { services[s] = ""; }
    }));
    if (!dialect) for (const t of Object.values(services)) dialect = typeDialect(t) || dialect;
    const actions = {}; const parts = new Set();
    if (dialect === "ros1") {
      for (const [n, t] of Object.entries(topics)) {
        const m = /^([^/]+)\/(.+)ActionGoal$/.exec(normType(t));
        if (n.endsWith("/goal") && m) {
          const a = n.slice(0, -5);
          if (["cancel", "status", "feedback", "result"].every((p) => (a + "/" + p) in topics)) {
            actions[a] = `${m[1]}/${m[2]}Action`;
            Object.keys(ros1ActionTopics(a, actions[a])).forEach((x) => parts.add(x));
          }
        }
      }
    } else if (dialect === "ros2" && services_list.includes("/rosapi/action_servers")) {
      const ar = await conn.call("/rosapi/action_servers");
      for (const a of ar.action_servers || []) actions[a] = null;
    }
    return { dialect, topics, services, actions, parts };
  }
  const has = (g, kind, name) => name in ({ topic: g.topics, service: g.services, action: g.actions }[kind]);
  const typeOf = (g, kind, name) => ({ topic: g.topics, service: g.services, action: g.actions }[kind])[name];
  const names = (g) => new Set([...Object.keys(g.topics), ...Object.keys(g.services), ...Object.keys(g.actions)]);

  // ------------------------------------------------------------------ discovery (mirrors discovery.py)
  function resolve(p, name, ns) {
    ns = String(ns || "").replace(/^\/+|\/+$/g, "");
    if (!ns || !p.namespace || p.namespace.global_names.includes(name)) return name;
    return p.namespace.compose.replace("{ns}", ns).replace("{name}", name);
  }
  function jointPrefix(p, ns) {
    ns = String(ns || "").replace(/^\/+|\/+$/g, "");
    return !ns || !p.namespace ? "" : (p.namespace.joint_prefix || "").replace("{ns}", ns);
  }
  const required = (p) => p.endpoints.filter((e) => !e.optional);
  function validate(p, ns, g) {
    const v = { missing: [], wrong: [], unexpected: [], unverified: [], dialect: null };
    if (g.dialect && g.dialect !== p.dialect) v.dialect = `the wire is ${g.dialect.toUpperCase()} but profile '${p.id}' is ${p.dialect.toUpperCase()}`;
    const claimed = new Set();
    for (const e of p.endpoints) {
      const wire = resolve(p, e.name, ns);
      claimed.add(wire); e.parts.forEach((x) => claimed.add(resolve(p, x.name, ns)));
      if (!has(g, e.kind, wire)) { if (!e.optional) v.missing.push(`${e.kind} ${wire} (${e.type})`); continue; }
      const got = typeOf(g, e.kind, wire);
      if (e.kind === "action" && got === null) { v.unverified.push(wire); continue; }
      if (!got || normType(got) !== normType(e.type)) v.wrong.push(`${e.kind} ${wire}: expected ${e.type}, found ${got || "unknown"}`);
    }
    const owned = p.owned_prefixes.map((o) => resolve(p, o, ns));
    const nsp = ns ? "/" + ns.replace(/^\/+|\/+$/g, "") + "/" : null;
    for (const [kind, table] of [["topic", g.topics], ["service", g.services], ["action", g.actions]]) {
      for (const n of Object.keys(table)) {
        if (claimed.has(n) || isInfra(n, kind)) continue;
        if (g.parts.has(n) && [...claimed].some((a) => n.startsWith(a + "/"))) continue;
        if (owned.some((o) => n.startsWith(o)) || (nsp && n.startsWith(nsp))) v.unexpected.push(`${kind} ${n}`);
      }
    }
    v.ok = !v.missing.length && !v.wrong.length && !v.unexpected.length && !v.dialect;
    v.problems = [...(v.dialect ? [v.dialect] : []), ...v.missing.map((x) => "missing " + x), ...v.wrong.map((x) => "wrong type on " + x),
      ...v.unexpected.map((x) => "unexpected " + x + " under the profile's names")];
    return v;
  }
  function namespacesFor(p, g) {
    if (!p.namespace) return [""];
    const out = [p.namespace.default.replace(/^\/+|\/+$/g, "")];
    const wire = names(g);
    for (const e of required(p)) {
      if (p.namespace.global_names.includes(e.name)) continue;
      for (const w of wire) if (w !== e.name && w.endsWith(e.name)) {
        const pre = w.slice(0, -e.name.length).replace(/^\/+|\/+$/g, "");
        if (pre && !out.includes(pre)) out.push(pre);
      }
    }
    return out;
  }
  const mkTarget = (p, ns, g) => {
    const t = { profile: p, namespace: ns, validation: validate(p, ns, g) };
    t.label = p.id + (ns ? ` (namespace /${ns})` : "");
    t.req = new Set(required(p).map((e) => resolve(p, e.name, ns)));
    t.wire = (n) => resolve(p, n, ns);
    return t;
  };
  function discover(g, profiles, ns) {
    const full = [], near = [];
    for (const p of profiles) {
      if (g.dialect && g.dialect !== p.dialect) continue;
      const nss = ns != null && p.namespace ? [ns.replace(/^\/+|\/+$/g, "")] : namespacesFor(p, g);
      for (const n of nss) {
        const req = required(p);
        const present = req.filter((e) => has(g, e.kind, resolve(p, e.name, n)));
        const t = mkTarget(p, n, g);
        if (req.length && present.length === req.length) full.push(t);
        else if (req.length && present.length * 2 >= req.length) near.push([t, req.filter((e) => !present.includes(e)).map((e) => resolve(p, e.name, n))]);
      }
    }
    const subset = (a, b) => a.size < b.size && [...a].every((x) => b.has(x));
    const kept = [], dominated = [];
    for (const a of full) { const by = full.find((b) => b !== a && subset(a.req, b.req)); if (by) dominated.push([a, by]); else kept.push(a); }
    const groups = [];
    for (const t of kept) {
      const grp = groups.find((gr) => gr.some((o) => [...t.req].some((x) => o.req.has(x))));
      if (grp) grp.push(t); else groups.push([t]);
    }
    return { targets: groups.filter((x) => x.length === 1).map((x) => x[0]), ambiguous: groups.filter((x) => x.length > 1), dominated, near };
  }
  class SelectionError extends Error { constructor(reason, candidates = []) { super(reason); this.reason = reason; this.candidates = candidates; } }
  const labels = (ts) => ts.map((t) => t.label).join(", ");
  function selectTarget(g, profiles, robotId, ns) {
    const robot = robotId ? profiles.find((p) => p.id === robotId) : null;
    if (robot && ns != null && !robot.namespace) throw new SelectionError(`'${robot.id}' offers no namespace override`);
    const d = discover(g, profiles, robot && !robot.namespace ? null : ns);
    const pool = [...d.targets, ...d.ambiguous.flat()];
    let t;
    if (robot) {
      let mine = pool.filter((x) => x.profile.id === robot.id);
      const dom = d.dominated.filter(([a]) => a.profile.id === robot.id);
      if (!mine.length && dom.length) throw new SelectionError(`'${robot.id}' is not uniquely identified: the wire presents ${dom[0][1].label}, whose interface contains all of ${robot.id}'s required names`, dom[0]);
      if (!mine.length) {
        const nm = d.near.filter(([x]) => x.profile.id === robot.id);
        if (nm.length) throw new SelectionError(`'${robot.id}' not found on the wire: missing ${nm[0][1].slice(0, 8).join(", ")}`, [nm[0][0]]);
        throw new SelectionError(`'${robot.id}' not found on the wire (none of its required interface is present)`);
      }
      if (mine.length > 1) {
        const def = robot.namespace ? robot.namespace.default.replace(/^\/+|\/+$/g, "") : "";
        const pick = ns == null ? mine.filter((x) => x.namespace === def) : [];
        if (pick.length === 1) mine = pick; else throw new SelectionError(`several '${robot.id}' targets found; choose a namespace: ${labels(mine)}`, mine);
      }
      t = mine[0];
      const grp = d.ambiguous.find((gr) => gr.includes(t));
      if (grp && !robot.namespace && grp.some((x) => x.profile.id !== robot.id)) throw new SelectionError(`ambiguous wire: ${labels(grp)} all match`, grp);
    } else {
      if (d.ambiguous.length) {
        let grp = d.ambiguous[0];
        if (new Set(grp.map((x) => x.profile.id)).size === 1 && ns == null) {
          const p = grp[0].profile; const def = p.namespace ? p.namespace.default.replace(/^\/+|\/+$/g, "") : "";
          const pick = grp.filter((x) => x.namespace === def); if (pick.length === 1) grp = pick;
        }
        if (grp.length > 1) throw new SelectionError(`ambiguous wire, select a robot explicitly; candidates: ${labels(grp)}`, grp);
        d.targets.push(grp[0]);
      }
      if (d.targets.length !== 1) {
        if (!d.targets.length) throw new SelectionError("no supported robot identified on the wire" + (d.near.length ? "; incomplete candidates: " + d.near.map(([x, m]) => `${x.label} (missing ${m.slice(0, 4).join(", ")})`).join("; ") : ""), d.near.map((x) => x[0]));
        throw new SelectionError("several robots on the wire, select one: " + labels(d.targets), d.targets);
      }
      t = d.targets[0];
    }
    if (ns != null && !t.profile.namespace) throw new SelectionError(`'${t.profile.id}' offers no namespace override`);
    if (!t.validation.ok) throw new SelectionError(`${t.label} failed typed validation: ${t.validation.problems.join("; ")}`, [t]);
    return t;
  }
  RC.lib = { normType, typeDialect, validate, discover, selectTarget, fetchGraph, RosConn };

  // ------------------------------------------------------------------ templates
  function fill(tpl, values, jp) {
    if (Array.isArray(tpl)) return tpl.map((x) => fill(x, values, jp));
    if (tpl && typeof tpl === "object") { const o = {}; for (const k of Object.keys(tpl)) o[k] = fill(tpl[k], values, jp); return o; }
    if (typeof tpl === "string") {
      const m = /^\$([A-Za-z_][A-Za-z0-9_]*)$/.exec(tpl);
      if (m) { if (!(m[1] in values)) throw new Error(`template field $${m[1]} has no value`); return values[m[1]]; }
      return tpl.split("{joint_prefix}").join(jp);
    }
    return tpl;
  }
  function coerce(f, raw) {
    if (f.choices) { if (!f.choices.includes(raw)) throw new Error(`${f.name}: ${raw} is not one of ${f.choices.join(", ")}`); return raw; }
    const v = Number(raw);
    if (raw === "" || !Number.isFinite(v)) throw new Error(`${f.label}: not a number`);
    if (f.min != null && v < f.min) throw new Error(`${f.label}: ${v} below documented minimum ${f.min}`);
    if (f.max != null && v > f.max) throw new Error(`${f.label}: ${v} above documented maximum ${f.max}`);
    if (f.integer && !Number.isInteger(v)) throw new Error(`${f.label}: must be an integer`);
    return v;
  }

  // ------------------------------------------------------------------ cameras
  const UNTIED_STALE_S = 2.0;
  function decodeImage(msg, allowed) {
    const enc = String(msg.encoding || "").toLowerCase();
    if (allowed && allowed.length && !allowed.map((x) => x.toLowerCase()).includes(enc)) throw Object.assign(new Error(`unsupported encoding '${enc}'`), { unsupported: true });
    const w = msg.width | 0, h = msg.height | 0, step = msg.step | 0;
    const bin = typeof msg.data === "string" ? atob(msg.data) : null;
    const len = bin ? bin.length : (msg.data || []).length;
    const at = bin ? (i) => bin.charCodeAt(i) : (i) => msg.data[i];
    if (!w || !h || len < step * h) throw new Error("truncated or empty image");
    const out = new Uint8ClampedArray(w * h * 4);
    const put = (o, r, g, b) => { out[o] = r; out[o + 1] = g; out[o + 2] = b; out[o + 3] = 255; };
    for (let y = 0; y < h; y++) {
      const row = y * step;
      for (let x = 0; x < w; x++) {
        const o = (y * w + x) * 4;
        switch (enc) {
          case "rgb8": case "8uc3": put(o, at(row + 3 * x), at(row + 3 * x + 1), at(row + 3 * x + 2)); break;
          case "bgr8": put(o, at(row + 3 * x + 2), at(row + 3 * x + 1), at(row + 3 * x)); break;
          case "rgba8": put(o, at(row + 4 * x), at(row + 4 * x + 1), at(row + 4 * x + 2)); break;
          case "bgra8": put(o, at(row + 4 * x + 2), at(row + 4 * x + 1), at(row + 4 * x)); break;
          case "mono8": case "8uc1": { const v = at(row + x); put(o, v, v, v); break; }
          case "mono16": case "16uc1": { const v = msg.is_bigendian ? at(row + 2 * x) : at(row + 2 * x + 1); put(o, v, v, v); break; }
          case "yuv422": case "uyvy": case "yuv422_yuy2": case "yuyv": {
            const b0 = row + 4 * (x >> 1); const yuy = enc === "yuv422_yuy2" || enc === "yuyv";
            const Y = at(yuy ? b0 + (x & 1 ? 2 : 0) : b0 + (x & 1 ? 3 : 1));
            const U = at(yuy ? b0 + 1 : b0) - 128, V = at(yuy ? b0 + 3 : b0 + 2) - 128;
            put(o, Y + 1.402 * V, Y - 0.344136 * U - 0.714136 * V, Y + 1.772 * U); break;
          }
          default: throw Object.assign(new Error(`unsupported encoding '${enc}'`), { unsupported: true });
        }
      }
    }
    return new ImageData(out, w, h);
  }

  const S = (RC.state = { config: null, profiles: [], conn: null, graph: null, target: null,
    lost: false, streams: [], controls: new Map(), feedback: new Map(), padGroup: null });

  // ------------------------------------------------------------------ DOM helpers
  function h(tag, props, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === "class") e.className = v;
      else if (k === "text") e.textContent = v;
      else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? "" : String(v));
    }
    for (const c of kids.flat()) if (c != null && c !== false) e.append(c);
    return e;
  }
  const exact = (v) => String(+Number(v).toFixed(6)).replace("-", "−");   // a documented limit, as documented
  const signed = (v, d) => (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(d);
  function decimals(f) {
    if (f.integer) return 0;
    const span = f.max - f.min;
    return span <= 5 ? 2 : span <= 50 ? 1 : 0;
  }

  // ------------------------------------------------------------------ status surfaces
  function setConn(state, text) {
    $("conn").textContent = text; $("conn").className = state === "ok" ? "ok" : state === "bad" ? "bad" : "dim";
    $("conn-pill").dataset.state = state;
  }
  function setValidation(state, title, sub, reason) {
    $("validation").dataset.state = state;
    $("validation-title").textContent = title;
    $("validation-sub").textContent = sub || "";
    $("reason").textContent = reason || "";
  }
  function showBanner(text) { const b = $("banner"); b.hidden = !text; b.textContent = text || ""; }

  // ------------------------------------------------------------------ cameras
  function clearCameras() {
    for (const s of S.streams) { try { if (s.sid && S.conn && S.conn.open) S.conn.unsubscribe(s.sid); } catch (e) {} }
    S.streams = []; $("cameras").innerHTML = ""; $("cameras").dataset.count = "0";
  }
  function addCamera(topic, type, staleS, encodings, tied, missing) {
    const canvas = h("canvas", { width: 640, height: 480 });
    const fig = h("figure", { class: "cam notlive", "data-testid": "camera", "data-topic": topic, "data-state": "waiting" },
      h("div", { class: "cam-view" }, canvas,
        h("div", { class: "cam-overlay" }, h("div", { class: "ov-title" }), h("div", { class: "ov-detail" })),
        h("span", { class: "cam-badge state", "data-testid": "camera-state", role: "status" })),
      h("figcaption", { class: "cam-caption" },
        h("code", { text: topic }),
        tied ? null : h("span", { class: "untied", "data-testid": "untied", text: "not tied to a target" }),
        h("span", { class: "cam-meta" })));
    $("cameras").appendChild(fig);
    const n = $("cameras").children.length;
    $("cameras").dataset.count = n > 3 ? "many" : String(n);
    const s = { topic, type, staleS, encodings, tied, div: fig, canvas, last: null, frames: 0, error: null,
      unsupported: false, missing, started: performance.now(), times: [], w: 0, h: 0, enc: "" };
    if (!missing) s.sid = S.conn.subscribe(topic, type, (m) => onFrame(s, m), 50);
    S.streams.push(s);
    return s;
  }
  function onFrame(s, m) {
    try {
      const img = decodeImage(m, s.encodings);
      if (s.canvas.width !== img.width || s.canvas.height !== img.height) {
        s.canvas.width = img.width; s.canvas.height = img.height;
        s.div.querySelector(".cam-view").style.setProperty("--ar", `${img.width} / ${img.height}`);
      }
      s.canvas.getContext("2d").putImageData(img, 0, 0);
      const now = performance.now();
      s.last = now; s.frames++; s.error = null; s.unsupported = false;
      s.w = img.width; s.h = img.height; s.enc = String(m.encoding || "");
      s.times.push(now); while (s.times.length > 2 && now - s.times[0] > 2000) s.times.shift();
    } catch (e) { s.error = e.message; s.unsupported = !!e.unsupported; s.last = null; }
  }
  // [state, badge, detail]. Never "live" unless a frame arrived within the stale threshold.
  function cameraState(s, now) {
    if (s.missing) return ["missing", "MISSING", "missing: topic not on the wire"];
    if (S.lost) return ["failed", "DISCONNECTED", "Connection lost: this image is frozen, not live. Reload the page to connect again."];
    if (s.unsupported) return ["unsupported", "UNSUPPORTED", s.error];
    if (s.error) return ["failed", "FAILED", "failed: " + s.error];
    if (s.last == null) {
      return (now - s.started) / 1000 > s.staleS
        ? ["stale", "STALE", `stale: no frame within ${s.staleS} s`] : ["waiting", "WAITING", "waiting for frames"];
    }
    const age = (now - s.last) / 1000;
    return age <= s.staleS ? ["live", "LIVE", "live"]
      : ["stale", "STALE", `stale: last frame ${age.toFixed(1)} s ago (threshold ${s.staleS} s); the image is frozen`];
  }
  function camMeta(s, now) {
    const parts = [];
    if (s.w) parts.push(`${s.w}×${s.h}`);
    if (s.enc) parts.push(s.enc);
    if (s.times.length > 1 && s.last != null && now - s.last < 2000) {
      const fps = (s.times.length - 1) / ((s.times[s.times.length - 1] - s.times[0]) / 1000);
      parts.push(`${fps.toFixed(fps < 10 ? 1 : 0)} fps`);
    }
    if (s.last != null) { const a = (now - s.last) / 1000; parts.push(a < 1 ? `${Math.round(a * 1000)} ms` : `${a.toFixed(1)} s ago`); }
    return parts.join(" · ");
  }
  function tick() {
    const now = performance.now();
    let live = 0;
    for (const s of S.streams) {
      const [st, badge, detail] = cameraState(s, now);
      if (st === "live") live++;
      const b = s.div.querySelector(".cam-badge");
      if (b.textContent !== badge) b.textContent = badge;
      b.className = "cam-badge state " + (st === "live" ? "ok" : st === "waiting" ? "dim" : "bad");
      s.div.querySelector(".ov-title").textContent = badge;
      s.div.querySelector(".ov-detail").textContent = detail;
      s.div.querySelector(".cam-meta").textContent = camMeta(s, now);
      s.div.dataset.state = st; s.div.classList.toggle("notlive", st !== "live");
    }
    const none = !S.streams.length;
    $("no-cameras").style.display = none ? "" : "none";
    $("no-cameras").textContent = none ? (S.target ? "No cameras for this robot's profile." : "No cameras discovered on the wire.") : "";
    $("cam-summary").textContent = none ? "" : `${live} of ${S.streams.length} live`;
    tickFeedback(now);
  }
  setInterval(tick, 200);

  // ------------------------------------------------------------------ measured joint positions
  // Shown only where a control's own prerequisites include a documented JointState output
  // topic and its template pairs joint names with field values; nothing is invented.
  function feedbackFor(c, t) {
    const fb = c.prerequisites.map((n) => t.profile.endpoints.find((e) => e.name === n))
      .find((e) => e && e.kind === "topic" && e.direction === "out" && normType(e.type) === "sensor_msgs/JointState");
    if (!fb) return null;
    const jp = jointPrefix(t.profile, t.namespace);
    const nameLists = [], refLists = [];
    (function walk(x) {
      if (Array.isArray(x)) {
        if (x.length && x.every((v) => typeof v === "string" && v.startsWith("$"))) refLists.push(x.map((v) => v.slice(1)));
        else if (x.length && x.every((v) => typeof v === "string" && !v.startsWith("$"))) nameLists.push(x.map((v) => v.split("{joint_prefix}").join(jp)));
        else x.forEach(walk);
      } else if (x && typeof x === "object") Object.values(x).forEach(walk);
    })(c.template);
    const map = {};
    for (const refs of refLists) {
      const names = nameLists.find((n) => n.length === refs.length);
      if (!names) continue;
      refs.forEach((f, i) => {
        const fld = c.fields.find((x) => x.name === f);
        if (fld && (fld.unit === "rad" || fld.unit === "m")) map[f] = names[i];
      });
    }
    return Object.keys(map).length ? { topic: t.wire(fb.name), type: fb.type, map } : null;
  }
  function subscribeFeedback(topic, type) {
    if (S.feedback.has(topic)) return;
    const f = { topic, last: null, pos: new Map(), sid: null };
    S.feedback.set(topic, f);
    try {
      f.sid = S.conn.subscribe(topic, type, (m) => {
        const names = m.name || [], pos = m.position || [];
        names.forEach((n, i) => { if (Number.isFinite(pos[i])) f.pos.set(n, pos[i]); });
        f.last = performance.now();
      }, 100);
    } catch (e) {}
  }
  function clearFeedback() {
    for (const f of S.feedback.values()) { try { if (f.sid && S.conn && S.conn.open) S.conn.unsubscribe(f.sid); } catch (e) {} }
    S.feedback.clear();
  }
  function measuredOf(cs, now) {
    if (!cs.fb) return null;
    const f = S.feedback.get(cs.fb.topic);
    const fresh = f && f.last != null && now - f.last < 1000 && !S.lost;
    const out = {};
    for (const [field, joint] of Object.entries(cs.fb.map)) out[field] = fresh && f.pos.has(joint) ? f.pos.get(joint) : null;
    return out;
  }
  function tickFeedback(now) {
    for (const cs of S.controls.values()) {
      if (!cs.fb) continue;
      const m = measuredOf(cs, now);
      let any = false;
      for (const fs of Object.values(cs.fields)) {
        if (!fs.meas) continue;
        const v = m[fs.f.name];
        if (v == null) {
          fs.measVal.textContent = "—"; fs.meas.dataset.state = "none"; fs.meas.title = `measured: no fresh ${cs.fb.topic} message`;
          fs.mark.hidden = true;
        } else {
          any = true;
          fs.measVal.textContent = `${signed(v, decimals(fs.f) + 1)} ${fs.f.unit}`; fs.meas.dataset.state = "live";
          fs.meas.title = `measured ${cs.fb.map[fs.f.name]} on ${cs.fb.topic}`;
          const p = Math.min(1, Math.max(0, (v - fs.f.min) / (fs.f.max - fs.f.min)));
          fs.mark.style.left = `${p * 100}%`;
        }
      }
      if (cs.useMeasured) cs.useMeasured.disabled = !any;
    }
  }

  // ------------------------------------------------------------------ sending
  // Values changed on the page go to the robot at once; nothing is ever stopped or cancelled
  // by the page. Topic publishes stream while a value is dragged; action goals stream at a
  // lower rate, each new goal preempting the previous one on the action server. Both use a
  // throttle with a leading and a trailing edge that reads the inputs when it fires, so the
  // last value is always sent. Service calls and choice buttons send once per click.
  const STREAM_MS = { publish: 100, action: 200 };
  const GOAL_ACK_MS = 1000;     // the next goal waits for the previous one to show life, at most this long
  const GOAL_CONNS_MAX = 3;     // per control: open goal connections (the latest goal and superseded ones)
  function randomId() { return "rc-" + Math.random().toString(16).slice(2) + Date.now().toString(16); }

  // "click": one send per click (services, controls without numeric fields, choice buttons);
  // "stream": a send per change, throttled (topic publishes and action goals).
  function sendMode(c) {
    if (c.kind === "call") return "click";
    return c.fields.some((f) => !f.choices) ? "stream" : "click";
  }
  const liveOk = (cs) => !!S.target && !!S.conn && !S.lost && cs.available && !cs.dead;

  function setStatus(id, cls, text) {
    const cs = S.controls.get(id); if (!cs) return;
    cs.status = { cls, text }; cs.statusEl.textContent = text; cs.statusEl.className = "status " + cls;
    cs.div.dataset.status = cls;
    log({ ev: "status", control: id, cls, text });
  }

  // A value of a streamed control changed: send it now, or as soon as the throttle allows.
  function trigger(cs) {
    if (!liveOk(cs) || cs.mode !== "stream") return;
    cs.th.pending = true; pump(cs);
  }
  function pump(cs) {
    const th = cs.th;
    if (!th.pending || th.timer || th.busy || !liveOk(cs)) return;
    if (cs.ctrl.kind === "action" && cs.goal && !cs.goal.acked && Date.now() - cs.goal.sentAt < GOAL_ACK_MS) return;
    const wait = th.last + STREAM_MS[cs.ctrl.kind === "action" ? "action" : "publish"] - Date.now();
    if (wait > 0) { th.timer = setTimeout(() => { th.timer = null; pump(cs); }, wait); return; }
    th.pending = false; th.last = Date.now(); th.busy = true;
    Promise.resolve(startControl(cs)).catch(() => {}).finally(() => { th.busy = false; pump(cs); });
  }
  // One send per click.
  function fire(cs) { if (liveOk(cs)) startControl(cs); }

  async function startControl(cs) {
    const c = cs.ctrl, t = S.target, conn = S.conn;
    if (!liveOk(cs)) return false;
    const values = {};
    try { for (const f of c.fields) values[f.name] = coerce(f, cs.inputs[f.name].value); }
    catch (e) { setStatus(c.id, "bad", "not sent: " + e.message); return false; }
    let payload;
    try { payload = fill(c.template, values, jointPrefix(t.profile, t.namespace)); }
    catch (e) { setStatus(c.id, "bad", "not sent: " + e.message); return false; }
    const name = t.wire(c.name);
    log({ ev: "send", control: c.id, name, values });
    try {
      if (c.kind === "publish") {
        conn.publish(name, payload, c.type); cs.sent++;
        setStatus(c.id, "ok", `sent (${cs.sent} message${cs.sent === 1 ? "" : "s"} so far; a publish is not acknowledged)`);
      } else if (c.kind === "call") {
        setStatus(c.id, "warn", "pending: waiting for the service answer…");
        conn.call(name, payload, 10000).then((v) => {
          if (v && (v.result === false || v.success === false)) setStatus(c.id, "bad", "failed: " + JSON.stringify(v));
          else setStatus(c.id, "ok", "done: " + JSON.stringify(v).slice(0, 120));
        }, (e) => setStatus(c.id, "bad", "failed: " + e.message));
      } else if (c.kind === "action") {
        if (t.profile.dialect === "ros2") await sendGoal2(cs, conn, name, payload);
        else sendGoal1(cs, conn, name, payload);
      }
      return true;
    } catch (e) { setStatus(c.id, "bad", "failed to send: " + e.message); return false; }
  }

  // ROS 2: stock rosbridge (Humble, Jazzy) handles one client's send_action_goal to completion
  // before that client's next op, so each goal gets its own connection. Its result or feedback
  // acknowledges it; once the latest goal is acknowledged the superseded goals' connections
  // close (closing one cancels nothing on stock rosbridge). At most GOAL_CONNS_MAX stay open.
  async function sendGoal2(cs, conn, name, payload) {
    const c = cs.ctrl;
    const gc = new RosConn(conn.url); await gc.connect(5000);
    if (!liveOk(cs) || S.conn !== conn) { gc.close(); return; }
    const g = { conn: gc, id: randomId(), sentAt: Date.now(), acked: false, done: false };
    cs.goal = g; cs.goals.push(g);
    const ack = () => {
      if (g.acked) return;
      g.acked = true;
      if (cs.goal === g) {
        if (!g.done) setStatus(c.id, "warn", "goal running");
        for (const o of [...cs.goals]) if (o !== g) retireGoal(cs, o);
      }
      pump(cs);
    };
    gc.handlers.set("action_feedback:" + g.id, ack);
    gc.handlers.set("action_result:" + g.id, (m) => {
      g.done = true; ack();
      if (cs.goal === g) {
        const st = { 4: "succeeded", 5: "canceled", 6: "aborted", 2: "canceled" }[m.status] || `ended (status ${m.status})`;
        setStatus(c.id, m.result === false || m.status === 6 ? "bad" : m.status === 4 ? "ok" : "warn",
          "goal " + st + (m.result === false ? ": " + JSON.stringify(m.values) : ""));
      }
      retireGoal(cs, g);
    });
    gc.onlost = () => {
      if (!g.done && cs.goal === g) setStatus(c.id, "bad", "goal connection lost; outcome unknown");
      g.done = true; retireGoal(cs, g);
    };
    gc.send({ op: "send_action_goal", id: g.id, action: name, action_type: c.type, args: payload, feedback: true });
    setStatus(c.id, "warn", "sending: goal sent, waiting for the action server…");
    setTimeout(() => pump(cs), GOAL_ACK_MS + 10);
    const old = cs.goals.filter((o) => o !== g);
    for (let i = 0; i < old.length && cs.goals.length > GOAL_CONNS_MAX; i++) retireGoal(cs, old[i]);
  }
  function retireGoal(cs, g) {
    const i = cs.goals.indexOf(g); if (i < 0) return;
    cs.goals.splice(i, 1);
    setTimeout(() => g.conn.close(), 50);
  }

  // ROS 1 actionlib: goals go out on the target's connection; a new goal preempts the previous
  // one on the action server. One result subscription per control reports the latest goal.
  function sendGoal1(cs, conn, name, payload) {
    const c = cs.ctrl, topics = ros1ActionTopics(name, c.type);
    if (!cs.resultSid) {
      cs.resultSid = conn.subscribe(name + "/result", topics[name + "/result"], (m) => {
        if (!cs.goal || !m.status || !m.status.goal_id || m.status.goal_id.id !== cs.goal.id) return;
        const st = { 3: "succeeded", 2: "canceled", 8: "canceled", 4: "aborted", 5: "rejected" }[m.status.status] || `ended (status ${m.status.status})`;
        setStatus(c.id, m.status.status === 3 ? "ok" : m.status.status === 4 || m.status.status === 5 ? "bad" : "warn", "goal " + st);
      });
    }
    const g = { conn, id: randomId(), sentAt: Date.now(), acked: true, done: false };
    cs.goal = g;
    conn.publish(name + "/goal", { header: {}, goal_id: { id: g.id, stamp: { secs: 0, nsecs: 0 } }, goal: payload }, topics[name + "/goal"]);
    setStatus(c.id, "warn", "goal sent, running…");
  }

  // Controls of a target that is being replaced: no further sends. The latest unfinished goal
  // keeps its connection until its result arrives; nothing is cancelled.
  function retireControls() {
    for (const cs of S.controls.values()) {
      cs.dead = true;
      if (cs.th.timer) { clearTimeout(cs.th.timer); cs.th.timer = null; }
      cs.th.pending = false;
      for (const g of [...cs.goals]) if (g !== cs.goal || g.done) retireGoal(cs, g);
    }
  }

  function refreshControls() {
    for (const cs of S.controls.values()) cs.body.disabled = !liveOk(cs);
    if (S.padGroup) S.padGroup.sync();
    syncLive();
  }
  function updateLock() {
    $("robot").disabled = S.lost; $("namespace").disabled = S.lost || !nsAllowed(); $("select").disabled = S.lost;
  }
  function nsAllowed() {
    const id = $("robot").value; const p = S.profiles.find((x) => x.id === id);
    return !!(p && p.namespace);
  }
  function syncNsField() {
    const ok = nsAllowed();
    $("namespace").placeholder = ok ? "(hardware default)" : "not supported";
    if (!ok) $("namespace").value = "";
  }

  function syncLive() {
    const t = S.target, label = t ? t.label : "";
    let state, title, hint;
    if (S.lost) { state = "fault"; title = "Controls disabled"; hint = "Disconnected. Reload the page to connect again."; }
    else if (!t) { state = "unavailable"; title = "Controls unavailable"; hint = "Validate a robot target first."; }
    else if (!S.controls.size) { state = "unavailable"; title = "Camera only"; hint = "This robot's profile lists no page controls."; }
    else { state = "live"; title = `Controls live on ${label}`; hint = `Values you change below are sent to ${label} at once.`; }
    const card = $("live-card");
    card.dataset.state = state; $("live-title").textContent = title; $("live-hint").textContent = hint;
  }

  // ------------------------------------------------------------------ controls UI
  function kindText(c) {
    if (c.kind === "call") return "Service call · once per click";
    if (c.kind === "action") return sendMode(c) === "stream"
      ? "Action goal · sent as you change it (≈5 per s); each goal preempts the previous one"
      : "Action goal · once per click";
    return sendMode(c) === "stream" ? "Topic publish · sent as you change it (≈10 per s)" : "Topic publish · once per click";
  }

  function validateField(fs) {
    const f = fs.f;
    let v = null, err = "";
    try { v = coerce(f, fs.num.value); } catch (e) { err = e.message.startsWith(f.label + ": ") ? e.message.slice(f.label.length + 2) : e.message; }
    fs.div.classList.toggle("invalid", !!err);
    fs.num.setAttribute("aria-invalid", String(!!err));
    fs.err.textContent = err;
    if (v != null) {
      if (Number(fs.range.value) !== v) fs.range.value = String(v);
    }
    for (const fn of fs.listeners) fn(v);
  }
  function setField(fs, v, fromRange) {
    const f = fs.f;
    v = Math.min(f.max, Math.max(f.min, Number(v)));
    let r = Number(v.toFixed(decimals(f)));
    if (r > f.max || r < f.min) r = v;            // never round past a documented limit
    if (f.integer) r = Math.round(r);
    fs.num.value = String(r);
    if (!fromRange) fs.range.value = String(r);
    validateField(fs);
  }

  function buildField(c, f, cs) {
    const id = `f-${c.id}-${f.name}`;
    if (f.choices) {
      let value = f.default != null && f.choices.includes(f.default) ? f.default : f.choices[0];
      cs.inputs[f.name] = { get value() { return value; } };
      if (cs.mode === "click") {
        // Each choice is a button that sends that choice once.
        const grid = h("div", { class: "choices", role: "group", "aria-label": f.label, "data-field": f.name });
        for (const ch of f.choices) {
          const b = h("button", { type: "button", class: "choice", "data-choice": ch, text: ch,
            title: `send ‘${ch}’ once` });
          b.addEventListener("click", () => { value = ch; fire(cs); });
          grid.append(b);
        }
        return h("div", { class: "field" }, h("div", { class: "field-top" }, h("label", { text: f.label })), grid);
      }
      const grid = h("div", { class: "choices", role: "radiogroup", "aria-label": f.label, "data-field": f.name });
      const btns = f.choices.map((ch) => h("button", { type: "button", class: "choice", role: "radio", "data-choice": ch, text: ch }));
      const sync = () => { for (const b of btns) { const on = b.dataset.choice === value; b.setAttribute("aria-checked", String(on)); b.tabIndex = on ? 0 : -1; } };
      btns.forEach((b, i) => {
        b.addEventListener("click", () => { value = b.dataset.choice; sync(); trigger(cs); });
        b.addEventListener("keydown", (e) => {
          const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
          if (!d) return;
          e.preventDefault(); const n = btns[(i + d + btns.length) % btns.length]; value = n.dataset.choice; sync(); n.focus(); trigger(cs);
        });
      });
      grid.append(...btns);
      cs.afterBuild.push(sync);
      return h("div", { class: "field" }, h("div", { class: "field-top" }, h("label", { text: f.label })), grid);
    }
    const num = h("input", { type: "number", id, "data-field": f.name, min: f.min, max: f.max, step: f.integer ? 1 : "any",
      "aria-describedby": id + "-lim" });
    if (f.default != null) num.value = f.default;
    const range = h("input", { type: "range", min: f.min, max: f.max, step: f.integer ? 1 : "any",
      "aria-label": `${f.label} (${f.unit || "value"})`, tabindex: "-1" });
    if (f.default != null) range.value = f.default;
    const lo = h("button", { type: "button", class: "limit", title: `documented minimum ${exact(f.min)} ${f.unit}`, text: exact(f.min) });
    const hi = h("button", { type: "button", class: "limit", title: `documented maximum ${exact(f.max)} ${f.unit}`, text: exact(f.max) });
    const mark = h("span", { class: "meas-mark", hidden: true, "aria-hidden": "true" });
    const div = h("div", { class: "field" },
      h("div", { class: "field-top" },
        h("label", { for: id, text: f.label }),
        null,
        h("span", { class: "num-wrap" }, num, h("span", { class: "unit", text: f.unit || "" }))),
      h("div", { class: "slider-row", id: id + "-lim" }, lo, h("div", { class: "track" }, range, mark), hi),
      h("div", { class: "field-err", role: "alert" }));
    const fs = { f, div, num, range, mark, err: div.querySelector(".field-err"), listeners: [], meas: null, measVal: null };
    if (cs.fb && cs.fb.map[f.name]) {
      fs.measVal = h("span", { class: "m-val", text: "—" });
      fs.meas = h("span", { class: "measured" }, h("span", { class: "m-label", text: "measured" }), fs.measVal);
      div.querySelector(".field-top").insertBefore(fs.meas, div.querySelector(".num-wrap"));
    }
    // Typing only validates; a typed value is sent when committed (Enter or leaving the field).
    num.addEventListener("input", () => validateField(fs));
    num.addEventListener("change", () => { validateField(fs); trigger(cs); });
    range.addEventListener("input", () => { setField(fs, range.value, true); trigger(cs); });
    lo.addEventListener("click", () => { setField(fs, f.min); trigger(cs); });
    hi.addEventListener("click", () => { setField(fs, f.max); trigger(cs); });
    cs.inputs[f.name] = num; cs.fields[f.name] = fs;
    cs.afterBuild.push(() => validateField(fs));
    return div;
  }

  function buildControl(c, t, g) {
    // availability: every prerequisite present with its documented type
    let reason = null;
    for (const pre of c.prerequisites) {
      const e = t.profile.endpoints.find((x) => x.name === pre);
      const wire = t.wire(pre);
      if (!e) { reason = `prerequisite ${pre} is not in the profile`; break; }
      if (!has(g, e.kind, wire)) { reason = `prerequisite ${e.kind} ${wire} is not on the wire`; break; }
      const got = typeOf(g, e.kind, wire);
      if (!(e.kind === "action" && got === null) && normType(got) !== normType(e.type)) { reason = `prerequisite ${wire} has type ${got}, expected ${e.type}`; break; }
    }
    const compact = !c.fields.length;
    const div = h("div", { class: "control" + (compact ? " compact" : "") + (reason ? " unavailable" : ""), "data-control": c.id, "data-testid": "control" });
    const cs = { ctrl: c, div, inputs: {}, fields: {}, available: !reason, status: null, afterBuild: [], fb: reason ? null : feedbackFor(c, t),
      mode: sendMode(c), th: { pending: false, timer: null, busy: false, last: 0 }, goal: null, goals: [], sent: 0, dead: false };
    div.append(h("div", { class: "ctl-head" },
      h("div", {}, h("h3", { text: c.label }), h("div", { class: "ctl-kind", text: kindText(c) }))));
    div.append(h("details", { class: "ctl-wire" }, h("summary", { text: "Interface" }),
      h("dl", {},
        h("dt", { text: "Name" }), h("dd", { text: t.wire(c.name) }),
        h("dt", { text: "Type" }), h("dd", { text: c.type }),
        cs.fb ? [h("dt", { text: "Measured" }), h("dd", { text: cs.fb.topic })] : null)));
    if (reason) div.append(h("div", { class: "unavail", "data-testid": "unavailable", text: "unavailable: " + reason }));
    // Everything that edits or sends sits in one fieldset, disabled while the control is not live.
    cs.body = h("fieldset", { class: "ctl-body", disabled: true });
    if (c.fields.length) cs.body.append(h("div", { class: "fields" }, c.fields.map((f) => buildField(c, f, cs))));
    const numeric = c.fields.filter((f) => !f.choices);
    if (numeric.length) {
      const tools = h("div", { class: "ctl-tools" });
      const reset = h("button", { type: "button", class: "btn btn-sm btn-ghost", text: "Reset to defaults" });
      reset.addEventListener("click", () => { for (const f of numeric) if (f.default != null) setField(cs.fields[f.name], f.default); trigger(cs); });
      tools.append(reset);
      if (cs.fb) {
        cs.useMeasured = h("button", { type: "button", class: "btn btn-sm btn-ghost", text: "Use measured", disabled: true,
          title: `set targets to the positions measured on ${cs.fb.topic}` });
        cs.useMeasured.addEventListener("click", () => {
          const m = measuredOf(cs, performance.now());
          for (const [k, v] of Object.entries(m)) if (v != null) setField(cs.fields[k], v);
          trigger(cs);
        });
        tools.append(cs.useMeasured);
      }
      cs.body.append(tools);
    }
    if (cs.mode === "click" && !c.fields.some((f) => f.choices)) {
      cs.button = h("button", { type: "button", class: "send", "data-testid": "send", "aria-describedby": `st-${c.id}`,
        text: c.kind === "call" ? "Call" : "Send" });
      cs.button.addEventListener("click", () => fire(cs));
      cs.body.append(cs.button);
    }
    div.append(cs.body);
    cs.statusEl = h("div", { class: "status dim", "data-testid": "status", id: `st-${c.id}`, role: "status" });
    div.append(cs.statusEl);
    S.controls.set(c.id, cs);
    for (const fn of cs.afterBuild) fn();
    return cs;
  }

  // AiNex-style head: one pad drives both head targets; dragging it streams head_pan and
  // head_tilt like their own sliders.
  function padPair(p) {
    const ok = (id) => { const c = p.controls.find((x) => x.id === id); return c && c.fields.some((f) => f.name === "position" && !f.choices && f.min != null && f.max != null); };
    return ok("head_pan") && ok("head_tilt") ? { pan: "head_pan", tilt: "head_tilt" } : null;
  }
  function buildPadGroup(pair) {
    const wrap = h("div", { class: "head-group" });
    const pad = h("div", { class: "pad", tabindex: "0", role: "group",
      "aria-label": "Head pad: dragging or the arrow keys send the pan and tilt targets" });
    const dot = h("span", { class: "pad-dot" });
    const zx = h("span", { class: "pad-zero" }), zy = h("span", { class: "pad-zero h" });
    const readout = h("div", { class: "pad-readout", "aria-live": "polite" });
    pad.append(zx, zy, dot);
    const card = h("div", { class: "pad-card" },
      h("div", { class: "ctl-head" }, h("div", {}, h("h3", { text: "Head" }),
        h("div", { class: "ctl-kind", text: "Drag to move the head: pan and tilt are sent as you drag (≈10 per s)." }))),
      pad, readout);
    wrap.append(card);
    const g = { div: wrap, sync: () => {} };
    g.init = () => {
      const PC = S.controls.get(pair.pan), TC = S.controls.get(pair.tilt);
      const P = PC.fields.position, T = TC.fields.position;
      const fp = P.f, ft = T.f;
      const live = () => liveOk(PC) && liveOk(TC);
      g.sync = () => { pad.classList.toggle("disabled", !live()); pad.setAttribute("aria-disabled", String(!live())); pad.tabIndex = live() ? 0 : -1; };
      // + pan = left, + tilt = up (profile labels): left edge is pan max, top edge is tilt max.
      const xOf = (v) => (fp.max - v) / (fp.max - fp.min), yOf = (v) => (ft.max - v) / (ft.max - ft.min);
      const x0 = xOf(Math.min(fp.max, Math.max(fp.min, 0))) * 100, y0 = yOf(Math.min(ft.max, Math.max(ft.min, 0))) * 100;
      zx.style.left = `${x0}%`; zy.style.top = `${y0}%`;
      const lbl = (txt, css) => { const e = h("span", { class: "pad-lbl", text: txt }); Object.assign(e.style, css); pad.append(e); };
      lbl(`← left ${exact(fp.max)}`, { left: "8px", top: `calc(${y0}% + 4px)` });
      lbl(`right ${exact(fp.min)} →`, { right: "8px", top: `calc(${y0}% + 4px)` });
      lbl(`↑ up ${exact(ft.max)}`, { top: "6px", left: `calc(${x0}% + 8px)` });
      lbl(`↓ down ${exact(ft.min)}`, { bottom: "6px", left: `calc(${x0}% + 8px)` });
      const paint = () => {
        const pv = Number(P.num.value), tv = Number(T.num.value);
        const okP = Number.isFinite(pv) && pv >= fp.min && pv <= fp.max, okT = Number.isFinite(tv) && tv >= ft.min && tv <= ft.max;
        dot.hidden = !(okP && okT);
        if (okP && okT) { dot.style.left = `${xOf(pv) * 100}%`; dot.style.top = `${yOf(tv) * 100}%`; }
        readout.textContent = `pan ${okP ? signed(pv, 2) : "invalid"} rad · tilt ${okT ? signed(tv, 2) : "invalid"} rad`;
      };
      P.listeners.push(paint); T.listeners.push(paint); paint();
      const setFrom = (ev) => {
        const r = pad.getBoundingClientRect();
        const x = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)), y = Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height));
        setField(P, fp.max - x * (fp.max - fp.min)); setField(T, ft.max - y * (ft.max - ft.min));
        trigger(PC); trigger(TC);
      };
      let drag = false;
      pad.addEventListener("pointerdown", (ev) => { if (!live()) return; drag = true; try { pad.setPointerCapture(ev.pointerId); } catch (e) {} setFrom(ev); });
      pad.addEventListener("pointermove", (ev) => { if (drag && live()) setFrom(ev); });
      const end = () => { drag = false; };
      pad.addEventListener("pointerup", end); pad.addEventListener("pointercancel", end);
      pad.addEventListener("keydown", (e) => {
        const st = e.shiftKey ? 0.2 : 0.05;
        const d = { ArrowLeft: [st, 0], ArrowRight: [-st, 0], ArrowUp: [0, st], ArrowDown: [0, -st] }[e.key];
        if (!d || !live()) return;
        e.preventDefault();
        const pv = Number(P.num.value) || 0, tv = Number(T.num.value) || 0;
        setField(P, pv + d[0]); setField(T, tv + d[1]);
        if (d[0]) trigger(PC); if (d[1]) trigger(TC);
      });
      g.sync();
    };
    return g;
  }

  function renderControls() {
    const root = $("controls"); root.innerHTML = ""; S.controls.clear(); S.padGroup = null;
    $("workspace").classList.toggle("no-controls", !S.target || !S.target.profile.controls.length);
    if (!S.target) { root.append(h("p", { class: "empty", text: "No validated target: no controls are offered." })); refreshControls(); return; }
    const t = S.target, g = S.graph;
    if (!t.profile.controls.length) root.append(h("p", { class: "empty", "data-testid": "no-controls", text: "This robot's profile lists no bounded page controls." }));
    const pair = padPair(t.profile);
    let group = null;
    for (const c of t.profile.controls) {
      const cs = buildControl(c, t, g);
      if (pair && (c.id === pair.pan || c.id === pair.tilt)) {
        if (!group) { group = buildPadGroup(pair); root.append(group.div); }
        group.div.append(cs.div);
      } else root.append(cs.div);
    }
    if (group) { group.init(); S.padGroup = group; }
    for (const el of root.querySelectorAll(".control.compact")) {      // side by side only in runs of two or more
      const pair = [el.previousElementSibling, el.nextElementSibling].some((x) => x && x.matches(".control.compact"));
      if (!pair) el.classList.add("solo");
    }
    for (const cs of S.controls.values()) if (cs.fb) subscribeFeedback(cs.fb.topic, cs.fb.type);
    refreshControls();
  }

  // ------------------------------------------------------------------ target lifecycle
  async function connectAndSelect(robotId, ns) {
    S.target = null;
    $("target").textContent = "none"; $("target-pill").dataset.state = "none";
    setValidation("pending", "Connecting to the wire…", S.config.url, "");
    setConn("pending", "connecting…");
    const conn = new RosConn(S.config.url);
    try { await conn.connect(5000); }
    catch (e) {
      setConn("bad", "unreachable - reload to retry");
      setValidation("bad", "Cannot connect", "", "cannot connect to " + S.config.url);
      S.lost = true; renderControls(); updateLock(); return;
    }
    S.conn = conn;
    conn.onlost = () => onLost(conn);
    setConn("ok", "connected");
    setValidation("pending", "Reading the wire…", "Discovering and type-checking the interface through rosapi", "");
    try { S.graph = await fetchGraph(conn); }
    catch (e) { setValidation("bad", "Cannot read the wire", "", "cannot read the wire through rosapi: " + e.message); renderControls(); return; }
    clearCameras(); clearFeedback();
    try {
      S.target = selectTarget(S.graph, S.profiles, robotId || null, ns == null || ns === "" ? null : ns);
    } catch (e) {
      S.target = null;
      setValidation("refused", "No validated target — controls unavailable",
        "Select a robot (and namespace, where supported) and press ‘Select target’.",
        (e.reason || e.message) + ".");
      $("target-pill").dataset.state = "warn";
      for (const [n, ty] of Object.entries(S.graph.topics).filter(([, ty]) => normType(ty) === "sensor_msgs/Image").sort()) addCamera(n, ty, UNTIED_STALE_S, null, false, false);
      renderControls(); updateLock();
      log({ ev: "no-target", reason: e.reason || e.message });
      return;
    }
    const t = S.target;
    $("target").textContent = t.label; $("target-pill").dataset.state = "ok";
    $("robot").value = t.profile.id; $("namespace").value = t.namespace; syncNsField(); $("namespace").value = t.namespace;
    setValidation("ok", `${t.label} validated`,
      `${t.profile.name} · ${t.profile.dialect === "ros2" ? "ROS 2" : "ROS 1"} · typed interface matches the packaged profile`, "");
    for (const c of t.profile.cameras) addCamera(t.wire(c.topic), c.type, c.stale_after_s, c.encodings, true, !(t.wire(c.topic) in S.graph.topics));
    renderControls(); updateLock();
    log({ ev: "target", label: t.label });
  }

  // Connection loss sends nothing: the controls are disabled and the page stays disconnected
  // until it is reloaded. Goals already sent keep running.
  function onLost(conn) {
    if (conn !== S.conn) return;
    S.lost = true;
    setConn("bad", "DISCONNECTED - reload the page to connect again");
    if (S.target) $("target-pill").dataset.state = "bad";
    showBanner(`Connection to ${conn.url} lost. Controls disabled; nothing was sent and nothing was stopped. Reload the page to connect again.`);
    refreshControls(); updateLock();
    log({ ev: "lost" });
  }

  async function changeTarget() {
    if (S.lost) return;
    const old = S.conn;
    retireControls(); showBanner("");
    clearCameras(); clearFeedback(); S.controls.clear(); S.padGroup = null; $("controls").innerHTML = "";
    if (old) setTimeout(() => old.close(), 300);
    const id = $("robot").value || null; const nsRaw = $("namespace").value.trim();
    await connectAndSelect(id, nsAllowed() ? (nsRaw === "" ? null : nsRaw) : null);
  }
  // ------------------------------------------------------------------ theme
  const THEMES = ["auto", "dark", "light"];
  function applyTheme(th) {
    if (th === "auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = th;
    const lbl = `Colour theme: ${th === "auto" ? "automatic" : th}`;
    $("theme").setAttribute("aria-label", lbl); $("theme").title = lbl + " (click to change)";
  }
  let theme = "auto";
  try { theme = localStorage.getItem("rc-theme") || "auto"; } catch (e) {}
  if (!THEMES.includes(theme)) theme = "auto";
  applyTheme(theme);
  $("theme").addEventListener("click", () => {
    theme = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length]; applyTheme(theme);
    try { localStorage.setItem("rc-theme", theme); } catch (e) {}
  });


  // ------------------------------------------------------------------ wiring
  // No arm switch, STOP, Esc or automatic stop/cancel: the page sends only what the user changes.
  $("select").addEventListener("click", changeTarget);
  $("robot").addEventListener("change", () => { $("namespace").disabled = !nsAllowed() || S.lost; syncNsField(); });

  async function boot() {
    S.config = await (await fetch("config.json")).json();
    const pj = await (await fetch("profiles.json")).json();
    INFRA = pj.infrastructure; S.profiles = pj.profiles;
    $("url").textContent = S.config.url;
    const sel = $("robot");
    sel.appendChild(h("option", { value: "", text: "Identify automatically" }));
    for (const p of S.profiles) sel.appendChild(h("option", { value: p.id, title: p.name, text: `${p.id} — ${p.name.split(" (")[0]}` }));
    sel.value = S.config.robot || ""; $("namespace").value = S.config.namespace || "";
    $("namespace").disabled = !nsAllowed(); syncNsField(); $("namespace").value = S.config.namespace || "";
    await connectAndSelect(S.config.robot, S.config.namespace);
    RC.ready = true;
  }
  boot().catch((e) => { setValidation("bad", "Page error", "", "page error: " + e.message); RC.ready = true; });
})();
