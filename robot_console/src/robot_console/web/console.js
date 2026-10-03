// Robot console page: rosbridge transport, typed discovery/validation (the same algorithm and
// the same packaged profiles as robot_console/discovery.py), live cameras and bounded controls
// whose changes are sent at once. Plain browser JavaScript, no external dependencies.
"use strict";
(() => {
  const RC = (window.RC = {});        // RC.state and RC.ready: the page state, for tests and tools
  const $ = (id) => document.getElementById(id);

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
    if (INFRA.nodes.includes(name)) return true;
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
      this.wsClosed = new Promise((r) => { this._wsClosed = r; });   // the socket finished closing
    }
    connect(timeoutMs = 5000) {
      return new Promise((resolve, reject) => {
        let done = false;
        const timer = setTimeout(() => { if (!done) { done = true; reject(new Error("timeout")); try { this.ws.close(); } catch (e) {} } }, timeoutMs);
        try { this.ws = new WebSocket(this.url); } catch (e) { clearTimeout(timer); reject(e); return; }
        this.ws.onopen = () => { if (!done) { done = true; clearTimeout(timer); resolve(this); } };
        this.ws.onerror = () => { if (!done) { done = true; clearTimeout(timer); reject(new Error("cannot connect to " + this.url)); } };
        this.ws.onclose = () => { this._wsClosed(); this._closed(); };
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
    // Close, resolving once the socket has closed (or after timeoutMs).
    closeAndWait(timeoutMs = 500) {
      this.close();
      return Promise.race([this.wsClosed, new Promise((r) => setTimeout(r, timeoutMs))]);
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
  // Bytes per pixel of the supported raw encodings (4:2:2 packed: 2), as camera.py.
  const BPP = { rgb8: 3, bgr8: 3, "8uc3": 3, rgba8: 4, bgra8: 4, mono8: 1, "8uc1": 1, mono16: 2, "16uc1": 2,
    yuv422: 2, uyvy: 2, yuv422_yuy2: 2, yuyv: 2 };
  const YUV422 = ["yuv422", "uyvy", "yuv422_yuy2", "yuyv"];
  function decodeImage(msg, allowed) {
    const enc = String(msg.encoding || "").toLowerCase();
    if (allowed && allowed.length && !allowed.map((x) => x.toLowerCase()).includes(enc)) throw Object.assign(new Error(`unsupported encoding '${enc}'`), { unsupported: true });
    if (!(enc in BPP)) throw Object.assign(new Error(`unsupported encoding '${enc}'`), { unsupported: true });
    const w = msg.width | 0, h = msg.height | 0, step = msg.step | 0;
    const bin = typeof msg.data === "string" ? atob(msg.data) : null;
    const len = bin ? bin.length : (msg.data || []).length;
    const at = bin ? (i) => bin.charCodeAt(i) : (i) => msg.data[i];
    if (w <= 0 || h <= 0) throw new Error("empty image");
    if (step < w * BPP[enc]) throw new Error(`row step ${step} shorter than ${w} pixels of ${enc}`);
    if (YUV422.includes(enc) && w % 2) throw new Error("odd width for a 4:2:2 image");
    if (len < step * h) throw new Error(`truncated image (${len} of ${step * h} bytes)`);
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
    lost: false, streams: [], controls: new Map(), feedback: new Map(), reads: new Map(), conflicts: new Map(),
    padGroup: null, model: null });

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
    if (s.missing) return ["missing", "MISSING", "unavailable: the topic is not on the wire"];
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
    drawModel();
  }
  setInterval(tick, 200);

  // ------------------------------------------------------------------ reported joint positions
  // Shown only where the profile documents them: explicitly, by its model's joints that name a
  // control field and the joint's name on the model's documented JointState feedback (with the
  // documented unit mapping, field = (rad - offset) / scale); otherwise where a control's own
  // prerequisites include a documented JointState output topic and its template pairs joint
  // names with field values. Nothing is invented. `measured` is false where the profile says the
  // robot reports something other than a measurement (e.g. an echo of its last commands).
  function feedbackFor(c, t) {
    const jp = jointPrefix(t.profile, t.namespace), m = t.profile.model;
    if (m && m.feedback) {
      const map = {}, conv = {};
      for (const j of m.joints) for (const b of j.command || []) {
        if (b.control !== c.id || !j.reported) continue;
        map[b.field] = j.reported.split("{joint_prefix}").join(jp);
        conv[b.field] = { scale: b.scale == null ? 1 : Number(b.scale), offset: Number(b.offset || 0) };
      }
      if (Object.keys(map).length) return { topic: t.wire(m.feedback.topic), type: m.feedback.type, map, conv, measured: !!m.feedback.measured };
    }
    const fb = c.prerequisites.map((n) => t.profile.endpoints.find((e) => e.name === n))
      .find((e) => e && e.kind === "topic" && e.direction === "out" && normType(e.type) === "sensor_msgs/JointState");
    if (!fb) return null;
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
    return Object.keys(map).length ? { topic: t.wire(fb.name), type: fb.type, map, conv: {}, measured: true } : null;
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
        scheduleModelDraw();
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
    for (const [field, joint] of Object.entries(cs.fb.map)) {
      const cv = cs.fb.conv[field] || { scale: 1, offset: 0 };
      out[field] = fresh && f.pos.has(joint) ? (f.pos.get(joint) - cv.offset) / cv.scale : null;
    }
    return out;
  }
  const fbWord = (fb) => (fb.measured ? "measured" : "reported");
  function tickFeedback(now) {
    for (const cs of S.controls.values()) {
      let any = false;
      if (cs.fb) {
        const m = measuredOf(cs, now);
        for (const fs of Object.values(cs.fields)) {
          if (!fs.meas) continue;
          const v = m[fs.f.name];
          if (v == null) {
            fs.measVal.textContent = "—"; fs.meas.dataset.state = "none"; fs.meas.title = `${fbWord(cs.fb)}: no fresh ${cs.fb.topic} message`;
            fs.mark.hidden = true;
          } else {
            if (cs.fb.measured) any = true;
            fs.measVal.textContent = `${signed(v, decimals(fs.f) + 1)} ${fs.f.unit}`; fs.meas.dataset.state = "live";
            fs.meas.title = cs.fb.measured ? `measured ${cs.fb.map[fs.f.name]} on ${cs.fb.topic}`
              : `reported ${cs.fb.map[fs.f.name]} on ${cs.fb.topic}: what the robot publishes for this joint, not a measurement`;
            const p = Math.min(1, Math.max(0, (v - fs.f.min) / (fs.f.max - fs.f.min)));
            fs.mark.style.left = `${p * 100}%`;
          }
        }
      }
      if (cs.reads.length && cs.readStatus) {
        const rv = readValues(cs);
        for (const fs of Object.values(cs.fields)) {
          if (!fs.read) continue;
          const r = rv[fs.f.name];
          if (!r || r.v == null) {
            fs.readVal.textContent = "—"; fs.read.dataset.state = r ? "invalid" : "none";
            fs.read.title = r ? `the last read of ${r.x.wire} gave no valid value for this field` : `not read yet: “${cs.readBtn.textContent}” asks the robot once`;
          } else {
            any = true;
            fs.readVal.textContent = `${signed(r.v, decimals(fs.f) + 1)} ${fs.f.unit}`; fs.read.dataset.state = "read";
            fs.read.title = `measured: read from ${r.x.wire} ${((now - r.x.last.at) / 1000).toFixed(1)} s ago (not live)`;
          }
        }
        cs.readStatus.textContent = readText(cs.reads, now);
      }
      if (cs.useMeasured) cs.useMeasured.disabled = !any;
    }
  }

  // ------------------------------------------------------------------ measured reads
  // Read services the profile documents as returning measured joint positions (e.g. a servo
  // readback). The page calls one only when the user clicks its button, once per click, never
  // on load or by itself; each answered value maps to a control field as
  // field = (raw - offset) / scale. A reading is a snapshot, shown with its age, not live.
  function readPath(v, path) {
    for (const seg of String(path).split(".")) {
      if (v == null) return undefined;
      const m = /^([A-Za-z_]\w*)\[([A-Za-z_]\w*)=(-?[\d.]+)\]$/.exec(seg);      // list element by key: name[key=value]
      if (m) { const a = v[m[1]]; v = Array.isArray(a) ? a.find((x) => x && Number(x[m[2]]) === Number(m[3])) : undefined; }
      else if (/^\d+$/.test(seg)) v = Array.isArray(v) ? v[Number(seg)] : undefined;
      else v = v[seg];
    }
    return v;
  }
  function prepareReads(t, g) {
    S.reads = new Map();
    for (const r of t.profile.reads || []) {
      const e = t.profile.endpoints.find((x) => x.name === r.service && x.kind === "service"), wire = t.wire(r.service);
      let reason = null;
      if (!e) reason = `${r.service} is not in the profile`;
      else if (!has(g, "service", wire)) reason = `${wire} is not on the wire`;
      else if (normType(typeOf(g, "service", wire)) !== normType(r.type)) reason = `${wire} has type ${typeOf(g, "service", wire)}, expected ${r.type}`;
      S.reads.set(r.id, { r, wire, reason, last: null, busy: false, err: null });
    }
  }
  // The available reads that give values for control `cid`.
  const readsOf = (cid) => [...S.reads.values()].filter((x) => !x.reason && x.r.values.some((m) => m.control === cid));
  function readValues(cs) {
    const out = {};
    for (const x of cs.reads) {
      if (!x.last || !x.last.vals[cs.ctrl.id]) continue;
      for (const [f, v] of Object.entries(x.last.vals[cs.ctrl.id])) if (!out[f] || out[f].x.last.at < x.last.at) out[f] = { v, x };
    }
    return out;
  }
  function readText(xs, now) {
    return xs.map((x) => x.busy ? `reading ${x.wire}…` : x.err ? `read of ${x.wire} failed: ${x.err}`
      : x.last ? `read from ${x.wire} ${((now - x.last.at) / 1000).toFixed(1)} s ago` : "").filter(Boolean).join("; ");
  }
  function doRead(x) {
    const conn = S.conn;
    if (!conn || S.lost || !S.target || x.busy) return;
    const r = x.r;
    x.busy = true; x.err = null;
    conn.call(x.wire, r.request || {}, 5000).then((v) => {
      if (r.ok_field && v && v[r.ok_field] === false) throw new Error(`the service answered ${JSON.stringify(v).slice(0, 120)}`);
      const raws = r.values.map((m) => readPath(v, m.path));
      const allZero = !!r.invalid_if_all_zero && raws.every((z) => Number(z) === 0);
      const vals = {};
      r.values.forEach((m, i) => {
        const raw = Number(raws[i]);
        const ok = raws[i] != null && Number.isFinite(raw) && !(r.invalid || []).includes(raw) && !allZero;
        (vals[m.control] = vals[m.control] || {})[m.field] = ok ? (raw - Number(m.offset || 0)) / (m.scale == null ? 1 : Number(m.scale)) : null;
      });
      x.last = { at: performance.now(), vals };
    }).catch((e) => { x.err = e.message; })
      .finally(() => { x.busy = false; tickFeedback(performance.now()); });
    tickFeedback(performance.now());
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

  // Every control shows the phase of its latest send: sending (on its way, or waiting for the
  // answer), running (a goal the action server acknowledged), done, failed, or invalid (a value
  // outside the documented limits, never sent). The text starts with the phase.
  const PHASE_CLS = { sending: "warn", running: "warn", done: "ok", failed: "bad", invalid: "bad" };
  function setStatus(id, phase, text) {
    const cs = S.controls.get(id); if (!cs) return;
    const cls = PHASE_CLS[phase] || "dim";
    text = `${phase}: ${text}`;
    cs.status = { phase, cls, text }; cs.statusEl.textContent = text; cs.statusEl.className = "status " + cls;
    cs.div.dataset.status = cls; cs.div.dataset.phase = phase;
  }

  // A value of a streamed control changed: send it now, or as soon as the throttle allows.
  function trigger(cs) {
    if (!liveOk(cs) || cs.mode !== "stream") return;
    cs.th.pending = true; pump(cs);
    if (cs.th.pending && (!cs.status || cs.status.phase !== "sending")) setStatus(cs.ctrl.id, "sending", "the new value goes out as the rate limit allows…");
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
    catch (e) { setStatus(c.id, "invalid", "not sent: " + e.message); return false; }
    let payload;
    try { payload = fill(c.template, values, jointPrefix(t.profile, t.namespace)); }
    catch (e) { setStatus(c.id, "invalid", "not sent: " + e.message); return false; }
    const name = t.wire(c.name);
    try {
      if (c.kind === "publish") {
        conn.publish(name, payload, c.type); cs.sent++;
        setStatus(c.id, "done", `sent (${cs.sent} message${cs.sent === 1 ? "" : "s"} so far; a publish is not acknowledged)`);
      } else if (c.kind === "call") {
        setStatus(c.id, "sending", "pending: waiting for the service answer…");
        conn.call(name, payload, 10000).then((v) => {
          // A documented response field (the profile's ok_field) that is false is a failure.
          const bad = v && ((c.ok_field && v[c.ok_field] === false) || v.result === false || v.success === false);
          if (bad) setStatus(c.id, "failed", "the service answered " + JSON.stringify(v));
          else setStatus(c.id, "done", "the service answered " + JSON.stringify(v).slice(0, 120));
        }, (e) => setStatus(c.id, "failed", e.message));
      } else if (c.kind === "action") {
        await sendGoal2(cs, conn, name, payload);       // action controls are ROS 2 only (profiles test)
      }
      return true;
    } catch (e) { setStatus(c.id, "failed", "could not send: " + e.message); return false; }
  }

  // ROS 2: stock rosbridge (Humble, Jazzy) handles one client's send_action_goal to completion
  // before that client's next op, so each goal gets its own connection. Its result or feedback
  // acknowledges it; once the latest goal is acknowledged the superseded goals' connections
  // close (closing one cancels nothing on stock rosbridge). At most GOAL_CONNS_MAX are open,
  // counting the new goal's: the oldest superseded ones close (and are closed) before it opens.
  async function sendGoal2(cs, conn, name, payload) {
    const c = cs.ctrl;
    const room = cs.goals.slice(0, Math.max(0, cs.goals.length - (GOAL_CONNS_MAX - 1)));
    await Promise.all(room.map((o) => retireGoal(cs, o, true)));
    if (!liveOk(cs) || S.conn !== conn) return;
    const gc = new RosConn(conn.url); await gc.connect(5000);
    if (!liveOk(cs) || S.conn !== conn) { gc.close(); return; }
    const g = { conn: gc, id: randomId(), sentAt: Date.now(), acked: false, done: false };
    cs.goal = g; cs.goals.push(g);
    const ack = () => {
      if (g.acked) return;
      g.acked = true;
      if (cs.goal === g) {
        if (!g.done) setStatus(c.id, "running", "goal running (acknowledged by the action server)");
        for (const o of [...cs.goals]) if (o !== g) retireGoal(cs, o);
      }
      pump(cs);
    };
    gc.handlers.set("action_feedback:" + g.id, ack);
    gc.handlers.set("action_result:" + g.id, (m) => {
      g.done = true; ack();
      if (cs.goal === g) {
        // GoalStatus: 4 succeeded, 5 canceled, 6 aborted; anything but success did not reach the target
        const st = { 4: "succeeded", 5: "canceled", 6: "aborted", 2: "canceled" }[m.status] || `ended (status ${m.status})`;
        setStatus(c.id, m.result !== false && m.status === 4 ? "done" : "failed",
          "goal " + st + (m.result === false ? ": " + JSON.stringify(m.values) : ""));
      }
      retireGoal(cs, g);
    });
    gc.onlost = () => {
      if (!g.done && cs.goal === g) setStatus(c.id, "failed", "goal connection lost; outcome unknown");
      g.done = true; retireGoal(cs, g);
    };
    gc.send({ op: "send_action_goal", id: g.id, action: name, action_type: c.type, args: payload, feedback: true });
    setStatus(c.id, "sending", "goal sent, waiting for the action server…");
    setTimeout(() => pump(cs), GOAL_ACK_MS + 10);
    const old = cs.goals.filter((o) => o !== g);
    for (let i = 0; i < old.length && cs.goals.length > GOAL_CONNS_MAX; i++) retireGoal(cs, old[i]);
  }
  // `now`: close at once and resolve when the socket has closed (making room for a new goal).
  function retireGoal(cs, g, now = false) {
    const i = cs.goals.indexOf(g); if (i < 0) return Promise.resolve();
    cs.goals.splice(i, 1);
    if (now) return g.conn.closeAndWait(500);
    setTimeout(() => g.conn.close(), 50);
    return Promise.resolve();
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
    drawModel();
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
    const fs = { f, div, num, range, mark, err: div.querySelector(".field-err"), listeners: [], meas: null, measVal: null, read: null, readVal: null };
    if (cs.fb && cs.fb.map[f.name]) {
      fs.measVal = h("span", { class: "m-val", text: "—" });
      fs.meas = h("span", { class: "measured" }, h("span", { class: "m-label", text: fbWord(cs.fb) }), fs.measVal);
      div.querySelector(".field-top").insertBefore(fs.meas, div.querySelector(".num-wrap"));
    }
    if (cs.reads.some((x) => x.r.values.some((m) => m.control === c.id && m.field === f.name))) {
      fs.readVal = h("span", { class: "m-val", text: "—" });
      fs.read = h("span", { class: "measured read", "data-testid": "read-value" }, h("span", { class: "m-label", text: "measured" }), fs.readVal);
      div.querySelector(".field-top").insertBefore(fs.read, div.querySelector(".num-wrap"));
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
    if (!reason && S.conflicts.has(c.id)) reason = S.conflicts.get(c.id);
    const compact = !c.fields.length;
    const div = h("div", { class: "control" + (compact ? " compact" : "") + (reason ? " unavailable" : ""), "data-control": c.id, "data-testid": "control" });
    const cs = { ctrl: c, div, inputs: {}, fields: {}, available: !reason, status: null, afterBuild: [], fb: reason ? null : feedbackFor(c, t),
      reads: reason ? [] : readsOf(c.id),
      mode: sendMode(c), th: { pending: false, timer: null, busy: false, last: 0 }, goal: null, goals: [], sent: 0, dead: false };
    div.append(h("div", { class: "ctl-head" },
      h("div", {}, h("h3", { text: c.label }), h("div", { class: "ctl-kind", text: kindText(c) }))));
    div.append(h("details", { class: "ctl-wire" }, h("summary", { text: "Interface" }),
      h("dl", {},
        h("dt", { text: "Name" }), h("dd", { text: t.wire(c.name) }),
        h("dt", { text: "Type" }), h("dd", { text: c.type }),
        cs.fb ? [h("dt", { text: cs.fb.measured ? "Measured" : "Reported" }), h("dd", { text: cs.fb.topic })] : null)));
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
      if (cs.reads.length) {
        // One call of each documented read service per click; never automatic.
        cs.readBtn = h("button", { type: "button", class: "btn btn-sm btn-ghost", "data-testid": "read", text: "Read measured",
          title: "ask the robot once for the measured positions: " + cs.reads.map((x) => x.wire).join(", ") });
        cs.readBtn.addEventListener("click", () => { for (const x of cs.reads) doRead(x); });
        tools.append(cs.readBtn);
      }
      if ((cs.fb && cs.fb.measured) || cs.reads.length) {
        const from = [cs.fb && cs.fb.measured ? cs.fb.topic : null, ...cs.reads.map((x) => x.wire)].filter(Boolean).join(" / ");
        cs.useMeasured = h("button", { type: "button", class: "btn btn-sm btn-ghost", text: "Use measured", disabled: true,
          title: `set targets to the positions measured on ${from}` });
        cs.useMeasured.addEventListener("click", () => {
          // A fresh measurement on the documented topic first, else the latest read.
          const m = cs.fb && cs.fb.measured ? measuredOf(cs, performance.now()) : {}, rv = readValues(cs);
          for (const k of Object.keys(cs.fields)) {
            const v = m[k] != null ? m[k] : rv[k] ? rv[k].v : null;
            if (v != null) setField(cs.fields[k], v);
          }
          trigger(cs);
        });
        tools.append(cs.useMeasured);
      }
      if (cs.reads.length) { cs.readStatus = h("span", { class: "read-status", "data-testid": "read-status", role: "status" }); tools.append(cs.readStatus); }
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
      // − pan = the robot's left (head_pan turns about the model's −Z axis, console spec §2.1),
      // + tilt = up: the left edge is pan min, the top edge is tilt max.
      const xOf = (v) => (v - fp.min) / (fp.max - fp.min), yOf = (v) => (ft.max - v) / (ft.max - ft.min);
      const x0 = xOf(Math.min(fp.max, Math.max(fp.min, 0))) * 100, y0 = yOf(Math.min(ft.max, Math.max(ft.min, 0))) * 100;
      zx.style.left = `${x0}%`; zy.style.top = `${y0}%`;
      const lbl = (txt, css) => { const e = h("span", { class: "pad-lbl", text: txt }); Object.assign(e.style, css); pad.append(e); };
      lbl(`← left ${exact(fp.min)}`, { left: "8px", top: `calc(${y0}% + 4px)` });
      lbl(`right ${exact(fp.max)} →`, { right: "8px", top: `calc(${y0}% + 4px)` });
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
        setField(P, fp.min + x * (fp.max - fp.min)); setField(T, ft.max - y * (ft.max - ft.min));
        trigger(PC); trigger(TC);
      };
      let drag = false;
      pad.addEventListener("pointerdown", (ev) => { if (!live()) return; drag = true; try { pad.setPointerCapture(ev.pointerId); } catch (e) {} setFrom(ev); });
      pad.addEventListener("pointermove", (ev) => { if (drag && live()) setFrom(ev); });
      const end = () => { drag = false; };
      pad.addEventListener("pointerup", end); pad.addEventListener("pointercancel", end);
      pad.addEventListener("keydown", (e) => {
        const st = e.shiftKey ? 0.2 : 0.05;
        const d = { ArrowLeft: [-st, 0], ArrowRight: [st, 0], ArrowUp: [0, st], ArrowDown: [0, -st] }[e.key];
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
    const root = $("controls"); root.innerHTML = ""; S.controls.clear(); S.padGroup = null; S.reads = new Map();
    $("workspace").classList.toggle("no-controls", !S.target || !S.target.profile.controls.length);
    if (!S.target) { root.append(h("p", { class: "empty", text: "No validated target: no controls are offered." })); renderModel(); refreshControls(); return; }
    const t = S.target, g = S.graph;
    prepareReads(t, g);
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
    renderModel();
    refreshControls();
  }

  // ------------------------------------------------------------------ 3D model
  // The validated target's embodiment from its profile's `model` (the joint tree of the pinned
  // vendor URDF), drawn on a plain canvas with an orthographic projection; dragging empty space
  // turns the view. Each joint is posed from the joint position the robot reports where the
  // profile documents feedback for it, otherwise at the page's target for it (or zero when no
  // control commands it), and the page says which. A joint that a control commands can be
  // clicked and dragged, or selected and moved with the arrow keys, to change that control's
  // field: the same clamped, validated and throttled path as the control's own inputs.
  let M = null;
  const ID3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const mm3 = (A, B) => A.map((r) => [0, 1, 2].map((j) => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
  const mv3 = (A, v) => A.map((r) => dot3(r, v));
  const at3 = (T, p) => mv3(T.R, p).map((x, i) => x + T.t[i]);
  function rpyMat([r, p, y]) {                                   // URDF: Rz(yaw) Ry(pitch) Rx(roll)
    const cr = Math.cos(r), sr = Math.sin(r), cp = Math.cos(p), sp = Math.sin(p), cy = Math.cos(y), sy = Math.sin(y);
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
      [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
      [-sp, cp * sr, cp * cr]];
  }
  function axisMat(a, q) {                                       // rotation by q about axis a
    const n = Math.hypot(a[0], a[1], a[2]) || 1, x = a[0] / n, y = a[1] / n, z = a[2] / n;
    const c = Math.cos(q), s = Math.sin(q), C = 1 - c;
    return [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
      [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
      [z * x * C - y * s, z * y * C + x * s, c + z * z * C]];
  }

  function buildModel(t) {
    const m = t.profile.model;
    if (!m || !(m.joints || []).length) return null;
    const jp = jointPrefix(t.profile, t.namespace);
    const joints = m.joints.map((j) => Object.assign({}, j, { R0: rpyMat(j.rpy || [0, 0, 0]), q: 0, src: "zero", active: null,
      wire: j.reported ? j.reported.split("{joint_prefix}").join(jp) : null }));
    const kids = new Map();
    for (const j of joints) { if (!kids.has(j.parent)) kids.set(j.parent, []); kids.get(j.parent).push(j); }
    return { m, joints, kids, byName: new Map(joints.map((j) => [j.name, j])),
      commanded: joints.filter((j) => !j.mimic && (j.command || []).length),
      fb: m.feedback ? { topic: t.wire(m.feedback.topic), type: m.feedback.type, measured: !!m.feedback.measured } : null,
      yaw: Math.PI - 0.7, pitch: 0.5, fit: null, selected: null, drag: null, pts: new Map(), chips: new Map() };
  }
  // A joint's command binding resolved to the field built on this page (null if not built).
  function bound(b) {
    const cs = S.controls.get(b.control), fs = cs && cs.fields[b.field];
    return fs ? { cs, fs, scale: b.scale == null ? 1 : Number(b.scale), offset: Number(b.offset || 0) } : null;
  }
  // The field the model edits for joint J: the one last changed while live, else the first live one.
  function editField(J) {
    const all = (J.command || []).map(bound).filter(Boolean);
    if (J.active && liveOk(J.active.cs)) return J.active;
    return all.find((x) => liveOk(x.cs)) || all.find((x) => x.cs.available) || all[0] || null;
  }

  function poseJoints(now) {
    const f = M.fb && S.feedback.get(M.fb.topic);
    const fresh = !!f && f.last != null && now - f.last < 1000 && !S.lost;
    for (const J of M.joints) {
      if (J.type === "fixed" || J.mimic) continue;
      if (fresh && J.wire && f.pos.has(J.wire)) { J.q = f.pos.get(J.wire); J.src = "reported"; continue; }
      const b = J.active || editField(J);
      if (!b) { J.q = 0; J.src = "zero"; continue; }
      const v = Number(b.fs.num.value);                          // an invalid entry keeps the last pose
      if (b.fs.num.value !== "" && Number.isFinite(v) && v >= b.fs.f.min && v <= b.fs.f.max) J.q = v * b.scale + b.offset;
      J.src = "target";
    }
    for (const J of M.joints) {
      if (!J.mimic) continue;
      const o = M.byName.get(J.mimic.joint);
      J.q = o ? Number(J.mimic.multiplier == null ? 1 : J.mimic.multiplier) * o.q + Number(J.mimic.offset || 0) : 0;
      J.src = o ? o.src : "zero";
    }
  }
  function forward() {
    const frames = new Map([[M.m.root, { R: ID3, t: [0, 0, 0] }]]), todo = [M.m.root];
    while (todo.length) {
      const link = todo.pop(), T = frames.get(link);
      for (const J of M.kids.get(link) || []) {
        let R = mm3(T.R, J.R0);
        if (J.type !== "fixed" && J.axis) R = mm3(R, axisMat(J.axis, J.q));
        J.T = { R, t: at3(T, J.xyz) };
        frames.set(J.child, J.T); todo.push(J.child);
      }
    }
    return frames;
  }
  // Line segments [a, b, kind] in the root frame: links between joint origins, links to their
  // documented tip point, and the edges of the documented boxes.
  function segments(frames) {
    const out = [];
    for (const J of M.joints) {
      const a = frames.get(J.parent);
      if (a && J.T && Math.hypot(...J.T.t.map((x, i) => x - a.t[i])) > 1e-6) out.push([a.t, J.T.t, "link"]);
    }
    for (const tp of M.m.tips || []) { const T = frames.get(tp.link); if (T) out.push([T.t, at3(T, tp.xyz), "link"]); }
    for (const bx of M.m.boxes || []) {
      const T = frames.get(bx.link); if (!T) continue;
      const c = [0, 1, 2, 3, 4, 5, 6, 7].map((k) => at3(T, [k & 1 ? bx.max[0] : bx.min[0], k & 2 ? bx.max[1] : bx.min[1], k & 4 ? bx.max[2] : bx.min[2]]));
      for (const [i, j] of [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]]) out.push([c[i], c[j], "box"]);
    }
    return out;
  }
  function fitOf(segs) {
    const pts = segs.flatMap((s) => [s[0], s[1]]); if (!pts.length) pts.push([0, 0, 0]);
    const lo = [0, 1, 2].map((i) => Math.min(...pts.map((p) => p[i]))), hi = [0, 1, 2].map((i) => Math.max(...pts.map((p) => p[i])));
    const center = lo.map((x, i) => (x + hi[i]) / 2);
    const radius = Math.max(0.05, ...pts.map((p) => Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2])));
    return { center, radius, floor: lo[2] };
  }
  function projector(w, h) {
    const cy = Math.cos(M.yaw), sy = Math.sin(M.yaw), cp = Math.cos(M.pitch), sp = Math.sin(M.pitch);
    const d = [cp * cy, cp * sy, sp], r = [-sy, cy, 0], u = [-sp * cy, -sp * sy, cp];
    const k = 0.45 * Math.min(w, h) / M.fit.radius, c = M.fit.center;
    return (p) => { const v = [p[0] - c[0], p[1] - c[1], p[2] - c[2]]; return [w / 2 + k * dot3(v, r), h / 2 - k * dot3(v, u), dot3(v, d)]; };
  }

  let modelQueued = false;
  function scheduleModelDraw() {
    if (!M || modelQueued) return;
    modelQueued = true;
    requestAnimationFrame(() => { modelQueued = false; drawModel(); });
  }
  const SRC_TEXT = { reported: "reported", target: "not reported: the page's target", zero: "not reported or commanded: drawn at 0" };
  function drawModel() {
    if (!M) return;
    const cv = $("model-canvas"), w = cv.clientWidth, hgt = cv.clientHeight;
    const now = performance.now();
    poseJoints(now);
    const frames = forward(), segs = segments(frames);
    if (!M.fit) M.fit = fitOf(segs);
    if (w && hgt) {
      const dpr = window.devicePixelRatio || 1;
      if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(hgt * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(hgt * dpr); }
      const g = cv.getContext("2d"), P = projector(w, hgt), css = getComputedStyle(document.documentElement);
      const col = (n) => css.getPropertyValue(n).trim();
      g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, hgt); g.lineCap = "round";
      const R = M.fit.radius, c = M.fit.center;                    // floor grid under the model
      g.strokeStyle = col("--border"); g.lineWidth = 1;
      for (let i = -4; i <= 4; i++) {
        for (const [a, b] of [[[c[0] + i * R / 4, c[1] - R, M.fit.floor], [c[0] + i * R / 4, c[1] + R, M.fit.floor]],
          [[c[0] - R, c[1] + i * R / 4, M.fit.floor], [c[0] + R, c[1] + i * R / 4, M.fit.floor]]]) {
          const A = P(a), B = P(b); g.beginPath(); g.moveTo(A[0], A[1]); g.lineTo(B[0], B[1]); g.stroke();
        }
      }
      const items = segs.map(([a, b, kind]) => { const A = P(a), B = P(b); return { z: (A[2] + B[2]) / 2, kind, A, B }; });
      M.pts.clear();
      for (const J of M.joints) {
        if (J.type === "fixed" || !J.T) continue;
        const s = P(J.T.t), cmd = M.commanded.includes(J);
        if (cmd) M.pts.set(J.name, s);
        items.push({ z: M.selected === J.name ? Infinity : s[2] + 1e-4, kind: cmd ? "cmd" : "joint", A: s, J });
      }
      items.sort((x, y) => x.z - y.z);
      for (const it of items) {
        if (it.kind === "link" || it.kind === "box") {
          g.strokeStyle = col(it.kind === "link" ? "--muted" : "--faint"); g.lineWidth = it.kind === "link" ? 5 : 1.5;
          g.beginPath(); g.moveTo(it.A[0], it.A[1]); g.lineTo(it.B[0], it.B[1]); g.stroke();
        } else if (it.kind === "joint") {
          g.fillStyle = col("--faint"); g.beginPath(); g.arc(it.A[0], it.A[1], 3, 0, 2 * Math.PI); g.fill();
        } else {
          const J = it.J, sel = M.selected === J.name;
          if (sel && J.axis) {                                      // the selected joint's axis
            const e = P(at3(J.T, J.axis.map((x) => x * R * 0.3)));
            g.strokeStyle = col("--focus"); g.lineWidth = 2; g.beginPath(); g.moveTo(it.A[0], it.A[1]); g.lineTo(e[0], e[1]); g.stroke();
          }
          g.fillStyle = col("--ink"); g.strokeStyle = col(sel ? "--focus" : J.src === "reported" ? "--ok" : "--warn");
          g.lineWidth = 3; g.beginPath(); g.arc(it.A[0], it.A[1], sel ? 9 : 7, 0, 2 * Math.PI); g.fill(); g.stroke();
          if (sel) { g.fillStyle = col("--text"); g.font = "12px " + col("--mono"); g.fillText(J.name, it.A[0] + 12, it.A[1] - 10); }
        }
      }
    }
    syncModelText(now);
  }
  function syncModelText(now) {
    const f = M.fb && S.feedback.get(M.fb.topic);
    const n = { reported: 0, target: 0, zero: 0 };
    for (const J of M.joints) if (J.type !== "fixed" && !J.mimic) n[J.src]++;
    const why = M.fb ? (f && f.last != null ? `no fresh ${M.fb.topic}` : `no ${M.fb.topic} yet`) : "the profile documents no joint-position topic";
    const rest = [];
    if (n.target) rest.push(`${n.target} commanded joint${n.target > 1 ? "s" : ""} drawn at the page's targets`);
    if (n.zero) rest.push(`${n.zero} other joint${n.zero > 1 ? "s" : ""} at 0`);
    const text = n.reported
      ? `Pose reported on ${M.fb.topic}` + (M.fb.measured ? "" : " (what the robot publishes, not a measurement)") +
        (rest.length ? `; not reported: ${rest.join(", ")}.` : ".")
      : `Pose not reported (${why}): ${rest.join(", ") || "no movable joints"}.`;
    const pose = $("model-pose");
    if (pose.textContent !== text) pose.textContent = text;
    pose.dataset.state = n.reported && !n.target ? "reported" : n.reported ? "partial" : "not-reported";
    for (const J of M.commanded) {
      const chip = M.chips.get(J.name), b = editField(J);
      const txt = `${J.name} ${signed(Math.round(J.q * 100) / 100 || 0, 2)}`;
      if (chip.firstChild.textContent !== txt) chip.firstChild.textContent = txt;
      chip.dataset.src = J.src; chip.setAttribute("aria-pressed", String(M.selected === J.name));
      chip.disabled = !(b && liveOk(b.cs));
      chip.title = `${J.name}: ${SRC_TEXT[J.src]}` + (b ? `; commanded by ${b.cs.ctrl.label} › ${b.fs.f.label}` : "");
    }
    const J = M.selected && M.byName.get(M.selected), sel = $("model-sel");
    let st = "";
    if (J) {
      const b = editField(J);
      st = `${J.name}: ${signed(J.q, 3)} rad, ${SRC_TEXT[J.src]}.`;
      if (b) st += ` Commands ${b.cs.ctrl.label} › ${b.fs.f.label} = ${b.fs.num.value} ${b.fs.f.unit}` +
        (liveOk(b.cs) ? "; drag the joint or use the arrow keys to change it." : " (control not live).");
    }
    if (sel.textContent !== st) sel.textContent = st;
    // For tests and tools: the pose drawn, where each joint came from, each joint origin in the
    // root frame, and the commanded joints' positions on the canvas (CSS px from its top-left).
    S.model = { rendered: true, selected: M.selected, joints: {}, sources: {}, world: {}, clickable: M.commanded.map((x) => x.name), points: {} };
    for (const x of M.joints) {
      if (x.T) S.model.world[x.name] = x.T.t;
      if (x.type !== "fixed") { S.model.joints[x.name] = x.q; S.model.sources[x.name] = x.src; }
    }
    for (const [name, s] of M.pts) S.model.points[name] = [s[0], s[1]];
  }

  function selectJoint(name) { if (M) { M.selected = name; drawModel(); } }
  // Set joint J's field to `v` (field units) through the control's own path; false if not live.
  function setJointField(J, b, v) {
    if (!b || !liveOk(b.cs) || !Number.isFinite(v)) return false;
    setField(b.fs, v); J.active = b; trigger(b.cs); drawModel();
    return true;
  }
  function nudgeJoint(J, n) {
    const b = editField(J); if (!b) return;
    setJointField(J, b, Number(b.fs.num.value) + n * (b.fs.f.max - b.fs.f.min) / 100);
  }
  function renderModel() {
    M = S.target ? buildModel(S.target) : null;
    S.model = null;
    $("model-view").hidden = !M; $("model-empty").hidden = !!M;
    $("model-empty").textContent = S.target ? "This robot's profile has no model." : "No validated target: no model is shown.";
    const list = $("model-joints"); list.innerHTML = "";
    if (!M) { $("model-pose").textContent = ""; $("model-sel").textContent = ""; return; }
    for (const J of M.commanded) {
      const chip = h("button", { type: "button", class: "joint-chip", "data-joint": J.name, "aria-pressed": "false" }, h("span"));
      chip.addEventListener("click", () => selectJoint(J.name));
      chip.addEventListener("keydown", (e) => {
        const k = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[e.key]; if (!k) return;
        e.preventDefault(); M.selected = J.name; nudgeJoint(J, k * (e.shiftKey ? 5 : 1));
      });
      M.chips.set(J.name, chip); list.append(chip);
      for (const b of J.command) {                               // the field changed last is the one shown
        const x = bound(b); if (x) x.fs.listeners.push(() => { J.active = x; scheduleModelDraw(); });
      }
    }
    if (M.fb) subscribeFeedback(M.fb.topic, M.fb.type);
    drawModel();
  }
  (function wireModelCanvas() {
    const cv = $("model-canvas");
    const xy = (ev) => { const r = cv.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; };
    // A press on a joint picks the selected joint if it is under the pointer, else the nearest;
    // a click without moving on joints drawn on top of each other picks the next of them.
    cv.addEventListener("pointerdown", (ev) => {
      if (!M) return;
      const [x, y] = xy(ev);
      const near = [...M.pts].map(([name, s]) => [Math.hypot(s[0] - x, s[1] - y), name]).filter(([d]) => d < 14)
        .sort((a, b) => a[0] - b[0]).map(([, name]) => name);
      try { cv.setPointerCapture(ev.pointerId); } catch (e) {}
      ev.preventDefault(); cv.focus();
      if (near.length) {
        const prev = near.includes(M.selected) ? M.selected : null, J = M.byName.get(prev || near[0]), b = editField(J);
        M.selected = J.name; M.drag = { J, b, x, y, v0: b ? Number(b.fs.num.value) : NaN, near, prev, moved: false };
        drawModel();
      } else M.drag = { x, y, yaw: M.yaw, pitch: M.pitch };
    });
    cv.addEventListener("pointermove", (ev) => {
      const d = M && M.drag; if (!d) return;
      const [x, y] = xy(ev);
      if (Math.hypot(x - d.x, y - d.y) > 3) d.moved = true;
      if (!d.J) { M.yaw = d.yaw - (x - d.x) * 0.01; M.pitch = Math.max(-1.4, Math.min(1.4, d.pitch + (y - d.y) * 0.01)); drawModel(); }
      else if (d.b && d.moved) setJointField(d.J, d.b, d.v0 + ((x - d.x) - (y - d.y)) * (d.b.fs.f.max - d.b.fs.f.min) / 300);
    });
    cv.addEventListener("pointerup", () => {
      const d = M && M.drag; if (!d) return;
      M.drag = null;
      if (d.J && d.prev && !d.moved && d.near.length > 1) selectJoint(d.near[(d.near.indexOf(d.prev) + 1) % d.near.length]);
    });
    const end = () => { if (M) M.drag = null; };
    cv.addEventListener("pointercancel", end); cv.addEventListener("lostpointercapture", end);
    cv.addEventListener("keydown", (e) => {
      const k = M && { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[e.key]; if (!k) return;
      e.preventDefault();
      const J = M.selected && M.byName.get(M.selected);
      if (J) nudgeJoint(J, k * (e.shiftKey ? 5 : 1));
      else { if (e.key === "ArrowLeft" || e.key === "ArrowRight") M.yaw -= k * 0.15; else M.pitch = Math.max(-1.4, Math.min(1.4, M.pitch + k * 0.15)); drawModel(); }
    });
    if (window.ResizeObserver) new ResizeObserver(() => drawModel()).observe(cv);
  })();

  // ------------------------------------------------------------------ target lifecycle
  async function connectAndSelect(robotId, ns) {
    S.target = null; renderModel();
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
      return;
    }
    const t = S.target;
    await checkSolePublishers(t, conn);
    if (S.conn !== conn || S.target !== t) return;
    $("target").textContent = t.label; $("target-pill").dataset.state = "ok";
    $("robot").value = t.profile.id; $("namespace").value = t.namespace; syncNsField(); $("namespace").value = t.namespace;
    setValidation("ok", `${t.label} validated`,
      `${t.profile.name} · ${t.profile.dialect === "ros2" ? "ROS 2" : "ROS 1"} · typed interface matches the packaged profile`, "");
    for (const c of t.profile.cameras) addCamera(t.wire(c.topic), c.type, c.stale_after_s, c.encodings, true, !(t.wire(c.topic) in S.graph.topics));
    renderControls(); updateLock();
  }

  // A publish control its profile marks `sole_publisher` is incompatible with any other node
  // publishing its topic, whose messages would re-command the robot (e.g. the myCobot 280 boot's
  // slider GUI). rosapi names the publishers when the target validates (a read, nothing is sent
  // to the robot); rosbridge's own node is ROS infrastructure and is ignored.
  async function checkSolePublishers(t, conn) {
    S.conflicts = new Map();
    for (const c of t.profile.controls) {
      if (!c.sole_publisher || c.kind !== "publish") continue;
      const topic = t.wire(c.name);
      try {
        const pubs = ((await conn.call("/rosapi/publishers", { topic })).publishers || []).filter((n) => !INFRA.nodes.includes(n));
        if (pubs.length) S.conflicts.set(c.id, `${topic} is also published by ${pubs.join(", ")}, whose messages would re-command the robot (incompatible prerequisite; see the profile)`);
      } catch (e) { S.conflicts.set(c.id, `cannot check which nodes publish ${topic}: ${e.message}`); }
    }
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
