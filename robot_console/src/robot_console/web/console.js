// Robot console page: rosbridge transport, typed discovery/validation (the same algorithm and
// the same packaged profiles as robot_console/discovery.py), live cameras, bounded controls
// whose changes are sent at once and, for a teleoperable profile, teleop.sh's hold-to-move base
// motion with its stops. Plain browser JavaScript, no external dependencies.
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
  // [topics, services] behind a ROS 2 action `name` of `pkg/action/X` (dialect.ros2_action_endpoints).
  function ros2ActionEndpoints(name, type) {
    const [pkg, base] = normType(type).split("/");
    return [
      { [name + "/_action/feedback"]: `${pkg}/action/${base}_FeedbackMessage`, [name + "/_action/status"]: "action_msgs/msg/GoalStatusArray" },
      { [name + "/_action/send_goal"]: `${pkg}/action/${base}_SendGoal`, [name + "/_action/cancel_goal"]: "action_msgs/srv/CancelGoal",
        [name + "/_action/get_result"]: `${pkg}/action/${base}_GetResult` },
    ];
  }
  // `/x/_action/send_goal` of `pkg/action/X_SendGoal` -> [`/x`, `pkg/action/X`] (dialect.ros2_action_from_send_goal).
  function ros2ActionFromSendGoal(service, type) {
    const suffix = "/_action/send_goal";
    const m = /^([^/]+)\/action\/(.+)_SendGoal$/.exec(String(type || "").trim());
    return service.endsWith(suffix) && m ? [service.slice(0, -suffix.length), `${m[1]}/action/${m[2]}`] : null;
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
    get bytesIn() { return this._bytesIn || 0; }
    set bytesIn(v) { this._bytesIn = v; }
    _closed() {
      if (this.closed) return;
      this.closed = true;
      for (const [, p] of this.pending) p.reject(new Error("connection lost"));
      this.pending.clear();
      if (!this.userClosed && this.onlost) this.onlost();
    }
    _message(data) {
      // A message with a NaN or infinite float (a LaserScan's out-of-range returns, say) arrives from
      // Python's json as bare NaN/Infinity, which JSON does not allow: read those as null (no value).
      let m; try { m = JSON.parse(data); } catch (e) {
        try { m = JSON.parse(nonFiniteToNull(String(data))); } catch (e2) { return; }
      }
      this.bytesIn += data.length;                       // the link's incoming load (shown on the page)
      if (m.op === "publish") { const cbs = this.subs.get(m.topic); if (cbs) for (const [, cb] of cbs) { try { cb(m.msg || {}, data.length); } catch (e) { console.error(e); } } }
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
    // `onLate(values)`: an answer that comes after the timeout is still handed there (not to the promise).
    call(service, args = {}, timeoutMs = 5000, onLate = null) {
      const id = "call:" + (++this.ids);
      return new Promise((resolve, reject) => {
        const t = setTimeout(() => {
          if (!this.pending.delete(id)) return;
          if (onLate) this.pending.set(id, { resolve: onLate, reject: () => {} });
          reject(new Error(`${service} did not answer within ${timeoutMs / 1000} s`));
        }, timeoutMs);
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

  // Python's json writes a non-finite float as the bare token NaN, Infinity or -Infinity, which JSON does not
  // allow: replace those tokens with null, outside strings only (string contents are copied untouched).
  function nonFiniteToNull(text) {
    let out = "", i = 0;
    const n = text.length;
    while (i < n) {
      const c = text[i];
      if (c === '"') {                                   // copy a string literal, escapes included
        let j = i + 1;
        while (j < n && text[j] !== '"') j += text[j] === "\\" ? 2 : 1;
        out += text.slice(i, j + 1); i = j + 1; continue;
      }
      const m = /^-?(?:NaN|Infinity)/.exec(text.slice(i, i + 9));
      if (m && !/[\w.]/.test(text[i - 1] || "")) { out += "null"; i += m[0].length; continue; }
      out += c; i++;
    }
    return out;
  }
  RC.nonFiniteToNull = nonFiniteToNull;

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
    } else if (dialect === "ros2") {
      if (services_list.includes("/rosapi/action_servers")) {
        const ar = await conn.call("/rosapi/action_servers");
        for (const a of ar.action_servers || []) actions[a] = null;
      }
      for (const [s, t] of Object.entries(services)) {        // a rosapi that does list hidden services
        const hit = ros2ActionFromSendGoal(s, t);
        if (hit) actions[hit[0]] = hit[1];
      }
      for (const [a, t] of Object.entries(actions)) {
        const [tp, sv] = ros2ActionEndpoints(a, t || "x/action/X");
        [...Object.keys(tp), ...Object.keys(sv)].forEach((x) => parts.add(x));
      }
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
  // A typed number: "." or "," as the decimal separator, "−" as a minus (the page writes "." and
  // shows "−"), so the value does not depend on the browser's locale.
  function parseNum(raw) {
    const s = String(raw == null ? "" : raw).trim().replace(/^[−–]/, "-").replace(",", ".");
    return s === "" ? NaN : Number(s);
  }
  function coerce(f, raw) {
    if (f.choices) { if (!f.choices.includes(raw)) throw new Error(`${f.name}: ${raw} is not one of ${f.choices.join(", ")}`); return raw; }
    const v = parseNum(raw);
    if (!Number.isFinite(v)) throw new Error(`${f.label}: not a number`);
    if (f.min != null && v < f.min) throw new Error(`${f.label}: ${v} below documented minimum ${f.min}`);
    if (f.max != null && v > f.max) throw new Error(`${f.label}: ${v} above documented maximum ${f.max}`);
    if (f.integer && !Number.isInteger(v)) throw new Error(`${f.label}: must be an integer`);
    return v;
  }
  // Python's round(): halves go to the even neighbour (the vendor code the profiles cite rounds so).
  function roundHalfEven(x) { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 ? r - 1 : r; }
  // A field's documented mapping onto the wire (profiles.py WireMap, the inverse of a read):
  // offset + scale * value, the scaled value rounded to an integer when `integer`; a value that
  // maps outside the documented wire range is invalid and never sent.
  function toWire(f, v) {
    const w = f.to_wire; if (!w) return v;
    const x = Number(w.offset || 0) + (w.integer ? roundHalfEven(Number(w.scale) * v) : Number(w.scale) * v);
    if ((w.min != null && x < w.min) || (w.max != null && x > w.max)) throw new Error(`${f.label}: ${v} ${f.unit} maps to ${x}, outside ${w.min}..${w.max} on the wire`);
    return x;
  }

  // ------------------------------------------------------------------ cameras
  const UNTIED_STALE_S = 2.0;
  // Bytes per pixel of the supported raw encodings (4:2:2 packed: 2), as camera.py.
  const BPP = { rgb8: 3, bgr8: 3, "8uc3": 3, rgba8: 4, bgra8: 4, mono8: 1, "8uc1": 1, mono16: 2, "16uc1": 2,
    yuv422: 2, uyvy: 2, yuv422_yuy2: 2, yuyv: 2 };
  const YUV422 = ["yuv422", "uyvy", "yuv422_yuy2", "yuyv"];
  // The depth false-colour map (stops and colour of "no depth" as camera.py's DEPTH_STOPS): 256 RGB
  // rows, near (warm) first. A depth camera's 16-bit value times its unit_m is metres, scaled to its
  // range_m; 0 is no depth and is black.
  const DEPTH_STOPS = [[0, [213, 32, 39]], [0.17, [244, 125, 32]], [0.34, [250, 224, 50]], [0.5, [80, 200, 100]],
    [0.67, [40, 180, 220]], [0.84, [50, 90, 210]], [1, [70, 40, 140]]];
  const DEPTH_LUT = (() => {
    const lut = [];
    for (let i = 0; i < 256; i++) {
      const x = i / 255;
      let k = 1;
      while (k < DEPTH_STOPS.length - 1 && x > DEPTH_STOPS[k][0]) k++;
      const [p0, c0] = DEPTH_STOPS[k - 1], [p1, c1] = DEPTH_STOPS[k];
      const f = (x - p0) / (p1 - p0);
      lut.push(c0.map((v, j) => Math.round(v + (c1[j] - v) * f)));
    }
    return lut;
  })();
  const depthColor = (v, unit, range) => {
    if (v === 0) return [0, 0, 0];
    const t = Math.min(1, Math.max(0, (v * unit - range[0]) / (range[1] - range[0])));
    return DEPTH_LUT[Math.floor(t * 255 + 0.5)];
  };
  // `depth` = { unit, range } of a depth camera: its 16-bit frames are drawn as that map. `ir`: an
  // infrared camera's single-channel frame is drawn in grey stretched from its own minimum to its own
  // maximum (rounded; all black for a flat frame); another 16-bit frame is stretched from 0 to its
  // maximum, as camera.py draws them.
  function decodeImage(msg, allowed, depth, ir) {
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
    const wide = enc === "mono16" || enc === "16uc1", mono = wide || enc === "mono8" || enc === "8uc1";
    const sample = (row, x) => {
      if (!wide) return at(row + x);
      const a = at(row + 2 * x), b = at(row + 2 * x + 1);
      return msg.is_bigendian ? (a << 8) | b : (b << 8) | a;
    };
    let grey = null;                                  // value -> grey level, for a stretched frame
    if (mono && !depth && (ir || wide)) {
      let lo = Infinity, hi = -Infinity;
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const v = sample(y * step, x); if (v < lo) lo = v; if (v > hi) hi = v; }
      grey = ir ? (v) => (hi > lo ? Math.round((v - lo) * 255 / (hi - lo)) : 0)
        : (v) => Math.min(255, Math.floor(v * (255 / (hi || 1))));
    }
    for (let y = 0; y < h; y++) {
      const row = y * step;
      for (let x = 0; x < w; x++) {
        const o = (y * w + x) * 4;
        switch (enc) {
          case "rgb8": case "8uc3": put(o, at(row + 3 * x), at(row + 3 * x + 1), at(row + 3 * x + 2)); break;
          case "bgr8": put(o, at(row + 3 * x + 2), at(row + 3 * x + 1), at(row + 3 * x)); break;
          case "rgba8": put(o, at(row + 4 * x), at(row + 4 * x + 1), at(row + 4 * x + 2)); break;
          case "bgra8": put(o, at(row + 4 * x + 2), at(row + 4 * x + 1), at(row + 4 * x)); break;
          case "mono8": case "8uc1": case "mono16": case "16uc1": {
            const v = sample(row, x);
            if (depth && wide) { const c = depthColor(v, depth.unit, depth.range); put(o, c[0], c[1], c[2]); break; }
            const g = grey ? grey(v) : v; put(o, g, g, g); break;
          }
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
    lost: false, streams: [], sensors: [], controls: new Map(), feedback: new Map(), reads: new Map(),
    padGroup: null, model: null, groups: [], poll: null, mo: null, changing: false, clock: null, camPause: null });

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
  const degText = (rad) => { const d = Math.round(rad * 180 / Math.PI); return (d < 0 ? "−" : "") + Math.abs(d) + "°"; };
  // A value with its unit, and degrees next to radians: "+0.82 rad (47°)".
  const valueText = (v, unit, d) => `${signed(v, d)} ${unit}` + (unit === "rad" ? ` (${degText(v)})` : "");
  // The resolution a field is entered, stepped and copied at. A joint angle keeps 0.01 rad over its
  // whole range (an arm's base or wrist joint spans more than 5 rad), so a measured position copied
  // into the target stays within 0.005 rad of the measurement.
  function decimals(f) {
    if (f.integer) return 0;
    const span = f.max - f.min;
    return span <= 5 || f.unit === "rad" ? 2 : span <= 50 ? 1 : 0;
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
    S.camPause = null; $("congestion").hidden = true; $("congestion-drive").hidden = true;
    clearSensors();
  }
  // The colour legend of a depth camera: the map near to far over the camera's range, 0 black.
  function depthLegend(depth) {
    const bar = h("canvas", { class: "depth-bar", width: 256, height: 1, "data-testid": "depth-bar", "aria-hidden": "true" });
    const g = bar.getContext("2d");
    DEPTH_LUT.forEach((c, i) => { g.fillStyle = `rgb(${c[0]}, ${c[1]}, ${c[2]})`; g.fillRect(i, 0, 1, 1); });
    const m = (v) => `${Number(v).toFixed(2)} m`;
    return h("div", { class: "depth-legend", "data-testid": "depth-legend" },
      h("span", { class: "depth-swatch", title: "no depth" }),
      h("span", { class: "depth-end", "data-testid": "depth-near", text: m(depth.range[0]) }),
      bar,
      h("span", { class: "depth-end", "data-testid": "depth-far", text: m(depth.range[1]) }),
      h("span", { class: "depth-note", text: "near to far · black = no depth" }));
  }
  // An infrared camera's note: how its grey levels are drawn.
  function irLegend() {
    return h("div", { class: "depth-legend ir-legend", "data-testid": "ir-legend" },
      h("span", { class: "ir-bar", "aria-hidden": "true" }),
      h("span", { class: "depth-note", text: "infrared · each frame stretched from its darkest (black) to its brightest (white) value" }));
  }
  // Console policy (camera.py): rosbridge throttle_rate of a stream not tied to a target, and of a
  // profile camera that records no max_rate_hz.
  const UNTIED_THROTTLE_MS = 500, DEFAULT_THROTTLE_MS = 100;
  // Link health (console spec §2.3; camera.py): a camera is live only while the image it shows is at most LAG_LIMIT_S
  // old in robot time, judged again on every tick; a frame that arrived already older is LAGGING. A frame stamped
  // more than STEP_TOLERANCE_S after its own arrival means the robot's clock stepped.
  const LAG_LIMIT_S = 1.0, STEP_TOLERANCE_S = 0.25;
  // A shown image is live while its age is at most LAG_LIMIT_S + LIVE_PERIODS frame periods at the stream's cap
  // (camera.live_bound_s): a steady lag below LAG_LIMIT_S never flickers between frames. A decoded frame is held until
  // a time answer has read the robot's clock after it was taken (clockConfirms), and its age is bounded with every
  // answer from CLOCK_STALE_S before it arrived until now (clockSpan), so a robot clock step never makes an image
  // older than the bound read live (console spec §2.3 names the one case left).
  const LIVE_PERIODS = 1.5;
  // The cards are judged every TICK_MS: an image reads LIVE only if it will still be within its bound at the next tick.
  const TICK_MS = 200;
  const liveBound = (s) => LAG_LIMIT_S + LIVE_PERIODS * s.throttle / 1000;
  // The clock reference, as camera.ClockRef: /rosapi/get_time every CLOCK_SYNC_MS (or at once when a frame's lag
  // must be checked); a call sent at console time t0 and answered at t1 with robot time T bounds the offset
  // (robot - console) to [T - t1, T - t0]; the estimate is the intersection of the recent samples' bounds, and a
  // sample whose bounds exclude the estimate's midpoint is a clock step (the older samples are dropped). Console
  // time is monotonic, so a step of the console's wall clock changes nothing. On ROS 1, /use_sim_time is read
  // first (again each period until known). No usable reference: the lag is unknown, never zero.
  const CLOCK_SYNC_MS = 2000, CLOCK_STALE_S = 10, CLOCK_WINDOW_S = 300, CLOCK_ANSWER_MS = 3000;
  // Congestion: the rules R1-R3 of camera.py, applied the same way (see congestion() below).
  const CONGESTED_RTT_S = 1.0, CONGESTED_WINDOW_S = 5;
  // A steady lag (spread <= DOMAIN_SPREAD_S over DOMAIN_SPAN_S, >= 5 frames, beyond DOMAIN_MIN_S, while the newest time
  // answer took at most CONGESTED_RTT_S and less than half that lag) is either a real delay or stamps on another
  // clock: it stays LAGGING, with that note, and R1 skips it. DOMAIN_RTT_S is R1's round trip.
  const DOMAIN_SPREAD_S = 0.25, DOMAIN_SPAN_S = 3, DOMAIN_MIN_S = 2, DOMAIN_RTT_S = 0.3;
  const monoS = () => (performance.timeOrigin + performance.now()) / 1000;
  function timeS(st) {
    if (!st) return null;
    const sec = st.secs != null ? st.secs : st.sec, ns = st.nsecs != null ? st.nsecs : (st.nanosec || 0);
    return typeof sec === "number" && (sec || ns) ? sec + ns * 1e-9 : null;
  }
  function makeClock(conn, dialect) {
    const c = { conn, dialect, samples: [], answers: [], lo: null, hi: null, lastOk: null, lastT0: -Infinity, steps: 0, resets: 0, lastRtt: null,
      lastRttAt: null, lastRttT0: null, lastTimeout: null, lastTimeoutT0: null, grace: [-Infinity, -Infinity],
      sentAt: null, inflight: false, sim: dialect === "ros1" ? null : false,
      reason: "measuring the robot's clock", timer: null };
    c.timer = setInterval(() => clockSample(c), CLOCK_SYNC_MS);
    clockSample(c);
    return c;
  }
  function stopClock(c) { if (c) clearInterval(c.timer); }
  async function clockSample(c) {
    if (c.inflight || !c.conn.open) return;
    c.inflight = true; c.sentAt = monoS();
    try {
      if (c.sim == null) {                                 // a ROS 1 robot on simulated time has no wall reference
        try {
          const v = await c.conn.call("/rosapi/get_param", { name: "/use_sim_time", default: "false" }, CLOCK_ANSWER_MS);
          c.sim = String(v.value).trim().toLowerCase() === "true";
        } catch (e) {
          // a question given up on counts for congestion (R3) as camera.ClockRef counts it, while the lag stays unknown
          if (/did not answer within/.test(e.message)) { c.lastTimeout = monoS(); c.lastTimeoutT0 = c.sentAt; }
          c.reason = `checking whether the robot runs on simulated time (/use_sim_time: ${e.message})`; return;
        }
        c.sentAt = monoS();                                // the time question below is a new one
      }
      if (c.sim) { c.reason = "the robot runs on simulated time (/use_sim_time)"; c.lastOk = null; }
      const t0 = monoS();
      // a late answer (after CLOCK_ANSWER_MS) still gives its round trip (R2), as camera.ClockRef keeps it
      const late = () => { const t1 = monoS(); c.lastRtt = t1 - t0; c.lastRttAt = t1; c.lastRttT0 = t0; };
      const v = await c.conn.call("/rosapi/get_time", {}, CLOCK_ANSWER_MS, late);
      const t1 = monoS(), T = timeS(v.time);
      c.lastRtt = t1 - t0; c.lastRttAt = t1; c.lastRttT0 = t0;
      if (c.sim) return;                                   // simulated time: only the round trip counts (R1-R3)
      if (T == null) { c.reason = "the robot's clock reads 0 (simulated time not started)"; return; }
      const lo = T - t1, hi = T - t0, mid = c.lo == null ? null : (c.lo + c.hi) / 2;
      let keep = c.samples.filter((x) => t1 - x.t <= CLOCK_WINDOW_S);
      let ilo = Math.max(lo, ...keep.map((x) => x.lo)), ihi = Math.min(hi, ...keep.map((x) => x.hi));
      // bounds excluding every recent sample prove a step (counted); excluding only the midpoint may hide one: reset
      if (ilo > ihi || (mid != null && !(lo <= mid && mid <= hi))) { if (ilo > ihi) c.steps++; c.resets++; keep = []; ilo = lo; ihi = hi; }
      keep.push({ t: t1, lo, hi });
      c.answers = c.answers.filter((x) => t1 - x.t <= CLOCK_WINDOW_S).concat([{ t: t1, lo, hi, T }]);
      Object.assign(c, { samples: keep, lo: ilo, hi: ihi, lastOk: t1, lastT0: t0 });
      for (const x of S.streams) promote(x);              // frames waiting for this answer are shown now
    } catch (e) {
      if (/did not answer within/.test(e.message)) { c.lastTimeout = monoS(); c.lastTimeoutT0 = c.sentAt; }
      c.reason = "no clock reference: /rosapi/get_time did not answer (" + e.message + ")";
    } finally { c.inflight = false; c.sentAt = null; }
  }
  // Offset bounds [lo, hi] of robot - console, or a reason there are none.
  function clockOffset(c) {
    if (!c) return [null, "no clock reference"];
    if (c.sim == null) return [null, c.reason.startsWith("checking") ? c.reason : "checking whether the robot runs on simulated time (/use_sim_time)"];
    if (c.sim) return [null, "the robot runs on simulated time (/use_sim_time)"];
    if (c.lastOk == null) return [null, c.reason];
    if (monoS() - c.lastOk > CLOCK_STALE_S) return [null, `the clock reference is older than ${CLOCK_STALE_S} s (${c.reason})`];
    return [[c.lo, c.hi], ""];
  }
  // camera.ClockRef.confirms: an answer read the robot's clock after the frame was taken (its question sent after the
  // frame arrived, or the time it read not before the frame's stamp).
  function clockConfirms(c, arrival, stamp) {
    return c.lastT0 >= arrival || c.answers.some((x) => x.t >= arrival - CLOCK_STALE_S && x.T >= stamp);
  }
  // camera.ClockRef.span: the lowest and highest offset of every answer from CLOCK_STALE_S before `arrival` until now.
  function clockSpan(c, arrival) {
    let got = c.answers.filter((x) => x.t >= arrival - CLOCK_STALE_S);
    if (!got.length) got = c.answers.slice(-1);
    return [Math.min(...got.map((x) => x.lo)), Math.max(...got.map((x) => x.hi))];
  }
  // ["known", arrivalLo, arrivalHi, ageLo, ageHi]: bounds of a shown frame's lag when it arrived and of its age now,
  // both in robot time (camera.frame_lag); or ["unknown", reason].
  function frameLag(c, arrival, stamp, nowS) {
    if (stamp == null) return ["unknown", "the frames carry no header stamp"];
    const [off, why] = clockOffset(c);
    if (!off) return ["unknown", why];
    if (!clockConfirms(c, arrival, stamp)) return ["unknown", "this frame arrived before the robot's clock could be checked"];
    const [olo, ohi] = clockSpan(c, arrival);
    const lo = arrival + olo - stamp, hi = arrival + ohi - stamp;
    if (hi < -STEP_TOLERANCE_S) return ["unknown", "the frame is stamped after its arrival: the robot's clock is inconsistent"];
    return ["known", lo, hi, nowS + olo - stamp, nowS + ohi - stamp];
  }
  // The steady lag of a stream's frames (a real delay or stamps on another clock), when they have sat at one offset
  // beyond DOMAIN_MIN_S (or before their arrival) while the newest time answer took at most CONGESTED_RTT_S and less
  // than half that lag (an unknown round trip does not count); else null (camera.CameraStream.steady_offset).
  function domainOffset(s) {
    const c = S.clock;
    if (!c || c.lastRtt == null || c.lastRtt > CONGESTED_RTT_S || s.lags.length < 5) return null;
    const last = s.lags[s.lags.length - 1][0], recent = s.lags.filter(([a]) => last - a <= DOMAIN_SPAN_S);
    if (recent.length < 5 || last - recent[0][0] < DOMAIN_SPAN_S * 0.8) return null;
    const v = recent.map(([, x]) => x), mean = v.reduce((a, x) => a + x, 0) / v.length;
    if (Math.max(...v) - Math.min(...v) > DOMAIN_SPREAD_S || !(mean > DOMAIN_MIN_S || mean < -STEP_TOLERANCE_S)) return null;
    if (c.lastRtt >= Math.abs(mean) / 2) return null;     // the answers wait about as long: a queue they share
    return mean;
  }
  const lagSpan = (s) => s.lags.length >= 5 ? s.lags[s.lags.length - 1][0] - s.lags[0][0] : 0;
  RC.clock = () => S.clock;
  function stampS(m) { return timeS(m && m.header && m.header.stamp); }
  // `o.kind`: color | depth | ir; `o.depth` = { unit, range } of a depth camera; `o.info`: its CameraInfo
  // topic; `o.note`: shown when the (optional) camera is not on the wire, e.g. which launch starts it;
  // `o.throttle`: the most frames rosbridge sends (ms between frames); `o.onDemand`: subscribed only
  // while the user has started it; `o.silent`: why an advertised stream may never deliver a frame.
  // A sensor_msgs/CompressedImage camera (`encodings`: its formats, jpeg or png) is decoded by the browser.
  function addCamera(topic, type, staleS, encodings, tied, missing, optional, o = {}) {
    const depth = o.depth || null, kind = depth ? "depth" : o.kind || "color", info = o.info, note = o.note;
    const throttle = o.throttle || DEFAULT_THROTTLE_MS, compressed = normType(type) === "sensor_msgs/CompressedImage";
    const toggle = o.onDemand && !missing ? h("button", { type: "button", class: "btn cam-toggle", "data-testid": "camera-toggle",
      text: tied ? "Start stream" : "Preview" }) : null;
    const canvas = h("canvas", { width: 640, height: 480 });
    const fig = h("figure", { class: "cam notlive", "data-testid": "camera", "data-topic": topic, "data-state": "waiting",
        "data-kind": kind, "data-compressed": compressed ? "true" : null },
      h("div", { class: "cam-view" }, canvas,
        h("div", { class: "cam-overlay" }, h("div", { class: "ov-title" }), h("div", { class: "ov-detail" })),
        h("span", { class: "cam-badge state", "data-testid": "camera-state", role: "status" })),
      depth ? depthLegend(depth) : kind === "ir" ? irLegend() : null,
      o.caption ? h("p", { class: "cam-caption-note", "data-testid": "camera-caption", text: o.caption }) : null,
      h("figcaption", { class: "cam-caption" },
        h("code", { text: topic }),
        kind !== "color" ? h("span", { class: "cam-kind", text: kind === "ir" ? "infrared" : kind }) : null,
        compressed ? h("span", { class: "cam-kind", text: "compressed" }) : null,
        tied ? null : h("span", { class: "untied", "data-testid": "untied", text: "not tied to a target" }),
        info ? h("span", { class: "cam-info", "data-testid": "camera-info", title: "the stream's sensor_msgs/CameraInfo topic", text: "info " + info }) : null,
        h("span", { class: "cam-rate", "data-testid": "camera-rate", text: `≤ ${+(1000 / throttle).toFixed(1)} fps` }),
        toggle,
        h("span", { class: "cam-meta" })));
    $("cameras").appendChild(fig);
    const n = $("cameras").children.length;
    $("cameras").dataset.count = n > 3 ? "many" : String(n);
    const s = { topic, type, staleS, encodings, tied, depth, ir: kind === "ir", compressed, note: note || "", silent: o.silent || "",
      onDemand: !!o.onDemand, throttle, lastStamp: null, lastArrival: null, lags: [], pending: null, decodeQueued: false,
      div: fig, canvas, last: null, frames: 0, error: null,
      unsupported: false, missing, optional: !!optional, started: performance.now(), times: [], w: 0, h: 0, enc: "" };
    if (!missing && !s.onDemand) streamOn(s);
    if (toggle) toggle.addEventListener("click", () => {
      if (s.sid) { streamOff(s); return; }
      if (S.camPause) { s.refused = "Not started: the cameras are paused (the link is congested); resume them first."; return; }
      // a preview not tied to a target: one at a time (no profile says what it costs the link)
      if (!s.tied) for (const x of S.streams) if (x !== s && !x.tied && x.sid) streamOff(x);
      s.refused = budgetRefusal(s);
      if (!s.refused) streamOn(s);
    });
    S.streams.push(s);
    return s;
  }
  // The link budget (console spec §2.3): a stream's load is its message size (measured, or before its first frame
  // a 640x480 frame of its first encoding as base64 JSON; a compressed frame taken as a tenth of rgb8) times its rate
  // cap. Starting an on-demand stream that would take the cameras over the profile's link_budget_mbps is refused,
  // with the numbers; so is one while a point-cloud sample is in flight.
  const NOMINAL_PX = 640 * 480;
  function streamMbps(s) {
    const enc = String((s.encodings && s.encodings[0]) || "rgb8").toLowerCase();
    const raw = s.compressed ? NOMINAL_PX * 3 / 10 : NOMINAL_PX * (BPP[enc] || 3);
    const bytes = s.lastBytes || Math.ceil(raw * 4 / 3) + 300;
    return bytes * 8 * (1000 / s.throttle) / 1e6;
  }
  const camsMbps = (except) => S.streams.filter((x) => x !== except && x.sid).reduce((a, x) => a + streamMbps(x), 0);
  function budgetRefusal(s) {
    const budget = S.target && S.target.profile.link_budget_mbps;
    if (!budget) return "";
    if (S.sensors.some((x) => x.sampling)) return "Not started: a point-cloud sample is being received; start the stream after it.";
    const used = camsMbps(s), add = streamMbps(s);
    if (used + add <= budget) return "";
    return `Not started: with it the cameras would take about ${(used + add).toFixed(0)} Mbit/s, over this robot's link budget of `
      + `${budget} Mbit/s (${used.toFixed(0)} in use, this stream about ${add.toFixed(0)}). Stop another stream first.`;
  }
  // Subscribe a stream at its rate cap; an on-demand one only while its user keeps it started.
  function streamOn(s) {
    if (!S.conn || !S.conn.open || S.lost) return;
    s.sid = S.conn.subscribe(s.topic, s.type, (m, size) => onFrame(s, m, size), s.throttle);
    s.started = performance.now(); s.error = null; s.unsupported = false; s.refused = ""; s.lags = []; s.held = null; s.arrivedAt = null; s.recheck = null;
    s.last = null; s.lastArrival = null; s.lastStamp = null; s.times = [];      // a fresh start: no earlier frame is current
    const b = s.div.querySelector(".cam-toggle"); if (b) b.textContent = s.tied ? "Stop stream" : "Stop preview";
  }
  function streamOff(s) {
    try { if (s.sid && S.conn && S.conn.open) S.conn.unsubscribe(s.sid); } catch (e) {}
    s.sid = null;
    const b = s.div.querySelector(".cam-toggle"); if (b) b.textContent = s.tied ? "Start stream" : "Preview";
  }
  // Congestion (console spec §2.3): every camera stream is unsubscribed, the page says why, prominently, and the
  // operator resumes them. rosbridge's throttle and queue_length bound what waits in rosbridge itself, and its
  // Autobahn server holds the next message until Twisted has written the previous one, but what is written still
  // waits in Twisted's, the kernel's and the Wi-Fi's buffers: on a link slower than the streams the delay can grow,
  // and only the client can stop sending it more.
  function pauseCameras(reason) {
    if (S.camPause) return;
    S.camPause = { reason, resume: S.streams.filter((x) => x.sid) };
    for (const x of S.camPause.resume) streamOff(x);
    for (const x of S.streams) { x.held = null; x.refused = ""; }     // an earlier refusal no longer applies
    const text = `Cameras paused: the link is congested (${reason}). Driving commands and the stop share this link, so no `
      + "images are received until you resume them.";
    for (const id of ["congestion", "congestion-drive"]) {      // also where the operator drives
      const b = $(id); b.hidden = false; b.querySelector(".congestion-text").textContent = text;
    }
  }
  function resumeCameras() {
    const p = S.camPause;
    if (!p) return;
    S.camPause = null; $("congestion").hidden = true; $("congestion-drive").hidden = true;
    if (S.clock) { S.clock.lastRtt = null; S.clock.lastTimeout = null; }
    for (const x of S.streams) x.refused = "";             // "paused" refusals are void now
    for (const x of p.resume) if (!x.missing) streamOn(x);
  }
  RC.resumeCameras = resumeCameras;
  // R1-R3 (camera.CameraSet.congestion): R1 a stream LAGGING while the newest time answer took >= DOMAIN_RTT_S, once
  // its lags span 0.8 DOMAIN_SPAN_S and show no steady offset (a steady offset is not a backlog); R2 a
  // time answer took > CONGESTED_RTT_S (within the last CONGESTED_WINDOW_S); R3 a time question unanswered for
  // CLOCK_ANSWER_MS, now or given up within the last CLOCK_ANSWER_MS, while frames arrived within the last second.
  // A question sent, or a frame that arrived, inside the grace window of a sample the operator asked for is not
  // counted (the sample's own burst delays them).
  function congestion(nowS) {
    const c = S.clock;
    if (!c || S.camPause || S.lost) return "";
    const graced = (t) => t != null && c.grace[0] <= t && t <= c.grace[1];
    const rtt = graced(c.lastRttT0) ? null : c.lastRtt;
    for (const x of S.streams) {
      if (x.sid && x.state === "lagging" && rtt != null && rtt >= DOMAIN_RTT_S && !graced(x.lastArrival)
          && lagSpan(x) >= DOMAIN_SPAN_S * 0.8 && domainOffset(x) == null)
        return `frames on ${x.topic} arrive late while the robot's time answer took ${rtt.toFixed(2)} s (at least ${DOMAIN_RTT_S} s): ${x.detail || "lagging"}`;
    }
    if (rtt != null && rtt > CONGESTED_RTT_S && nowS - c.lastRttAt <= CONGESTED_WINDOW_S) return `the robot's time answer took ${rtt.toFixed(1)} s`;
    const arriving = S.streams.some((x) => x.sid && x.arrivedAt != null && nowS - x.arrivedAt < 1);   // shown yet or not
    const unanswered = c.sentAt != null && nowS - c.sentAt > CLOCK_ANSWER_MS / 1000 && !graced(c.sentAt);
    const gaveUp = c.lastTimeout != null && nowS - c.lastTimeout <= CLOCK_ANSWER_MS / 1000 && !graced(c.lastTimeoutT0);
    if (arriving && (unanswered || gaveUp)) return `the robot's time answer is over ${CLOCK_ANSWER_MS / 1000} s late while frames keep arriving`;
    return "";
  }

  // A frame is only stored when it arrives; it is decoded in a later task, the latest one only, so input
  // events (a released key, Space, the Stop button) are handled between frames and never wait behind a backlog.
  function onFrame(s, m, size) {
    s.pending = m; s.pendingArrival = s.arrivedAt = monoS(); if (size) s.lastBytes = size;
    if (!s.decodeQueued) { s.decodeQueued = true; setTimeout(() => { s.decodeQueued = false; decodePending(s); }, 0); }
  }
  function shown(s, w, hgt, enc) {
    const now = performance.now();
    s.last = now; s.frames++; s.error = null; s.unsupported = false;
    s.w = w; s.h = hgt; s.enc = enc;
    s.times.push(now); while (s.times.length > 2 && now - s.times[0] > 2000) s.times.shift();
  }
  // The frame's offset from the robot's clock (for the steady-lag note), while there is a reference.
  function noteLag(s) {
    const [off] = clockOffset(S.clock);
    if (off && s.lastStamp != null) {
      s.lags = s.lags.filter(([a]) => s.lastArrival - a <= 2 * DOMAIN_SPAN_S);
      s.lags.push([s.lastArrival, s.lastArrival + (off[0] + off[1]) / 2 - s.lastStamp]);
    } else s.lags = [];
  }
  function decodePending(s) {
    const m = s.pending; s.pending = null;
    if (!m || (s.onDemand && !s.sid) || !s.sid) return;
    const stamp = stampS(m), arrival = s.pendingArrival;
    if (s.compressed) { decodeCompressed(s, m, stamp, arrival); return; }
    try {
      const img = decodeImage(m, s.encodings, s.depth, s.ir);
      s.held = { img, stamp, arrival, enc: String(m.encoding || "") };
      s.error = null; s.unsupported = false;
      promote(s);
    } catch (e) { s.error = e.message; s.unsupported = !!e.unsupported; s.last = null; s.held = null; }
  }
  // A decoded frame is drawn once confirmed (clockConfirms), or at once with no usable reference (its age is then
  // unknown); until then the clock is asked and the image on screen keeps ageing (camera.CameraStream._promote).
  function promote(s) {
    const f = s.held;
    if (!f || !s.sid) { s.held = null; return; }
    const usable = S.clock && f.stamp != null && clockOffset(S.clock)[0];
    if (usable && !clockConfirms(S.clock, f.arrival, f.stamp)) { if (!S.clock.inflight) clockSample(S.clock); return; }
    s.held = null;
    const w = f.img.width, hgt = f.img.height;
    if (s.canvas.width !== w || s.canvas.height !== hgt) {
      s.canvas.width = w; s.canvas.height = hgt;
      s.div.querySelector(".cam-view").style.setProperty("--ar", `${w} / ${hgt}`);
    }
    if (f.bmp) s.canvas.getContext("2d").drawImage(f.img, 0, 0); else s.canvas.getContext("2d").putImageData(f.img, 0, 0);
    s.lastStamp = f.stamp; s.lastArrival = f.arrival;       // the frame now on screen
    noteLag(s);
    shown(s, w, hgt, f.enc);
  }
  // sensor_msgs/CompressedImage: `format` names the codec (image_transport writes e.g. "rgb8; jpeg compressed bgr8");
  // the camera's encodings list the codecs it documents. The browser decodes it.
  function decodeCompressed(s, m, stamp, arrival) {
    const fmt = String(m.format || "").toLowerCase(), codec = ["jpeg", "png"].find((k) => fmt.includes(k));
    const allowed = (s.encodings || []).map((x) => String(x).toLowerCase());
    if (!codec || (allowed.length && !allowed.includes(codec))) {
      s.error = `unsupported format '${m.format || ""}'`; s.unsupported = true; s.last = null; return;
    }
    let bytes;
    try { bytes = Uint8Array.from(atob(String(m.data || "")), (ch) => ch.charCodeAt(0)); }
    catch (e) { s.error = "undecodable image data"; s.last = null; return; }
    // decodes may finish out of order: one that resolves for an older frame than the one shown is dropped
    const seq = s.decodeSeq = (s.decodeSeq || 0) + 1;
    createImageBitmap(new Blob([bytes], { type: `image/${codec}` })).then((bmp) => {
      if (!s.sid || seq < (s.shownSeq || 0)) return;
      s.shownSeq = seq;
      s.held = { img: bmp, bmp: true, stamp, arrival, enc: codec };
      s.error = null; s.unsupported = false;
      promote(s);
    }).catch(() => { if (seq >= (s.shownSeq || 0)) { s.error = `undecodable ${codec} image`; s.last = null; } });
  }

  // [state, badge, detail]: judged on every tick from the shown image's age now, never only when it arrived.
  function cameraState(s, now) {
    s.lagNow = null;                       // shown only while it describes the image on screen now
    if (s.missing) return ["missing", "MISSING", (s.optional ? "unavailable: this optional camera is not on the wire" : "unavailable: the topic is not on the wire") + (s.note ? ". " + s.note : "")];
    if (S.lost) return ["failed", "DISCONNECTED", "Connection lost: this image is frozen, not live. Reload the page to connect again."];
    if (S.camPause && S.camPause.resume.includes(s)) return ["paused", "PAUSED", `Paused: the link is congested (${S.camPause.reason}). Resume the cameras above.`];
    promote(s);                            // a held frame confirmed since the last tick is drawn now
    if (s.unsupported) return ["unsupported", "UNSUPPORTED", s.error];
    if (s.error) return ["failed", "FAILED", "failed: " + s.error];
    if (s.onDemand && !s.sid) {
      if (s.refused) return ["off", "OFF", s.refused];
      if (!s.tied) return ["off", "OFF", "Not tied to a target: previewed only on request, one at a time (at most 2 frames/s)."];
      return ["off", "OFF", `Off: this stream is sent only while you have it started (Start stream; at most ${+(1000 / s.throttle).toFixed(1)} frames/s), `
        + "to spare the robot's link for driving." + (s.last != null ? " The last image shown is not live." : "")];
    }
    if (s.last == null) {
      if ((now - s.started) / 1000 <= s.staleS) return ["waiting", "WAITING", "waiting for frames"];
      return s.silent ? ["silent", "NO FRAMES", `No frames yet. ${s.silent}`] : ["stale", "STALE", `stale: no frame within ${s.staleS} s`];
    }
    const since = (now - s.last) / 1000, bound = liveBound(s);
    if (since > s.staleS) return ["stale", "STALE", `stale: last frame ${since.toFixed(1)} s ago (threshold ${s.staleS} s); the image is frozen`];
    const steady = domainOffset(s);
    const note = steady != null ? ` A steady offset of ${steady >= 0 ? "+" : ""}${steady.toFixed(1)} s: either a real delay or stamps on another clock `
      + "than the robot's ROS clock (the Astra SDK stamps frames on its host clock)." : "";
    const lag = s.lagNow = frameLag(S.clock, s.lastArrival, s.lastStamp, monoS());
    if (lag[0] !== "known") {
      if (steady != null && steady > 0) return ["lagging", "LAGGING", `lagging: frames arrive about ${steady.toFixed(1)} s after their stamps; not current.${note}`];
      if (since > bound) return ["stale", "STALE", `stale: last frame ${since.toFixed(1)} s ago; its age is unknown (${lag[1]})`];
      return ["unknown", "LAG UNKNOWN", `Frames arrive, but whether they are current cannot be checked: ${lag[1]}.${note}`];
    }
    const [, alo, ahi, glo, ghi] = lag;
    if (alo <= LAG_LIMIT_S) s.recheck = null;
    else if (S.clock.lastT0 < s.lastArrival) {
      // confirmed by a reading that may predate a backward clock step: the clock is asked again, and the stream is first
      // called late once an answer asked after its first late-looking frame arrived has come (a stream already
      // LAGGING stays so meanwhile, rather than flicker)
      clockSample(S.clock);
      if (s.recheck == null) s.recheck = s.lastArrival;
      if (s.state !== "lagging" && S.clock.lastT0 < s.recheck)
        return ["unknown", "LAG UNKNOWN", `This image looks ${alo.toFixed(1)}-${ahi.toFixed(1)} s old when it arrived; re-checking the robot's clock.`];
    }
    if (alo > LAG_LIMIT_S) return ["lagging", "LAGGING", `lagging: this image was ${alo.toFixed(1)}-${ahi.toFixed(1)} s old when it arrived; not current.`
      + (note || " The link or the camera is behind.")];
    if (ghi + TICK_MS / 1000 <= bound) return ["live", "LIVE", `live: this image is at most ${ghi.toFixed(2)} s old (live while at most ${bound.toFixed(2)} s)`];
    if (glo > bound) return ["stale", "STALE", `stale: no newer frame; this image is ${glo.toFixed(1)}-${ghi.toFixed(1)} s old (live only to ${bound.toFixed(2)} s)`];
    const why = ghi - glo > 2 * STEP_TOLERANCE_S ? ` The robot's time answers since ${CLOCK_STALE_S} s before it arrived disagree by ${(ghi - glo).toFixed(1)} s: a clock step or slow answers.` : "";
    return ["unknown", "LAG UNKNOWN", `This image is ${glo.toFixed(2)}-${ghi.toFixed(2)} s old (live only to ${bound.toFixed(2)} s): whether it is current cannot be decided.${why}`];
  }
  // The meta line: size, encoding, rate, the last frame's arrival and, only while the image on screen is judged
  // against the clock (live, lagging, stale with a known age), its age now and the bound up to which it is live.
  function camMeta(s, now) {
    const parts = [];
    if (s.w) parts.push(`${s.w}×${s.h}`);
    if (s.enc) parts.push(s.enc);
    if (s.sid && s.times.length > 1 && s.last != null && now - s.last < 2000) {
      const fps = (s.times.length - 1) / ((s.times[s.times.length - 1] - s.times[0]) / 1000);
      parts.push(`${fps.toFixed(fps < 10 ? 1 : 0)} fps`);
    }
    if (s.last != null) { const a = (now - s.last) / 1000; parts.push(`last frame ${a < 1 ? Math.round(a * 1000) + " ms" : a.toFixed(1) + " s"} ago`); }
    if (s.lagNow && s.lagNow[0] === "known") {
      const l = s.lagNow;
      parts.push(`age ${Math.max(0, (l[3] + l[4]) / 2).toFixed(2)} ± ${((l[4] - l[3]) / 2).toFixed(2)} s · live to ${liveBound(s).toFixed(2)} s`);
    } else if (s.lagNow) parts.push("age unknown");
    return parts.join(" · ");
  }
  function tick() {
    const now = performance.now();
    let live = 0;
    for (const s of S.streams) {
      const [st, badge, detail] = cameraState(s, now);
      s.state = st; s.detail = detail;
      if (st === "live") live++;
      const b = s.div.querySelector(".cam-badge");
      if (b.textContent !== badge) b.textContent = badge;
      b.className = "cam-badge state " + (st === "live" ? "ok" : ["waiting", "off", "silent"].includes(st) ? "dim"
        : ["unknown", "paused"].includes(st) ? "warn" : "bad");
      s.div.querySelector(".ov-title").textContent = badge;
      s.div.querySelector(".ov-detail").textContent = detail;
      s.div.querySelector(".cam-meta").textContent = camMeta(s, now);
      s.div.dataset.state = st; s.div.classList.toggle("notlive", st !== "live");
    }
    const why = congestion(monoS());
    if (why) pauseCameras(why);
    const none = !S.streams.length;
    $("no-cameras").style.display = none ? "" : "none";
    $("no-cameras").textContent = none ? (S.target ? "No cameras for this robot's profile." : "No cameras discovered on the wire.") : "";
    $("cam-summary").textContent = none ? "" : `${live} of ${S.streams.length} live`;
    linkRate(now);
    tickSensors(now);
    tickFeedback(now);
    drawModel();
  }
  setInterval(tick, TICK_MS);
  // The link's measured incoming load over the last 2 s (all messages), and the cameras' share of the budget:
  // ok below 80 % of it, near from 80 %, over above it (warning colours).
  const linkSamples = [];
  function linkRate(now) {
    const el = $("link-rate");
    if (!S.conn) { el.textContent = ""; return; }
    linkSamples.push([now, S.conn.bytesIn]);
    while (linkSamples.length > 2 && now - linkSamples[0][0] > 2000) linkSamples.shift();
    const [t0, b0] = linkSamples[0], [t1, b1] = linkSamples[linkSamples.length - 1];
    const mbps = t1 > t0 ? (b1 - b0) * 8 / ((t1 - t0) / 1000) / 1e6 : 0;
    const budget = S.target && S.target.profile.link_budget_mbps;
    const cams = camsMbps(null), sampling = S.sensors.filter((x) => x.sampling);
    el.dataset.mbps = mbps.toFixed(3);
    el.dataset.level = !budget ? "" : Math.max(cams, mbps) > budget ? "over" : Math.max(cams, mbps) >= 0.8 * budget ? "near" : "ok";
    el.textContent = `link in ${mbps.toFixed(1)} Mbit/s` + (budget ? ` · cameras ≤ ${cams.toFixed(0)} of ${budget} Mbit/s budget` : "")
      + (sampling.length ? ` · receiving a point-cloud sample (about ${(sampling.reduce((a, x) => a + sampleBytes(x), 0) / 1e6).toFixed(0)} MB)` : "");
  }

  // ------------------------------------------------------------------ sensors (read-only)
  // The profile's `sensors` (console spec §2.5): manufacturer topics the page only subscribes to
  // and shows. Nothing here sends: a sensor has no control and no stop.
  function clearSensors() {
    for (const s of S.sensors) { clearTimeout(s.sampleTimer); try { if (s.sid && S.conn && S.conn.open) S.conn.unsubscribe(s.sid); } catch (e) {} }
    S.sensors = [];
    $("sensor-list").innerHTML = ""; $("sensor-headlines").innerHTML = ""; $("sensor-headlines").hidden = true;
    $("sensors-panel").hidden = true; $("sensor-summary").textContent = "";
  }
  const SENSOR_THROTTLE_MS = 100;
  // An array field: a JSON array, or the base64 string rosbridge may use for a byte array.
  function asList(v) {
    if (Array.isArray(v)) return v;
    if (typeof v === "string") { try { return Array.from(atob(v), (c) => c.charCodeAt(0)); } catch (e) { return null; } }
    return null;
  }
  const fin = (v) => typeof v === "number" && Number.isFinite(v);
  const fixed = (v, d) => v.toFixed(d).replace("-", "−");
  // Roll, pitch and yaw (rad) of a quaternion {x, y, z, w} (ZYX: yaw about z, then pitch, then roll);
  // null for a missing or zero quaternion (a message that carries no orientation).
  function quatRpy(q) {
    if (!q || ![q.x, q.y, q.z, q.w].every(fin)) return null;
    const n = Math.hypot(q.x, q.y, q.z, q.w);
    if (n < 1e-9) return null;
    const [x, y, z, w] = [q.x / n, q.y / n, q.z / n, q.w / n];
    return [Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
      Math.asin(Math.max(-1, Math.min(1, 2 * (w * y - z * x)))),
      Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))];
  }
  // Quaternions {x, y, z, w}: product, conjugate, and the URDF rpy rotation Rz(yaw) Ry(pitch) Rx(roll).
  const quatMul = (a, b) => ({ w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z, x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x, z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w });
  const quatConj = (q) => ({ x: -q.x, y: -q.y, z: -q.z, w: q.w });
  function quatFromRpy([r, p, y]) {
    const ax = (x, y2, z, a) => ({ x: x * Math.sin(a / 2), y: y2 * Math.sin(a / 2), z: z * Math.sin(a / 2), w: Math.cos(a / 2) });
    return quatMul(quatMul(ax(0, 0, 1, y), ax(0, 1, 0, p)), ax(1, 0, 0, r));
  }
  function addSensor(t, def) {
    const topic = t.wire(def.topic), missing = !(topic in S.graph.topics);
    const groups = [];                       // fields of one group are shown together, in profile order
    for (const f of def.fields) {
      let g = groups.find((x) => x.id === f.group);
      if (!g) groups.push(g = { id: f.group, fields: [] });
      g.fields.push(f);
    }
    const cells = new Map();                 // field name -> { f, value, row, bits: Map(bit -> element), tile }
    const body = h("div", { class: "sensor-body" });
    for (const g of groups) {
      const row = h("div", { class: g.id ? "sensor-group" : "sensor-fields sensor-group", "data-group": g.id || null },
        g.id ? h("span", { class: "sensor-group-label", text: g.id }) : null);
      for (const f of g.fields) {
        const value = h("span", { class: "sf-value", "data-testid": "sensor-value", "data-value": "", text: "—" });
        const kids = [h("span", { class: "sf-label", text: f.label }), value];
        if ((f.kind === "number" || f.kind === "vector3") && f.unit) kids.push(h("span", { class: "sf-unit", text: f.unit }));
        if (f.warn_below != null) kids.push(h("span", { class: "sf-warn", "data-testid": "sensor-warn", text: `low: below ${f.warn_below} ${f.unit}`.trim() }));
        const cell = { f, value, bits: new Map(), tile: null };
        if (f.kind === "bits") {
          const wrap = h("span", { class: "sf-bits" });
          for (const b of f.bits) {
            const el = h("span", { class: "sf-bit", "data-testid": "sensor-bit", "data-bit": b.bit, "data-set": "unknown", text: b.label });
            cell.bits.set(b.bit, el); wrap.append(el);
          }
          kids.push(wrap);
        }
        cell.row = h("div", { class: "sensor-field", "data-testid": "sensor-field", "data-field": f.name, "data-kind": f.kind }, kids);
        row.append(cell.row);
        if (f.headline) {                    // also shown large at the top of the panel
          const tv = h("span", { class: "hl-value", "data-testid": "headline-value", text: "—" });
          const ts = h("span", { class: "hl-state", "data-testid": "headline-state" });
          const tile = h("div", { class: "headline", "data-testid": "headline", "data-sensor": def.id, "data-field": f.name, "data-state": "waiting" },
            h("div", { class: "hl-head" }, h("span", { class: "hl-label", text: f.label }), ts),
            h("div", { class: "hl-reading" }, tv, f.unit ? h("span", { class: "hl-unit", text: f.unit }) : null),
            f.warn_below != null ? h("div", { class: "hl-warn", text: `below ${f.warn_below} ${f.unit}: charge or replace the battery`.replace(/ +:/, ":") }) : null,
            h("code", { class: "hl-topic", text: topic }));
          $("sensor-headlines").append(tile); $("sensor-headlines").hidden = false;
          cell.tile = { el: tile, value: tv, state: ts };
        }
        cells.set(f.name, cell);
      }
      body.append(row);
    }
    let plot = null;
    if (def.plot === "scan") {
      const canvas = h("canvas", { class: "scan-canvas", width: 360, height: 360, "data-testid": "scan-canvas",
        role: "img", "aria-label": `top-down plot of ${topic}` });
      const text = h("div", { class: "scan-text", "data-testid": "scan-text", text: "no scan yet" });
      const zoom = (k) => h("button", { type: "button", class: "btn scan-zoom", "data-testid": `scan-zoom-${k}`,
        text: { in: "Zoom in", out: "Zoom out", auto: "Auto" }[k] });
      const zin = zoom("in"), zout = zoom("out"), zauto = zoom("auto");
      plot = { el: h("div", { class: "scan-plot", "data-testid": "scan-plot" }, canvas,
        h("div", { class: "scan-side" }, text, h("div", { class: "scan-zooms" }, zin, zout, zauto))), canvas, text, zoom: 1, msg: null };
      const rezoom = (z) => { plot.zoom = z; if (plot.msg) drawScan(plot, plot.msg); };
      zin.addEventListener("click", () => rezoom(Math.max(1 / 64, plot.zoom / 2)));
      zout.addEventListener("click", () => rezoom(Math.min(64, plot.zoom * 2)));
      zauto.addEventListener("click", () => rezoom(1));
      body.append(plot.el);
    }
    const state = h("span", { class: "sensor-state", "data-testid": "sensor-state", role: "status" });
    const meta = h("span", { class: "sensor-meta", "data-testid": "sensor-meta" });
    const sample = def.on_demand && !missing ? h("button", { type: "button", class: "btn sensor-sample", "data-testid": "sensor-sample",
      text: "Sample once" }) : null;
    const el = h("article", { class: "sensor", "data-testid": "sensor", "data-sensor": def.id, "data-topic": topic,
        "data-state": "waiting", "data-wide": def.fields.length > 8 || plot ? "true" : null },
      h("div", { class: "sensor-head" },
        h("div", {}, h("div", { class: "sensor-title", text: def.label }), h("code", { class: "sensor-topic", text: topic })),
        state),
      def.caption ? h("p", { class: "sensor-caption", "data-testid": "sensor-caption", text: def.caption }) : null,
      def.on_demand ? h("p", { class: "sensor-caption", text: "On demand: subscribing makes the robot compute this topic, "
        + "so the page subscribes only when you sample it and unsubscribes after the first message." }) : null,
      body, sample, meta);
    $("sensor-list").appendChild(el);
    $("sensors-panel").hidden = false;
    const s = { def, topic, el, state, meta, cells, plot, msg: null, last: null, count: 0, times: [], dirty: false, missing,
      started: performance.now() };
    const throttle = def.throttle_s != null ? Math.round(def.throttle_s * 1000) : SENSOR_THROTTLE_MS;
    const got = (m) => {
      const now = performance.now();
      s.msg = m; s.last = now; s.count++; s.dirty = true;
      s.times.push(now); while (s.times.length > 20) s.times.shift();
    };
    if (!missing && !def.on_demand) s.sid = S.conn.subscribe(topic, def.type, got, throttle);
    if (sample) sample.addEventListener("click", () => sampleOnce(s, got));
    S.sensors.push(s);
    return s;
  }
  // An on-demand sensor: subscribe, keep the first message, unsubscribe at once (or after
  // SAMPLE_TIMEOUT_MS without one). Nothing is published or called.
  const SAMPLE_TIMEOUT_MS = 10000, SAMPLE_THROTTLE_MS = 3600000, SAMPLE_GRACE_MAX_S = 5;
  // A sample counts against the link budget (console spec §2.3): a cloud of up to 640x480 points is megabytes in one
  // burst, so it is refused while the cameras take more than half the budget, and no stream starts while it is in
  // flight. Its size is the last sample's, or before one a 640x480 cloud of 20-byte points as base64 JSON.
  function sampleBytes(s) { return s.lastSampleBytes || Math.ceil(NOMINAL_PX * 20 * 4 / 3); }
  function sampleRefusal(s) {
    const budget = S.target && S.target.profile.link_budget_mbps;
    if (!budget) return "";
    if (S.camPause) return "Not sampled: the cameras are paused because the link is congested.";
    const cams = camsMbps(null);
    if (cams <= budget / 2) return "";
    return `Not sampled: a cloud is about ${(sampleBytes(s) / 1e6).toFixed(0)} MB in one burst, and the cameras already take about `
      + `${cams.toFixed(0)} of the ${budget} Mbit/s link budget. Stop the depth, infrared or USB stream first.`;
  }
  function sampleOnce(s, got) {
    if (s.sid || !S.conn || !S.conn.open || S.lost) return;
    s.refused = sampleRefusal(s);
    if (s.refused) return;
    // the sample's own burst delays what shares the link: its expected time at the budget (or 20 Mbit/s), plus 1 s,
    // at most SAMPLE_GRACE_MAX_S, is a grace window for the congestion rules (never longer)
    if (S.clock) {
      const mbps = (S.target && S.target.profile.link_budget_mbps) || 20, t = monoS();
      S.clock.grace = [t, t + Math.min(SAMPLE_GRACE_MAX_S, sampleBytes(s) * 8 / (mbps * 1e6) + 1)];
    }
    const done = () => { clearTimeout(s.sampleTimer); try { if (s.sid && S.conn.open) S.conn.unsubscribe(s.sid); } catch (e) {} s.sid = null; s.sampling = false; };
    s.sampling = true; s.timedOut = false; s.sampleStarted = performance.now();
    // throttle_rate SAMPLE_THROTTLE_MS with queue_length 1: rosbridge sends the first message and no other
    // before the unsubscribe arrives (a cloud is megabytes).
    s.sid = S.conn.subscribe(s.topic, s.def.type, (m, size) => { if (!s.sampling) return; done(); if (size) s.lastSampleBytes = size; got(m); }, SAMPLE_THROTTLE_MS);
    s.sampleTimer = setTimeout(() => { if (s.sampling) { done(); s.timedOut = true; } }, RC.sampleTimeoutMs || SAMPLE_TIMEOUT_MS);
  }
  function sensorValue(cell, msg) {
    const { f, value } = cell;
    const v = readPath(msg, f.path);
    const d = f.decimals == null ? 3 : f.decimals;
    let text = "—", raw = "", low = false, parts = null, none = false;
    if (f.kind === "number") {
      if (fin(v) && (f.no_reading || []).some((x) => Math.abs(v - x) < 1e-6)) { text = f.no_reading_label; none = true; }
      else if (fin(v)) { raw = String(v); text = fixed(v, d); low = f.warn_below != null && v < f.warn_below; }
    } else if (f.kind === "text") {
      if (typeof v === "string") { raw = v; text = v === "" ? "(empty)" : v; }
    } else if (f.kind === "bool") {
      if (typeof v === "boolean") { raw = text = String(v); }
    } else if (f.kind === "list") {
      const a = asList(v);
      if (a) { raw = a.join(","); text = a.length ? a.join(", ") : "(none)"; }
    } else if (f.kind === "count") {
      const a = asList(v);
      if (a) { raw = text = String(a.length); }
    } else if (f.kind === "vector3") {
      if (v && [v.x, v.y, v.z].every(fin)) { raw = [v.x, v.y, v.z].join(","); parts = [`x ${fixed(v.x, d)}`, `y ${fixed(v.y, d)}`, `z ${fixed(v.z, d)}`]; }
    } else if (f.kind === "rpy") {
      const r = quatRpy(f.mount_rpy && v ? quatMul(v, quatConj(quatFromRpy(f.mount_rpy))) : v);
      if (r) {
        const deg = r.map((a) => a * 180 / Math.PI), dd = f.decimals == null ? 1 : f.decimals;
        raw = deg.join(","); parts = [`roll ${fixed(deg[0], dd)}°`, `pitch ${fixed(deg[1], dd)}°`, `yaw ${fixed(deg[2], dd)}°`];
      } else if (v) text = "(no orientation)";
    } else if (f.kind === "bits") {
      const n = typeof v === "number" ? v : Array.isArray(v) || v == null ? NaN : Number(v);
      const ok = Number.isInteger(n) && n >= 0;
      if (ok) { raw = text = String(n); }
      for (const [bit, el] of cell.bits) el.dataset.set = ok ? String(((n >>> bit) & 1) === 1) : "unknown";
    }
    if (parts) {                       // each component kept whole; lines break only between components
      text = parts.join(" · ");
      if (value.textContent !== text) value.replaceChildren(...parts.flatMap((p, i) => [i ? h("span", { class: "vsep", text: " · " }) : null,
        h("span", { class: "vc", text: p })]).filter(Boolean));
    } else if (value.textContent !== text) value.textContent = text;
    value.dataset.value = raw;
    value.dataset.noreading = none ? "true" : "";
    cell.row.dataset.warn = low ? "low" : "";
    if (cell.tile) {
      cell.tile.value.textContent = text; cell.tile.value.dataset.value = raw;
      cell.tile.el.dataset.warn = low ? "low" : ""; cell.tile.el.dataset.noreading = none ? "true" : "";
    }
  }
  // A "nice" plot radius (1, 2 or 5 times a power of ten) at or above r.
  function niceRadius(r) {
    const p = Math.pow(10, Math.floor(Math.log10(r)));
    return [1, 2, 5, 10].map((k) => k * p).find((x) => x >= r - 1e-9);
  }
  // The returns of a LaserScan within its range_min..range_max: [x forward, y left, range, angle].
  function scanPoints(m) {
    const out = [], rs = Array.isArray(m.ranges) ? m.ranges : [];
    const lo = fin(m.range_min) ? m.range_min : 0, hi = fin(m.range_max) ? m.range_max : Infinity;
    if (!fin(m.angle_min) || !fin(m.angle_increment)) return out;
    rs.forEach((r, i) => {
      if (!fin(r) || r < lo || r > hi) return;
      const a = m.angle_min + i * m.angle_increment;
      out.push([r * Math.cos(a), r * Math.sin(a), r, a]);
    });
    return out;
  }
  // Top-down: the scan frame's x (forward) up, y (left) to the left, rings at quarters of the radius. The radius
  // is robust to a few far returns: the 90th percentile of the in-range returns, made "nice", times the user's
  // zoom (Zoom in/out halves/doubles it, Auto resets); returns beyond it are drawn on the rim, hollow.
  function drawScan(plot, m) {
    plot.msg = m;
    const pts = scanPoints(m), c = plot.canvas, g = c.getContext("2d"), W = c.width, H = c.height;
    const css = getComputedStyle(document.documentElement), col = (n) => css.getPropertyValue(n).trim();
    const cx = W / 2, cy = H / 2;
    const sorted = pts.map((p) => p[2]).sort((a, b) => a - b);
    const p90 = sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(0.9 * (sorted.length - 1)))] : 0;
    const auto = niceRadius(Math.max(p90, 0.5)), R = auto * plot.zoom, k = (Math.min(W, H) / 2 - 14) / R;
    g.clearRect(0, 0, W, H);
    g.strokeStyle = col("--border"); g.fillStyle = col("--faint"); g.lineWidth = 1; g.font = "11px system-ui, sans-serif";
    for (let q = 1; q <= 4; q++) {
      g.beginPath(); g.arc(cx, cy, (R * q / 4) * k, 0, 2 * Math.PI); g.stroke();
      g.fillText(`${+(R * q / 4).toFixed(2)} m`, cx + 3, cy - (R * q / 4) * k + 12);
    }
    g.beginPath(); g.moveTo(cx, 6); g.lineTo(cx, H - 6); g.moveTo(6, cy); g.lineTo(W - 6, cy); g.stroke();
    let beyond = 0;
    for (const [x, y, r] of pts) {
      if (r <= R) { g.fillStyle = col("--info"); g.fillRect(cx - y * k - 1.5, cy - x * k - 1.5, 3, 3); continue; }
      beyond++;
      const f = R / r; g.strokeStyle = col("--muted"); g.strokeRect(cx - y * f * k - 1.5, cy - x * f * k - 1.5, 3, 3);
    }
    g.fillStyle = col("--text"); g.beginPath();                // the sensor, pointing forward (up)
    g.moveTo(cx, cy - 9); g.lineTo(cx - 6, cy + 6); g.lineTo(cx + 6, cy + 6); g.closePath(); g.fill();
    let near = null;
    for (const p of pts) if (!near || p[2] < near[2]) near = p;
    plot.el.dataset.points = String(pts.length);
    plot.el.dataset.radius = String(+R.toPrecision(6));
    plot.el.dataset.beyond = String(beyond);
    plot.el.dataset.nearest = near ? String(near[2]) : "";
    plot.text.textContent = `${pts.length} of ${(m.ranges || []).length} returns in range · rings every ${+(R / 4).toFixed(2)} m`
      + (beyond ? ` · ${beyond} beyond ${+R.toFixed(2)} m shown on the rim` : "") + " · forward is up"
      + (near ? ` · nearest ${near[2].toFixed(2)} m at ${fixed(near[3] * 180 / Math.PI, 0)}°` : "");
  }

  // [state, badge]: never "live" unless a message arrived within the sensor's stale threshold; a
  // latched or non-periodic topic (stale_after_s null) is "latched" with its last message's age.
  function sensorState(s, now) {
    if (s.missing) return ["missing", "MISSING"];
    if (S.lost) return ["lost", "DISCONNECTED"];
    if (s.def.event_only) return s.last == null ? ["event", "EVENT-ONLY · NONE YET"] : ["lastevent", "LAST EVENT"];
    if (s.def.on_demand) {
      if (s.sampling) return ["sampling", "SAMPLING"];
      if (s.timedOut) return ["nomessage", "NO MESSAGE"];
      return s.last == null ? ["off", "NOT SAMPLED"] : ["sampled", "SAMPLED"];
    }
    if (s.last == null) return ["waiting", "NO MESSAGE YET"];
    if (s.def.stale_after_s == null) return ["latched", "LAST VALUE"];
    return (now - s.last) / 1000 <= s.def.stale_after_s ? ["live", "LIVE"] : ["stale", "STALE"];
  }
  // Messages per second over the recent arrivals (as delivered: at most one per the sensor's throttle).
  function sensorRate(s, now) {
    const ts = s.times.filter((x) => now - x < Math.max(3000, 3000 * (s.def.throttle_s || 0)));
    if (ts.length < 2 || s.last == null || now - s.last > 2000 * Math.max(1, s.def.throttle_s || 0)) return "";
    const r = (ts.length - 1) / ((ts[ts.length - 1] - ts[0]) / 1000);
    return ` · ${r.toFixed(r < 10 ? 1 : 0)}/s`;
  }
  function tickSensors(now) {
    let live = 0;
    for (const s of S.sensors) {
      if (s.dirty) {
        s.dirty = false;
        for (const c of s.cells.values()) sensorValue(c, s.msg);
        if (s.plot) drawScan(s.plot, s.msg);
      }
      const [st, badge] = sensorState(s, now);
      if (!s.def.event_only && !s.def.on_demand && (st === "live" || st === "latched")) live++;
      if (s.state.textContent !== badge) s.state.textContent = badge;
      s.el.dataset.state = st;
      for (const c of s.cells.values()) if (c.tile) { c.tile.el.dataset.state = st; c.tile.state.textContent = badge; }
      const age = s.last == null ? "" : (now - s.last) / 1000;
      if (s.def.on_demand && s.el.querySelector(".sensor-sample")) s.el.querySelector(".sensor-sample").disabled = !!s.sampling || S.lost;
      s.meta.textContent = st === "missing" ? "the topic is not on the wire"
        : s.def.on_demand && s.refused && !s.sampling ? s.refused
        : st === "nomessage" ? `no message within ${(RC.sampleTimeoutMs || SAMPLE_TIMEOUT_MS) / 1000} s; unsubscribed`
        : st === "sampling" ? "subscribed until the first message arrives"
        : s.last == null ? (s.def.on_demand ? "not subscribed: press Sample once"
          : s.def.event_only ? "published only on an event: none since the page subscribed (normal)" : "no message received yet")
        : s.def.on_demand ? `sampled ${(now - s.last) / 1000 < 1 ? "just now" : ((now - s.last) / 1000).toFixed(0) + " s ago"}; not subscribed (a snapshot, not live)`
        : `${s.count} message${s.count === 1 ? "" : "s"}${sensorRate(s, now)} · last ${age < 1 ? Math.round(age * 1000) + " ms" : age.toFixed(1) + " s"} ago`
          + (st === "stale" ? ` (threshold ${s.def.stale_after_s} s; the values are frozen)` : st === "lost" ? " (connection lost; the values are frozen)" : "")
          + (s.def.throttle_s != null ? ` · at most one per ${s.def.throttle_s} s` : "");
    }
    // periodic and latched sensors report; event-only and on-demand ones are counted apart (waiting is normal)
    const main = S.sensors.filter((s) => !s.def.event_only && !s.def.on_demand).length;
    const ev = S.sensors.filter((s) => s.def.event_only), od = S.sensors.filter((s) => s.def.on_demand);
    $("sensor-summary").textContent = !S.sensors.length ? "" : `${live} of ${main} reporting`
      + (ev.length ? ` · ${ev.length} event-only (${ev.filter((s) => s.last != null).length} seen)` : "")
      + (od.length ? ` · ${od.length} on demand` : "");
  }

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
            fs.measVal.textContent = valueText(v, fs.f.unit, decimals(fs.f) + 1); fs.meas.dataset.state = "live";
            fs.meas.title = cs.fb.measured ? `measured ${cs.fb.map[fs.f.name]} on ${cs.fb.topic}`
              : `reported ${cs.fb.map[fs.f.name]} on ${cs.fb.topic}: what the robot publishes for this joint, not a measurement`;
            const p = Math.min(1, Math.max(0, (v - fs.f.min) / (fs.f.max - fs.f.min)));
            fs.mark.style.left = `${p * 100}%`;
          }
        }
      }
      if (cs.reads.length) {
        const rv = readValues(cs);
        for (const fs of Object.values(cs.fields)) {
          if (!fs.read) continue;
          const r = rv[fs.f.name];
          const polled = r && r.x.r.poll_s, age = r ? (now - r.x.last.at) / 1000 : 0;
          if (!r || r.v == null) {
            fs.readVal.textContent = "—"; fs.read.dataset.state = r ? "invalid" : "none";
            fs.read.title = r ? `the last read of ${r.x.wire} gave no valid value for this field`
              : cs.polled.length ? `not measured yet: the page asks ${cs.polled[0].wire} about every ${cs.polled[0].r.poll_s} s`
              : cs.readBtn ? `not read yet: “${cs.readBtn.textContent}” asks the robot once` : "not read yet";
          } else {
            any = true;
            fs.readVal.textContent = valueText(r.v, fs.f.unit, decimals(fs.f) + (cs.group ? 0 : 1));
            fs.read.dataset.state = !polled ? "read" : pollFresh(r.x, now) ? "live" : "stale";
            fs.read.title = polled
              ? `measured by ${r.x.wire} ${age.toFixed(1)} s ago (asked about every ${r.x.r.poll_s} s)` + (fs.read.tagName === "BUTTON" ? "; click to make it the target (sends it)" : "")
              : `measured: read from ${r.x.wire} ${age.toFixed(1)} s ago (not live)`;
          }
          if (fs.read.tagName === "BUTTON") fs.read.disabled = !(r && r.v != null);
        }
        if (cs.readStatus) cs.readStatus.textContent = readText(cs.reads, now);
      }
      if (cs.useMeasured) cs.useMeasured.disabled = !any;
    }
    for (const gr of S.groups) syncGroup(gr, now);
  }
  // A polled measurement is fresh while younger than three periods (and the connection is up).
  const pollFresh = (x, now) => !!x.last && !S.lost && (now - x.last.at) / 1000 <= 3 * x.r.poll_s;

  // ------------------------------------------------------------------ measured reads
  // Read services the profile documents as returning measured joint positions (e.g. a servo
  // readback). The page calls one only when the user clicks its button, once per click, never
  // on load or by itself, except the read the profile marks `poll_s` (polled positions, below);
  // each answered value maps to a control field as field = (raw - offset) / scale. A reading is
  // a snapshot, shown with its age.
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
  // A read's answer as {control: {field: value in the field's unit, or null when not read}}
  // (profiles.py Read.parse); throws when the answer reports a failed read.
  function parseRead(r, v) {
    if (!v || typeof v !== "object" || (r.ok_field && v[r.ok_field] === false)) throw new Error(`the service answered ${JSON.stringify(v).slice(0, 120)}`);
    const raws = r.values.map((m) => readPath(v, m.path));
    const vals = {};
    r.values.forEach((m, i) => {
      const raw = Number(raws[i]);
      const ok = raws[i] != null && Number.isFinite(raw) && !(r.invalid || []).includes(raw);
      (vals[m.control] = vals[m.control] || {})[m.field] = ok ? (raw - Number(m.offset || 0)) / (m.scale == null ? 1 : Number(m.scale)) : null;
    });
    return vals;
  }
  function doRead(x) {
    const conn = S.conn, t = S.target;
    if (!conn || S.lost || !t || x.busy) return Promise.resolve();
    const r = x.r, sentAt = performance.now();
    x.busy = true; x.err = null;
    const done = conn.call(x.wire, r.request || {}, 5000).then((v) => {
      const vals = parseRead(r, v);
      x.last = { at: performance.now(), sentAt, vals };
      if (r.poll_s && S.target === t) followMeasured(vals);
    }).catch((e) => { x.err = e.message; })
      .finally(() => { x.busy = false; tickFeedback(performance.now()); scheduleModelDraw(); });
    tickFeedback(performance.now());
    return done;
  }
  // Targets the user has not changed on this page follow the polled measurement (nothing is
  // sent), so a slider, the head pad or a joint dragged in the model starts where the robot is.
  function followMeasured(vals) {
    for (const [cid, fields] of Object.entries(vals)) {
      const cs = S.controls.get(cid); if (!cs || cs.dead) continue;
      for (const [name, v] of Object.entries(fields)) {
        const fs = cs.fields[name];
        if (fs && v != null && fs.pristine && !fs.held) setField(fs, v, false, true);
      }
    }
  }

  // ------------------------------------------------------------------ polled positions
  // The read the profile marks `poll_s` is its documented read-only joint-position query (console
  // spec §2.2): once the target is validated the page calls it about every
  // poll_s seconds, one call at a time, so the model and the measured values follow a robot that
  // publishes no joint states. It commands nothing, is never called before validation, and stops
  // on connection loss and on a target change.
  let pollTimer = null;
  function startPoll(t, conn) {
    stopPoll();
    const x = [...S.reads.values()].find((y) => y.r.poll_s && !y.reason);
    if (!x || S.target !== t || S.conn !== conn || S.lost) return;
    const live = () => S.poll === x && S.target === t && S.conn === conn && !S.lost;
    S.poll = x;
    const step = () => {
      pollTimer = null;
      if (!live()) return;
      const t0 = performance.now();
      doRead(x).finally(() => {
        if (live() && !pollTimer) pollTimer = setTimeout(step, Math.max(0, x.r.poll_s * 1000 - (performance.now() - t0)));
      });
    };
    step();
  }
  function stopPoll() { if (pollTimer) clearTimeout(pollTimer); pollTimer = null; S.poll = null; }

  // ------------------------------------------------------------------ sending
  // Values changed on the page go to the robot at once; no bounded control is ever stopped or
  // cancelled by the page. Topic publishes stream while a value is dragged; action goals stream at a
  // lower rate, as the profile records the action server's handling of a goal that arrives while
  // another of its goals runs (the control's `busy`). A server that preempts gets each new goal
  // as it comes, ending the previous one. A server that rejects it, or runs it beside the running
  // one, gets one goal of the control at a time ("serial"): while a goal runs, changes are held
  // (only the latest value counts) and go out when that goal's result arrives, whatever its
  // status, if they differ from the last goal sent; a change made while no goal runs is sent even
  // when it equals the last goal's. Nothing is cancelled to make room. Both use a throttle (for
  // action goals, either kind, about 5 per second) with a leading and a trailing edge that reads
  // the inputs when it fires, so the last value is always sent. Service calls and choice buttons
  // send once per click.
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
  // One goal at a time: an action control whose server does not preempt (profile `busy`).
  const isSerial = (c) => c.kind === "action" && c.busy !== "preempt";
  const liveOk = (cs) => !!S.target && !!S.conn && !S.lost && cs.available && !cs.dead;

  // Every control shows the phase of its latest send: sending (on its way, or waiting for the
  // answer), running (a goal the action server acknowledged), done, failed, rejected (a goal the
  // action server refused) or invalid (a value outside the documented limits, never sent). The
  // text starts with the phase.
  const PHASE_CLS = { sending: "warn", running: "warn", done: "ok", failed: "bad", rejected: "bad", invalid: "bad" };
  function setStatus(id, phase, text) {
    const cs = S.controls.get(id); if (!cs) return;
    const cls = PHASE_CLS[phase] || "dim";
    text = `${phase}: ${text}`;
    cs.status = { phase, cls, text }; cs.statusEl.textContent = text; cs.statusEl.className = "status " + cls;
    cs.div.dataset.status = cls; cs.div.dataset.phase = phase;
  }
  function heldText(c) {
    return "goal running; the action server " + (c.busy === "reject" ? "rejects a new goal while one runs"
      : "would run a new goal beside it") + ", so your latest value is held and sent when this goal's result arrives";
  }

  // A value of a streamed control changed: send it now, or as soon as the throttle (and, for a
  // serial control, the running goal) allows.
  function trigger(cs) {
    if (!liveOk(cs) || cs.mode !== "stream") return;
    queue(cs);
  }
  function queue(cs) {
    // A change made while no goal is outstanding is sent as the next goal (as the pacing allows)
    // even when it equals the last goal's: it replaces a held value still waiting for the pacing,
    // which would otherwise be dropped as unchanged.
    if (cs.serial && !goalHolds(cs)) cs.th.held = false;
    cs.th.pending = true; pump(cs);
    if (!cs.th.pending) return;
    if (cs.serial && cs.goal && cs.goal.acked && !cs.goal.done) setStatus(cs.ctrl.id, "running", heldText(cs.ctrl));
    else if (!cs.status || cs.status.phase !== "sending") setStatus(cs.ctrl.id, "sending", "the new value goes out as the rate limit allows…");
  }
  // The latest goal holds the next one back: for at most GOAL_ACK_MS until the action server
  // acknowledges it (feedback or result) and, for a serial control, once acknowledged, until its
  // result. A goal that is still unacknowledged after GOAL_ACK_MS holds nothing back (rosbridge
  // before 2.5 never answers a goal the server rejected).
  function goalHolds(cs) {
    const g = cs.goal;
    if (!g || g.done) return false;
    return g.acked ? cs.serial : Date.now() - g.sentAt < GOAL_ACK_MS;
  }
  function pump(cs) {
    const th = cs.th;
    if (!th.pending || th.timer || th.busy || !liveOk(cs)) return;
    if (cs.ctrl.kind === "action" && goalHolds(cs)) { if (cs.serial) th.held = true; return; }
    const wait = th.last + STREAM_MS[cs.ctrl.kind === "action" ? "action" : "publish"] - Date.now();
    if (wait > 0) { th.timer = setTimeout(() => { th.timer = null; pump(cs); }, wait); return; }
    const held = th.held;
    th.pending = false; th.held = false; th.last = Date.now(); th.busy = true;
    Promise.resolve(startControl(cs, held)).catch(() => {}).finally(() => { th.busy = false; pump(cs); });
  }
  // One send per click (a serial control's click waits for its running goal, as a change does).
  function fire(cs) {
    if (!liveOk(cs)) return;
    if (cs.serial) queue(cs); else startControl(cs);
  }

  // `held`: the value was held while a goal ran; it is sent only if it differs from that goal's.
  async function startControl(cs, held = false) {
    const c = cs.ctrl, t = S.target, conn = S.conn;
    if (!liveOk(cs)) return false;
    const values = {};
    try { for (const f of c.fields) values[f.name] = toWire(f, coerce(f, cs.inputs[f.name].value)); }
    catch (e) { setStatus(c.id, "invalid", "not sent: " + e.message); return false; }
    let payload;
    try { payload = fill(c.template, values, jointPrefix(t.profile, t.namespace)); }
    catch (e) { setStatus(c.id, "invalid", "not sent: " + e.message); return false; }
    const name = t.wire(c.name);
    if (held && cs.goal && cs.goal.key === JSON.stringify(payload)) return false;    // nothing new to send
    try {
      if (c.kind === "publish") {
        conn.publish(name, payload, c.type); cs.sent++;
        setStatus(c.id, "done", `sent (${cs.sent} message${cs.sent === 1 ? "" : "s"} so far; a publish is not acknowledged)`);
      } else if (c.kind === "call") {
        setStatus(c.id, "sending", "pending: waiting for the service answer…");
        conn.call(name, payload, 10000).then((v) => {
          const bad = v && (v.result === false || v.success === false);
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
  // close (closing one cancels nothing on stock rosbridge), and a goal's connection closes after
  // its result. At most GOAL_CONNS_MAX are open, counting the new goal's and those of retired
  // goals whose sockets are still closing (cs.closing): the oldest superseded goals close at
  // once, and the new goal's connection opens only once enough of the others have closed.
  async function sendGoal2(cs, conn, name, payload) {
    const c = cs.ctrl;
    const oldest = cs.goals.slice(0, Math.max(0, cs.goals.length - (GOAL_CONNS_MAX - 1)));
    oldest.forEach((o) => retireGoal(cs, o, true));
    while (cs.goals.length + cs.closing.size > GOAL_CONNS_MAX - 1) await Promise.race([...cs.closing].map((o) => o.closed));
    if (!liveOk(cs) || S.conn !== conn) return;
    const gc = new RosConn(conn.url); await gc.connect(5000);
    if (!liveOk(cs) || S.conn !== conn) { gc.close(); return; }
    const g = { conn: gc, id: randomId(), key: JSON.stringify(payload), sentAt: Date.now(), acked: false, done: false };
    cs.goal = g; cs.goals.push(g);
    const ack = () => {
      if (g.acked) return;
      g.acked = true;
      if (cs.goal === g) {
        if (!g.done) setStatus(c.id, "running", cs.th.pending && cs.serial ? heldText(c) : "goal running (acknowledged by the action server)");
        for (const o of [...cs.goals]) if (o !== g) retireGoal(cs, o);
      }
    };
    gc.handlers.set("action_feedback:" + g.id, () => { if (!g.acked) { ack(); pump(cs); } });
    gc.handlers.set("action_result:" + g.id, (m) => {
      g.done = true; ack();
      if (cs.goal === g) showResult(cs, m);
      retireGoal(cs, g);
      pump(cs);                           // a value held for a serial control goes out now
    });
    gc.onlost = () => {
      if (!g.done && cs.goal === g) setStatus(c.id, "failed", "goal connection lost; outcome unknown");
      g.done = true; retireGoal(cs, g); pump(cs);
    };
    gc.send({ op: "send_action_goal", id: g.id, action: name, action_type: c.type, args: payload, feedback: true });
    setStatus(c.id, "sending", "goal sent, waiting for the action server…");
    setTimeout(() => pump(cs), GOAL_ACK_MS + 10);
  }
  // A goal's result. GoalStatus: 4 succeeded, 5 canceled, 6 aborted (2 canceling); anything but
  // success did not reach the target. rosbridge answers a goal it could not run with result false
  // and its reason; from rosbridge_suite 2.5 that includes a goal the action server rejected
  // ("Action goal was rejected", status 0, rosbridge_library internal/actions.py).
  function showResult(cs, m) {
    const c = cs.ctrl;
    if (m.result === false) {
      const why = typeof m.values === "string" ? m.values : JSON.stringify(m.values);
      if (/reject/i.test(why)) {
        setStatus(c.id, "rejected", `the action server rejected the goal (rosbridge: ${why})` + (c.busy === "reject"
          ? "; it rejects a goal while another of its goals runs, from this page or any other client" : ""));
      } else setStatus(c.id, "failed", "rosbridge could not run the goal: " + why);
      return;
    }
    const st = { 4: "succeeded", 5: "canceled", 6: "aborted", 2: "canceled" }[m.status]
      || (m.status == null ? "ended without a status" : `ended (status ${m.status})`);
    setStatus(c.id, m.status === 4 ? "done" : "failed", "goal " + st);
  }
  // A retired goal leaves cs.goals and its connection closes (`now`: at once, else shortly);
  // until its socket has closed it still counts as open (cs.closing). Resolves then.
  function retireGoal(cs, g, now = false) {
    const i = cs.goals.indexOf(g); if (i < 0) return g.closed || Promise.resolve();
    cs.goals.splice(i, 1);
    cs.closing.add(g);
    g.closed = g.conn.wsClosed.then(() => { cs.closing.delete(g); });
    if (now) g.conn.close(); else setTimeout(() => g.conn.close(), 50);
    return g.closed;
  }

  // Controls of a target that is being replaced: no further sends. The latest unfinished goal
  // keeps its connection until its result arrives; nothing is cancelled.
  function retireControls() {
    for (const cs of S.controls.values()) {
      cs.dead = true;
      if (cs.th.timer) { clearTimeout(cs.th.timer); cs.th.timer = null; }
      cs.th.pending = false; cs.th.held = false;                 // a held value is not sent
      for (const g of [...cs.goals]) if (g !== cs.goal || g.done) retireGoal(cs, g);
    }
  }

  function refreshControls() {
    for (const cs of S.controls.values()) cs.body.disabled = !liveOk(cs);
    for (const gr of S.groups) syncGroup(gr, performance.now());
    if (S.padGroup) S.padGroup.sync();
    drawModel();
    syncLive();
    syncMotion();
  }
  function updateLock() {
    $("robot").disabled = S.lost; $("namespace").disabled = S.lost || !nsAllowed(); $("select").disabled = S.lost || S.changing;
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
    else if (!S.controls.size && S.mo) { state = "unavailable"; title = "No bounded controls"; hint = "This robot's profile lists no bounded page controls; its base motion is offered beside the cameras."; }
    else if (!S.controls.size) { state = "unavailable"; title = "Camera only"; hint = "This robot's profile lists no page controls."; }
    else { state = "live"; title = `Controls live on ${label}`; hint = `Values you change below are sent to ${label} at once.`; }
    const card = $("live-card");
    card.dataset.state = state; $("live-title").textContent = title; $("live-hint").textContent = hint;
  }

  // ------------------------------------------------------------------ controls UI
  function kindText(c) {
    if (c.kind === "call") return "Service call · once per click";
    if (c.kind === "action") {
      const one = c.busy === "reject" ? "the action server rejects a new goal while one runs"
        : "the action server would run a new goal beside the running one";
      if (sendMode(c) !== "stream") return "Action goal · once per click" + (isSerial(c) ? `; one goal at a time (${one})` : "");
      return isSerial(c)
        ? `Action goal · one at a time: ${one}, so while a goal runs only your latest value is kept, and it is sent when that goal ends`
        : "Action goal · sent as you change it (≈5 per s); the action server preempts the running goal with each new one";
    }
    return sendMode(c) === "stream" ? "Topic publish · sent as you change it (≈10 per s)" : "Topic publish · once per click";
  }

  function validateField(fs) {
    const f = fs.f;
    let v = null, err = "";
    try { v = coerce(f, fs.num.value); toWire(f, v); }
    catch (e) { v = null; err = e.message.startsWith(f.label + ": ") ? e.message.slice(f.label.length + 2) : e.message; }
    fs.div.classList.toggle("invalid", !!err);
    fs.num.setAttribute("aria-invalid", String(!!err));
    fs.err.textContent = err;
    if (v != null && fs.range && Number(fs.range.value) !== v) fs.range.value = String(v);
    if (fs.deg) fs.deg.textContent = v != null ? degText(v) : "";
    for (const fn of fs.listeners) fn(v);
  }
  // A change by the user: the field stops following the measurement, and the model draws its
  // target until a measurement asked for after the change has arrived.
  function touched(fs) { fs.pristine = false; fs.changedAt = performance.now(); }
  // `quiet`: set by the page itself (a measurement followed), not a change by the user.
  function setField(fs, v, fromRange, quiet) {
    const f = fs.f;
    v = Math.min(f.max, Math.max(f.min, Number(v)));
    let r = Number(v.toFixed(decimals(f)));
    if (r > f.max || r < f.min) r = v;            // never round past a documented limit
    if (f.integer) r = Math.round(r);
    if (!quiet) touched(fs);
    fs.num.value = String(r);
    if (!fromRange && fs.range) fs.range.value = String(r);
    validateField(fs);
  }
  const stepOf = (f) => (f.integer ? 1 : Math.pow(10, -decimals(f)));
  // Pointer releases end every hold (a slider, the head pad or a joint dragged in the model).
  window.addEventListener("pointerup", () => { for (const cs of S.controls.values()) for (const fs of Object.values(cs.fields)) fs.held = false; }, true);

  // mode "card": label, number and slider rows of a control card; "row": one compact line of a
  // grouped joint list (the control's label, measured value, number, slider); "mini": a number
  // only (a secondary field of a row, e.g. its duration).
  function buildField(c, f, cs, mode = "card") {
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
    // A text field, not type=number: the value reads the same in every locale ("." decimals).
    const num = h("input", { type: "text", class: "num", id, "data-field": f.name, "data-min": f.min, "data-max": f.max,
      autocomplete: "off", spellcheck: "false", "aria-describedby": mode === "mini" ? null : id + "-lim" });
    if (f.default != null) num.value = String(f.default);
    const deg = f.unit === "rad" ? h("span", { class: "deg", "aria-hidden": "true" }) : null;
    const wrap = h("span", { class: "num-wrap" }, num, h("span", { class: "unit", text: f.unit || "" }), deg);
    let div, range = null, mark = null;
    if (mode === "mini") {
      div = h("div", { class: "field mini" }, h("label", { for: id, text: f.label }), wrap, h("div", { class: "field-err", role: "alert" }));
    } else {
      range = h("input", { type: "range", min: f.min, max: f.max, step: f.integer ? 1 : "any",
        "aria-label": `${mode === "row" ? c.label : f.label} (${f.unit || "value"})`, tabindex: "-1" });
      if (f.default != null) range.value = f.default;
      const lo = h("button", { type: "button", class: "limit", title: `documented minimum ${exact(f.min)} ${f.unit}`, text: exact(f.min) });
      const hi = h("button", { type: "button", class: "limit", title: `documented maximum ${exact(f.max)} ${f.unit}`, text: exact(f.max) });
      mark = h("span", { class: "meas-mark", hidden: true, "aria-hidden": "true" });
      // A row: the control's label over the field's label (and the measured value), the number beside them.
      const label = mode === "row"
        ? h("div", { class: "row-id" }, h("label", { for: id }, h("span", { class: "row-name", text: c.label }), h("span", { class: "row-hint", text: f.label })))
        : h("label", { for: id, text: f.label });
      div = h("div", { class: "field" + (mode === "row" ? " main" : "") },
        h("div", { class: "field-top" }, label, wrap),
        h("div", { class: "slider-row", id: id + "-lim" }, lo, h("div", { class: "track" }, range, mark), hi),
        h("div", { class: "field-err", role: "alert" }));
      lo.addEventListener("click", () => { setField(fs, f.min); trigger(cs); });
      hi.addEventListener("click", () => { setField(fs, f.max); trigger(cs); });
    }
    const fs = { f, div, num, range, mark, deg, err: div.querySelector(".field-err"), listeners: [], meas: null, measVal: null,
      read: null, readVal: null, pristine: true, changedAt: 0, held: false };
    if (mode !== "mini" && cs.fb && cs.fb.map[f.name]) {
      fs.measVal = h("span", { class: "m-val", text: "—" });
      fs.meas = h("span", { class: "measured" }, h("span", { class: "m-label", text: fbWord(cs.fb) }), fs.measVal);
      div.querySelector(".field-top").insertBefore(fs.meas, wrap);
    }
    if (mode !== "mini" && cs.reads.some((x) => x.r.values.some((m) => m.control === c.id && m.field === f.name))) {
      fs.readVal = h("span", { class: "m-val", text: "—" });
      if (mode === "row") {
        // The measured value; clicking it makes it the target (and sends it, like "Use measured").
        fs.read = h("button", { type: "button", class: "measured read", "data-testid": "read-value", disabled: true },
          h("span", { class: "m-label", text: "measured" }), fs.readVal);
        fs.read.addEventListener("click", () => { const r = readValues(cs)[f.name]; if (r && r.v != null) { setField(fs, r.v); trigger(cs); } });
        div.querySelector(".row-id").append(fs.read);
      } else {
        fs.read = h("span", { class: "measured read", "data-testid": "read-value" }, h("span", { class: "m-label", text: "measured" }), fs.readVal);
        div.querySelector(".field-top").insertBefore(fs.read, wrap);
      }
    }
    // Typing only validates; a typed value is sent when committed (Enter or leaving the field).
    // The arrow keys step it by its display resolution (Shift: ten steps) and send it.
    num.addEventListener("input", () => { touched(fs); validateField(fs); });
    num.addEventListener("change", () => { validateField(fs); trigger(cs); });
    num.addEventListener("keydown", (e) => {
      const d = { ArrowUp: 1, ArrowDown: -1 }[e.key]; if (!d) return;
      e.preventDefault();
      const cur = parseNum(num.value);
      setField(fs, (Number.isFinite(cur) ? cur : Number(f.default || 0)) + d * stepOf(f) * (e.shiftKey ? 10 : 1));
      trigger(cs);
    });
    if (range) {
      range.addEventListener("pointerdown", () => { fs.held = true; });
      range.addEventListener("change", () => { fs.held = false; });
      range.addEventListener("input", () => { setField(fs, range.value, true); trigger(cs); });
    }
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
    const numeric = c.fields.filter((f) => !f.choices);
    // A control of a profile group is one compact row of the group's joint list.
    const row = !!c.group && (t.profile.groups || []).some((x) => x.id === c.group) && numeric.length > 0 && numeric.length === c.fields.length;
    const compact = !c.fields.length;
    const div = h("div", { class: "control" + (row ? " jrow" : compact ? " compact" : "") + (reason ? " unavailable" : ""), "data-control": c.id, "data-testid": "control" });
    const reads = reason ? [] : readsOf(c.id);
    const cs = { ctrl: c, div, inputs: {}, fields: {}, available: !reason, status: null, afterBuild: [], fb: reason ? null : feedbackFor(c, t),
      reads, polled: reads.filter((x) => x.r.poll_s), group: row ? c.group : null,
      mode: sendMode(c), serial: isSerial(c), th: { pending: false, held: false, timer: null, busy: false, last: 0 },
      goal: null, goals: [], closing: new Set(), sent: 0, dead: false };
    if (!row) {
      div.append(h("div", { class: "ctl-head" },
        h("div", {}, h("h3", { text: c.label }), h("div", { class: "ctl-kind", text: kindText(c) }))));
      div.append(h("details", { class: "ctl-wire" }, h("summary", { text: "Interface" }),
        h("dl", {},
          h("dt", { text: "Name" }), h("dd", { text: t.wire(c.name) }),
          h("dt", { text: "Type" }), h("dd", { text: c.type }),
          cs.fb ? [h("dt", { text: cs.fb.measured ? "Measured" : "Reported" }), h("dd", { text: cs.fb.topic })] : null)));
    }
    if (reason) div.append(h("div", { class: "unavail", "data-testid": "unavailable", text: "unavailable: " + reason }));
    // Everything that edits or sends sits in one fieldset, disabled while the control is not live.
    cs.body = h("fieldset", { class: "ctl-body", disabled: true });
    if (row) {
      cs.body.append(buildField(c, numeric[0], cs, "row"), ...numeric.slice(1).map((f) => buildField(c, f, cs, "mini")));
    } else {
      if (c.fields.length) cs.body.append(h("div", { class: "fields" }, c.fields.map((f) => buildField(c, f, cs))));
      if (numeric.length) {
        const tools = h("div", { class: "ctl-tools" });
        const reset = h("button", { type: "button", class: "btn btn-sm btn-ghost", text: "Reset to defaults" });
        reset.addEventListener("click", () => { for (const f of numeric) if (f.default != null) setField(cs.fields[f.name], f.default); trigger(cs); });
        tools.append(reset);
        // One call of each documented read service per click; never automatic. A polled read
        // refreshes the measured values by itself, so its controls have no read button.
        const clickReads = cs.polled.length ? [] : cs.reads;
        if (clickReads.length) {
          cs.readBtn = h("button", { type: "button", class: "btn btn-sm btn-ghost", "data-testid": "read", text: "Read measured",
            title: "ask the robot once for the measured positions: " + clickReads.map((x) => x.wire).join(", ") });
          cs.readBtn.addEventListener("click", () => { for (const x of clickReads) doRead(x); });
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
        if (clickReads.length) { cs.readStatus = h("span", { class: "read-status", "data-testid": "read-status", role: "status" }); tools.append(cs.readStatus); }
        cs.body.append(tools);
      }
      if (cs.mode === "click" && !c.fields.some((f) => f.choices)) {
        cs.button = h("button", { type: "button", class: "send", "data-testid": "send", "aria-describedby": `st-${c.id}`,
          text: c.kind === "call" ? "Call" : "Send" });
        cs.button.addEventListener("click", () => fire(cs));
        cs.body.append(cs.button);
      }
    }
    div.append(cs.body);
    cs.statusEl = h("div", { class: "status dim", "data-testid": "status", id: `st-${c.id}`, role: "status" });
    div.append(cs.statusEl);
    S.controls.set(c.id, cs);
    for (const fn of cs.afterBuild) fn();
    return cs;
  }

  // A profile group: its controls as one collapsible joint list, with the group's note, how its
  // controls send, their interface and a summary that keeps every row's failed, invalid and
  // sending state visible while the group is collapsed.
  function buildGroup(gd) {
    const meta = h("span", { class: "jg-meta" }), state = h("span", { class: "jg-state", role: "status" });
    const kind = h("div", { class: "ctl-kind" }), list = h("div", { class: "jlist" });
    const dl = h("dl");
    const body = h("div", { class: "jg-body" }, gd.note ? h("p", { class: "jg-note", text: gd.note }) : null, kind, list,
      h("details", { class: "ctl-wire" }, h("summary", { text: "Interface" }), dl));
    const el = h("details", { class: "jgroup", open: true, "data-group": gd.id, "data-testid": "group" },
      h("summary", {}, h("span", { class: "jg-title", text: gd.label }), state, meta), body);
    return { id: gd.id, def: gd, el, body, kind, list, dl, meta, state, members: [] };
  }
  function finishGroup(gr, t) {
    gr.kind.textContent = [...new Set(gr.members.map((cs) => kindText(cs.ctrl)))].join(" / ") +
      " · each row is one joint; values in rad (degrees in brackets)";
    const seen = new Set();
    for (const cs of gr.members) {
      const key = cs.ctrl.name + " " + cs.ctrl.type; if (seen.has(key)) continue; seen.add(key);
      gr.dl.append(h("dt", { text: "Name" }), h("dd", { text: t.wire(cs.ctrl.name) }), h("dt", { text: "Type" }), h("dd", { text: cs.ctrl.type }));
    }
    const x = gr.members.flatMap((cs) => cs.polled)[0];
    if (x) gr.dl.append(h("dt", { text: "Measured" }), h("dd", { text: `${x.wire} (${x.r.type}), asked about every ${x.r.poll_s} s` }));
    syncGroup(gr, performance.now());
  }
  function syncGroup(gr, now) {
    const n = gr.members.length, ph = { failed: 0, rejected: 0, invalid: 0, sending: 0 };
    for (const cs of gr.members) {
      const typedBad = Object.values(cs.fields).some((fs) => fs.div.classList.contains("invalid"));
      const p = typedBad ? "invalid" : cs.status ? cs.status.phase : null;
      if (p in ph) ph[p]++;
    }
    const x = gr.members.flatMap((cs) => cs.polled)[0];
    let meta = `${n} joint${n === 1 ? "" : "s"}`;
    if (x) meta += x.last ? ` · measured ${((now - x.last.at) / 1000).toFixed(1)} s ago` : " · not measured yet";
    if (x && x.err) meta += " · last read failed";
    if (gr.meta.textContent !== meta) gr.meta.textContent = meta;
    gr.meta.title = x && x.err ? `read of ${x.wire} failed: ${x.err}` : "";
    const parts = [];
    for (const k of ["failed", "rejected", "invalid", "sending"]) if (ph[k]) parts.push(`${ph[k]} ${k}`);
    const st = parts.join(" · ");
    if (gr.state.textContent !== st) gr.state.textContent = st;
    gr.state.dataset.state = ph.failed || ph.rejected || ph.invalid ? "bad" : ph.sending ? "warn" : "";
  }

  // AiNex-style head: one pad drives both head targets; dragging it streams head_pan and
  // head_tilt like their own sliders. In a profile group the pad heads the group's joint list.
  function padPair(p) {
    const ok = (id) => { const c = p.controls.find((x) => x.id === id); return c && c.fields.some((f) => f.name === "position" && !f.choices && f.min != null && f.max != null); };
    return ok("head_pan") && ok("head_tilt") ? { pan: "head_pan", tilt: "head_tilt" } : null;
  }
  function buildPadGroup(pair, inGroup) {
    const wrap = h("div", { class: "head-group" });
    const pad = h("div", { class: "pad", tabindex: "0", role: "group",
      "aria-label": "Head pad: dragging or the arrow keys send the pan and tilt targets" });
    const dot = h("span", { class: "pad-dot" });
    const zx = h("span", { class: "pad-zero" }), zy = h("span", { class: "pad-zero h" });
    const readout = h("div", { class: "pad-readout", "aria-live": "polite" });
    pad.append(zx, zy, dot);
    const kind = h("div", { class: "ctl-kind", text: "Drag to move the head: pan and tilt are sent as you drag (≈10 per s)." });
    const card = h("div", { class: "pad-card" + (inGroup ? " in-group" : "") },
      inGroup ? kind : h("div", { class: "ctl-head" }, h("div", {}, h("h3", { text: "Head" }), kind)),
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
        const pv = parseNum(P.num.value), tv = parseNum(T.num.value);
        const okP = Number.isFinite(pv) && pv >= fp.min && pv <= fp.max, okT = Number.isFinite(tv) && tv >= ft.min && tv <= ft.max;
        dot.hidden = !(okP && okT);
        if (okP && okT) { dot.style.left = `${xOf(pv) * 100}%`; dot.style.top = `${yOf(tv) * 100}%`; }
        readout.textContent = `pan ${okP ? valueText(pv, "rad", 2) : "invalid"} · tilt ${okT ? valueText(tv, "rad", 2) : "invalid"}`;
      };
      P.listeners.push(paint); T.listeners.push(paint); paint();
      const setFrom = (ev) => {
        const r = pad.getBoundingClientRect();
        const x = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)), y = Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height));
        setField(P, fp.min + x * (fp.max - fp.min)); setField(T, ft.max - y * (ft.max - ft.min));
        trigger(PC); trigger(TC);
      };
      let drag = false;
      pad.addEventListener("pointerdown", (ev) => { if (!live()) return; drag = true; P.held = T.held = true; try { pad.setPointerCapture(ev.pointerId); } catch (e) {} setFrom(ev); });
      pad.addEventListener("pointermove", (ev) => { if (drag && live()) setFrom(ev); });
      const end = () => { drag = false; P.held = T.held = false; };
      pad.addEventListener("pointerup", end); pad.addEventListener("pointercancel", end);
      pad.addEventListener("keydown", (e) => {
        const st = e.shiftKey ? 0.2 : 0.05;
        const d = { ArrowLeft: [-st, 0], ArrowRight: [st, 0], ArrowUp: [0, st], ArrowDown: [0, -st] }[e.key];
        if (!d || !live()) return;
        e.preventDefault();
        const pv = parseNum(P.num.value) || 0, tv = parseNum(T.num.value) || 0;
        if (d[0]) setField(P, pv + d[0]);
        if (d[1]) setField(T, tv + d[1]);
        if (d[0]) trigger(PC); if (d[1]) trigger(TC);
      });
      g.sync();
    };
    return g;
  }

  function renderControls() {
    renderMotion();
    const root = $("controls"); root.innerHTML = ""; S.controls.clear(); S.padGroup = null; S.reads = new Map(); S.groups = [];
    $("workspace").classList.toggle("no-controls", !S.target || !S.target.profile.controls.length);
    if (!S.target) { root.append(h("p", { class: "empty", text: "No validated target: no controls are offered." })); renderModel(); refreshControls(); return; }
    const t = S.target, g = S.graph;
    prepareReads(t, g);
    if (!t.profile.controls.length) root.append(h("p", { class: "empty", "data-testid": "no-controls", text: "This robot's profile lists no bounded page controls." }));
    const groups = new Map((t.profile.groups || []).map((x) => [x.id, x]));
    const pair = padPair(t.profile);
    let group = null, cur = null;
    for (const c of t.profile.controls) {
      const cs = buildControl(c, t, g);
      const isPad = pair && (c.id === pair.pan || c.id === pair.tilt);
      if (cs.group) {                                           // a row of its group's joint list
        if (!cur || cur.id !== cs.group) { cur = buildGroup(groups.get(cs.group)); S.groups.push(cur); root.append(cur.el); }
        cur.list.append(cs.div); cur.members.push(cs);
        if (isPad && !group) { group = buildPadGroup(pair, true); cur.list.before(group.div); }
        continue;
      }
      cur = null;
      if (isPad) {
        if (!group) { group = buildPadGroup(pair, false); root.append(group.div); }
        group.div.append(cs.div);
      } else root.append(cs.div);
    }
    if (group) { group.init(); S.padGroup = group; }
    for (const gr of S.groups) finishGroup(gr, t);
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
  // turns the view, and an arrow on the floor marks the robot's front (+x of the root link,
  // REP 103). Each joint is posed from the joint position the robot reports where the profile
  // documents a joint-state topic for it; else from the polled measurement of the control field
  // that commands it, except while the user is changing that field (drawn at the target until a
  // measurement asked for after the change arrives); else at the page's target for it (or zero
  // when no control commands it). The page says which. A joint that a control commands can be
  // clicked and dragged, or selected and moved with the arrow keys, to change that control's
  // field: the same clamped, validated and throttled path as the control's own inputs. Clicking
  // a joint highlights its control and scrolls it into view.
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
      yaw: Math.PI - 0.7, pitch: 0.3, fit: null, selected: null, drag: null, pts: new Map(), chips: new Map(), hl: null };
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
  // The joint's name on the page: its control's label where that control commands only this
  // joint (an AiNex servo), else the commanding field's label (one joint of an arm trajectory), or,
  // where that field moves several joints (one gripper motor driving both fingers), the control's
  // label and the joint's name.
  function jointLabel(J) {
    const b = editField(J);
    if (!b) return J.name;
    const binds = (pred) => M.commanded.filter((x) => (x.command || []).some(pred)).length;
    if (binds((c) => c.control === b.cs.ctrl.id) === 1) return b.cs.ctrl.label;
    return binds((c) => c.control === b.cs.ctrl.id && c.field === b.fs.f.name) > 1 ? `${b.cs.ctrl.label} (${J.name})` : b.fs.f.label;
  }
  // The latest polled measurement of a bound field: {v, x, sentAt}, or null.
  function polledOf(b) {
    let best = null;
    for (const x of b.cs.polled || []) {
      const vals = x.last && x.last.vals[b.cs.ctrl.id];
      const v = vals ? vals[b.fs.f.name] : null;
      if (v != null && (!best || x.last.at > best.x.last.at)) best = { v, x, sentAt: x.last.sentAt };
    }
    return best;
  }

  function poseJoints(now) {
    const f = M.fb && S.feedback.get(M.fb.topic);
    const fresh = !!f && f.last != null && now - f.last < 1000 && !S.lost;
    for (const J of M.joints) {
      if (J.type === "fixed" || J.mimic) continue;
      if (fresh && J.wire && f.pos.has(J.wire)) { J.q = f.pos.get(J.wire); J.src = "reported"; continue; }
      const b = J.active || editField(J);
      if (!b) { J.q = 0; J.src = "zero"; continue; }
      const m = polledOf(b);
      // Measured, unless the user is changing this joint: then its target until a measurement
      // asked for after the change has arrived.
      if (m && pollFresh(m.x, now) && !b.fs.held && m.sentAt > b.fs.changedAt) { J.q = m.v * b.scale + b.offset; J.src = "measured"; continue; }
      const v = parseNum(b.fs.num.value);                        // an invalid entry keeps the last pose
      if (Number.isFinite(v) && v >= b.fs.f.min && v <= b.fs.f.max) J.q = v * b.scale + b.offset;
      J.src = "target";
    }
    for (const J of M.joints) {
      if (!J.mimic) continue;
      const o = M.byName.get(J.mimic.joint);
      J.q = o ? Number(J.mimic.multiplier == null ? 1 : J.mimic.multiplier) * o.q + Number(J.mimic.offset || 0) : 0;
      J.src = o ? o.src : "zero";
    }
  }
  // Each child link's frame with URDF joint semantics: the joint frame is the origin (translate by
  // xyz, then rotate by rpy) in the parent's frame; a revolute or continuous joint then turns by q
  // about its axis, a prismatic one slides q along it (both in the joint frame).
  function forward() {
    const frames = new Map([[M.m.root, { R: ID3, t: [0, 0, 0] }]]), todo = [M.m.root];
    while (todo.length) {
      const link = todo.pop(), T = frames.get(link);
      for (const J of M.kids.get(link) || []) {
        const F = { R: mm3(T.R, J.R0), t: at3(T, J.xyz) };
        if ((J.type === "revolute" || J.type === "continuous") && J.axis) F.R = mm3(F.R, axisMat(J.axis, J.q));
        else if (J.type === "prismatic" && J.axis) {
          const n = Math.hypot(J.axis[0], J.axis[1], J.axis[2]) || 1;
          F.t = at3(F, J.axis.map((x) => x * J.q / n));
        }
        J.T = F;
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
    const k = 0.47 * Math.min(w, h) / M.fit.radius, c = M.fit.center;
    return (p) => { const v = [p[0] - c[0], p[1] - c[1], p[2] - c[2]]; return [w / 2 + k * dot3(v, r), h / 2 - k * dot3(v, u), dot3(v, d)]; };
  }

  let modelQueued = false;
  function scheduleModelDraw() {
    if (!M || modelQueued) return;
    modelQueued = true;
    requestAnimationFrame(() => { modelQueued = false; drawModel(); });
  }
  const SRC_TEXT = { reported: "reported", measured: "measured", target: "drawn at the page's target",
    zero: "not reported or commanded: drawn at 0" };
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
      const R = M.fit.radius, c = M.fit.center, fl = M.fit.floor;   // floor grid under the model
      g.strokeStyle = col("--border"); g.lineWidth = 1;
      for (let i = -4; i <= 4; i++) {
        for (const [a, b] of [[[c[0] + i * R / 4, c[1] - R, fl], [c[0] + i * R / 4, c[1] + R, fl]],
          [[c[0] - R, c[1] + i * R / 4, fl], [c[0] + R, c[1] + i * R / 4, fl]]]) {
          const A = P(a), B = P(b); g.beginPath(); g.moveTo(A[0], A[1]); g.lineTo(B[0], B[1]); g.stroke();
        }
      }
      // The robot's front: +x of the root link, on the floor.
      const F0 = P([0, 0, fl]), F1 = P([R * 0.8, 0, fl]), ang = Math.atan2(F1[1] - F0[1], F1[0] - F0[0]);
      g.strokeStyle = g.fillStyle = col("--info"); g.lineWidth = 2;
      g.beginPath(); g.moveTo(F0[0], F0[1]); g.lineTo(F1[0], F1[1]); g.stroke();
      g.beginPath(); g.moveTo(F1[0], F1[1]);
      for (const s of [-1, 1]) g.lineTo(F1[0] - 10 * Math.cos(ang + s * 0.45), F1[1] - 10 * Math.sin(ang + s * 0.45));
      g.closePath(); g.fill();
      g.font = "600 11px " + col("--font"); g.fillText("front", F1[0] + 8 * Math.cos(ang) - 12, F1[1] + 8 * Math.sin(ang) + 14);
      const items = segs.map(([a, b, kind]) => { const A = P(a), B = P(b); return { z: (A[2] + B[2]) / 2, kind, A, B }; });
      const xs = items.flatMap((it) => [it.A[0], it.B[0]]), ys = items.flatMap((it) => [it.A[1], it.B[1]]);
      M.drawn = xs.length ? [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)] : null;   // the robot on the canvas
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
          g.fillStyle = col("--ink"); g.strokeStyle = col(sel ? "--focus" : J.src === "reported" || J.src === "measured" ? "--ok" : "--warn");
          g.lineWidth = 3; g.beginPath(); g.arc(it.A[0], it.A[1], sel ? 8 : 6, 0, 2 * Math.PI); g.fill(); g.stroke();
          if (sel) {
            const txt = `${jointLabel(J)} ${valueText(J.q, J.type === "prismatic" ? "m" : "rad", 2)}`;
            g.font = "600 12.5px " + col("--font");
            const tw = g.measureText(txt).width, x = Math.min(it.A[0] + 12, w - tw - 8), y = Math.max(it.A[1] - 12, 16);
            g.fillStyle = col("--surface"); g.globalAlpha = 0.85; g.fillRect(x - 4, y - 13, tw + 8, 18); g.globalAlpha = 1;
            g.fillStyle = col("--text"); g.fillText(txt, x, y);
          }
        }
      }
    }
    syncModelText(now);
  }
  function syncModelText(now) {
    const f = M.fb && S.feedback.get(M.fb.topic), x = S.poll;
    const n = { reported: 0, measured: 0, target: 0, zero: 0 };
    for (const J of M.joints) if (J.type !== "fixed" && !J.mimic) n[J.src]++;
    const why = M.fb ? (f && f.last != null ? `no fresh ${M.fb.topic}` : `no ${M.fb.topic} yet`)
      : x ? (S.lost ? "connection lost"
        : x.last && !Object.values(x.last.vals).some((o) => Object.values(o).some((v) => v != null)) ? `${x.wire} answered no positions`
        : x.last ? `the last measurement is ${((now - x.last.at) / 1000).toFixed(1)} s old`
        : x.err ? `reading ${x.wire} failed: ${x.err}` : `waiting for the first ${x.wire} answer`)
      : "the profile documents no joint-position topic";
    const rest = [];
    if (n.target) rest.push(n.measured || n.reported ? `${n.target} joint${n.target > 1 ? "s" : ""} at the target you are changing`
      : `${n.target} commanded joint${n.target > 1 ? "s" : ""} drawn at the page's targets`);
    if (n.zero) rest.push(`${n.zero} other joint${n.zero > 1 ? "s" : ""} at 0`);
    let text;
    if (n.reported) {
      text = `Pose reported on ${M.fb.topic}` + (M.fb.measured ? "" : " (what the robot publishes, not a measurement)") +
        (rest.length ? `; not reported: ${rest.join(", ")}.` : ".");
    } else if (n.measured) {
      text = `Pose measured ${((now - x.last.at) / 1000).toFixed(1)} s ago by ${x.wire} (asked about every ${x.r.poll_s} s)` +
        (rest.length ? `; ${rest.join(", ")}.` : ".");
    } else text = `Pose not ${x && !M.fb ? "measured" : "reported"} (${why}): ${rest.join(", ") || "no movable joints"}.`;
    const pose = $("model-pose");
    if (pose.textContent !== text) pose.textContent = text;
    pose.dataset.state = (n.reported || n.measured) && !n.target ? "reported" : n.reported || n.measured ? "partial" : "not-reported";
    for (const J of M.commanded) {
      const chip = M.chips.get(J.name), b = editField(J);
      const d = J.type === "prismatic" ? 3 : 2;                    // m (a finger's travel is centimetres) or rad
      const txt = `${jointLabel(J)} ${signed(Math.round(J.q * 10 ** d) / 10 ** d || 0, d)}`;
      if (chip.firstChild.textContent !== txt) chip.firstChild.textContent = txt;
      chip.dataset.src = J.src; chip.setAttribute("aria-pressed", String(M.selected === J.name));
      chip.disabled = !(b && liveOk(b.cs));
      chip.title = `${J.name}: ${SRC_TEXT[J.src]}` + (b ? `; commanded by ${b.cs.ctrl.label} › ${b.fs.f.label}` : "");
    }
    const J = M.selected && M.byName.get(M.selected), sel = $("model-sel");
    let st = "";
    if (J) {
      const b = editField(J);
      st = `${jointLabel(J)} (${J.name}): ${valueText(J.q, J.type === "prismatic" ? "m" : "rad", 3)}, ${SRC_TEXT[J.src]}.`;
      if (b) st += ` Commands ${b.cs.ctrl.label} › ${b.fs.f.label} = ${b.fs.num.value} ${b.fs.f.unit}` +
        (liveOk(b.cs) ? "; drag the joint or use the arrow keys to change it." : " (control not live).");
    }
    if (sel.textContent !== st) sel.textContent = st;
    // For tests and tools: the pose drawn, where each joint came from, each joint origin in the
    // root frame, the commanded joints' positions on the canvas (CSS px from its top-left) and
    // their labels on the page.
    S.model = { rendered: true, selected: M.selected, joints: {}, sources: {}, world: {}, clickable: M.commanded.map((y) => y.name),
      points: {}, labels: {}, measured_age: x && x.last ? (now - x.last.at) / 1000 : null, drawn: M.drawn || null };
    for (const y of M.joints) {
      if (y.T) S.model.world[y.name] = y.T.t;
      if (y.type !== "fixed") { S.model.joints[y.name] = y.q; S.model.sources[y.name] = y.src; }
    }
    for (const y of M.commanded) S.model.labels[y.name] = jointLabel(y);
    for (const [name, s] of M.pts) S.model.points[name] = [s[0], s[1]];
  }

  // Select joint `name` in the model and highlight the control (or field) that commands it;
  // `reveal` also scrolls it into view: in the wide layout only the controls panel scrolls, so the
  // model stays under the pointer.
  function selectJoint(name, reveal) {
    if (!M) return;
    M.selected = name;
    const J = name && M.byName.get(name), b = J && editField(J);
    const el = b ? (b.cs.group ? b.cs.div : b.fs.div) : null;
    if (M.hl && M.hl !== el) M.hl.classList.remove("hl");
    M.hl = el;
    if (el) { el.classList.add("hl"); if (reveal) revealControl(el); }
    drawModel();
  }
  function revealControl(el) {
    const grp = el.closest("details.jgroup"); if (grp && !grp.open) grp.open = true;
    const panel = $("control-panel"), cs = getComputedStyle(panel);
    const behavior = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
    if (cs.position !== "sticky") { el.scrollIntoView({ block: "nearest", behavior }); return; }
    const r = el.getBoundingClientRect(), pr = panel.getBoundingClientRect();
    const top = Math.max(pr.top, 0), bottom = Math.min(pr.bottom, window.innerHeight);
    if (r.top < top || r.bottom > bottom) panel.scrollBy({ top: r.top - top - Math.max(0, (bottom - top - r.height) / 2), behavior });
  }
  // Set joint J's field to `v` (field units) through the control's own path; false if not live.
  function setJointField(J, b, v) {
    if (!b || !liveOk(b.cs) || !Number.isFinite(v)) return false;
    setField(b.fs, v); J.active = b; trigger(b.cs); drawModel();
    return true;
  }
  function nudgeJoint(J, n) {
    const b = editField(J); if (!b) return;
    setJointField(J, b, parseNum(b.fs.num.value) + n * (b.fs.f.max - b.fs.f.min) / 100);
  }
  function renderModel() {
    M = S.target ? buildModel(S.target) : null;
    S.model = null;
    $("model-view").hidden = !M; $("model-empty").hidden = !!M;
    $("model-empty").textContent = S.target ? "This robot's profile has no model." : "No validated target: no model is shown.";
    const list = $("model-joints"); list.innerHTML = "";
    if (!M) { $("model-pose").textContent = ""; $("model-sel").textContent = ""; $("model-joints-wrap").hidden = true; return; }
    $("model-joints-wrap").hidden = !M.commanded.length;
    $("model-joints-wrap").open = M.commanded.length <= 8;
    $("model-joints-count").textContent = `Joints the controls command (${M.commanded.length})`;
    for (const J of M.commanded) {
      const chip = h("button", { type: "button", class: "joint-chip", "data-joint": J.name, "aria-pressed": "false" }, h("span"));
      chip.addEventListener("click", () => selectJoint(J.name, true));
      chip.addEventListener("keydown", (e) => {
        const k = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[e.key]; if (!k) return;
        e.preventDefault(); M.selected = J.name; nudgeJoint(J, k * (e.shiftKey ? 5 : 1));
      });
      M.chips.set(J.name, chip); list.append(chip);
      for (const b of J.command) {
        const x = bound(b); if (!x) continue;
        x.fs.listeners.push(() => { J.active = x; scheduleModelDraw(); });   // the field changed last is the one shown
        // Working on a joint's control selects that joint in the model.
        x.fs.div.addEventListener("focusin", () => { if (M && M.selected !== J.name && M.byName.get(J.name) === J) selectJoint(J.name, false); });
      }
    }
    if (M.fb) subscribeFeedback(M.fb.topic, M.fb.type);
    drawModel();
  }
  (function wireModelCanvas() {
    const cv = $("model-canvas");
    const xy = (ev) => { const r = cv.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; };
    // A press on a joint picks the selected joint if it is under the pointer, else the nearest;
    // a click without moving on joints drawn on top of each other picks the next of them. A
    // click (no drag) also brings the joint's control into view.
    cv.addEventListener("pointerdown", (ev) => {
      if (!M) return;
      const [x, y] = xy(ev);
      const near = [...M.pts].map(([name, s]) => [Math.hypot(s[0] - x, s[1] - y), name]).filter(([d]) => d < 14)
        .sort((a, b) => a[0] - b[0]).map(([, name]) => name);
      try { cv.setPointerCapture(ev.pointerId); } catch (e) {}
      ev.preventDefault(); cv.focus({ preventScroll: true });
      if (near.length) {
        const prev = near.includes(M.selected) ? M.selected : null, J = M.byName.get(prev || near[0]), b = editField(J);
        M.drag = { J, b, x, y, v0: b ? parseNum(b.fs.num.value) : NaN, near, prev, moved: false };
        if (b) b.fs.held = true;
        selectJoint(J.name, false);
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
      if (d.b) d.b.fs.held = false;
      if (d.J && !d.moved) selectJoint(d.prev && d.near.length > 1 ? d.near[(d.near.indexOf(d.prev) + 1) % d.near.length] : d.J.name, true);
    });
    const end = () => { if (M && M.drag) { if (M.drag.b) M.drag.b.fs.held = false; M.drag = null; } };
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

  // ------------------------------------------------------------------ base motion
  // teleop.sh's base motion and AiNex head keys (console spec §2.1, §2.2),
  // for a teleoperable profile only. The contract is the profile's `teleop` section, built by
  // teleop_core.page_contract from the same profile data teleop uses (§3), and the state machine
  // below is teleop_core.TeleopCore's: held keys commanded at their recorded speed within the
  // documented limits (opposite keys cancel), re-sent at the profile's rate while held; the AiNex
  // walks through its gait (start ops once, then the parameters); release of the last motion
  // input, Space, the Stop button, focus loss, page hide and a target change clear the intent and
  // request the documented stop; an explicitly failed stop (the transport rejects or cannot send
  // it, or the stop service errs or times out; a publish is never acknowledged, so never fails)
  // disables base motion until an explicit re-enable and fresh input; connection loss clears the
  // intent and says that no stop could be delivered. Events are handled one at a time, in order,
  // each finishing its sends (and their answers) before the next, as teleop's single loop does.
  // Two deliberate differences, both from the page spec: teleop's walking enable calls (sent by
  // teleop right after a delivered start-up stop) go out with the first fresh motion press
  // instead, because the page sends nothing unprompted but the start-up stop; and keys act only
  // while the page, not an editable field, has keyboard focus, with on-page hold buttons for
  // pointer and touch input. Bounded controls are never stopped or cancelled.
  const nowS = () => performance.now() / 1000;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const KEY_OF = { " ": "space", Enter: "enter", ArrowLeft: "left", ArrowRight: "right", ArrowUp: "up", ArrowDown: "down" };
  const keyName = (e) => KEY_OF[e.key] || (/^[a-z]$/i.test(e.key || "") ? e.key.toLowerCase() : null);
  const isEditable = (el) => !!el && (el.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName));
  // Elements with arrow-key behaviour of their own: the head keys leave them alone.
  const ownsArrows = (el) => !!(el && el.closest && el.closest(".joint-chip, #model-canvas, .pad, [role=radiogroup], .choices"));

  function makeMotion(t, conn) {
    const c = t.profile.teleop;
    return { t, conn, c, enabled: false, connected: true, held: [], heldHead: [], walking: false,
      enablePending: !!(c.walk && c.walk.enable.length), lastCmd: null, lastSentAt: 0,
      headPos: {}, headLastAt: null, headSent: {}, headSendAt: {}, log: [], outcomes: [],
      chain: Promise.resolve(), pending: 0, stopping: false, started: false, leaving: false, dead: false, ui: null };
  }
  // Run `fn` after every earlier event (teleop's loop handles one event at a time).
  function moQueue(mo, fn) {
    mo.pending++;
    const p = mo.chain.then(() => (mo.dead || !mo.connected ? undefined : fn()))
      .catch((e) => { console.error(e); })
      .finally(() => { mo.pending--; syncMotion(); });
    mo.chain = p;
    return p;
  }
  function moLog(mo, text) {
    mo.log.push({ text, at: Date.now() });
    mo.log.splice(0, Math.max(0, mo.log.length - 12));
    syncMotion();
  }
  const heldKeys = (list) => new Set(list.map((x) => x.key));
  function moAxes(mo) {
    const h = heldKeys(mo.held), out = {};
    for (const [pos, neg, axis] of mo.c.axis_keys) {
      const a = mo.c.axes[axis];
      if (!a) { out[axis] = 0; continue; }
      const v = a.speed * ((h.has(pos) ? 1 : 0) - (h.has(neg) ? 1 : 0));
      out[axis] = Math.max(-a.limit, Math.min(a.limit, v));
    }
    return out;
  }
  function moCommand(mo) {
    const a = moAxes(mo);
    if (mo.c.base) return { linear: { x: a.x, y: a.y, z: 0.0 }, angular: { x: 0.0, y: 0.0, z: a.yaw } };
    const msg = Object.assign({}, mo.c.walk.param_template);
    for (const [axis, spec] of Object.entries(mo.c.axes)) msg[spec.field] = a[axis];
    return msg;
  }
  // Documented ops: a publish is sent (never acknowledged), a call must answer within the
  // timeout and not report failure (teleop_core._run_ops).
  async function moRunOps(mo, ops, param) {
    const texts = [];
    for (const o of ops) {
      if (o.op === "publish") {
        mo.conn.publish(mo.t.wire(o.name), o.msg === "$param" ? param : o.msg, o.type);
        texts.push(`published ${o.name} (no acknowledgement)`);
      } else if (o.op === "call") {
        const v = await mo.conn.call(mo.t.wire(o.name), o.msg, mo.c.call_timeout_s * 1000);
        if (v && typeof v === "object" && (v.result === false || v.success === false)) throw new Error(`${o.name} answered ${JSON.stringify(v)}`);
        texts.push(`${o.name} answered`);
      }
    }
    return texts.join("; ");
  }
  async function moSendMotion(mo) {
    if (!mo.held.length) return;
    const cmd = moCommand(mo);
    try {
      if (mo.c.base) mo.conn.publish(mo.t.wire(mo.c.base.topic), cmd, mo.c.base.type);
      else if (!mo.walking) { await moRunOps(mo, mo.c.walk.start, cmd); mo.walking = true; }
      else mo.conn.publish(mo.t.wire(mo.c.walk.param_topic), cmd, mo.c.walk.param_type);
    } catch (e) {
      moLog(mo, `motion command failed: ${e.message}`);
      await moRequestStop(mo, "motion command failed");
      return;
    }
    mo.lastCmd = cmd; mo.lastSentAt = nowS();
  }
  function moDisabledReason(mo) {
    if (!mo.connected) return "base motion is disabled: not connected";
    if (!mo.outcomes.length) return "base motion is not enabled yet: waiting for the start-up stop";
    if (!mo.outcomes[mo.outcomes.length - 1].ok) return "base motion is disabled after a failed stop: press Re-enable (or Enter), then press the keys again";
    return "base motion is disabled: press Re-enable (or Enter) to enable it";
  }
  async function moKeyDown(mo, key, src) {
    if (key === "space") return moClearAndStop(mo, "Space");
    if (key === "enter") return moEnable(mo);
    if (mo.c.motion_keys.includes(key)) {
      if (!mo.enabled) { moLog(mo, moDisabledReason(mo)); return; }
      if (mo.held.some((x) => x.key === key)) return;          // held already: not a fresh press
      if (mo.enablePending && !(await moWalkEnable(mo))) return;
      mo.held.push({ key, src });
      await moSendMotion(mo);
    } else if (mo.c.head && mo.c.head_keys.includes(key)) {
      if (!mo.enabled) { moLog(mo, moDisabledReason(mo)); return; }
      if (mo.heldHead.some((x) => x.key === key)) return;
      const axis = mo.c.head_axis[key];
      if (!mo.c.head[axis]) return;
      if (!mo.heldHead.some((x) => mo.c.head_axis[x.key] === axis) && !(await moReadHead(mo, axis))) return;
      mo.heldHead.push({ key, src });
      mo.headLastAt = nowS();
    }
  }
  async function moKeyUp(mo, src) {
    const i = mo.held.findIndex((x) => x.src === src);
    if (i >= 0) {
      mo.held.splice(i, 1);
      if (mo.held.length) await moSendMotion(mo);
      else await moRequestStop(mo, `last motion ${src.startsWith("k") ? "key" : "button"} released`);
      return;
    }
    const j = mo.heldHead.findIndex((x) => x.src === src);
    if (j >= 0) mo.heldHead.splice(j, 1);                       // the head target stops changing
  }
  // Inputs that stop acting (the keys once an editable field has focus) are released.
  async function moRelease(mo, pred, reason) {
    const had = mo.held.length;
    mo.held = mo.held.filter((x) => !pred(x.src)); mo.heldHead = mo.heldHead.filter((x) => !pred(x.src));
    if (had && !mo.held.length) await moRequestStop(mo, reason);
    else if (had !== mo.held.length) await moSendMotion(mo);
  }
  function moClearIntent(mo) { mo.held = []; mo.heldHead = []; }
  function moClearAndStop(mo, reason) { moClearIntent(mo); return moRequestStop(mo, reason); }
  // The walking enable teleop sends when it enables commands (TeleopCore.enable).
  async function moWalkEnable(mo) {
    try { await moRunOps(mo, mo.c.walk.enable); }
    catch (e) { mo.enabled = false; moClearIntent(mo); moLog(mo, `enable failed: ${e.message}`); return false; }
    mo.enablePending = false;
    return true;
  }
  async function moEnable(mo) {
    if (!mo.connected) { moLog(mo, "cannot enable: not connected"); return; }
    if (mo.enabled) return;
    moClearIntent(mo);                                          // only fresh presses after re-enabling count
    if (mo.c.walk && mo.c.walk.enable.length) {
      try { await moRunOps(mo, mo.c.walk.enable); }
      catch (e) { moLog(mo, `enable failed: ${e.message}`); return; }
      mo.enablePending = false;
    }
    mo.enabled = true;
    moLog(mo, "base motion ENABLED: hold the motion keys or buttons to move");
  }
  // Attempt the documented stop; an explicit failure disables base motion.
  async function moRequestStop(mo, reason) {
    moClearIntent(mo);
    mo.lastCmd = null; mo.stopping = true; syncMotion();
    let out;
    try {
      const text = await moRunOps(mo, mo.c.stop);
      mo.walking = false;
      out = { ok: true, text: `stop requested (${reason}): ${text}` };
    } catch (e) {
      mo.enabled = false;
      out = { ok: false, text: `STOP FAILED (${reason}): ${e.message}. Base motion disabled; press Re-enable (or Enter) and ` +
        `press the keys again to continue. ${mo.c.reenable_limitation}` };
    }
    mo.stopping = false;
    mo.outcomes.push(out); mo.outcomes.splice(0, Math.max(0, mo.outcomes.length - 20));
    moLog(mo, out.text);
    return out;
  }
  // Once the target is validated: advertise teleop's command topics, let a fresh ROS 1 publisher
  // connect, attempt the documented stop once and show its outcome; a delivered stop enables
  // base motion at once (teleop.py), a failed one leaves it disabled until Re-enable.
  async function moStartup(mo) {
    for (const [topic, type] of mo.c.command_topics) { try { mo.conn.advertise(mo.t.wire(topic), type); } catch (e) {} }
    await sleep(mo.c.advertise_settle_s * 1000);
    if (mo.dead || !mo.connected) return;
    const out = await moRequestStop(mo, "start-up, after validation");
    mo.started = true;
    if (out.ok) { mo.enabled = true; moLog(mo, "base motion ENABLED: hold the motion keys or buttons to move"); }
    else moLog(mo, "press Re-enable (or Enter) to enable base motion");
  }
  function moTick(mo) {
    const now = nowS();
    if (!(mo.enabled && mo.connected)) return undefined;
    let p;
    if (mo.held.length && now - mo.lastSentAt >= 1 / mo.c.rate_hz) p = moSendMotion(mo);
    return Promise.resolve(p).then(() => {
      if (mo.heldHead.length && mo.c.head) {
        const t = nowS(), dt = t - (mo.headLastAt == null ? t : mo.headLastAt);
        mo.headLastAt = t;
        const h = heldKeys(mo.heldHead);
        for (const [axis, hd] of Object.entries(mo.c.head)) {
          const [pos, neg] = mo.c.head_dirs[axis] || [];
          const d = (h.has(pos) ? 1 : 0) - (h.has(neg) ? 1 : 0);
          if (!d) continue;
          mo.headPos[axis] = Math.max(hd.min, Math.min(hd.max, mo.headPos[axis] + d * hd.rate * dt));
          const last = mo.headSent[axis];
          if (last == null || (Math.abs(last - mo.headPos[axis]) > 1e-6 && t - (mo.headSendAt[axis] || 0) >= mo.c.head_period_s)) moSendHead(mo, axis);
        }
      } else if (mo.headLastAt != null) mo.headLastAt = null;
    });
  }
  // Start a head axis from the head's measured position (the profile's documented read), read
  // when its first arrow key is pressed; while the position is unknown the key is ignored.
  async function moReadHead(mo, axis) {
    const hd = mo.c.head[axis], hr = hd.read;
    let pos = null;
    try {
      const r = hr && (mo.t.profile.reads || []).find((x) => x.id === hr.read);
      if (!r) throw new Error("the profile documents no position read");
      const v = await mo.conn.call(mo.t.wire(r.service), r.request || {}, mo.c.call_timeout_s * 1000);
      const vals = parseRead(r, v);
      pos = vals[hr.control] ? vals[hr.control][hr.field] : null;
      if (pos == null) throw new Error(`${r.service} reported no position for it`);
    } catch (e) { moLog(mo, `head position unknown (${axis}): ${e.message}; key ignored`); return false; }
    mo.headPos[axis] = Math.max(hd.min, Math.min(hd.max, pos));
    return true;
  }
  function moSendHead(mo, axis) {
    const hd = mo.c.head[axis], msg = Object.assign({}, hd.template);
    msg[hd.field] = Math.round(mo.headPos[axis] * 1e4) / 1e4;
    try { mo.conn.publish(mo.t.wire(hd.topic), msg, hd.type); }
    catch (e) { moLog(mo, `head command failed: ${e.message}`); return; }
    mo.headSent[axis] = mo.headPos[axis]; mo.headSendAt[axis] = nowS();
    // The page's own head target (its slider and the head pad) shows what the keys sent.
    for (const cs of S.controls.values()) {
      if (cs.dead || cs.ctrl.kind !== "publish" || cs.ctrl.name !== hd.topic || !cs.fields[hd.field]) continue;
      setField(cs.fields[hd.field], msg[hd.field]);
    }
  }
  // Page hide or unload: the page may be gone before an answer, so the stop is sent at once,
  // best effort, and is never confirmed.
  function moUnload(mo, reason) {
    if (!mo || mo.dead || !mo.connected) return;
    moClearIntent(mo); mo.enabled = false;
    try {
      for (const o of mo.c.stop) {
        if (o.op === "publish") mo.conn.publish(mo.t.wire(o.name), o.msg, o.type);
        else if (o.op === "call") mo.conn.call(mo.t.wire(o.name), o.msg, mo.c.call_timeout_s * 1000).catch(() => {});
      }
      moLog(mo, `stop sent (${reason}), not confirmed: a stop when the page is hidden or closed is not guaranteed`);
    } catch (e) { moLog(mo, `STOP NOT SENT (${reason}): ${e.message}. ${mo.c.no_delivery}`); }
  }
  function moConnectionLost(mo, reason) {
    if (!mo || mo.dead || !mo.connected) return;
    mo.connected = false; mo.enabled = false; moClearIntent(mo);
    moLog(mo, `CONNECTION LOST (${reason}). ${mo.c.no_delivery}`);
  }
  // A target change: stop the old target before leaving it; resolves with the outcome (or null).
  async function moLeave(mo, reason) {
    if (!mo || mo.dead) return null;
    mo.leaving = true; syncMotion();
    const out = mo.connected ? await moQueue(mo, () => moClearAndStop(mo, reason)) : null;
    mo.dead = true;
    return out || null;
  }

  // The state the page shows (and tests read): moving, stopped, stop failed, no connection...
  function motionState(mo) {
    if (!mo.connected) return ["lost", "No connection", "No stop could be delivered. " + mo.c.no_delivery];
    if (mo.leaving) return ["stopping", "Leaving this target", "stopping it first…"];
    if (!mo.started) return ["startup", "Starting", "sending the start-up stop…"];
    const last = mo.outcomes[mo.outcomes.length - 1];
    if (!mo.enabled && last && !last.ok) return ["failed", "Stop failed: base motion disabled", last.text];
    if (!mo.enabled) return ["disabled", "Disabled", moDisabledReason(mo)];
    if (mo.stopping) return ["stopping", "Stopping", "the documented stop is on its way…"];
    if (mo.held.length) return ["moving", "Moving", `held: ${mo.held.map((x) => x.key.toUpperCase()).join(" ")}`];
    if (mo.heldHead.length) return ["head", "Moving the head", `held: ${mo.heldHead.map((x) => x.key).join(" ")}`];
    return ["stopped", "Stopped", last ? last.text : ""];
  }
  RC.motion = () => {
    const mo = S.mo;
    if (!mo || mo.dead) return null;
    const [state] = motionState(mo);
    return { state, enabled: mo.enabled, connected: mo.connected, started: mo.started, walking: mo.walking,
      held: mo.held.map((x) => x.key), heldHead: mo.heldHead.map((x) => x.key), axes: moAxes(mo), headPos: Object.assign({}, mo.headPos),
      outcomes: mo.outcomes.map((o) => Object.assign({}, o)), log: mo.log.map((l) => l.text), pending: mo.pending };
  };

  const AXIS_TEXT = { x: ["forward", "back"], y: ["strafe left", "strafe right"], yaw: ["rotate left (counter-clockwise)", "rotate right (clockwise)"] };
  const KEY_GLYPH = { w: "↑", s: "↓", a: "←", d: "→", q: "↺", e: "↻", up: "↑", down: "↓", left: "←", right: "→" };
  const num3 = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(3);
  function axisText(c, axis) {
    const a = c.axes[axis]; if (!a) return "";
    const what = c.walk ? `${a.field} ` : "";
    return `${what}${exact(a.speed)} ${a.unit} (documented limit ${exact(a.limit)} ${a.unit})`;
  }
  function renderMotion() {
    const card = $("motion-card"), mo = S.mo;
    card.innerHTML = "";
    card.hidden = !mo || mo.dead;
    $("workspace").classList.toggle("has-motion", !card.hidden);
    if (card.hidden) return;
    const c = mo.c, walk = !!c.walk;
    card.dataset.kind = c.kind;
    const ui = (mo.ui = { holds: new Map() });
    const hold = (key, title, extra) => {
      const b = h("button", { type: "button", class: "hold" + (extra ? " " + extra : ""), "data-key": key, "data-testid": `hold-${key}`,
        tabindex: "-1", title, "aria-label": title, "data-held": "false" },
      h("span", { class: "glyph", "aria-hidden": "true", text: KEY_GLYPH[key] }), h("kbd", { text: key.length === 1 ? key.toUpperCase() : KEY_GLYPH[key] }));
      wireHold(b, key);
      ui.holds.set(key, b);
      return b;
    };
    const axisTitle = (key) => {
      for (const [pos, neg, axis] of c.axis_keys) {
        if (key === pos || key === neg) return `${key.toUpperCase()}: ${AXIS_TEXT[axis][key === pos ? 0 : 1]}, ${axisText(c, axis)}; hold to move`;
      }
      return key;
    };
    ui.stop = h("button", { type: "button", class: "stop-btn", "data-testid": "stop", title: "Stop: request the documented stop (Space)" },
      h("span", { text: "STOP" }), h("kbd", { text: "Space" }));
    ui.stop.addEventListener("pointerdown", (ev) => { if (ev.button === 0 || ev.pointerType !== "mouse") stopButton(); });
    ui.stop.addEventListener("click", (ev) => { if (ev.detail === 0) stopButton(); });    // keyboard activation
    const dpad = h("div", { class: "dpad", role: "group", "aria-label": "Base motion: hold a button to move" },
      hold("q", axisTitle("q")), hold("w", axisTitle("w")), hold("e", axisTitle("e")),
      hold("a", axisTitle("a")), ui.stop, hold("d", axisTitle("d")),
      h("span"), hold("s", axisTitle("s")), h("span"));
    const pads = h("div", { class: "motion-pads" }, h("div", { class: "pad-col" }, h("div", { class: "pad-cap", text: walk ? "Walk" : "Drive" }), dpad));
    if (c.head) {
      const ht = (key) => {
        const axis = c.head_axis[key], hd = c.head[axis];
        const dir = { left: "pan left (the robot's left)", right: "pan right", up: "tilt up", down: "tilt down" }[key];
        return `${KEY_GLYPH[key]}: head ${dir}, ${exact(hd.rate)} ${hd.unit}/s within ${exact(hd.min)}…${exact(hd.max)} ${hd.unit}; hold to move`;
      };
      pads.append(h("div", { class: "pad-col" }, h("div", { class: "pad-cap", text: "Head" }),
        h("div", { class: "dpad head", role: "group", "aria-label": "Head: hold a button to move it" },
          h("span"), hold("up", ht("up"), "small"), h("span"),
          hold("left", ht("left"), "small"), h("span"), hold("right", ht("right"), "small"),
          h("span"), hold("down", ht("down"), "small"), h("span"))));
    }
    ui.dot = h("span", { class: "mo-dot", "aria-hidden": "true" });
    ui.title = h("span", { class: "mo-title" }); ui.detail = h("span", { class: "mo-detail" });
    ui.state = h("div", { class: "mo-state", "data-testid": "motion-state", role: "status" }, ui.dot, h("span", { class: "mo-text" }, ui.title, ui.detail));
    ui.cmd = h("div", { class: "mo-cmd mono", "data-testid": "motion-cmd" });
    ui.reenable = h("button", { type: "button", class: "btn btn-sm", "data-testid": "reenable", hidden: true, text: "Re-enable base motion" });
    ui.reenable.addEventListener("click", () => reenable());
    ui.limit = h("p", { class: "mo-limit", "data-testid": "motion-limit", hidden: true, text: c.reenable_limitation });
    const legend = h("dl", { class: "mo-legend" });
    const row = (k, v) => legend.append(h("dt", {}, ...k.map((x) => h("kbd", { text: x }))), h("dd", { text: v }));
    for (const [pos, neg, axis] of c.axis_keys) if (c.axes[axis]) row([pos.toUpperCase(), neg.toUpperCase()], `${AXIS_TEXT[axis].join(" / ")}: ${axisText(c, axis)}`);
    if (c.head) {
      row(["←", "→"], `head pan left / right, ${exact(c.head.pan.rate)} ${c.head.pan.unit}/s within ${exact(c.head.pan.min)}…${exact(c.head.pan.max)} ${c.head.pan.unit}`);
      row(["↑", "↓"], `head tilt up / down, ${exact(c.head.tilt.rate)} ${c.head.tilt.unit}/s within ${exact(c.head.tilt.min)}…${exact(c.head.tilt.max)} ${c.head.tilt.unit}`);
    }
    row(["Space"], "stop (also the STOP button)");
    row(["Enter"], "re-enable after a failed stop (also the Re-enable button)");
    ui.log = h("ol", { class: "mo-log", "data-testid": "motion-log" });
    const wire = h("dl", {},
      h("dt", { text: "Command" }), h("dd", { text: c.base ? `${mo.t.wire(c.base.topic)} (${c.base.type}), ${c.rate_hz} per s while held`
        : `${mo.t.wire(c.walk.param_topic)} (${c.walk.param_type}), ${c.rate_hz} per s while held, after ${c.walk.start.map((o) => o.op === "call" ? `${o.name} ${JSON.stringify(o.msg)}` : o.name).join(", ")}` }),
      walk && c.walk.enable.length ? [h("dt", { text: "Enable" }), h("dd", { text: c.walk.enable.map((o) => `${o.name} ${JSON.stringify(o.msg)}`).join(", ") + " (with the first press after enabling)" })] : null,
      h("dt", { text: "Stop" }), h("dd", { text: c.stop_description }),
      h("dt", { text: "No stop" }), h("dd", { text: c.no_delivery }));
    const info = h("div", { class: "motion-info" }, ui.state, ui.cmd, h("div", { class: "mo-actions" }, ui.reenable), ui.limit,
      h("p", { class: "mo-note" }, h("strong", { text: "Hold to move; " }),
        "releasing, Space, STOP, leaving the window, hiding the page or changing the target request the documented stop. " +
        "Keys act while this page, not a number or text field, list or slider, has keyboard focus. A stop when the page is " +
        "hidden or closed is sent if possible but not guaranteed, and a lost connection delivers none."),
      legend,
      h("details", { class: "mo-logwrap", open: true }, h("summary", { text: "Motion log" }), ui.log),
      h("details", { class: "ctl-wire" }, h("summary", { text: "Interface" }), wire));
    card.append(h("div", { class: "panel-head" },
      h("h2", { id: "motion-h", class: "section-title", text: walk ? "Walking" : "Base motion" }),
      h("span", { class: "panel-meta", text: `${mo.t.label} · hold to move, release to stop` })),
    h("div", { class: "motion-body" }, pads, info));
    syncMotion();
  }
  function syncMotion() {
    const mo = S.mo, ui = mo && mo.ui;
    if (!ui || mo.dead) return;
    const [state, title, detail] = motionState(mo);
    const card = $("motion-card");
    card.dataset.state = state;
    if (ui.title.textContent !== title) ui.title.textContent = title;
    if (ui.detail.textContent !== detail) ui.detail.textContent = detail;
    ui.state.dataset.state = state;
    const a = moAxes(mo), u = (k) => (mo.c.axes[k] ? mo.c.axes[k].unit : "");
    const cmd = mo.held.length ? `x ${num3(a.x)} ${u("x")} · y ${num3(a.y)} ${u("y")} · yaw ${num3(a.yaw)} ${u("yaw")}` : "";
    if (ui.cmd.textContent !== cmd) ui.cmd.textContent = cmd;
    const live = mo.connected && mo.enabled && !mo.leaving;
    const held = new Set([...mo.held, ...mo.heldHead].map((x) => x.key));
    for (const [key, b] of ui.holds) {
      b.classList.toggle("off", !live); b.setAttribute("aria-disabled", String(!live));
      b.dataset.held = String(held.has(key));
    }
    ui.stop.disabled = !mo.connected || mo.leaving;
    ui.reenable.hidden = !(mo.connected && mo.started && !mo.enabled && !mo.leaving);
    ui.limit.hidden = !(state === "failed");
    const n = mo.log.length;
    if (ui.log.dataset.n !== String(n) || ui.log.dataset.last !== String(n && mo.log[n - 1].at)) {
      ui.log.dataset.n = String(n); ui.log.dataset.last = String(n && mo.log[n - 1].at);
      ui.log.innerHTML = "";
      for (const l of mo.log.slice(-8).reverse()) {
        const bad = /FAIL|LOST|NOT /.test(l.text);
        ui.log.append(h("li", { class: bad ? "bad" : "", text: `${new Date(l.at).toLocaleTimeString()}  ${l.text}` }));
      }
    }
  }

  // ---- input: keys while the page (not an editable field) has focus, hold buttons for pointers
  const usable = () => { const mo = S.mo; return mo && !mo.dead && !mo.leaving && mo.connected ? mo : null; };
  function press(key, src) { const mo = usable(); if (mo) moQueue(mo, () => moKeyDown(mo, key, src)); }
  function release(src) { const mo = S.mo; if (mo && !mo.dead && mo.connected) moQueue(mo, () => moKeyUp(mo, src)); }
  function stopNow(reason) { const mo = usable(); if (mo) moQueue(mo, () => moClearAndStop(mo, reason)); }
  const stopButton = () => stopNow("Stop button");
  function reenable() { const mo = usable(); if (mo) moQueue(mo, () => moEnable(mo)); }
  function wireHold(b, key) {
    b.addEventListener("pointerdown", (ev) => {
      if (ev.pointerType === "mouse" && ev.button !== 0) return;
      ev.preventDefault();
      try { b.setPointerCapture(ev.pointerId); } catch (e) {}
      press(key, "p" + ev.pointerId);
    });
    const end = (ev) => release("p" + ev.pointerId);
    b.addEventListener("pointerup", end); b.addEventListener("pointercancel", end); b.addEventListener("lostpointercapture", end);
    b.addEventListener("contextmenu", (ev) => ev.preventDefault());
  }
  window.addEventListener("keydown", (e) => {
    // With Meta (⌘) down, macOS browsers do not report key releases: the keys' holds end (input loss).
    if (e.key === "Meta" || e.key === "OS") {
      const m = usable();
      if (m) moQueue(m, () => moRelease(m, (src) => src.startsWith("k"), "keyboard input lost: key releases are not reported while ⌘ is held"));
      return;
    }
    const mo = S.mo, k = keyName(e);
    if (!k || !mo || mo.dead || e.isComposing || e.ctrlKey || e.metaKey || e.altKey) return;
    const isMotion = mo.c.motion_keys.includes(k), isHead = mo.c.head_keys.includes(k);
    if (!(isMotion || isHead || k === "space" || k === "enter")) return;
    const el = e.target;
    if (isEditable(el)) return;                                     // typing moves nothing
    if (isHead && (!mo.c.head || e.defaultPrevented || ownsArrows(el))) return;
    if (k === "enter" && el && el.closest && el.closest("button, a, summary, [role=button]")) return;
    e.preventDefault();                                             // no scrolling, no button activation
    if (e.repeat) return;                                           // key repeat is not a fresh press
    if (k === "space") stopNow("Space");
    else if (k === "enter") reenable();
    else press(k, "k" + k);
  });
  window.addEventListener("keyup", (e) => {
    const mo = S.mo, k = keyName(e);
    if (!k || !mo || mo.dead) return;
    if (k === "space" && !isEditable(e.target)) e.preventDefault();
    if (mo.c.motion_keys.includes(k) || mo.c.head_keys.includes(k)) release("k" + k);
  });
  // Focus moving to an editable field: the keys no longer act, so their holds end.
  document.addEventListener("focusin", (e) => {
    const mo = usable();
    if (mo && isEditable(e.target)) moQueue(mo, () => moRelease(mo, (src) => src.startsWith("k"), "keys released: keyboard focus moved to an editable field"));
  });
  window.addEventListener("blur", () => { const mo = usable(); if (mo) moQueue(mo, () => moClearAndStop(mo, "window lost keyboard focus")); });
  document.addEventListener("visibilitychange", () => {
    const mo = usable();
    if (mo && document.visibilityState === "hidden") moQueue(mo, () => moClearAndStop(mo, "page hidden"));
  });
  window.addEventListener("pagehide", () => moUnload(usable(), "page hide or unload"));
  setInterval(() => {
    const mo = S.mo;
    if (mo && !mo.dead && mo.connected && mo.enabled && !mo.pending && (mo.held.length || mo.heldHead.length || mo.headLastAt != null)) moQueue(mo, () => moTick(mo));
  }, 1000 / 30);

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
    stopClock(S.clock); S.clock = null;
    setConn("ok", "connected");
    setValidation("pending", "Reading the wire…", "Discovering and type-checking the interface through rosapi", "");
    try { S.graph = await fetchGraph(conn); }
    catch (e) { setValidation("bad", "Cannot read the wire", "", "cannot read the wire through rosapi: " + e.message); renderControls(); return; }
    S.clock = makeClock(conn, S.graph.dialect);           // read-only rosapi time calls, for each frame's lag
    clearCameras(); clearFeedback();
    try {
      S.target = selectTarget(S.graph, S.profiles, robotId || null, ns == null || ns === "" ? null : ns);
    } catch (e) {
      S.target = null;
      setValidation("refused", "No validated target — controls unavailable",
        "Select a robot (and namespace, where supported) and press ‘Select target’.",
        (e.reason || e.message) + ".");
      $("target-pill").dataset.state = "warn";
      for (const [n, ty] of Object.entries(S.graph.topics).filter(([, ty]) => normType(ty) === "sensor_msgs/Image").sort()) addCamera(n, ty, UNTIED_STALE_S, null, false, false, false, { throttle: UNTIED_THROTTLE_MS, onDemand: true });
      renderControls(); updateLock();
      return;
    }
    const t = S.target;
    $("target").textContent = t.label; $("target-pill").dataset.state = "ok";
    $("robot").value = t.profile.id; $("namespace").value = t.namespace; syncNsField(); $("namespace").value = t.namespace;
    setValidation("ok", `${t.label} validated`,
      `${t.profile.name} · ${t.profile.dialect === "ros2" ? "ROS 2" : "ROS 1"} · typed interface matches the packaged profile`, "");
    for (const c of t.profile.cameras) addCamera(t.wire(c.topic), c.type, c.stale_after_s, c.encodings, true, !(t.wire(c.topic) in S.graph.topics), c.optional, {
      depth: c.kind === "depth" ? { unit: c.unit_m, range: c.range_m } : null, kind: c.kind, info: c.camera_info ? t.wire(c.camera_info) : null,
      note: c.missing_note, throttle: c.throttle_ms, onDemand: c.on_demand, silent: c.silent_note, caption: c.caption });
    for (const sd of t.profile.sensors || []) addSensor(t, sd);
    if (t.profile.teleop) S.mo = makeMotion(t, conn);             // base motion: shown at once, enabled by the start-up stop
    renderControls(); updateLock();
    startPoll(t, conn);                                         // validated: the read-only position poll
    if (S.mo && S.mo.t === t) moQueue(S.mo, () => moStartup(S.mo));   // validated: the start-up stop
  }

  // Connection loss sends nothing: the controls are disabled and the page stays disconnected
  // until it is reloaded. Goals already sent keep running; base motion's intent is cleared and
  // no stop could be delivered.
  function onLost(conn) {
    if (conn !== S.conn) return;
    S.lost = true; stopPoll(); stopClock(S.clock);
    const mo = S.mo && S.mo.conn === conn && !S.mo.dead ? S.mo : null;
    if (mo) moConnectionLost(mo, "the websocket closed");
    setConn("bad", "DISCONNECTED - reload the page to connect again");
    if (S.target) $("target-pill").dataset.state = "bad";
    showBanner(`Connection to ${conn.url} lost. Controls disabled; nothing was sent and nothing was stopped. Reload the page to connect again.` +
      (mo ? ` Base motion: ${mo.c.no_delivery}` : ""));
    refreshControls(); updateLock();
  }

  // A target change validates afresh on a new connection. Base motion on the old target is
  // stopped before leaving it (its outcome is shown if it failed); nothing else is sent to it.
  async function changeTarget() {
    if (S.lost || S.changing) return;
    S.changing = true; updateLock();
    try {
      const old = S.conn;
      const out = await moLeave(S.mo, "target change");
      if (S.lost) return;
      S.mo = null; renderMotion();
      stopPoll(); retireControls(); showBanner(out && !out.ok ? `Previous target ${out.text}` : "");
      clearCameras(); clearFeedback(); S.controls.clear(); S.padGroup = null; S.groups = []; $("controls").innerHTML = "";
      if (old) setTimeout(() => old.close(), 300);
      const id = $("robot").value || null; const nsRaw = $("namespace").value.trim();
      await connectAndSelect(id, nsAllowed() ? (nsRaw === "" ? null : nsRaw) : null);
    } finally { S.changing = false; updateLock(); }
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
  // No arm switch, Esc or stop/cancel of a bounded control: apart from base motion (above), the
  // page sends only what the user changes.
  $("select").addEventListener("click", changeTarget);
  $("resume-cameras").addEventListener("click", resumeCameras);
  $("resume-cameras-drive").addEventListener("click", resumeCameras);
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
    tick();                     // the camera states (or "no cameras") laid out now, not on the next tick
    RC.ready = true;
  }
  boot().catch((e) => { setValidation("bad", "Page error", "", "page error: " + e.message); RC.ready = true; });
})();
